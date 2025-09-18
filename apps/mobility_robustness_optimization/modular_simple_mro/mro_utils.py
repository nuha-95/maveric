import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from gpytorch.utils.warnings import NumericalWarning

from notebooks.radp_library import (
    add_cell_info,
    calc_log_distance,
    calc_relative_bearing,
    check_cartesian_format,
    get_percell_data,
    normalize_cell_ids,
    preprocess_ue_data,
)
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, NormMethod
from radp.digital_twin.utils import constants
from radp.digital_twin.utils.cell_selection import perform_attachment

# Suppress the specific NumericalWarning from gpytorch
warnings.filterwarnings("ignore", category=NumericalWarning)

logger = logging.getLogger(__name__)


def train_or_update_rf_twins(bayesian_digital_twins: dict, topology: pd.DataFrame, new_data: pd.DataFrame, device: torch.device) -> dict:
    """
    Updates the Bayesian Digital Twins with new observations if they exist.
    If not, it trains new twins from scratch.
    """
    try:
        if not isinstance(new_data, pd.DataFrame):
            logger.error("The input 'new_data' must be a pandas DataFrame.")

        expected_columns = {"longitude", "latitude", "cell_id", "cell_rxpwr_dbm"}
        if not expected_columns.issubset(new_data.columns):
            logger.error(f"The input DataFrame must contain the following columns: {expected_columns}")

        # normalize cell_id format - regardless of dtype
        topology = normalize_cell_ids(topology)
        new_data = normalize_cell_ids(new_data)

        # Check if the new data is in the expected cartesian format
        check_cartesian_format(new_data, topology)

        # Prepare the new data for training or updating
        prepared_data = _prepare_train_or_update_data(new_data, topology)

        # update if bayesian digital twins exist already
        if bayesian_digital_twins:
            logger.info("Updating existing Bayesian Digital Twins with new data.")

            for cell_id, df in prepared_data.items():
                _update(bayesian_digital_twins, cell_id, df)
            logger.info("Bayesian Digital Twins updated successfully.")

        # If no Bayesian Digital Twins exist, train from scratch
        else:
            logger.info("No Bayesian Digital Twins available for update. Training from scratch.")
            bayesian_digital_twins = _training(maxiter=100, train_data=prepared_data, device=device)
            logger.info("\nBayesian Digital Twins trained successfully.")

        return bayesian_digital_twins

    except TypeError as te:
        logger.error(f"TypeError: {te}")
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
    except KeyError as ke:
        logger.error(f"KeyError: {ke}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


def _training(maxiter: int, train_data: Dict[str, pd.DataFrame], device: torch.device) -> dict:
    """Trains the Bayesian Digital Twins for each cell in the topology."""
    bayesian_digital_twins = {}

    for train_cell_id, training_data_idx in train_data.items():
        bayesian_digital_twins[train_cell_id] = BayesianDigitalTwin(
            data_in=[training_data_idx],
            x_columns=["log_distance", "relative_bearing"],
            y_columns=["cell_rxpwr_dbm"],
            norm_method=NormMethod.MINMAX,
        )

        bayesian_digital_twins[train_cell_id].model = bayesian_digital_twins[train_cell_id].model.to(device)

        bayesian_digital_twins[train_cell_id].train_distributed_gpmodel(maxiter=maxiter)

    return bayesian_digital_twins


def _update(bayesian_digital_twins: dict, cell_id: str, df: pd.DataFrame) -> None:
    """Updates the Bayesian Digital Twin (BDT) model for a specific cell."""
    # Remove near-duplicates in feature space
    df = df.drop_duplicates(subset=["log_distance", "relative_bearing"])

    # Subsample to at most 500 strongest samples per cell
    if df.shape[0] > 500:
        df = get_percell_data(data_in=df, choose_strongest_samples_percell=True, n_samples=500,)[0][0]

    twin = bayesian_digital_twins[cell_id]
    twin.update_trained_gpmodel([df])


def _prepare_train_or_update_data(df: pd.DataFrame, topology: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Returns key value pairs of cell_id and processed DataFrame for each cell_id."""
    required_columns = {"cell_lat", "cell_lon", "cell_az_deg"}
    if not required_columns.issubset(df.columns):
        df = add_cell_info(df, topology)

    update_data = calc_log_distance(df)
    update_data = calc_relative_bearing(update_data)

    update_data = update_data.loc[:, ["cell_id", "log_distance", "relative_bearing", "cell_rxpwr_dbm"]]

    # anything refering as training indicates training or update data
    train_per_cell_df = [x for _, x in update_data.groupby("cell_id")]
    n_cell = len(topology.index)

    metadata_df = pd.DataFrame(
        {
            "cell_id": [cell_id for cell_id in topology.cell_id],
            "idx": [i + 1 for i in range(n_cell)],
        }
    )

    idx_cell_id_mapping = dict(zip(metadata_df.idx, metadata_df.cell_id))
    n_samples_train = []

    for df in train_per_cell_df:
        n_samples_train.append(df.shape[0])

    train_per_cell_df_processed = []
    for i in range(n_cell):
        train_per_cell_df_processed.append(
            get_percell_data(
                data_in=train_per_cell_df[i],
                choose_strongest_samples_percell=False,
                n_samples=n_samples_train[i],
            )[0][0]
        )

    training_data = {}

    for i, df in enumerate(train_per_cell_df_processed):
        train_cell_id = idx_cell_id_mapping[i + 1]
        training_data[train_cell_id] = df

    return training_data


def predictions(bayesian_digital_twins: dict, pred_data: pd.DataFrame, topology: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Predicts the received power for each User Equipment (UE) at different locations and ticks using Bayesian Digital Twins."""
    prediction_data = preprocess_ue_data(pred_data, topology)
    prediction_data = calc_relative_bearing(prediction_data)
    full_prediction_df = pd.DataFrame()

    # Loop over each 'tick'
    for tick, tick_df in prediction_data.groupby("tick"):
        # Loop over each 'cell_id' within the current 'tick'
        for cell_id, cell_df in tick_df.groupby("cell_id"):
            cell_id = f"cell_{cell_id}"  # FIXME: should look better
            # Check if the Bayesian model for this cell_id exists
            if cell_id in bayesian_digital_twins:
                # Perform the Bayesian prediction
                pred_means_percell, _ = bayesian_digital_twins[cell_id].predict_distributed_gpmodel(
                    prediction_dfs=[cell_df]
                )

                # Assuming 'pred_means_percell' returns a list of predictions corresponding to the DataFrame index
                cell_df["pred_means"] = pred_means_percell[0]

                # Include additional necessary columns for the final DataFrame
                cell_df["tick"] = tick
                cell_df["cell_id"] = cell_id

                # Append the predictions to the full DataFrame
                full_prediction_df = pd.concat([full_prediction_df, cell_df], ignore_index=True)

            else:
                # Handle missing models, e.g., log a warning or initialize a default model
                logger.error(f"No model available for cell_id {cell_id}, skipping prediction.")

    full_prediction_df = full_prediction_df.rename(columns={"latitude": "loc_y", "longitude": "loc_x"})
    if full_prediction_df["cell_id"].dtype == object:
        full_prediction_df["cell_id"] = full_prediction_df["cell_id"].str.extract(r"(\d+)").astype(int)
    predicted = perform_attachment(full_prediction_df, topology)
    if full_prediction_df["cell_id"].dtype == int:
        full_prediction_df["cell_id"] = full_prediction_df["cell_id"].apply(lambda x: f"cell_{x}")
    return predicted, full_prediction_df


def preprocess_simulation_data(df: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    """Preprocess simulation data for MRO analysis."""
    df.drop(
        columns=["rxpower_stddev_dbm", "rxpower_dbm", "cell_rxpwr_dbm"],
        inplace=True,
        errors='ignore'
    )
    df.rename(
        columns={
            "mock_ue_id": "ue_id",
            "log_distance": "distance_km",
            "pred_means": "cell_rxpower_dbm",
        },
        inplace=True,
    )
    if topology["cell_id"].dtype == object:
        topology["cell_id"] = topology["cell_id"].str.replace("cell_", "").astype(int)
    if df["cell_id"].dtype == object:
        df["cell_id"] = df["cell_id"].str.extract(r"(\d+)").astype(int)
    df = add_sinr_column(df, topology)
    return df


def add_sinr_column(df: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'sinr_db' column to the input DataFrame."""
    df = df.copy()
    sinr_column = []

    # Group by location
    for _, group in df.groupby(["ue_id", "tick"]):

        # Group further by frequency layer within the same location
        freq_groups = group.groupby("cell_carrier_freq_mhz")

        # Create a temporary Series to store sinr values for current group
        group_sinr_values = pd.Series(index=group.index, dtype=float)

        for freq, freq_group in freq_groups:
            # List of all rx powers in this frequency group
            all_rxpowers = freq_group["cell_rxpower_dbm"].tolist()
            noise_db = constants.LATENT_BACKGROUND_NOISE_DB

            for idx, row in freq_group.iterrows():
                serving_power = row["cell_rxpower_dbm"]
                # Remove this row's signal from interference
                interference_others = [p for p in all_rxpowers if p != serving_power or all_rxpowers.count(p) > 1]
                sinr_db = compute_row_level_sinr(serving_power, interference_others, noise_db)
                group_sinr_values.at[idx] = sinr_db

        sinr_column.append(group_sinr_values)

    # Combine all the sinr values and add to DataFrame
    df["sinr_db"] = pd.concat(sinr_column).sort_index()

    return df


def compute_row_level_sinr(signal_dbm: float, interference_dbm_list: list, noise_db: float) -> float:
    """Computes the SINR for a single UE–cell pair by removing interference and noise from the received signal power."""
    signal_linear = 10 ** (signal_dbm / 10)
    interference_linear = sum(10 ** (p / 10) for p in interference_dbm_list)
    noise_linear = 10 ** (noise_db / 10)

    sinr_linear = signal_linear / (interference_linear + noise_linear)
    return 10 * np.log10(sinr_linear)


def count_handovers(df: pd.DataFrame) -> int:
    """Count the number of seamless cell handovers (cell to cell switches) for user equipment (UE)."""
    count = 0
    df = df.sort_values(by=["ue_id", "tick"])  # Ensure correct order
    prev_cells = {}
    prev_ticks = {}

    for _, row in df.iterrows():
        ue_id, cell_id, tick = row["ue_id"], row["cell_id"], row["tick"]

        if ue_id in prev_cells and prev_cells[ue_id] != cell_id and prev_cells[ue_id] is not None:
            if tick == prev_ticks[ue_id] + 1 and cell_id != "RLF":
                count += 1

        prev_cells[ue_id] = cell_id
        prev_ticks[ue_id] = tick
    return count


def count_rlf(df: pd.DataFrame) -> int:
    """Counts the number of Radio Link Failures (RLF) by analyzing cell handovers onto RLF for UE."""
    return (df["cell_id"] == "RLF").sum()


def calculate_mro_metric(data: pd.DataFrame) -> float:
    """Calculated total operational cellular time remaining after loss due to cell handovers (including RLF)."""
    # Constants for interruption times
    ts = 50 / 1000  # Convert ms to seconds
    t_nas = 1000 / 1000  # Convert ms to seconds

    # Calculate total time (T) based on ticks
    ticks = len(data["tick"].unique())
    tick_duration_seconds = 1  # 1 second per tick
    T = ticks * tick_duration_seconds

    ns_handover_count = count_handovers(data)  # Count of handovers to different cells (excluding RLF)
    nf_handover_count = count_rlf(data)  # Count of handovers to RLF

    # Calculate D
    D = T - (ns_handover_count * ts + nf_handover_count * t_nas)

    return D