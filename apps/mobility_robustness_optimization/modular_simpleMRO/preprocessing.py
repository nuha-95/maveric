import pandas as pd
import numpy as np
import sys
import os
import argparse
# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)
from notebooks.radp_library import find_sim_boundary, get_ue_data, preprocess_ue_data, calc_relative_bearing
from radp.digital_twin.utils.cell_selection import perform_attachment
from radp.digital_twin.utils import constants


def prepare_simulation_data(mobility_model_params, topology, new_data):
    """Prepare simulation data by determining boundaries and generating UE data."""
    bounds = find_sim_boundary(topology, new_data)
    mobility_model_params["ue_tracks_generation"]["params"]["lat_lon_boundaries"].update(bounds)
    
    simulation_data = get_ue_data(mobility_model_params)
    simulation_data = simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})
    
    if topology["cell_id"].dtype == int:
        topology["cell_id"] = topology["cell_id"].apply(lambda x: f"cell_{int(x)}")
    
    return simulation_data


def preprocess_prediction_data(pred_data, topology):
    """Preprocess prediction data for RF prediction."""
    # The MRO data already has cell_id, longitude, latitude, cell_rxpwr_dbm
    # We need to add topology info and create the cartesian format
    
    # Ensure topology has the right format
    if topology["cell_id"].dtype == object and not topology["cell_id"].str.startswith("cell_").all():
        topology = topology.copy()
        topology["cell_id"] = topology["cell_id"].apply(lambda x: f"cell_{x}" if not str(x).startswith("cell_") else str(x))
    
    # Convert pred_data cell_id to match topology format
    pred_data = pred_data.copy()
    if pred_data["cell_id"].dtype != object:
        pred_data["cell_id"] = pred_data["cell_id"].apply(lambda x: f"cell_{x}")
    
    # Add topology information to prediction data
    prediction_data = pred_data.merge(
        topology[["cell_id", "cell_lat", "cell_lon", "cell_az_deg", "cell_carrier_freq_mhz"]], 
        on="cell_id", 
        how="left"
    )
    
    # Add required columns for RF prediction
    from radp.digital_twin.utils.gis_tools import GISTools
    
    prediction_data["log_distance"] = prediction_data.apply(
        lambda row: GISTools.get_log_distance(
            row["cell_lat"], row["cell_lon"], row["latitude"], row["longitude"]
        ), axis=1
    )
    
    prediction_data["relative_bearing"] = prediction_data.apply(
        lambda row: GISTools.get_relative_bearing(
            row["cell_az_deg"], row["cell_lat"], row["cell_lon"], 
            row["latitude"], row["longitude"]
        ), axis=1
    )
    
    return prediction_data


def preprocess_simulation_data(df, topology):
    """Preprocess simulation data for MRO analysis."""
    df.drop(
        columns=["rxpower_stddev_dbm", "rxpower_dbm", "cell_rxpwr_dbm"],
        inplace=True,
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
    
    return df


def perform_attachment_prediction(full_prediction_df, topology):
    """Perform attachment prediction on the full prediction dataframe."""
    full_prediction_df = full_prediction_df.rename(columns={"latitude": "loc_y", "longitude": "loc_x"})
    if full_prediction_df["cell_id"].dtype == object:
        full_prediction_df["cell_id"] = full_prediction_df["cell_id"].str.extract(r"(\d+)").astype(int)
    
    predicted = perform_attachment(full_prediction_df, topology)
    
    if full_prediction_df["cell_id"].dtype == int:
        full_prediction_df["cell_id"] = full_prediction_df["cell_id"].apply(lambda x: f"cell_{x}")
    
    return predicted, full_prediction_df


def add_sinr_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add SINR column to the dataframe."""
    df = df.copy()
    
    # Check which columns exist in the dataframe
    ue_col = "mock_ue_id" if "mock_ue_id" in df.columns else "ue_id"
    power_col = "predicted_rxpower_dbm" if "predicted_rxpower_dbm" in df.columns else ("pred_means" if "pred_means" in df.columns else "cell_rxpower_dbm")
    
    # If required columns don't exist, return original dataframe
    if ue_col not in df.columns or "tick" not in df.columns or power_col not in df.columns:
        print(f"Warning: Required columns not found. Available columns: {df.columns.tolist()}")
        # For MRO data without tick/ue_id, create simple SINR based on power differences
        if power_col in df.columns:
            df["sinr_db"] = df[power_col] + 10  # Simple approximation
        return df
    
    sinr_column = []

    for _, group in df.groupby([ue_col, "tick"]):
        # Use a default frequency if cell_carrier_freq_mhz doesn't exist
        if "cell_carrier_freq_mhz" in df.columns:
            freq_groups = group.groupby("cell_carrier_freq_mhz")
        else:
            freq_groups = [(2100, group)]  # Default frequency
            
        group_sinr_values = pd.Series(index=group.index, dtype=float)

        for freq, freq_group in freq_groups:
            all_rxpowers = freq_group[power_col].tolist()
            noise_db = constants.LATENT_BACKGROUND_NOISE_DB

            for idx, row in freq_group.iterrows():
                serving_power = row[power_col]
                interference_others = [p for p in all_rxpowers if p != serving_power or all_rxpowers.count(p) > 1]
                sinr_db = compute_row_level_sinr(serving_power, interference_others, noise_db)
                group_sinr_values.at[idx] = sinr_db

        sinr_column.append(group_sinr_values)

    df["sinr_db"] = pd.concat(sinr_column).sort_index()
    return df


def compute_row_level_sinr(signal_dbm: float, interference_dbm_list: list, noise_db: float) -> float:
    """Compute SINR for a single UE-cell pair."""
    signal_linear = 10 ** (signal_dbm / 10)
    interference_linear = sum(10 ** (p / 10) for p in interference_dbm_list)
    noise_linear = 10 ** (noise_db / 10)
    
    sinr_linear = signal_linear / (interference_linear + noise_linear)
    return 10 * np.log10(sinr_linear)


def main():
    """Command line interface for preprocessing."""
    parser = argparse.ArgumentParser(description='Data preprocessing for SimpleMRO')
    parser.add_argument('--topology', required=True, help='Path to topology CSV file')
    parser.add_argument('--ue-data', required=True, help='Path to UE data CSV file')
    parser.add_argument('--output', required=True, help='Path to output processed data CSV file')
    parser.add_argument('--add-sinr', action='store_true', help='Add SINR column to processed data')
    
    args = parser.parse_args()
    
    # Load data
    topology = pd.read_csv(args.topology)
    ue_data = pd.read_csv(args.ue_data)
    
    # Process data
    processed_data = preprocess_prediction_data(ue_data, topology)
    
    # Add tick column if missing (required for MRO)
    if 'tick' not in processed_data.columns:
        processed_data['tick'] = 0  # Single time point
    
    # Add ue_id column if missing (required for MRO)
    if 'ue_id' not in processed_data.columns and 'mock_ue_id' not in processed_data.columns:
        # Create UE IDs based on unique location combinations
        processed_data['ue_id'] = processed_data.groupby(['longitude', 'latitude']).ngroup()
    
    # Add SINR if requested
    if args.add_sinr:
        processed_data = add_sinr_column(processed_data)
        print("SINR column added to processed data")
    
    # Save processed data
    processed_data.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}")
    print(f"Data shape: {processed_data.shape}")
    print(f"Columns: {processed_data.columns.tolist()}")


if __name__ == "__main__":
    main()