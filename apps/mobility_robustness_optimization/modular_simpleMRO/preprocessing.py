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
    """Preprocess prediction data for RF prediction using the same approach as simple_mro.py"""
    # Use the same preprocessing as in MRO system
    prediction_data = preprocess_ue_data(pred_data, topology)
    prediction_data = calc_relative_bearing(prediction_data)
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
    """Add SINR column using the exact same method as simple_mro.py _add_sinr_column"""
    df = df.copy()
    sinr_column = []

    # Group by location (same as simple_mro.py)
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
    """Compute SINR for a single UE-cell pair."""
    signal_linear = 10 ** (signal_dbm / 10)
    interference_linear = sum(10 ** (p / 10) for p in interference_dbm_list)
    noise_linear = 10 ** (noise_db / 10)
    
    sinr_linear = signal_linear / (interference_linear + noise_linear)
    return 10 * np.log10(sinr_linear)


def simulate_mro_preprocessing(topology, new_data, mobility_model_params, trained_models):
    """Simulate the exact preprocessing workflow from simple_mro.py"""
    from notebooks.radp_library import find_sim_boundary, get_ue_data
    from radp.digital_twin.utils.cell_selection import perform_attachment
    
    # Step 1: Determine simulation boundaries (same as simple_mro.py)
    bounds = find_sim_boundary(topology, new_data)
    mobility_model_params["ue_tracks_generation"]["params"]["lat_lon_boundaries"].update(bounds)
    
    # Step 2: Generate and preprocess simulation data (same as simple_mro.py)
    simulation_data = get_ue_data(mobility_model_params)
    simulation_data = simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})
    
    if topology["cell_id"].dtype == int:
        topology["cell_id"] = topology["cell_id"].apply(lambda x: f"cell_{int(x)}")
    
    # Step 3: Predict power and perform attachment (same as simple_mro.py _predictions method)
    prediction_data = preprocess_ue_data(simulation_data, topology)
    prediction_data = calc_relative_bearing(prediction_data)
    full_prediction_df = pd.DataFrame()
    
    # Loop over each 'tick'
    for tick, tick_df in prediction_data.groupby("tick"):
        # Loop over each 'cell_id' within the current 'tick'
        for cell_id, cell_df in tick_df.groupby("cell_id"):
            cell_id_str = f"cell_{cell_id}"
            # Check if the Bayesian model for this cell_id exists
            if cell_id_str in trained_models:
                # Perform the Bayesian prediction
                pred_means_percell, _ = trained_models[cell_id_str].predict_distributed_gpmodel(
                    prediction_dfs=[cell_df]
                )
                
                # Add predictions
                cell_df["pred_means"] = pred_means_percell[0]
                cell_df["tick"] = tick
                cell_df["cell_id"] = cell_id_str
                
                # Append the predictions to the full DataFrame
                full_prediction_df = pd.concat([full_prediction_df, cell_df], ignore_index=True)
    
    # Step 4: Perform attachment
    full_prediction_df = full_prediction_df.rename(columns={"latitude": "loc_y", "longitude": "loc_x"})
    if full_prediction_df["cell_id"].dtype == object:
        full_prediction_df["cell_id"] = full_prediction_df["cell_id"].str.extract(r"(\d+)").astype(int)
    
    predicted = perform_attachment(full_prediction_df, topology)
    
    if full_prediction_df["cell_id"].dtype == int:
        full_prediction_df["cell_id"] = full_prediction_df["cell_id"].apply(lambda x: f"cell_{x}")
    
    # Step 5: Preprocess simulation data (same as simple_mro.py _preprocess_simulation_data)
    simulation_data = full_prediction_df
    simulation_data.drop(
        columns=["rxpower_stddev_dbm", "rxpower_dbm", "cell_rxpwr_dbm"],
        inplace=True,
        errors='ignore'
    )
    simulation_data.rename(
        columns={
            "mock_ue_id": "ue_id",
            "log_distance": "distance_km",
            "pred_means": "cell_rxpower_dbm",
        },
        inplace=True,
    )
    if topology["cell_id"].dtype == object:
        topology["cell_id"] = topology["cell_id"].str.replace("cell_", "").astype(int)
    if simulation_data["cell_id"].dtype == object:
        simulation_data["cell_id"] = simulation_data["cell_id"].str.extract(r"(\d+)").astype(int)
    
    # Step 6: Add SINR column (same as simple_mro.py _add_sinr_column)
    simulation_data = add_sinr_column(simulation_data)
    
    return simulation_data


def main():
    """Command line interface for preprocessing."""
    parser = argparse.ArgumentParser(description='Data preprocessing for SimpleMRO')
    parser.add_argument('--topology', required=True, help='Path to topology CSV file')
    parser.add_argument('--ue-data', required=True, help='Path to UE data CSV file')
    parser.add_argument('--trained-model', required=True, help='Path to trained model pickle file')
    parser.add_argument('--output', required=True, help='Path to output processed data CSV file')
    
    args = parser.parse_args()
    
    # Load data
    topology = pd.read_csv(args.topology)
    new_data = pd.read_csv(args.ue_data)
    
    # Load trained models
    import pickle
    with open(args.trained_model, 'rb') as f:
        trained_models = pickle.load(f)
    
    # Create mobility model params (same as notebook)
    mobility_model_params = {
        "ue_tracks_generation": {
            "params": {
                "simulation_duration": 3600,
                "simulation_time_interval_seconds": 0.01,
                "num_ticks": 50,
                "num_batches": 1,
                "ue_class_distribution": {
                    "stationary": {"count": 10, "velocity": 0, "velocity_variance": 1},
                    "pedestrian": {"count": 5, "velocity": 2, "velocity_variance": 1},
                    "cyclist": {"count": 5, "velocity": 5, "velocity_variance": 1},
                    "car": {"count": 12, "velocity": 20, "velocity_variance": 1}
                },
                "lat_lon_boundaries": {
                    "min_lat": -90, "max_lat": 90, "min_lon": -180, "max_lon": 180
                },
                "gauss_markov_params": {
                    "alpha": 0.5, "variance": 0.8, "rng_seed": 42,
                    "lon_x_dims": 100, "lon_y_dims": 100
                }
            }
        }
    }
    
    # Process data using the exact same workflow as simple_mro.py
    processed_data = simulate_mro_preprocessing(topology, new_data, mobility_model_params, trained_models)
    
    # Save processed data
    processed_data.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}")
    print(f"Data shape: {processed_data.shape}")
    print(f"Columns: {processed_data.columns.tolist()}")


if __name__ == "__main__":
    main()