import pandas as pd
import sys
import os
import argparse
import pickle
# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin


def train_bdt_models(topology, ue_training_data):
    """Train Bayesian Digital Twin models for each cell"""
    from notebooks.radp_library import add_cell_info, calc_log_distance, calc_relative_bearing, get_percell_data, normalize_cell_ids
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, NormMethod
    
    # Normalize cell IDs
    topology = normalize_cell_ids(topology)
    ue_training_data = normalize_cell_ids(ue_training_data)
    
    # Add cell info if missing
    required_columns = {"cell_lat", "cell_lon", "cell_az_deg"}
    if not required_columns.issubset(ue_training_data.columns):
        ue_training_data = add_cell_info(ue_training_data, topology)
    
    # Calculate features
    update_data = calc_log_distance(ue_training_data)
    update_data = calc_relative_bearing(update_data)
    update_data = update_data.loc[:, ["cell_id", "log_distance", "relative_bearing", "cell_rxpwr_dbm"]]
    
    # Group by cell and train models
    trained_models = {}
    for cell_id, cell_data in update_data.groupby("cell_id"):
        print(f"Training BDT model for {cell_id}...")
        
        # Process data for this cell
        processed_data = get_percell_data(
            data_in=cell_data,
            choose_strongest_samples_percell=False,
            n_samples=cell_data.shape[0]
        )[0][0]
        
        # Create and train BDT model
        bdt = BayesianDigitalTwin(
            data_in=[processed_data],
            x_columns=["log_distance", "relative_bearing"],
            y_columns=["cell_rxpwr_dbm"],
            norm_method=NormMethod.MINMAX,
        )
        
        # Train the model
        bdt.train_distributed_gpmodel(maxiter=100)
        
        trained_models[cell_id] = bdt
        print(f"Completed training for {cell_id}")
    
    return trained_models


def main():
    """Command line interface for BDT training."""
    parser = argparse.ArgumentParser(description='Train Bayesian Digital Twin models')
    parser.add_argument('--topology', required=True, help='Path to topology CSV file')
    parser.add_argument('--training-data', required=True, help='Path to UE training data CSV file')
    parser.add_argument('--output', required=True, help='Path to output trained models pickle file')
    
    args = parser.parse_args()
    
    # Load data
    topology = pd.read_csv(args.topology)
    ue_training_data = pd.read_csv(args.training_data)
    
    # Train BDT models
    trained_models = train_bdt_models(topology, ue_training_data)
    
    # Save trained models
    with open(args.output, 'wb') as f:
        pickle.dump(trained_models, f)
    
    print(f"Trained BDT models saved to {args.output}")
    print(f"Trained models for cells: {list(trained_models.keys())}")


if __name__ == "__main__":
    main()