import pandas as pd
import numpy as np
import sys
import os
import argparse
import pickle
# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)
from notebooks.radp_library import find_sim_boundary, get_ue_data
from apps.mobility_robustness_optimization.mobility_robustness_optimization import MobilityRobustnessOptimization


class Preprocessor(MobilityRobustnessOptimization):
    """Simple preprocessor that inherits from MobilityRobustnessOptimization like SimpleMRO"""
    
    def __init__(self, mobility_model_params, topology, new_data=None, bdt=None):
        super().__init__(mobility_model_params, topology, new_data, bdt)
    
    def solve(self):
        """Required abstract method - just calls preprocess_data"""
        return self.preprocess_data()
    
    def preprocess_data(self):
        """Use the exact same preprocessing steps as SimpleMRO.solve()"""
        
        # Ensure we have trained models
        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not available. Need trained models for preprocessing.")
        
        bounds = find_sim_boundary(self.topology, self.new_data)
        self.mobility_model_params["ue_tracks_generation"]["params"]["lat_lon_boundaries"].update(bounds)
        
        self.simulation_data = get_ue_data(self.mobility_model_params)
        self.simulation_data = self.simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})
        
        if self.topology["cell_id"].dtype == int:
            self.topology["cell_id"] = self.topology["cell_id"].apply(lambda x: f"cell_{int(x)}")
        
        predictions, full_prediction_df = self._predictions(self.simulation_data)
        self.simulation_data = full_prediction_df
        self.simulation_data = self._preprocess_simulation_data(self.simulation_data)
        
        return self.simulation_data


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
    with open(args.trained_model, 'rb') as f:
        bdt_models = pickle.load(f)
    
    
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
    
    # Create preprocessor and process data
    preprocessor = Preprocessor(mobility_model_params, topology, new_data, bdt_models)
    processed_data = preprocessor.preprocess_data()
    
    # Save processed data
    processed_data.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}")
    print(f"Data shape: {processed_data.shape}")
    print(f"Columns: {processed_data.columns.tolist()}")


if __name__ == "__main__":
    main()