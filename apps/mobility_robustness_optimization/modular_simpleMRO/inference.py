import logging
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import sys
import os
import argparse
import pickle
import json

# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)

from preprocessing import Preprocessor
import numpy as np

# Add parent directory to path for MRO imports
sys.path.insert(0, os.path.join(maveric_root, 'apps', 'mobility_robustness_optimization'))
from mobility_robustness_optimization import calculate_mro_metric
from radp.digital_twin.utils.cell_selection import perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD

logger = logging.getLogger(__name__)


class SimpleMROInference:
    """Inference module for SimpleMRO - handles RF prediction and MRO optimization inference."""

    def __init__(self, topology: pd.DataFrame, trained_models: dict, optimal_hyst: float = None, optimal_ttt: int = None):
        self.topology = topology
        self.trained_models = trained_models
        self.optimal_hyst = optimal_hyst
        self.optimal_ttt = optimal_ttt
        self.logger = logging.getLogger(__name__)

    def get_preprocessed_data(self, ue_data: pd.DataFrame = None) -> pd.DataFrame:
        """Get preprocessed simulation data using Preprocessor."""
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
        
        preprocessor = Preprocessor(mobility_model_params, self.topology, ue_data, self.trained_models)
        return preprocessor.preprocess_data()

    def mro_inference(self, ue_data: pd.DataFrame = None, n_epochs: int = 5) -> dict:
        """Perform MRO inference using optimal parameters or optimization."""
        # Get preprocessed simulation data
        simulation_data = self.get_preprocessed_data(ue_data)
        
        # Use optimal parameters if available, otherwise optimize
        if self.optimal_hyst is not None and self.optimal_ttt is not None:
            print(f"Using pre-trained optimal parameters: Hyst={self.optimal_hyst:.6f}, TTT={self.optimal_ttt}")
            attached_df = perform_attachment_hyst_ttt(simulation_data, self.optimal_hyst, self.optimal_ttt, RLF_THRESHOLD)
            mro_metric = calculate_mro_metric(attached_df)
            print(f"MRO Metric Score: {mro_metric:.6f}")
            
            return {
                'hysteresis': self.optimal_hyst,
                'ttt': self.optimal_ttt,
                'mro_metric': mro_metric,
                'attached_data': attached_df,
                'all_scores': [{'hyst': self.optimal_hyst, 'ttt': self.optimal_ttt, 'score': mro_metric}]
            }
        else:
            print("No optimal parameters found. Running optimization...")
            # Initialize optimization parameters
            from radp.digital_twin.utils.cell_selection import find_hyst_diff
            max_diff = find_hyst_diff(simulation_data)
            num_ticks = simulation_data["tick"].nunique()
            hyst_range = [0, max_diff]
            ttt_range = [2, num_ticks + 1]
            
            # Track scores
            scores = []
            best_score = float('-inf')
            best_hyst = 0.01
            best_ttt = 5
            
            # Run optimization epochs
            for epoch in range(n_epochs):
                # Generate random parameters
                hyst = np.random.uniform(hyst_range[0], hyst_range[1])
                ttt = np.random.randint(ttt_range[0], ttt_range[1])
                
                # Perform attachment and calculate MRO metric
                attached_df = perform_attachment_hyst_ttt(simulation_data, hyst, ttt, RLF_THRESHOLD)
                mro_metric = calculate_mro_metric(attached_df)
                
                # Track best score
                if mro_metric > best_score:
                    best_score = mro_metric
                    best_hyst = hyst
                    best_ttt = ttt
                
                scores.append({'hyst': hyst, 'ttt': ttt, 'score': mro_metric})
                self.logger.info(f"{epoch:<6} {hyst:<14.10f} {ttt:<6} {mro_metric:<12.6f}")
            
            # Final attachment with best parameters
            final_attached_df = perform_attachment_hyst_ttt(simulation_data, best_hyst, best_ttt, RLF_THRESHOLD)
            
            return {
                'hysteresis': best_hyst,
                'ttt': best_ttt,
                'mro_metric': best_score,
                'attached_data': final_attached_df,
                'all_scores': scores
            }
    



def main():
    """Command line interface for MRO inference."""
    parser = argparse.ArgumentParser(description='Perform MRO optimization using trained models')
    parser.add_argument('--topology', required=True, help='Path to topology CSV file')
    parser.add_argument('--ue-data', help='Path to UE data CSV file (optional, uses mobility params if not provided)')
    parser.add_argument('--trained-model', required=True, help='Path to pre-trained model pickle file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of optimization epochs')
    parser.add_argument('--output', required=True, help='Path to save MRO inference results CSV file')
    parser.add_argument('--mode', choices=['rf', 'mro'], default='mro', help='Inference mode: rf (RF prediction only) or mro (full MRO optimization)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        topology = pd.read_csv(args.topology)
        ue_data = pd.read_csv(args.ue_data) if args.ue_data else None
        
        # Load pre-trained model
        with open(args.trained_model, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if it's a complete MRO model or just BDT models
        if isinstance(model_data, dict) and 'bdt_models' in model_data:
            # Complete MRO model with optimal parameters
            trained_models = model_data['bdt_models']
            optimal_hyst = model_data.get('optimal_hysteresis')
            optimal_ttt = model_data.get('optimal_ttt')
            print(f"Loaded complete MRO model with optimal parameters")
        else:
            # Just BDT models
            trained_models = model_data
            optimal_hyst = None
            optimal_ttt = None
            print(f"Loaded BDT models only")
        
        # Initialize inference engine
        inference_engine = SimpleMROInference(topology, trained_models, optimal_hyst, optimal_ttt)
        
        if args.mode == 'rf':
            # RF prediction only (preprocessed data)
            print("Getting preprocessed simulation data...")
            results = inference_engine.get_preprocessed_data(ue_data)
            results.to_csv(args.output, index=False)
            print(f"Preprocessed data with {len(results)} rows saved to {args.output}")
        else:
            # Full MRO optimization
            print("Solving MRO optimization...")
            
            mro_results = inference_engine.mro_inference(ue_data, args.epochs)
            
            print("\nResults:")
            print(f"Optimal Hysteresis: {mro_results['hysteresis']:.6f}")
            print(f"Optimal TTT: {mro_results['ttt']}")
            print(f"Best MRO Metric: {mro_results['mro_metric']:.6f}")
            
            # Save detailed results
            mro_results['attached_data'].to_csv(args.output, index=False)
            print(f"Detailed results saved to {args.output}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()