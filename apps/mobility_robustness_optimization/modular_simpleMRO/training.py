import logging
import warnings
import pandas as pd
import numpy as np
import sys
import os
import argparse
import pickle
# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)
from gpytorch.utils.warnings import NumericalWarning
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD
from apps.mobility_robustness_optimization.mobility_robustness_optimization import calculate_mro_metric, MobilityRobustnessOptimization
from preprocessing import Preprocessor


class MROTrainer(MobilityRobustnessOptimization):
    
    
    def __init__(self, mobility_model_params, topology, new_data=None, training_data=None):
        super().__init__(mobility_model_params, topology, new_data)
        self.training_data = training_data
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
    
    def solve(self):
        """Required abstract method implementation"""
        return self.train_mro()
    
    def train_bdt(self):
        """Train BDT models if training data is provided"""
        if self.training_data is not None:
            print("Training BDT models from scratch...")
            self.train_or_update_rf_twins(self.training_data)
        elif not self.bayesian_digital_twins:
            raise ValueError("No BDT models available and no training data provided")
    
    def train_mro(self, n_epochs=100):
        """Train complete MRO: BDT + optimization"""
        
        # Train BDT models if needed
        self.train_bdt()
        
        # Use Preprocessor for consistent data preparation
        preprocessor = Preprocessor(
            self.mobility_model_params,
            self.topology,
            self.new_data,
            self.bayesian_digital_twins
        )
        
        self.simulation_data = preprocessor.preprocess_data()
        
        # MRO optimization part (from SimpleMRO.solve lines 77-101)
        epochs = n_epochs
        hyst = 0.01
        ttt = 5
        rlf_threshold = RLF_THRESHOLD

        attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)
        max_diff = find_hyst_diff(self.simulation_data)
        num_ticks = self.simulation_data["tick"].nunique()
        hyst_range = [0, max_diff]
        ttt_range = [2, num_ticks + 1]

        # Suppress the specific NumericalWarning from gpytorch
        warnings.filterwarnings("ignore", category=NumericalWarning)

        self.score = pd.DataFrame(columns=["hyst", "ttt", "score"])

        header = f"{'Epoch':<6} {'Hyst':<14} {'TTT':<6} {'MRO Metric':<12}"
        self.logger.info(header)
        self.logger.info("-" * len(header))
        self.score.loc[len(self.score)] = [hyst, ttt, calculate_mro_metric(attached_df)]
        
        for i in range(epochs):
            while True:
                hyst = np.random.uniform(hyst_range[0], hyst_range[1])
                ttt = np.random.randint(ttt_range[0], ttt_range[1])
                if ttt not in self.score["ttt"].values or hyst not in self.score["hyst"].values:
                    break
            # Perform attachment and calculate MRO Metric
            attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)
            mro_metric = calculate_mro_metric(attached_df)

            # Store the data in the score DataFrame
            self.score.loc[len(self.score)] = [hyst, ttt, mro_metric]
            self.logger.info(f"{i:<6} {hyst:<14.10f} {ttt:<6} {mro_metric:<12.6f}")

        self.logger.info(f"\nOptimized Hyst: {self.score.loc[self.score['score'].idxmax(), 'hyst']}")
        self.logger.info(f"Optimized TTT: {int(self.score.loc[self.score['score'].idxmax(), 'ttt'])}")
        
        optimal_hyst = self.score.loc[self.score["score"].idxmax(), "hyst"]
        optimal_ttt = int(self.score.loc[self.score["score"].idxmax(), "ttt"])
        
        return optimal_hyst, optimal_ttt


def main():
    """Command line interface for MRO training."""
    parser = argparse.ArgumentParser(description='MRO training using trained BDT models')
    parser.add_argument('--topology', required=True, help='Path to topology CSV file')
    parser.add_argument('--ue-data', help='Path to UE data CSV file (optional, uses mobility params if not provided)')
    parser.add_argument('--bdt-models', help='Path to trained BDT models pickle file (optional if training-data provided)')
    parser.add_argument('--training-data', help='Path to training data CSV file for BDT training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--output', required=True, help='Path to output results CSV file')
    parser.add_argument('--save-models', action='store_true', help='Save complete MRO model (BDT + optimal parameters) to .pkl file')
    
    args = parser.parse_args()
    
    # Load data
    topology = pd.read_csv(args.topology)
    new_data = pd.read_csv(args.ue_data) if args.ue_data else None
    training_data = pd.read_csv(args.training_data) if args.training_data else None
    
    # Load BDT models if provided
    bdt_models = None
    if args.bdt_models:
        with open(args.bdt_models, 'rb') as f:
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
    
    # Create MRO trainer
    trainer = MROTrainer(mobility_model_params, topology, new_data, training_data)
    
    # Set BDT models if loaded
    if bdt_models:
        trainer.bayesian_digital_twins = bdt_models
    
    # Train complete MRO
    optimal_hyst, optimal_ttt = trainer.train_mro(args.epochs)
    
    # Save results
    trainer.score.to_csv(args.output, index=False)
    
    # Save complete MRO model (BDT + optimal parameters)
    if args.save_models:
        model_output = args.output.replace('.csv', '_complete.pkl')
        mro_complete_model = {
            'bdt_models': trainer.bayesian_digital_twins,
            'optimal_hysteresis': optimal_hyst,
            'optimal_ttt': optimal_ttt,
            'mobility_model_params': mobility_model_params,
            'topology': topology
        }
        with open(model_output, 'wb') as f:
            pickle.dump(mro_complete_model, f)
        print(f"Complete MRO model saved to {model_output}")
    
    print(f"MRO training completed!")
    print(f"Results saved to {args.output}")
    print(f"Optimal Hyst: {optimal_hyst}")
    print(f"Optimal TTT: {optimal_ttt}")


if __name__ == "__main__":
    main()