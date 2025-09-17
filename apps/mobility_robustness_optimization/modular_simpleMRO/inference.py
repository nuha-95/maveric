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

from preprocessing import preprocess_prediction_data, add_sinr_column, compute_row_level_sinr
import numpy as np
from training import BayesianTrainer

# Add parent directory to path for MRO imports
sys.path.insert(0, os.path.join(maveric_root, 'apps', 'mobility_robustness_optimization'))
from mobility_robustness_optimization import calculate_mro_metric
from radp.digital_twin.utils.cell_selection import perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD

logger = logging.getLogger(__name__)


class SimpleMROInference:
    """Inference module for SimpleMRO - handles RF prediction and MRO optimization inference."""

    def __init__(self, topology: pd.DataFrame, trained_models: dict):
        self.topology = topology
        self.trainer = BayesianTrainer(topology)
        self.trainer.bayesian_digital_twins = trained_models
        self.logger = logging.getLogger(__name__)

    def predict_rf(self, location_data: pd.DataFrame) -> pd.DataFrame:
        """Predict RF power for given locations using trained models."""
        if not self.trainer.bayesian_digital_twins:
            raise ValueError("No trained models available for prediction.")

        # Check if data is already processed (has required columns)
        required_prediction_cols = ['cell_id', 'log_distance', 'relative_bearing']
        is_preprocessed = all(col in location_data.columns for col in required_prediction_cols)
        
        if is_preprocessed:
            # Data is already preprocessed, use it directly
            prediction_data = location_data.copy()
        else:
            # Ensure required columns exist for preprocessing
            required_cols = ['longitude', 'latitude']
            if not all(col in location_data.columns for col in required_cols):
                raise ValueError(f"Input data must contain columns: {required_cols}")

            # Add tick column if missing
            if 'tick' not in location_data.columns:
                location_data = location_data.copy()
                location_data['tick'] = 0  # Single time point

            # Preprocess data for prediction
            prediction_data = preprocess_prediction_data(location_data, self.topology)
        
        results = pd.DataFrame()

        # Make predictions for each cell
        for cell_id, cell_df in prediction_data.groupby("cell_id"):
            # Handle both string and integer cell_id formats
            if isinstance(cell_id, str) and cell_id.startswith('cell_'):
                cell_id_str = cell_id
            else:
                cell_id_str = f"cell_{cell_id}"
            
            if cell_id_str in self.trainer.bayesian_digital_twins:
                try:
                    pred_means, _ = self.trainer.bayesian_digital_twins[cell_id_str].predict_distributed_gpmodel(
                        prediction_dfs=[cell_df]
                    )
                    cell_df = cell_df.copy()
                    cell_df["predicted_rxpower_dbm"] = pred_means[0]
                    cell_df["cell_id"] = cell_id_str
                    results = pd.concat([results, cell_df], ignore_index=True)
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {cell_id_str}: {e}")
            else:
                self.logger.warning(f"No model for {cell_id_str}")

        return results

    def mro_inference(self, ue_data: pd.DataFrame, n_epochs: int = 5) -> dict:
        """Perform MRO optimization to find optimal hysteresis and TTT parameters."""
        # Get RF predictions and prepare simulation data
        rf_predictions = self.predict_rf(ue_data)
        rf_with_sinr = add_sinr_column(rf_predictions)
        simulation_data = self._prepare_for_attachment(rf_with_sinr)
        
        # Initialize optimization parameters
        max_diff = self._find_hyst_diff(simulation_data)
        num_ticks = simulation_data["tick"].nunique() if "tick" in simulation_data.columns else 50
        hyst_range = [0.01, max(max_diff, 5.0)]  # Ensure minimum range
        ttt_range = [2, max(10, min(num_ticks + 1, 50))]  # Ensure minimum range
        
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
    
    def _prepare_for_attachment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for attachment function by converting to expected format."""
        attachment_data = data.copy()
        
        # Rename columns to match expected format
        column_mapping = {
            'mock_ue_id': 'ue_id',
            'predicted_rxpower_dbm': 'cell_rxpower_dbm',
            'log_distance': 'distance_km'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in attachment_data.columns:
                attachment_data = attachment_data.rename(columns={old_col: new_col})
        
        # Convert cell_id to integer if it's string format
        if attachment_data['cell_id'].dtype == object:
            attachment_data['cell_id'] = attachment_data['cell_id'].str.extract(r'(\d+)').astype(int)
        
        return attachment_data
    
    def _find_hyst_diff(self, data: pd.DataFrame) -> float:
        """Find maximum hysteresis difference from simulation data."""
        if 'cell_rxpower_dbm' in data.columns:
            power_values = data['cell_rxpower_dbm'].values
            return float(np.max(power_values) - np.min(power_values))
        return 10.0  # Default value


def main():
    """Command line interface for MRO inference."""
    parser = argparse.ArgumentParser(description='Perform MRO optimization using trained models')
    parser.add_argument('--topology', required=True, help='Path to topology CSV file')
    parser.add_argument('--ue-data', required=True, help='Path to UE data CSV file')
    parser.add_argument('--trained-model', required=True, help='Path to pre-trained model pickle file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of optimization epochs')
    parser.add_argument('--output', required=True, help='Path to save MRO inference results CSV file')
    parser.add_argument('--mode', choices=['rf', 'mro'], default='mro', help='Inference mode: rf (RF prediction only) or mro (full MRO optimization)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        topology = pd.read_csv(args.topology)
        ue_data = pd.read_csv(args.ue_data)
        
        # Load pre-trained model
        with open(args.trained_model, 'rb') as f:
            trained_models = pickle.load(f)
        
        # Initialize inference engine
        inference_engine = SimpleMROInference(topology, trained_models)
        
        if args.mode == 'rf':
            # RF prediction only
            print(f"Predicting RF power for {len(ue_data)} locations...")
            results = inference_engine.predict_rf(ue_data)
            results.to_csv(args.output, index=False)
            print(f"RF predictions completed for {len(results)} location-cell combinations")
            print(f"Results saved to {args.output}")
        else:
            # Full MRO optimization
            print("Solving MRO optimization...")
            logger.info("Epoch  Hyst           TTT    MRO Metric  ")
            logger.info("-----------------------------------------")
            
            mro_results = inference_engine.mro_inference(ue_data, args.epochs)
            
            logger.info(f"\nOptimized Hyst: {mro_results['hysteresis']}")
            logger.info(f"Optimized TTT: {mro_results['ttt']}")
            
            print("\nResults:")
            print(f"Optimal Hysteresis: {mro_results['hysteresis']:.6f}")
            print(f"Optimal TTT: {mro_results['ttt']}")
            
            # Save detailed results
            mro_results['attached_data'].to_csv(args.output, index=False)
            print(f"Detailed results saved to {args.output}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()