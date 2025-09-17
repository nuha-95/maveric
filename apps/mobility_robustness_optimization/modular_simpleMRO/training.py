import logging
import pandas as pd
import torch
import numpy as np
import warnings
import sys
import os
import argparse
import pickle
from typing import Dict, List, Tuple
from gpytorch.utils.warnings import NumericalWarning
# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, NormMethod
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD
from notebooks.radp_library import (
    add_cell_info,
    calc_log_distance,
    calc_relative_bearing,
    get_percell_data,
    normalize_cell_ids,
)
from apps.mobility_robustness_optimization.mobility_robustness_optimization import calculate_mro_metric

logger = logging.getLogger(__name__)


class BayesianTrainer:
    """Handles training and updating of Bayesian Digital Twins."""
    
    def __init__(self, topology):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.topology = topology
        self.bayesian_digital_twins = {}
    
    def train_bayesian_twins(self, new_data: pd.DataFrame, maxiter: int = 100) -> List[float]:
        """Train Bayesian Digital Twins from scratch."""
        self.topology = normalize_cell_ids(self.topology)
        new_data = normalize_cell_ids(new_data)
        
        prepared_data = self._prepare_train_data(new_data)
        return self._training(maxiter, prepared_data)
    
    def update_bayesian_twins(self, new_data: pd.DataFrame):
        """Update existing Bayesian Digital Twins with new data."""
        self.topology = normalize_cell_ids(self.topology)
        new_data = normalize_cell_ids(new_data)
        
        prepared_data = self._prepare_train_data(new_data)
        
        for cell_id, df in prepared_data.items():
            self._update(cell_id, df)
    
    def _training(self, maxiter: int, train_data: Dict[str, pd.DataFrame]) -> List[float]:
        """Train the Bayesian Digital Twins for each cell."""
        bayesian_digital_twins = {}
        loss_vs_iters = []

        for train_cell_id, training_data_idx in train_data.items():
            bayesian_digital_twins[train_cell_id] = BayesianDigitalTwin(
                data_in=[training_data_idx],
                x_columns=["log_distance", "relative_bearing"],
                y_columns=["cell_rxpwr_dbm"],
                norm_method=NormMethod.MINMAX,
            )

            bayesian_digital_twins[train_cell_id].model = bayesian_digital_twins[train_cell_id].model.to(self.device)
            self.bayesian_digital_twins[train_cell_id] = bayesian_digital_twins[train_cell_id]

            loss_vs_iters.append(
                bayesian_digital_twins[train_cell_id].train_distributed_gpmodel(
                    maxiter=maxiter,
                )
            )

        return loss_vs_iters
    
    def _update(self, cell_id: str, df: pd.DataFrame) -> None:
        """Update the Bayesian Digital Twin model for a specific cell."""
        df = df.drop_duplicates(subset=["log_distance", "relative_bearing"])

        if df.shape[0] > 500:
            df = get_percell_data(data_in=df, choose_strongest_samples_percell=True, n_samples=500)[0][0]

        twin = self.bayesian_digital_twins[cell_id]
        twin.update_trained_gpmodel([df])
    
    def _prepare_train_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare training data for each cell."""
        required_columns = {"cell_lat", "cell_lon", "cell_az_deg"}
        if not required_columns.issubset(df.columns):
            df = add_cell_info(df, self.topology)

        update_data = calc_log_distance(df)
        update_data = calc_relative_bearing(update_data)
        update_data = update_data.loc[:, ["cell_id", "log_distance", "relative_bearing", "cell_rxpwr_dbm"]]

        train_per_cell_df = [x for _, x in update_data.groupby("cell_id")]
        n_cell = len(self.topology.index)

        metadata_df = pd.DataFrame({
            "cell_id": [cell_id for cell_id in self.topology.cell_id],
            "idx": [i + 1 for i in range(n_cell)],
        })

        idx_cell_id_mapping = dict(zip(metadata_df.idx, metadata_df.cell_id))
        n_samples_train = [df.shape[0] for df in train_per_cell_df]

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

    def optimize_attachment_parameters(self, simulation_data: pd.DataFrame, epochs: int = 100) -> Tuple[float, int]:
        """Optimize hysteresis and TTT parameters for attachment."""
        hyst = 0.01
        ttt = 5
        rlf_threshold = RLF_THRESHOLD

        attached_df = perform_attachment_hyst_ttt(simulation_data, hyst, ttt, rlf_threshold)
        max_diff = find_hyst_diff(simulation_data)
        num_ticks = simulation_data["tick"].nunique()
        hyst_range = [0, max_diff]
        ttt_range = [2, num_ticks + 1]

        warnings.filterwarnings("ignore", category=NumericalWarning)
        score = pd.DataFrame(columns=["hyst", "ttt", "score"])

        header = f"{'Epoch':<6} {'Hyst':<14} {'TTT':<6} {'MRO Metric':<12}"
        logger.info(header)
        logger.info("-" * len(header))
        
        score.loc[len(score)] = [hyst, ttt, calculate_mro_metric(attached_df)]
        
        for i in range(epochs):
            while True:
                hyst = np.random.uniform(hyst_range[0], hyst_range[1])
                ttt = np.random.randint(ttt_range[0], ttt_range[1])
                if ttt not in score["ttt"].values or hyst not in score["hyst"].values:
                    break
            
            attached_df = perform_attachment_hyst_ttt(simulation_data, hyst, ttt, rlf_threshold)
            mro_metric = calculate_mro_metric(attached_df)
            
            score.loc[len(score)] = [hyst, ttt, mro_metric]
            logger.info(f"{i:<6} {hyst:<14.10f} {ttt:<6} {mro_metric:<12.6f}")

        best_idx = score["score"].idxmax()
        logger.info(f"\nOptimized Hyst: {score.loc[best_idx, 'hyst']}")
        logger.info(f"Optimized TTT: {int(score.loc[best_idx, 'ttt'])}")
        
        return score.loc[best_idx, "hyst"], int(score.loc[best_idx, "ttt"])


def main():
    """Command line interface for training."""
    parser = argparse.ArgumentParser(description='Train Bayesian Digital Twins for SimpleMRO')
    parser.add_argument('--topology', required=True, help='Path to topology CSV file')
    parser.add_argument('--training-data', required=True, help='Path to training data CSV file')
    parser.add_argument('--output-model', required=True, help='Path to save trained model (pickle file)')
    parser.add_argument('--maxiter', type=int, default=100, help='Maximum training iterations')
    parser.add_argument('--optimize-attachment', action='store_true', help='Also optimize attachment parameters')
    parser.add_argument('--simulation-data', help='Path to simulation data for attachment optimization')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs for attachment optimization')
    
    args = parser.parse_args()
    
    # Load data
    topology = pd.read_csv(args.topology)
    training_data = pd.read_csv(args.training_data)
    
    # Initialize trainer
    trainer = BayesianTrainer(topology)
    
    # Train models
    print(f"Training Bayesian Digital Twins with {args.maxiter} iterations...")
    loss_vs_iters = trainer.train_bayesian_twins(training_data, args.maxiter)
    
    # Save trained models
    with open(args.output_model, 'wb') as f:
        pickle.dump(trainer.bayesian_digital_twins, f)
    
    print(f"Training completed. Loss iterations: {len(loss_vs_iters)}")
    print(f"Trained models saved to {args.output_model}")
    
    # Optimize attachment parameters if requested
    if args.optimize_attachment and args.simulation_data:
        simulation_data = pd.read_csv(args.simulation_data)
        print(f"\nOptimizing attachment parameters with {args.epochs} epochs...")
        optimal_hyst, optimal_ttt = trainer.optimize_attachment_parameters(simulation_data, args.epochs)
        print(f"Optimal Hysteresis: {optimal_hyst:.6f}")
        print(f"Optimal TTT: {optimal_ttt}")


if __name__ == "__main__":
    main()