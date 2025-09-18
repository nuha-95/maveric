import logging
import warnings
import pandas as pd
import numpy as np
import sys
import os
# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)
from gpytorch.utils.warnings import NumericalWarning
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD
from mro_utils import calculate_mro_metric

# Suppress warnings
warnings.filterwarnings("ignore", category=NumericalWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_mro(simulation_data, n_epochs=100):
    """Train MRO by generating score dataframe through parameter exploration."""
    
    # Initialize parameters
    max_diff = find_hyst_diff(simulation_data)
    num_ticks = simulation_data["tick"].nunique()
    hyst_range = [0, max_diff]
    ttt_range = [2, num_ticks + 1]
    
    score = pd.DataFrame(columns=["hyst", "ttt", "score"])
    
    header = f"{'Epoch':<6} {'Hyst':<14} {'TTT':<6} {'MRO Metric':<12}"
    logger.info(header)
    logger.info("-" * len(header))
    
    # Generate score dataframe through parameter exploration
    for i in range(n_epochs):
        hyst = np.random.uniform(hyst_range[0], hyst_range[1])
        ttt = np.random.randint(ttt_range[0], ttt_range[1])
        
        # Perform attachment and calculate MRO Metric
        attached_df = perform_attachment_hyst_ttt(simulation_data, hyst, ttt, RLF_THRESHOLD)
        mro_metric = calculate_mro_metric(attached_df)
        
        # Store the data in the score DataFrame
        score.loc[len(score)] = [hyst, ttt, mro_metric]
        logger.info(f"{i:<6} {hyst:<14.10f} {ttt:<6} {mro_metric:<12.6f}")
    
    logger.info(f"\nGenerated {len(score)} parameter combinations for training.")
    
    return score