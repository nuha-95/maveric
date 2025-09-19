import logging
import warnings
import pandas as pd
import numpy as np
import torch
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)

from gpytorch.utils.warnings import NumericalWarning
from radp.digital_twin.utils.cell_selection import find_hyst_diff
from radp.digital_twin.utils.constants import RLF_THRESHOLD
from rl_environment import ReinforcedMROEnv

# Suppress warnings
warnings.filterwarnings("ignore", category=NumericalWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_rl_mro(simulation_data, total_timesteps=100):
    """Train RL-based MRO using PPO algorithm."""
    
    # Define parameter ranges
    max_diff = find_hyst_diff(simulation_data)
    num_ticks = simulation_data["tick"].nunique()
    hyst_range = [0, max_diff]
    ttt_range = [2, num_ticks + 1]

    # Create and vectorize RL environment
    env = DummyVecEnv([lambda: ReinforcedMROEnv(simulation_data, RLF_THRESHOLD, hyst_range, ttt_range)])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # PPO agent
    model = PPO("MlpPolicy", env, verbose=2, n_steps=64, batch_size=64, device=device)
    model.learn(total_timesteps)
    
    logger.info(f"\nRL Training completed with {total_timesteps} timesteps.")
    
    return model


