import logging
import pandas as pd
import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)

from radp.digital_twin.utils.cell_selection import find_hyst_diff
from radp.digital_twin.utils.constants import RLF_THRESHOLD
from rl_utils import ReinforcedMROEnv

logger = logging.getLogger(__name__)


def rl_mro_inference(simulation_data, ppo_model):
    
    
    print("Loading trained PPO model for parameter prediction...")
    
    # Load trained PPO model
    ppo_model = PPO.load(ppo_model)
    
    # Define parameter ranges from simulation data 
    max_diff = find_hyst_diff(simulation_data)
    num_ticks = simulation_data["tick"].nunique()
    hyst_range = [0, max_diff]
    ttt_range = [2, num_ticks + 1]
    
    # Create dummy environment 
    dummy_env = DummyVecEnv([lambda: ReinforcedMROEnv(simulation_data, RLF_THRESHOLD, hyst_range, ttt_range)])
    
    # Predict optimal action using trained model
    obs = dummy_env.reset()
    action, _ = ppo_model.predict(obs, deterministic=True)
    
    # Ensure ttt is an integer
    hyst, ttt = action[0]
    ttt = int(round(ttt))
    
    print(f"PPO predicted parameters: Hyst={hyst:.6f}, TTT={ttt}")
    
    return {
        'hysteresis': hyst,
        'ttt': ttt
    }