import logging
import pandas as pd
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)

from mro_utils import calculate_mro_metric

from radp.digital_twin.utils.cell_selection import perform_attachment_hyst_ttt, find_hyst_diff
from radp.digital_twin.utils.constants import RLF_THRESHOLD
from rl_environment import ReinforcedMROEnv

logger = logging.getLogger(__name__)


def rl_mro_inference(ppo_model):
    """Get optimal parameters from trained PPO model."""
    
    print("Loading trained PPO model for parameter prediction...")
    
    # Load trained PPO model
    ppo_model = PPO.load(ppo_model)
    
    # Create dummy environment for prediction (parameters will be extracted from model)
    
    dummy_env = DummyVecEnv([lambda: ReinforcedMROEnv(None, RLF_THRESHOLD, [0, 1], [2, 50])])
    
    # Predict optimal action using trained PPO agent
    obs = dummy_env.reset()
    action, _ = ppo_model.predict(obs, deterministic=True)

    # Extract parameters
    hyst, ttt = action[0]
    ttt = int(round(ttt))
    
    print(f"PPO predicted parameters: Hyst={hyst:.6f}, TTT={ttt}")
    
    return {
        'hysteresis': hyst,
        'ttt': ttt
    }


