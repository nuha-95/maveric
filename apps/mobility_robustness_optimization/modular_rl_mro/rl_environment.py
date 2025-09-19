import logging
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from radp.digital_twin.utils.cell_selection import perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD
import sys
import os

# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)

from mro_utils import calculate_mro_metric

logger = logging.getLogger(__name__)


class ReinforcedMROEnv(Env):
    """RL Environment for MRO optimization using PPO."""
    
    def __init__(self, df, rlf_threshold, hyst_range, ttt_range):
        super().__init__()
        self.df = df
        self.rlf_threshold = rlf_threshold
        self.hyst_range = hyst_range
        self.ttt_range = ttt_range

        self.action_space = Box(
            low=np.array([hyst_range[0], ttt_range[0]]),
            high=np.array([hyst_range[1], ttt_range[1]]),
            dtype=np.float64,
        )
        self.observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float64)

        self.state = np.array([0.0])
        self.current_step = 0
        self.max_steps = 20
        self.episode_num = 1
        self.episode_reward = 0.0
        self.logger = logging.getLogger(__name__)

    def step(self, action):
        hyst, ttt = action
        ttt = int(round(ttt))

        attached_df = perform_attachment_hyst_ttt(self.df, hyst, ttt, self.rlf_threshold)
        mro_metric = calculate_mro_metric(attached_df)

        reward = mro_metric
        self.episode_reward += reward
        self.state = np.array([reward])
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        self.logger.info(
            f"Episode: {self.episode_num}, Timestep: {self.current_step}, "
            f"Hyst: {hyst:.6f}, TTT: {ttt}, Reward: {reward:.6f}, Done: {terminated}"
        )

        if terminated:
            avg_reward = self.episode_reward / self.max_steps
            self.logger.info(f"Episode {self.episode_num} average reward: {avg_reward:.6f}\n")
            self.episode_num += 1
            self.episode_reward = 0.0

        return self.state, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        self.state = np.array([0.0])
        self.current_step = 0
        return self.state, {}

    def render(self):
        self.logger.info(f"Current State: {self.state}, Current Step: {self.current_step}")