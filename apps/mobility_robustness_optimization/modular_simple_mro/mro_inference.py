import logging
import pandas as pd
import sys
import os

# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)


import pickle

logger = logging.getLogger(__name__)


def mro_inference(trained_model):
    """Get best parameters from trained model score dataframe."""
    
    # Load trained model
    with open(trained_model, 'rb') as f:
        trained_model = pickle.load(f)
    
    score_dataframe = trained_model['score_dataframe']
    
    # Find best parameters from score dataframe
    best_idx = score_dataframe['score'].idxmax()
    best_hyst = score_dataframe.loc[best_idx, 'hyst']
    best_ttt = int(score_dataframe.loc[best_idx, 'ttt'])
    best_score = score_dataframe.loc[best_idx, 'score']
    
    print(f"Best parameters: Hyst={best_hyst:.6f}, TTT={best_ttt}")
    print(f"Best MRO Score: {best_score:.6f}")
    
    return {
        'hysteresis': best_hyst,
        'ttt': best_ttt,
        
    }