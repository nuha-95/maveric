import pandas as pd
import numpy as np
import sys
import os
import pickle
import torch
# Add maveric root to path
maveric_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maveric_root)
from notebooks.radp_library import find_sim_boundary, get_ue_data
from mro_utils import predictions, preprocess_simulation_data


def preprocess_mro(mobility_model_params, topology, bdt_model_path, new_data=None):
    """Preprocess simulation data using trained BDT models."""
    
    # Load trained models
    with open(bdt_model_path, 'rb') as f:
        bayesian_digital_twins = pickle.load(f)
    
    # Ensure we have trained models
    if not bayesian_digital_twins:
        raise ValueError("Bayesian Digital Twins are not available. Need trained models for preprocessing.")
    
    bounds = find_sim_boundary(topology, new_data)
    mobility_model_params["ue_tracks_generation"]["params"]["lat_lon_boundaries"].update(bounds)
    
    simulation_data = get_ue_data(mobility_model_params)
    simulation_data = simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})
    
    if topology["cell_id"].dtype == int:
        topology["cell_id"] = topology["cell_id"].apply(lambda x: f"cell_{int(x)}")
    
    predicted, full_prediction_df = predictions(bayesian_digital_twins, simulation_data, topology)
    simulation_data = preprocess_simulation_data(full_prediction_df, topology)
    
    return simulation_data