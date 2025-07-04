#!/usr/bin/env python3
"""
INTERNCRANE Stacking Model API

This module provides simple function call interfaces to use trained INTERNCRANE stacking models.

Usage example:
    from stacking_api import load_stacker_from_config, stack_predict
    
    # Load stacker from YAML config
    stacker = load_stacker_from_config("config_stacking.yaml")
    
    # Make predictions
    result = stack_predict(stacker, {"SMILES": "CCO"})
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress warnings
warnings.filterwarnings("ignore")

from model_stacking import ModelStacker
from utils.stacking_config import (
    load_yaml_config, validate_stacking_config, extract_stacking_config,
    extract_model_configs, get_model_weights
)

class StackingPredictor:
    """Stacking model predictor class"""
    
    def __init__(self, stacker: ModelStacker):
        self.stacker = stacker
        self.task_type = stacker.task_type
        self.stacking_method = stacker.stacking_method
        self.model_names = list(stacker.base_models.keys())
        self.n_models = len(stacker.base_models)
    
    def predict(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
        """Perform stacking prediction"""
        return self.stacker.predict(data)
    
    def predict_single(self, sample: Dict[str, Any]) -> Union[float, str, int]:
        """Predict single sample"""
        return self.stacker.predict_single(sample)
    
    def evaluate(self, auto_load: bool = True) -> Dict[str, Any]:
        """Evaluate stacking model"""
        return self.stacker.evaluate(auto_load=auto_load)
    
    def save(self, filepath: str) -> None:
        """Save stacker"""
        self.stacker.save(filepath)
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'stacking_method': self.stacking_method,
            'task_type': self.task_type,
            'n_models': self.n_models,
            'model_names': self.model_names,
            'model_weights': self.stacker.model_weights
        }

def load_stacker_from_config(config_path: str) -> StackingPredictor:
    """Load stacker from YAML configuration file"""
    config = load_yaml_config(config_path)
    validate_stacking_config(config)
    
    stacking_config = extract_stacking_config(config)
    experiment_dir = stacking_config['experiment_dir']
    method = stacking_config.get('method', 'weighted_average')
    
    stacker = ModelStacker(experiment_dir=experiment_dir)
    stacker.set_stacking_method(method)
    
    model_configs = extract_model_configs(stacking_config)
    weights = get_model_weights(model_configs)
    
    for model_config in model_configs:
        model_name = model_config['name']
        weight = weights[model_name]
        stacker.add_model(model_name, weight)
    
    return StackingPredictor(stacker)

def create_stacker(experiment_dir: str, model_names: List[str], 
                  weights: Optional[List[float]] = None,
                  method: str = "weighted_average") -> StackingPredictor:
    """Create stacker programmatically"""
    if weights is None:
        weights = [1.0] * len(model_names)
    
    if len(weights) != len(model_names):
        raise ValueError("Number of weights must match number of models")
    
    stacker = ModelStacker(experiment_dir=experiment_dir)
    stacker.set_stacking_method(method)
    
    for model_name, weight in zip(model_names, weights):
        stacker.add_model(model_name, weight)
    
    return StackingPredictor(stacker)

def stack_predict(predictor: StackingPredictor, 
                 data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
    """Use stacking predictor to make predictions"""
    return predictor.predict(data)

def stack_predict_single(predictor: StackingPredictor, 
                        sample: Dict[str, Any]) -> Union[float, str, int]:
    """Predict single sample"""
    return predictor.predict_single(sample)

def quick_stack_predict(config_path: str, 
                       data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
    """Complete config loading and prediction in one step"""
    predictor = load_stacker_from_config(config_path)
    return predictor.predict(data)

# Example usage
if __name__ == "__main__":
    print("INTERNCRANE Stacking Model API Example Usage")
    print("-" * 50)
    
    print("1. Load stacker from config file:")
    print("predictor = load_stacker_from_config('config_stacking.yaml')")
    
    print("\n2. Create stacker programmatically:")
    print("predictor = create_stacker('output/my_experiment', ['xgb', 'lgbm'])")
    
    print("\n3. Make predictions:")
    print("result = stack_predict(predictor, {'SMILES': 'CCO'})")
