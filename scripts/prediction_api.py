#!/usr/bin/env python3
"""
INTERNCRANE Prediction API

This module provides simple function call interfaces to use trained INTERNCRANE models.
Can be directly imported and used in other Python code without command line or configuration files.

Usage example:
    from prediction_api import load_model, predict, predict_single, quick_predict
    
    # Load model
    predictor = load_model("output/my_experiment", "xgb")
    
    # Predict single sample
    result = predict_single(predictor, {"SMILES": "CCO"})
    print(result)
    
    # Predict multiple samples
    samples = [
        {"SMILES": "CCO"},
        {"SMILES": "c1ccccc1"}
    ]
    results = predict(predictor, samples)
    print(results['predictions'])
"""

import os
import sys
import warnings
import contextlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 抑制警告
warnings.filterwarnings("ignore")

from utils.io_handler import (
    load_model_from_path, load_scaler_from_path, load_label_encoder_from_path,
    load_config_from_path, get_full_model_name, find_model_file
)
from core.run_manager import process_dataframe

class INTERNCRANEPredictor:
    """INTERNCRANE model predictor class"""
    
    def __init__(self, model, training_config, scaler=None, label_encoder=None):
        self.model = model
        self.training_config = training_config
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.task_type = training_config.get('task_type', 'regression')
    
    @contextlib.contextmanager
    def _suppress_output(self):
        """Context manager to suppress output"""
        original_stdout, original_stderr = sys.stdout, sys.stderr
        devnull = open(os.devnull, 'w')
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            devnull.close()
    
    def _process_input_data(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> pd.DataFrame:
        """Process input data, convert to DataFrame format"""
        if isinstance(data, dict):
            # Single sample, convert to single-row DataFrame
            return pd.DataFrame([data])
        elif isinstance(data, list):
            # Multiple samples, convert to DataFrame
            return pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            # Already a DataFrame
            return data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _generate_features(self, df: pd.DataFrame) -> np.ndarray:
        """Generate features"""
        # Get configuration
        common_cfg = self.training_config.get('data', {}).get('single_file_config', {})
        feature_gen_cfg = self.training_config.get('features', {})
        
        # Add dummy target column (real target values not needed for prediction)
        target_col = common_cfg.get('target_col', 'target')
        if target_col not in df.columns:
            df[target_col] = 0
        
        # Generate features (suppress output)
        with self._suppress_output():
            X, _, _, _ = process_dataframe(
                df=df.copy(),
                common_cfg=common_cfg,
                feature_gen_cfg=feature_gen_cfg,
                output_dir="."
            )
        
        # Apply feature scaling
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def predict(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
        """
        Make predictions
        
        Args:
            data: Input data, can be:
                - dict: Feature dictionary for single sample
                - list[dict]: List of feature dictionaries for multiple samples
                - pd.DataFrame: DataFrame containing features
        
        Returns:
            dict: Dictionary containing prediction results, format:
                {
                    'predictions': np.ndarray,  # Prediction values
                    'probabilities': np.ndarray or None,  # Classification probabilities (classification tasks only)
                    'task_type': str,  # Task type
                    'n_samples': int  # Number of samples
                }
        """
        # Process input data
        df = self._process_input_data(data)
        
        # Generate features
        X = self._generate_features(df)
        
        # Make predictions
        predictions = self.model.predict(X)
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        # Get classification probabilities (if classification task)
        probabilities = None
        if self.task_type != 'regression':
            try:
                if hasattr(self.model, 'predict_proba'):
                    predict_proba_method = getattr(self.model, 'predict_proba')
                    if callable(predict_proba_method):
                        prob_result = predict_proba_method(X)
                        if prob_result is not None:
                            probabilities = np.array(prob_result) if not isinstance(prob_result, np.ndarray) else prob_result
            except Exception:
                pass  # If unable to get probabilities, continue execution
        
        # Decode classification labels (if label encoder exists)
        if self.task_type != 'regression' and self.label_encoder is not None:
            try:
                decoded_predictions = self.label_encoder.inverse_transform(predictions)
                return {
                    'predictions': decoded_predictions,  # Decoded labels
                    'predictions_encoded': predictions,  # Encoded prediction values
                    'probabilities': probabilities,
                    'task_type': self.task_type,
                    'n_samples': len(predictions)
                }
            except Exception:
                pass  # If decoding fails, return original prediction values
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'task_type': self.task_type,
            'n_samples': len(predictions)
        }
    
    def predict_single(self, sample: Dict[str, Any]) -> Union[float, str, int]:
        """
        Predict single sample, directly return prediction value
        
        Args:
            sample: Feature dictionary for single sample
        
        Returns:
            Prediction value (scalar)
        """
        result = self.predict(sample)
        predictions = result['predictions']
        
        if len(predictions) == 1:
            return predictions[0]
        else:
            raise ValueError(f"Expected single prediction value, but got {len(predictions)}")

def load_model(experiment_dir: str, model_name: str) -> INTERNCRANEPredictor:
    """
    Load trained INTERNCRANE model
    
    Args:
        experiment_dir: Experiment directory path
        model_name: Model name (e.g., 'xgb', 'lgbm', 'catboost')
    
    Returns:
        INTERNCRANEPredictor: Predictor object
    
    Raises:
        FileNotFoundError: If model files cannot be found
        ValueError: If model loading fails
    """
    # Check experiment directory
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    
    # Load training configuration
    config_path = os.path.join(experiment_dir, "run_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    
    training_config = load_config_from_path(config_path)
    task_type = training_config.get('task_type', 'regression')
    
    # Load model
    full_model_name = get_full_model_name(model_name)
    model_dir = os.path.join(experiment_dir, 'models', full_model_name)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    
    model_path = find_model_file(model_dir, full_model_name)
    model = load_model_from_path(model_path, task_type)
    
    # Load preprocessing tools
    data_splits_dir = os.path.join(experiment_dir, 'data_splits')
    
    # Load feature scaler
    scaler = None
    scaler_path = os.path.join(data_splits_dir, "processed_dataset_scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = load_scaler_from_path(scaler_path)
    
    # Load label encoder (classification tasks)
    label_encoder = None
    if task_type != 'regression':
        encoder_path = os.path.join(data_splits_dir, "processed_dataset_label_encoder.joblib")
        if os.path.exists(encoder_path):
            label_encoder = load_label_encoder_from_path(encoder_path)
    
    return INTERNCRANEPredictor(model, training_config, scaler, label_encoder)

def load_model_from_files(model_path: str, config_path: str, 
                         scaler_path: Optional[str] = None, 
                         encoder_path: Optional[str] = None) -> INTERNCRANEPredictor:
    """
    Load model from direct file paths
    
    Args:
        model_path: Model file path
        config_path: Configuration file path
        scaler_path: Scaler file path (optional)
        encoder_path: Label encoder file path (optional)
    
    Returns:
        INTERNCRANEPredictor: Predictor object
    """
    # Load configuration
    training_config = load_config_from_path(config_path)
    task_type = training_config.get('task_type', 'regression')
    
    # Load model
    model = load_model_from_path(model_path, task_type)
    
    # Load preprocessing tools
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        scaler = load_scaler_from_path(scaler_path)
    
    label_encoder = None
    if encoder_path and os.path.exists(encoder_path):
        label_encoder = load_label_encoder_from_path(encoder_path)
    
    return INTERNCRANEPredictor(model, training_config, scaler, label_encoder)

# Convenience functions
def predict(predictor: INTERNCRANEPredictor, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
    """
    Use predictor to make predictions
    
    Args:
        predictor: INTERNCRANEPredictor object
        data: Input data
    
    Returns:
        Prediction results dictionary
    """
    return predictor.predict(data)

def predict_single(predictor: INTERNCRANEPredictor, sample: Dict[str, Any]) -> Union[float, str, int]:
    """
    Predict single sample
    
    Args:
        predictor: INTERNCRANEPredictor object
        sample: Feature dictionary for single sample
    
    Returns:
        Prediction value (scalar)
    """
    return predictor.predict_single(sample)

def quick_predict(experiment_dir: str, model_name: str, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
    """
    Complete model loading and prediction in one step
    
    Args:
        experiment_dir: Experiment directory path
        model_name: Model name
        data: Input data
    
    Returns:
        Prediction results dictionary
    """
    predictor = load_model(experiment_dir, model_name)
    return predictor.predict(data)

# Example usage
if __name__ == "__main__":
    # Example: How to use this API
    print("INTERNCRANE Prediction API Example Usage")
    print("-" * 50)
    
    # 1. Load model
    print("1. Load model:")
    print("predictor = load_model('output/my_experiment', 'xgb')")
    
    # 2. Predict single sample
    print("\n2. Predict single sample:")
    print("""
sample = {
    'SMILES': 'CCO',
    'Solvent_1_SMILES': 'CC(=O)O',
    'Solvent_2_SMILES': 'CCN',
    'feature_1': 1.2,
    'feature_2': 3.4
}
result = predict_single(predictor, sample)
print(f"Prediction value: {result}")
    """)
    
    # 3. Predict multiple samples
    print("\n3. Predict multiple samples:")
    print("""
data = [
    {'SMILES': 'CCO', 'Solvent_1_SMILES': 'CC(=O)O', 'feature_1': 1.2},
    {'SMILES': 'c1ccccc1', 'Solvent_1_SMILES': 'CNC(=O)N', 'feature_1': 2.1}
]
results = predict(predictor, data)
print(f"Prediction results: {results['predictions']}")
    """)
    
    # 4. One-step prediction
    print("\n4. One-step prediction:")
    print("""
results = quick_predict('output/my_experiment', 'xgb', sample)
print(f"Prediction results: {results}")
    """) 