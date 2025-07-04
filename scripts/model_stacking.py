#!/usr/bin/env python3
"""
INTERNCRANE Model Stacking Module

This module provides the core model stacking class ModelStacker, supporting multiple stacking strategies:
1. Simple Average
2. Weighted Average  
3. Learned Meta-Model

Utility functions have been moved to utils modules:
- utils.stacking_ensemble: Ensemble creation tools
- utils.stacking_config: Configuration processing tools
- utils.stacking_evaluation: Evaluation analysis tools
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
import yaml
from datetime import datetime
from typing import Dict, List, Any, Union, Optional
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss
import contextlib

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prediction_api import load_model, INTERNCRANEPredictor
from data_loader import create_validation_dataset

class ModelStacker:
    """
    INTERNCRANE Model Stacker Core Class
    
    Supports multiple stacking strategies:
    1. Simple Average
    2. Weighted Average
    3. Learned Meta-Model
    """
    
    def __init__(self, experiment_dir: Optional[str] = None, models: Optional[List[INTERNCRANEPredictor]] = None):
        """Initialize model stacker"""
        self.experiment_dir = experiment_dir
        self.base_models = {}  # Base models dictionary
        self.model_weights = {}  # Model weights
        self.meta_model = None  # Meta-model
        self.stacking_method = "weighted_average"  # Default stacking method
        self.task_type = None
        self.is_fitted = False
        self._config_path = None  # YAML config file path
        self._config = None  # YAML config content
        
        # If pre-loaded models are provided, add them
        if models:
            for i, model in enumerate(models):
                self.add_model_instance(f"model_{i}", model)
    
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
    
    def add_model(self, model_name: str, weight: float = 1.0):
        """Add model from experiment directory"""
        if self.experiment_dir is None:
            raise ValueError("Need to set experiment_dir to add models")
        
        try:
            with self._suppress_output():
                predictor = load_model(self.experiment_dir, model_name)
            
            self.base_models[model_name] = predictor
            self.model_weights[model_name] = weight
            
            # Set task type (from first model)
            if self.task_type is None:
                self.task_type = predictor.task_type
            
            print(f"✓ Added model: {model_name} (weight: {weight})")
            
        except Exception as e:
            print(f"❌ Failed to add model {model_name}: {e}")
    
    def add_model_instance(self, model_name: str, predictor: INTERNCRANEPredictor, weight: float = 1.0):
        """Add model instance directly"""
        self.base_models[model_name] = predictor
        self.model_weights[model_name] = weight
        
        # Set task type (from first model)
        if self.task_type is None:
            self.task_type = predictor.task_type
        
        print(f"✓ Added model instance: {model_name} (weight: {weight})")
    
    @classmethod
    def from_yaml_config(cls, config_path: str):
        """Create model stacker from YAML configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate config file
        from utils.stacking_config import validate_stacking_config
        validate_stacking_config(config)
        
        # Get basic configuration
        stacking_config = config['stacking']
        experiment_dir = stacking_config.get('experiment_dir')
        
        # Create stacker instance
        stacker = cls(experiment_dir=experiment_dir)
        
        # Set stacking method
        method = stacking_config.get('method', 'weighted_average')
        stacker.set_stacking_method(method)
        
        # Add models
        models = stacking_config.get('models', [])
        for model_config in models:
            model_name = model_config['name']
            weight = model_config.get('weight', 1.0)
            enabled = model_config.get('enabled', True)
            
            if enabled:
                stacker.add_model(model_name, weight)
        
        # Meta-model configuration
        meta_config = stacking_config.get('meta_model', {})
        if meta_config.get('auto_train', False):
            validation_config = meta_config.get('validation', {})
            validation_size = validation_config.get('size', 100)
            auto_load = validation_config.get('auto_load', True)
            split_aware = validation_config.get('split_aware', False)
            
            if auto_load:
                stacker.fit_meta_model(
                    auto_load=True, 
                    validation_size=validation_size,
                    split_aware=split_aware
                )
            else:
                print("⚠️  Meta-model configured for auto-training but auto data loading is disabled")
        
        # Save config info to stacker
        stacker._config_path = config_path
        stacker._config = config
        
        print(f"✓ Created stacker from YAML config: {config_path}")
        return stacker
    
    def set_stacking_method(self, method: str):
        """Set stacking method"""
        valid_methods = ["simple_average", "weighted_average", "ridge", "rf", "logistic"]
        if method not in valid_methods:
            raise ValueError(f"Unsupported stacking method: {method}. Supported methods: {valid_methods}")
        
        self.stacking_method = method
        print(f"✓ Set stacking method: {method}")
    
    def _get_base_predictions(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> np.ndarray:
        """Get predictions from all base models"""
        if not self.base_models:
            raise ValueError("No base models added")
        
        predictions = []
        model_names = list(self.base_models.keys())
        
        for model_name in model_names:
            predictor = self.base_models[model_name]
            try:
                with self._suppress_output():
                    result = predictor.predict(data)
                pred = result['predictions']
                predictions.append(pred)
            except Exception as e:
                print(f"❌ Model {model_name} prediction failed: {e}")
                # Fill with zeros for failed predictions
                if predictions:
                    pred = np.zeros_like(predictions[0])
                else:
                    # Estimate number of samples
                    if isinstance(data, dict):
                        n_samples = 1
                    elif isinstance(data, list):
                        n_samples = len(data)
                    elif isinstance(data, pd.DataFrame):
                        n_samples = len(data)
                    else:
                        n_samples = 1
                    pred = np.zeros(n_samples)
                predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def _get_base_probabilities(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> Optional[np.ndarray]:
        """Get probability predictions from all base models (classification tasks only)"""
        if self.task_type == 'regression':
            return None
        
        probabilities = []
        model_names = list(self.base_models.keys())
        
        for model_name in model_names:
            predictor = self.base_models[model_name]
            try:
                with self._suppress_output():
                    result = predictor.predict(data)
                prob = result.get('probabilities')
                if prob is not None:
                    probabilities.append(prob)
                else:
                    # If model has no probability output, use one-hot encoding instead
                    pred = result['predictions']
                    if hasattr(predictor.label_encoder, 'classes_'):
                        n_classes = len(predictor.label_encoder.classes_)
                        prob = np.zeros((len(pred), n_classes))
                        for i, p in enumerate(pred):
                            prob[i, p] = 1.0
                        probabilities.append(prob)
            except Exception as e:
                print(f"❌ Model {model_name} probability prediction failed: {e}")
                # Fill with uniform distribution for failed predictions
                if probabilities:
                    prob = np.ones_like(probabilities[0]) / probabilities[0].shape[1]
                else:
                    # Estimate number of samples and classes
                    n_samples = 1 if isinstance(data, dict) else len(data)
                    n_classes = 2  # Default binary classification
                    prob = np.ones((n_samples, n_classes)) / n_classes
                probabilities.append(prob)
        
        if probabilities:
            return np.stack(probabilities, axis=1)  # (n_samples, n_models, n_classes)
        return None
    
    def fit_meta_model(self, validation_data: Optional[Union[Dict, List[Dict], pd.DataFrame]] = None, 
                      true_labels: Optional[Union[List, np.ndarray]] = None,
                      auto_load: bool = True,
                      validation_size: int = 100,
                      split_aware: bool = False):
        """Train meta-model"""
        if not self.base_models:
            raise ValueError("No base models added")
        
        # Auto-load validation data
        if validation_data is None and true_labels is None and auto_load:
            if self.experiment_dir is None:
                raise ValueError("Need to set experiment_dir or provide validation_data and true_labels")
            
            print("🔄 Auto-loading validation data from experiment directory...")
            try:
                validation_data, true_labels = create_validation_dataset(
                    self.experiment_dir, 
                    validation_size=validation_size,
                    split_aware=split_aware
                )
                if validation_data is not None:
                    print(f"✓ Auto-loaded validation data: {len(validation_data)} samples")
                else:
                    raise ValueError("Auto-loaded validation data is empty")
            except Exception as e:
                raise ValueError(f"Auto-loading validation data failed: {e}")
        
        if validation_data is None or true_labels is None:
            raise ValueError("Must provide validation_data and true_labels, or set auto_load=True")
        
        print(f"Training meta-model ({self.stacking_method})...")
        
        # Get base model predictions
        base_predictions = self._get_base_predictions(validation_data)
        y_true = np.array(true_labels)
        
        # Create meta-model based on stacking method
        if self.stacking_method == "ridge":
            if self.task_type == 'regression':
                self.meta_model = Ridge(alpha=1.0)
            else:
                self.meta_model = LogisticRegression(max_iter=1000)
        elif self.stacking_method == "rf":
            if self.task_type == 'regression':
                self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                self.meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.stacking_method == "logistic":
            if self.task_type != 'regression':
                self.meta_model = LogisticRegression(max_iter=1000)
            else:
                raise ValueError("Logistic regression meta-model only applies to classification tasks")
        
        # Train meta-model
        if self.meta_model is not None:
            self.meta_model.fit(base_predictions, y_true)
            print(f"✓ Meta-model training completed")
        
        self.is_fitted = True
    
    def predict(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
        """Make predictions using stacked model"""
        if not self.base_models:
            raise ValueError("No base models added")
        
        # Get base model predictions
        base_predictions = self._get_base_predictions(data)
        
        # Combine based on stacking method
        if self.stacking_method == "simple_average":
            final_predictions = np.mean(base_predictions, axis=1)
        
        elif self.stacking_method == "weighted_average":
            # Normalize weights
            model_names = list(self.base_models.keys())
            weights = np.array([self.model_weights[name] for name in model_names])
            weights = weights / np.sum(weights)
            final_predictions = np.average(base_predictions, axis=1, weights=weights)
        
        else:
            # Use meta-model
            if self.meta_model is None:
                raise ValueError("Meta-model not trained, please call fit_meta_model() first")
            final_predictions = self.meta_model.predict(base_predictions)
        
        # Handle probability predictions (classification tasks)
        probabilities = None
        if self.task_type != 'regression':
            base_probabilities = self._get_base_probabilities(data)
            if base_probabilities is not None:
                if self.stacking_method == "simple_average":
                    probabilities = np.mean(base_probabilities, axis=1)
                elif self.stacking_method == "weighted_average":
                    model_names = list(self.base_models.keys())
                    weights = np.array([self.model_weights[name] for name in model_names])
                    weights = weights / np.sum(weights)
                    probabilities = np.average(base_probabilities, axis=1, weights=weights)
                elif self.meta_model is not None:
                    try:
                        # Try to use meta-model's probability prediction (only classifiers have this)
                        probabilities = self.meta_model.predict_proba(base_predictions)  # type: ignore
                    except (AttributeError, TypeError):
                        # Regression models or models without probability output, use simple average
                        probabilities = np.mean(base_probabilities, axis=1) if base_probabilities is not None else None
                    except Exception:
                        # If meta-model probability prediction fails, use simple average
                        probabilities = np.mean(base_probabilities, axis=1) if base_probabilities is not None else None
        
        # Get first model's label_encoder for decoding
        first_model = list(self.base_models.values())[0]
        
        # Build return result
        result = {
            'predictions': final_predictions,
            'probabilities': probabilities,
            'task_type': self.task_type,
            'n_samples': len(final_predictions),
            'stacking_method': self.stacking_method,
            'base_predictions': base_predictions,  # Include base predictions for analysis
            'model_names': list(self.base_models.keys())
        }
        
        # Label decoding for classification tasks
        if self.task_type != 'regression' and first_model.label_encoder is not None:
            try:
                decoded_predictions = first_model.label_encoder.inverse_transform(final_predictions.astype(int))
                result['predictions_decoded'] = decoded_predictions
                result['predictions_encoded'] = final_predictions
                result['predictions'] = decoded_predictions  # Main return is decoded labels
            except Exception:
                pass
        
        return result
    
    def predict_single(self, sample: Dict[str, Any]) -> Union[float, str, int]:
        """Predict single sample, directly return prediction value"""
        result = self.predict(sample)
        predictions = result['predictions']
        
        if len(predictions) == 1:
            return predictions[0]
        else:
            raise ValueError(f"Expected single prediction value, but got {len(predictions)}")
    
    def evaluate(self, test_data: Optional[Union[Dict, List[Dict], pd.DataFrame]] = None, 
                true_labels: Optional[Union[List, np.ndarray]] = None,
                auto_load: bool = True,
                use_test_set: bool = True,
                evaluate_both_sets: bool = True) -> Dict[str, Any]:
        """
        Evaluate stacking model performance (using evaluation functions from utils module)
        
        Args:
            test_data: Test data (optional)
            true_labels: True labels (optional)
            auto_load: Whether to auto-load data
            use_test_set: Whether to use test set (effective when evaluate_both_sets=False)
            evaluate_both_sets: Whether to evaluate both validation and test datasets
        
        Returns:
            dict: Evaluation results
        """
        from utils.stacking_evaluation import evaluate_stacking_performance
        return evaluate_stacking_performance(self, test_data, true_labels, auto_load, use_test_set, evaluate_both_sets)
    
    def save(self, filepath: str):
        """Save stacked model"""
        stacker_data = {
            'experiment_dir': self.experiment_dir,
            'model_weights': self.model_weights,
            'stacking_method': self.stacking_method,
            'task_type': self.task_type,
            'is_fitted': self.is_fitted,
            'model_names': list(self.base_models.keys()),
            'meta_model': self.meta_model,
            'created_at': datetime.now().isoformat()
        }
        
        # Create directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(stacker_data, f)
        
        print(f"✓ Stacked model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load stacked model"""
        with open(filepath, 'rb') as f:
            stacker_data = pickle.load(f)
        
        # Recreate stacker
        stacker = cls(experiment_dir=stacker_data['experiment_dir'])
        stacker.model_weights = stacker_data['model_weights']
        stacker.stacking_method = stacker_data['stacking_method']
        stacker.task_type = stacker_data['task_type']
        stacker.is_fitted = stacker_data['is_fitted']
        stacker.meta_model = stacker_data['meta_model']
        
        # Reload base models
        for model_name in stacker_data['model_names']:
            try:
                stacker.add_model(model_name, stacker.model_weights[model_name])
            except Exception as e:
                print(f"❌ Cannot load model {model_name}: {e}")
        
        print(f"✓ Stacked model loaded from {filepath}")
        return stacker

# Convenience functions, imported from utils modules
def create_ensemble(experiment_dir: str, model_names: List[str], 
                   weights: Optional[List[float]] = None,
                   method: str = "weighted_average") -> ModelStacker:
    """Quick create model ensemble (imported from utils.stacking_ensemble)"""
    from utils.stacking_ensemble import create_ensemble as _create_ensemble
    return _create_ensemble(experiment_dir, model_names, weights, method)

def auto_ensemble(experiment_dir: str, **kwargs) -> ModelStacker:
    """Auto-create optimal ensemble (imported from utils.stacking_ensemble)"""
    from utils.stacking_ensemble import auto_ensemble as _auto_ensemble
    return _auto_ensemble(experiment_dir, **kwargs)

def smart_ensemble_with_meta_learner(experiment_dir: str, **kwargs) -> ModelStacker:
    """Smart meta-learner ensemble (imported from utils.stacking_ensemble)"""
    from utils.stacking_ensemble import smart_ensemble_with_meta_learner as _smart_ensemble
    return _smart_ensemble(experiment_dir, **kwargs)

def load_stacking_config_from_yaml(config_path: str) -> ModelStacker:
    """Load model stacking configuration from YAML config file (convenience function)"""
    return ModelStacker.from_yaml_config(config_path)

# Example usage
if __name__ == "__main__":
    print("CRAFT Model Stacking Core Module")
    print("-" * 50)
    print("Main Features:")
    print("1. ModelStacker - Core stacking class")
    print("2. Support multiple stacking methods (simple average, weighted average, meta-learner)")
    print("3. YAML configuration support")
    print("4. Auto data loading and evaluation")
    print()
    print("Utility functions located in:")
    print("- utils.stacking_ensemble: Ensemble creation tools")
    print("- utils.stacking_config: Configuration processing tools")
    print("- utils.stacking_evaluation: Evaluation analysis tools")
    print()
    print("Use API interface: stacking_api.py")
    print("Command line tool: stacking_yaml_demo.py") 