# utils/shap_integration.py
"""
SHAP Integration Module

Integrates SHAP analysis into the training pipeline.
Automatically performs SHAP analysis on compatible models after training.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from rich.console import Console

try:
    from utils.shap_analyzer import SHAPAnalyzer, SHAP_AVAILABLE
except ImportError:
    SHAP_AVAILABLE = False

console = Console(width=120)

# Models that support SHAP analysis
SHAP_COMPATIBLE_MODELS = {
    'xgboost', 'xgbregressor', 'xgbclassifier',
    'lightgbm', 'lgbmregressor', 'lgbmclassifier',
    'catboost', 'catboostregressor', 'catboostclassifier',
    'randomforest', 'randomforestregressor', 'randomforestclassifier',
    'extratrees', 'extratreesregressor', 'extratreesclassifier',
    'gradientboosting', 'gradientboostingregressor', 'gradientboostingclassifier',
    'histgradientboosting', 'histgradientboostingregressor', 'histgradientboostingclassifier',
    'ridge', 'lasso', 'elasticnet', 'bayesianridge',
    'svr', 'svc', 'knn', 'kneighborsregressor', 'kneighborsclassifier',
    'kernelridge', 'gaussianprocess', 'gaussianprocessregressor', 'gaussianprocessclassifier',
    'tabnet', 'tabnetregressor', 'tabnetclassifier',
    'ann', 'mlpregressor', 'mlpclassifier',
}

def is_model_shap_compatible(model_name: str) -> bool:
    """Check if a model supports SHAP analysis."""
    return model_name.lower() in SHAP_COMPATIBLE_MODELS

def perform_shap_analysis(
    model,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Optional[list] = None,
    output_dir: Optional[str] = None,
    max_samples: int = 100,
    console: Optional[Console] = None
) -> bool:
    """
    Perform SHAP analysis on a trained model.

    Args:
        model: Trained model instance
        model_name: Name of the model
        X_train: Training data for background dataset
        X_test: Test data to explain
        feature_names: List of feature names
        output_dir: Directory to save SHAP analysis results
        max_samples: Maximum samples to analyze
        console: Rich console for logging

    Returns:
        True if analysis successful, False otherwise
    """
    if console is None:
        console = Console(width=120)

    if not SHAP_AVAILABLE:
        console.print(f"[yellow]⚠ SHAP not available. Skipping SHAP analysis for {model_name}[/yellow]")
        return False

    if not is_model_shap_compatible(model_name):
        console.print(f"[dim]ℹ {model_name} does not support SHAP analysis[/dim]")
        return False

    try:
        console.print(f"[cyan]  Performing SHAP analysis for {model_name}...[/cyan]")

        # Create analyzer
        analyzer = SHAPAnalyzer(model, X_train, feature_names)

        # Generate report
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            analyzer.generate_report(X_test[:max_samples], output_dir)
            console.print(f"[green]  ✓ SHAP analysis complete[/green]")
            return True
        else:
            console.print(f"[yellow]  ⚠ No output directory specified[/yellow]")
            return False

    except Exception as e:
        console.print(f"[yellow]  ⚠ SHAP analysis failed: {e}[/yellow]")
        return False

def get_shap_feature_importance(
    model,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Optional[list] = None,
    top_n: int = 20,
    console: Optional[Console] = None
) -> Optional[pd.DataFrame]:
    """
    Get feature importance from SHAP values.

    Args:
        model: Trained model instance
        model_name: Name of the model
        X_train: Training data for background dataset
        X_test: Test data to explain
        feature_names: List of feature names
        top_n: Number of top features to return
        console: Rich console for logging

    Returns:
        DataFrame with feature importance or None if failed
    """
    if console is None:
        console = Console(width=120)

    if not SHAP_AVAILABLE:
        return None

    if not is_model_shap_compatible(model_name):
        return None

    try:
        analyzer = SHAPAnalyzer(model, X_train, feature_names)

        if not analyzer.create_explainer():
            return None

        if not analyzer.explain(X_test):
            return None

        return analyzer.get_feature_importance(top_n)

    except Exception as e:
        console.print(f"[yellow]  ⚠ Failed to get SHAP feature importance: {e}[/yellow]")
        return None

def should_perform_shap_analysis(config: Dict[str, Any]) -> bool:
    """
    Determine if SHAP analysis should be performed based on config.

    Args:
        config: Training configuration dictionary

    Returns:
        True if SHAP analysis should be performed
    """
    if not SHAP_AVAILABLE:
        return False

    # Check if SHAP analysis is explicitly enabled in config
    shap_config = config.get('shap_analysis', {})
    if isinstance(shap_config, dict):
        return shap_config.get('enabled', False)

    return False

def get_shap_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get SHAP analysis configuration from training config.

    Args:
        config: Training configuration dictionary

    Returns:
        SHAP configuration dictionary
    """
    default_config = {
        'enabled': False,
        'max_samples': 100,
        'models': None,  # None means all compatible models
        'generate_plots': True,
        'save_feature_importance': True,
    }

    shap_config = config.get('shap_analysis', {})
    if isinstance(shap_config, dict):
        default_config.update(shap_config)

    return default_config
