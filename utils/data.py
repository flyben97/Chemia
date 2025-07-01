# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from rich.console import Console

default_console = Console()

def encode_labels(y, task_type='regression', console=None):
    """
    Encodes the target variable y. For classification, it uses LabelEncoder.
    For regression, it ensures the output is a 2D column vector.
    
    Returns the processed y and the fitted label_encoder instance (or None for regression).
    """
    if console is None:
        console = default_console

    if y.size == 0:
        return np.array([]), None

    label_encoder = None
    if task_type == 'regression':
        y_processed = y.reshape(-1, 1) if y.ndim == 1 else y
    elif task_type in ['binary_classification', 'multiclass_classification']:
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit_transform(np.array(y).ravel())
        if hasattr(label_encoder, 'classes_') and label_encoder.classes_ is not None:
            try:
                original_classes = list(label_encoder.classes_)
                if y_processed is not None and len(y_processed) > 0:
                    encoded_classes = list(np.unique(y_processed))
                    console.print(f"  [dim]Original y classes: {original_classes}, Encoded y classes: {encoded_classes}[/dim]")
                else:
                    console.print(f"  [dim]Original y classes: {original_classes}[/dim]")
            except Exception as e:
                console.print(f"  [dim]Label encoding completed (details unavailable: {e})[/dim]")
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
        
    return y_processed, label_encoder

def split_data(X, y, train_size=0.6, valid_size=0.2, test_size=0.2, random_state=42, task_type='regression'):
    """
    Splits features X and labels y into training, validation, and test sets.
    Handles stratification for classification tasks to maintain label balance.
    """
    if not abs(train_size + valid_size + test_size - 1.0) < 1e-6:
         raise ValueError("Split ratios (train_size, valid_size, test_size) must sum to 1.0.")
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive.")
    if valid_size < 0:
        raise ValueError("valid_size must be non-negative.")

    stratify_option = y if task_type in ['binary_classification', 'multiclass_classification'] and y is not None and len(np.unique(y)) > 1 else None
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=stratify_option
    )
    
    if valid_size == 0:
        X_train, y_train = X_temp, y_temp
        X_val = np.empty((0, X.shape[1]), dtype=X.dtype)
        y_val = np.empty((0,), dtype=y.dtype)
    else:
        stratify_temp_option = y_temp if task_type in ['binary_classification', 'multiclass_classification'] and y_temp is not None and len(np.unique(y_temp)) > 1 else None
        valid_ratio_of_temp = valid_size / (train_size + valid_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=valid_ratio_of_temp, random_state=random_state, shuffle=True, stratify=stratify_temp_option
        )
        
    return X_train, X_val, X_test, y_train, y_val, y_test
