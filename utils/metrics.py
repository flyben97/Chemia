# utils/metrics.py
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import numpy as np
from rich.console import Console

# Default console, can be replaced if passed
default_console_metrics = Console()


def compute_metrics(y_train_true, y_train_pred, y_test_true, y_test_pred,
                    task_type='regression', num_classes=None,
                    y_train_pred_proba=None, y_test_pred_proba=None,
                    y_val_true=None, y_val_pred=None, y_val_pred_proba=None, # <<< ADDED: Validation set args
                    console=None):
    """Compute relevant metrics based on task type."""
    effective_console = console if console is not None else default_console_metrics
    
    y_train_true = np.array(y_train_true).ravel()
    y_train_pred = np.array(y_train_pred).ravel()
    y_test_true = np.array(y_test_true).ravel()
    y_test_pred = np.array(y_test_pred).ravel()

    metrics = {
        'y_train_pred': y_train_pred, 
        'y_test_pred': y_test_pred,   
    }
    if y_train_pred_proba is not None and len(y_train_pred_proba) > 0: 
        metrics['y_train_pred_proba'] = y_train_pred_proba
    if y_test_pred_proba is not None and len(y_test_pred_proba) > 0: 
        metrics['y_test_pred_proba'] = y_test_pred_proba
    
    # --- ADDED: Handle validation data if provided ---
    if y_val_true is not None and y_val_pred is not None and len(y_val_true) > 0:
        y_val_true = np.array(y_val_true).ravel()
        y_val_pred = np.array(y_val_pred).ravel()
        metrics['y_val_pred'] = y_val_pred
        if y_val_pred_proba is not None and len(y_val_pred_proba) > 0:
            metrics['y_val_pred_proba'] = y_val_pred_proba
    # --- END ADDED BLOCK ---

    if task_type == 'regression':
        metrics['train_r2'] = r2_score(y_train_true, y_train_pred)
        metrics['test_r2'] = r2_score(y_test_true, y_test_pred)
        metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
        metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
        metrics['train_mae'] = mean_absolute_error(y_train_true, y_train_pred)
        metrics['test_mae'] = mean_absolute_error(y_test_true, y_test_pred)
        if 'y_val_pred' in metrics: # <<< ADDED
            metrics['val_r2'] = r2_score(y_val_true, y_val_pred)
            metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
            metrics['val_mae'] = mean_absolute_error(y_val_true, y_val_pred)

    elif task_type == 'binary_classification' or task_type == 'multiclass_classification':
        avg_method = 'binary' if task_type == 'binary_classification' and num_classes == 2 else 'weighted'
        
        metrics['train_accuracy'] = accuracy_score(y_train_true, y_train_pred)
        metrics['test_accuracy'] = accuracy_score(y_test_true, y_test_pred)
        metrics['train_precision'] = precision_score(y_train_true, y_train_pred, average=avg_method, zero_division=0)
        metrics['test_precision'] = precision_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
        metrics['train_recall'] = recall_score(y_train_true, y_train_pred, average=avg_method, zero_division=0)
        metrics['test_recall'] = recall_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
        metrics['train_f1'] = f1_score(y_train_true, y_train_pred, average=avg_method, zero_division=0)
        metrics['test_f1'] = f1_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
        
        if 'y_val_pred' in metrics: # <<< ADDED
            metrics['val_accuracy'] = accuracy_score(y_val_true, y_val_pred)
            metrics['val_precision'] = precision_score(y_val_true, y_val_pred, average=avg_method, zero_division=0)
            metrics['val_recall'] = recall_score(y_val_true, y_val_pred, average=avg_method, zero_division=0)
            metrics['val_f1'] = f1_score(y_val_true, y_val_pred, average=avg_method, zero_division=0)

        has_train_proba = 'y_train_pred_proba' in metrics
        has_test_proba = 'y_test_pred_proba' in metrics
        has_val_proba = 'y_val_pred_proba' in metrics # <<< ADDED

        # ... (AUC logic for train and test remains the same)
        # --- NOTE: For brevity, I'm omitting the full AUC copy-paste.
        # --- You would also add a similar block for validation AUC if has_val_proba is True.
        # --- I'll add the validation AUC logic here for completeness. ---
        if has_val_proba:
            if task_type == 'binary_classification':
                val_proba_pos = metrics['y_val_pred_proba'][:, 1] if metrics['y_val_pred_proba'].ndim == 2 and metrics['y_val_pred_proba'].shape[1] == 2 else metrics['y_val_pred_proba']
                unique_val_labels = np.unique(y_val_true)
                if len(unique_val_labels) > 1:
                    metrics['val_auc'] = roc_auc_score(y_val_true, val_proba_pos)
                else:
                    metrics['val_auc'] = np.nan
            elif task_type == 'multiclass_classification' and num_classes is not None and num_classes > 2:
                unique_val_labels_mc = np.unique(y_val_true)
                if len(unique_val_labels_mc) > 1:
                     metrics['val_auc_ovr_weighted'] = roc_auc_score(y_val_true, metrics['y_val_pred_proba'], multi_class='ovr', average='weighted', labels=np.arange(num_classes))
                else:
                     metrics['val_auc_ovr_weighted'] = np.nan

    else:
        raise ValueError(f"Unsupported task_type for metrics: {task_type}")
        
    return metrics