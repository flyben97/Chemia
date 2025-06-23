# utils/io.py
import os
import torch
import joblib
from datetime import datetime
import numpy as np 
import pandas as pd 
import json 
from rich.console import Console
import textwrap

default_console = Console()

CRAFT_BANNER = r"""
   ██████╗ ██████╗   █████╗ ███████╗████████╗
  ██╔════╝ ██╔══██╗ ██╔══██╗██╔════╝╚══██╔══╝
  ██║      ██████╔╝███████║█████╗     ██║   
  ██║      ██╔══██╗██╔══██║██╔══╝     ██║   
  ╚██████╗ ██║  ██║██║  ██║██║        ██║   
   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝   
"""

def _format_config_to_str(config_dict: dict) -> str:
    """
    Formats a configuration dictionary into a readable string for logging.
    This function flattens the nested dictionary for clarity.
    """
    if not isinstance(config_dict, dict):
        return ""
    
    try:
        # Flatten the nested dictionary for a cleaner log display
        # e.g., data: {smiles_col: 'SMILES'} becomes 'data_smiles_col: SMILES'
        flat_config = pd.json_normalize(config_dict, sep='.').to_dict(orient='records')[0]
    except (IndexError, TypeError):
        # Fallback for empty or non-standard dict
        flat_config = config_dict

    parts = []
    for key, value in flat_config.items():
        if value is None:
            continue
        if isinstance(value, list):
            # For lists, just show the key and number of items or the items themselves if short
            if len(value) > 3:
                parts.append(f"{key}=[{len(value)} items]")
            else:
                parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={value}")
    
    return ", ".join(parts)


def ensure_experiment_directories(base_output_dir, experiment_run_name, console=None):
    if console is None: console = default_console
    run_dir = os.path.join(base_output_dir, experiment_run_name)
    models_dir = os.path.join(run_dir, 'models') 
    data_splits_dir = os.path.join(run_dir, 'data_splits')
    for d in [base_output_dir, run_dir, models_dir, data_splits_dir]:
        os.makedirs(d, exist_ok=True)
    return run_dir, models_dir, data_splits_dir

def ensure_model_specific_directory(models_base_dir, model_name_short, console=None):
    if console is None: console = default_console
    model_dir = os.path.join(models_base_dir, model_name_short)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def save_model_artifact(model_object, model_artifact_name, model_specific_output_dir, is_pytorch_model=False, console=None):
    if console is None: console = default_console
    if is_pytorch_model:
        model_path = os.path.join(model_specific_output_dir, f"{model_artifact_name}.pth")
        torch.save(model_object.state_dict(), model_path)
    else: 
        model_path = os.path.join(model_specific_output_dir, f"{model_artifact_name}.joblib")
        joblib.dump(model_object, model_path)
    console.print(f"  [green]✓ Saved model artifact[/green] '{model_artifact_name}' to [dim]{model_path}[/dim]")
    return model_path

def save_hyperparameters(hyperparameters, model_name_short, model_specific_output_dir, console=None):
    if console is None: console = default_console
    params_path = os.path.join(model_specific_output_dir, f"{model_name_short}_hyperparameters.json")
    with open(params_path, 'w', encoding='utf-8') as f:
        serializable_params = {}
        for k, v in hyperparameters.items():
            if isinstance(v, np.integer): serializable_params[k] = int(v)
            elif isinstance(v, np.floating): serializable_params[k] = float(v)
            elif isinstance(v, np.ndarray): serializable_params[k] = v.tolist()
            else: serializable_params[k] = v
        json.dump(serializable_params, f, indent=4)
    console.print(f"  [green]✓ Saved hyperparameters[/green] for {model_name_short} to [dim]{params_path}[/dim]")
    return params_path

def save_predictions(metrics_dict, model_name_short, model_specific_output_dir, y_train_true, y_test_true, y_val_true=None, console=None):
    if console is None: console = default_console
    predictions_dir = os.path.join(model_specific_output_dir, "predictions_final_model") 
    os.makedirs(predictions_dir, exist_ok=True)

    def save_df(df, filename):
        path = os.path.join(predictions_dir, filename)
        df.to_csv(path, index=False, encoding='utf-8')
        console.print(f"  [green]✓ Saved predictions[/green] for {model_name_short} to [dim]{path}[/dim]")

    if 'y_train_pred' in metrics_dict:
        train_df = pd.DataFrame({'y_true': y_train_true.ravel(), 'y_pred': metrics_dict['y_train_pred'].ravel()})
        if 'y_train_pred_proba' in metrics_dict and metrics_dict['y_train_pred_proba'] is not None:
            proba_df = pd.DataFrame(metrics_dict['y_train_pred_proba'], columns=[f'proba_class_{i}' for i in range(metrics_dict['y_train_pred_proba'].shape[1])])
            train_df = pd.concat([train_df, proba_df], axis=1)
        save_df(train_df, f"{model_name_short}_train_predictions.csv")

    if 'y_test_pred' in metrics_dict and y_test_true is not None and y_test_true.size > 0:
        test_df = pd.DataFrame({'y_true': y_test_true.ravel(), 'y_pred': metrics_dict['y_test_pred'].ravel()})
        if 'y_test_pred_proba' in metrics_dict and metrics_dict['y_test_pred_proba'] is not None:
            proba_df = pd.DataFrame(metrics_dict['y_test_pred_proba'], columns=[f'proba_class_{i}' for i in range(metrics_dict['y_test_pred_proba'].shape[1])])
            test_df = pd.concat([test_df, proba_df], axis=1)
        save_df(test_df, f"{model_name_short}_test_predictions.csv")

    if 'y_val_pred' in metrics_dict and y_val_true is not None and y_val_true.size > 0:
        val_df = pd.DataFrame({'y_true': y_val_true.ravel(), 'y_pred': metrics_dict['y_val_pred'].ravel()})
        if 'y_val_pred_proba' in metrics_dict and metrics_dict['y_val_pred_proba'] is not None:
            proba_df = pd.DataFrame(metrics_dict['y_val_pred_proba'], columns=[f'proba_class_{i}' for i in range(metrics_dict['y_val_pred_proba'].shape[1])])
            val_df = pd.concat([val_df, proba_df], axis=1)
        save_df(val_df, f"{model_name_short}_validation_predictions.csv")

def save_cv_fold_predictions(cv_predictions_dict, model_name_short, model_specific_output_dir, console=None):
    if console is None: console = default_console
    if cv_predictions_dict is None:
        console.print(f"  [dim]No CV predictions to save for {model_name_short}.[/dim]")
        return
    predictions_dir = os.path.join(model_specific_output_dir, "predictions_cv_oof") 
    os.makedirs(predictions_dir, exist_ok=True)
    df_data = {'y_true_oof': cv_predictions_dict['y_true_oof'].ravel(), 'y_pred_oof': cv_predictions_dict['y_pred_oof'].ravel()}
    cv_df = pd.DataFrame(df_data)
    if cv_predictions_dict.get('y_proba_oof') is not None:
        proba_df_cv = pd.DataFrame(cv_predictions_dict['y_proba_oof'], columns=[f'proba_class_{i}' for i in range(cv_predictions_dict['y_proba_oof'].shape[1])])
        cv_df = pd.concat([cv_df, proba_df_cv], axis=1)
    cv_pred_path = os.path.join(predictions_dir, f"{model_name_short}_cv_oof_predictions.csv")
    cv_df.to_csv(cv_pred_path, index=False, encoding='utf-8') 
    console.print(f"  [green]✓ Saved CV OOF predictions[/green] for {model_name_short} to [dim]{cv_pred_path}[/dim]")


def log_results(model_name_short, best_params, best_score, metrics_dict, 
                model_specific_output_dir, task_type='regression', 
                best_trial_fold_scores=None, console=None,
                experiment_run_name=None, model_runtime_seconds=None, config=None,
                data_shapes=None):
    if console is None: console = default_console
    
    log_path = os.path.join(model_specific_output_dir, f"{model_name_short}_results.log") 
    
    main_metric_name = "R²" if task_type == 'regression' else "F1 Score" 
    if task_type == 'binary_classification': main_metric_name += " (Binary)"
    elif task_type == 'multiclass_classification': main_metric_name += " (Weighted)"

    log_content = []
    log_content.append(CRAFT_BANNER)
    log_content.append("Chemical Representation and Analysis for Functional Targets (CRAFT) - v1.1.1\n")
    log_content.append("="*88)
    log_content.append(f"{'Log Generated:':<25} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if experiment_run_name: log_content.append(f"{'Experiment Run Name:':<25} {experiment_run_name}")
    if config:
        config_str = _format_config_to_str(config)
        wrapped_config = textwrap.fill(config_str, width=86, initial_indent=' ' * 27, subsequent_indent=' ' * 27).strip()
        log_content.append(f"{'Run Configuration:':<25} {wrapped_config}")

    log_content.append("-" * 88)
    log_content.append(f"{'Model Trained:':<25} {model_name_short.upper()}")
    log_content.append(f"{'Task Type:':<25} {task_type}")

    if data_shapes:
        log_content.append("\n" + "--- Data & Split Information ---")
        train_shape = data_shapes.get('train', ('N/A', 'N/A'))
        log_content.append(f"{'Training Set Dimensions:':<25} {train_shape[0]} samples x {train_shape[1]} features")

        if data_shapes.get('hpo_method') == 'CV HPO':
            folds = data_shapes.get('cv_folds', 'N/A')
            log_content.append(f"{'HPO Method:':<25} Cross-Validation ({folds} folds on training data)")
        else:
            val_shape = data_shapes.get('val', ('N/A', 'N/A'))
            log_content.append(f"{'Validation Set Dimensions:':<25} {val_shape[0]} samples x {val_shape[1]} features")
            log_content.append(f"{'HPO Method:':<25} Hold-out Validation")
        
        test_shape = data_shapes.get('test', ('N/A', 'N/A'))
        log_content.append(f"{'Test Set Dimensions:':<25} {test_shape[0]} samples x {test_shape[1]} features")
    log_content.append("="*88 + "\n")
    
    log_content.append("--- Hyperparameter Optimization (Optuna) ---")
    log_content.append(f"Best HPO Score ({main_metric_name}): {best_score:.4f}")
    if best_trial_fold_scores: 
        log_content.append(f"  Scores from individual CV folds: [{', '.join([f'{s:.4f}' for s in best_trial_fold_scores])}]")
    
    log_content.append("\nBest Hyperparameters Found:")
    log_content.append(json.dumps(best_params, indent=4))
    log_content.append("\n" + "="*88 + "\n")
    log_content.append("--- Final Model Performance ---")
    
    header = f"| {'Metric':<25} | {'Train Set':^15} | {'Validation Set':^15} | {'Test Set':^15} |"
    separator = f"|{'-'*27}|{'-'*17}|{'-'*17}|{'-'*17}|"
    log_content.append(header)
    log_content.append(separator)

    metric_display_map = {'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE', 'accuracy': 'Accuracy', 'f1': 'F1 Score (Weighted)', 'precision': 'Precision (Weighted)', 'recall': 'Recall (Weighted)', 'auc': 'AUC (Binary)', 'auc_ovr_weighted': 'AUC (OvR Weighted)'}
    if task_type == 'binary_classification':
        metric_display_map.update({'f1': 'F1 Score (Binary)', 'precision': 'Precision (Binary)', 'recall': 'Recall (Binary)'})
    
    metric_keys_order_reg = ['r2', 'rmse', 'mae']
    metric_keys_order_cls = ['accuracy', 'f1', 'precision', 'recall', 'auc', 'auc_ovr_weighted']
    metrics_to_log_stems = metric_keys_order_reg if task_type == 'regression' else metric_keys_order_cls

    for stem in metrics_to_log_stems:
        train_key, val_key, test_key = f'train_{stem}', f'val_{stem}', f'test_{stem}'
        if train_key in metrics_dict or test_key in metrics_dict:
            train_val = metrics_dict.get(train_key)
            val_val = metrics_dict.get(val_key)
            test_val = metrics_dict.get(test_key)
            train_str = f"{train_val:.4f}" if train_val is not None and not np.isnan(train_val) else "N/A"
            val_str = f"{val_val:.4f}" if val_val is not None and not np.isnan(val_val) else "N/A"
            test_str = f"{test_val:.4f}" if test_val is not None and not np.isnan(test_val) else "N/A"
            display_name = metric_display_map.get(stem, stem.title())
            log_content.append(f"| {display_name:<25} | {train_str:^15} | {val_str:^15} | {test_str:^15} |")

    log_content.append(separator)
    log_content.append("\n" + "="*88 + "\n")
    if model_runtime_seconds is not None: log_content.append(f"Model-Specific Runtime (HPO, Fit, Eval): {model_runtime_seconds:.2f} seconds")
    log_content.append("\n*** End of CRAFT Log for this Model ***")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_content))
    console.print(f"  [green]✓ Logged results[/green] for {model_name_short} ({task_type}) to [dim]{log_path}[/dim]")
    return log_path

def log_experiment_summary(run_base_dir, experiment_run_name, config, total_duration_seconds, start_timestamp, all_results, console=None):
    if console is None: console = default_console
    summary_log_path = os.path.join(run_base_dir, "_experiment_summary.log")
    
    start_date_str = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
    days, rem = divmod(total_duration_seconds, 86400); hours, rem = divmod(rem, 3600); mins, secs = divmod(rem, 60)
    formatted_duration = f"{int(days)}d {int(hours)}h {int(mins)}m {int(secs)}s"

    log_content = [CRAFT_BANNER, "Chemical Representation and Analysis for Functional Targets (CRAFT) - v1.1.1\n", "=" * 88, "  EXPERIMENT RUN SUMMARY", "=" * 88]
    log_content.append(f"{'Experiment Name:':<25} {experiment_run_name}")
    log_content.append(f"{'Run Started:':<25} {datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append(f"{'Total Runtime:':<25} {formatted_duration}")
    if config:
        config_str = _format_config_to_str(config)
        wrapped_config = textwrap.fill(config_str, width=86, initial_indent=' ' * 27, subsequent_indent=' ' * 27).strip()
        log_content.append(f"{'Run Configuration:':<25} {wrapped_config}")
    log_content.extend(["\n" + "="*88, "  MODEL PERFORMANCE SUMMARY", "="*88])

    if not all_results:
        log_content.append("\nNo models were successfully run to summarize.")
    else:
        for result in all_results:
            model_name, task_type = result.get('model_name', 'Unknown'), result.get('task_type', 'unknown')
            main_metric_name = "R²" if task_type == 'regression' else "F1 (Weighted)"
            if task_type == 'binary_classification': main_metric_name = "F1 (Binary)"
            log_content.extend([f"\n----- Model: {model_name.upper()} -----", f"Task Type: {task_type}"])
            
            data_shapes = result.get('data_shapes')
            if data_shapes:
                train_shape = data_shapes.get('train', ('N/A', 'N/A'))
                dim_str = f"Data Dimensions: Train={train_shape}"
                if data_shapes.get('hpo_method') == 'CV HPO':
                    folds = data_shapes.get('cv_folds', 'N/A')
                    dim_str += f", HPO on {folds}-Folds CV"
                else:
                    val_shape = data_shapes.get('val', ('N/A', 'N/A'))
                    dim_str += f", Val={val_shape}"
                
                test_shape = data_shapes.get('test', ('N/A', 'N/A'))
                dim_str += f", Test={test_shape}"
                log_content.append(dim_str)
            
            log_content.extend([f"Best HPO Score ({main_metric_name}): {result.get('best_optuna_score', float('nan')):.4f}", "Best Hyperparameters:", json.dumps(result.get('best_params', {}), indent=2), "\nFinal Metrics:"])
            header = f"| {'Metric':<25} | {'Train Set':^15} | {'Validation Set':^15} | {'Test Set':^15} |"
            separator = f"|{'-'*27}|{'-'*17}|{'-'*17}|{'-'*17}|"
            log_content.extend([header, separator])
            metric_display_map = {'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE', 'accuracy': 'Accuracy', 'f1': 'F1 (Weighted)', 'precision': 'Precision (W)', 'recall': 'Recall (W)', 'auc': 'AUC (Binary)', 'auc_ovr_weighted': 'AUC (OvR W)'}
            if task_type == 'binary_classification':
                metric_display_map.update({'f1': 'F1 (Binary)', 'precision': 'Precision (B)', 'recall': 'Recall (B)'})
            metric_keys = ['r2', 'rmse', 'mae'] if task_type == 'regression' else ['accuracy', 'f1', 'precision', 'recall', 'auc', 'auc_ovr_weighted']
            for stem in metric_keys:
                if f'train_{stem}' in result or f'test_{stem}' in result:
                    train_val, val_val, test_val = result.get(f'train_{stem}'), result.get(f'val_{stem}'), result.get(f'test_{stem}')
                    train_str = f"{train_val:.4f}" if train_val is not None and not np.isnan(train_val) else "N/A"
                    val_str = f"{val_val:.4f}" if val_val is not None and not np.isnan(val_val) else "N/A"
                    test_str = f"{test_val:.4f}" if test_val is not None and not np.isnan(test_val) else "N/A"
                    display_name = metric_display_map.get(stem, stem.title())
                    log_content.append(f"| {display_name:<25} | {train_str:^15} | {val_str:^15} | {test_str:^15} |")
            log_content.append(separator)

    log_content.extend(["\n\n" + "="*88, "*** End of Summary Log ***"])
    with open(summary_log_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_content))

    console.print(f"[bold green]✓ Experiment summary log created at:[/bold green] [dim]{summary_log_path}[/dim]")
    return summary_log_path

def save_data_splits_csv(data_splits_dir, dataset_name_prefix,
                         X_train, y_train, X_val, y_val, X_test, y_test,
                         scaler=None, label_encoder=None, console=None):
    from sklearn.preprocessing import LabelEncoder
    
    if console is None:
        console = default_console
    csv_paths = {}
    
    datasets_to_save = {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test
    }
    if X_val is not None and X_val.size > 0:
        datasets_to_save['X_val'] = X_val
    if y_val is not None and y_val.size > 0:
        datasets_to_save['y_val'] = y_val

    for name, data_array in datasets_to_save.items():
        if data_array is None or data_array.size == 0:
            console.print(f"  [dim]Skipping save for {name} as it is empty.[/dim]")
            csv_paths[name] = None
            continue
        df_to_save = pd.DataFrame(data_array)
        if data_array.ndim == 1 and name.startswith('y_'):
            df_to_save.columns = ['target']
        
        csv_file_path = os.path.join(data_splits_dir, f"{dataset_name_prefix}_{name}.csv")
        df_to_save.to_csv(csv_file_path, index=False, encoding='utf-8')
        csv_paths[name] = csv_file_path
        console.print(f"  [green]✓ Saved {name}[/green] (CSV format) to [dim]{csv_file_path}[/dim]")
    
    if label_encoder is not None and isinstance(label_encoder, LabelEncoder):
        le_classes_path = os.path.join(data_splits_dir, f"{dataset_name_prefix}_label_encoder_classes.txt")
        with open(le_classes_path, 'w', encoding='utf-8') as f:
            for cls_item in label_encoder.classes_:
                f.write(f"{cls_item}\n")
        console.print(f"  [green]✓ Saved label encoder classes[/green] to [dim]{le_classes_path}[/dim]")
        
        le_joblib_path = os.path.join(data_splits_dir, f"{dataset_name_prefix}_label_encoder.joblib")
        joblib.dump(label_encoder, le_joblib_path)
        console.print(f"  [green]✓ Saved label encoder object[/green] to [dim]{le_joblib_path}[/dim]")

    if scaler is not None:
        scaler_joblib_path = os.path.join(data_splits_dir, f"{dataset_name_prefix}_scaler.joblib")
        joblib.dump(scaler, scaler_joblib_path)
        console.print(f"  [green]✓ Saved scaler[/green] to [dim]{scaler_joblib_path}[/dim]")
    
    return csv_paths

def load_data_splits_csv(csv_dir, dataset_name_prefix, console=None):
    # ... (此函数是读取，通常不需要修改，但为了稳健可以加上 encoding) ...
    if console is None:
        console = default_console
    data_dict = {}
    console.print(f"  [cyan]Attempting to load data splits from CSV files in {csv_dir} with prefix {dataset_name_prefix}...[/cyan]")
    
    expected_base_names = ['X_train', 'y_train', 'X_test', 'y_test']
    optional_base_names = ['X_val', 'y_val']

    for name_part in expected_base_names + optional_base_names:
        fname = f"{dataset_name_prefix}_{name_part}.csv"
        fpath = os.path.join(csv_dir, fname)
        if os.path.exists(fpath):
            try:
                loaded_array = pd.read_csv(fpath, encoding='utf-8').values
            except Exception:
                loaded_array = pd.read_csv(fpath).values # Fallback to default
            if loaded_array.shape[1] == 1 and name_part.startswith('y_'):
                data_dict[name_part] = loaded_array.squeeze()
            else:
                data_dict[name_part] = loaded_array
            console.print(f"    [green]✓ Loaded {name_part}[/green] from [dim]{fpath}[/dim]")
        elif name_part in optional_base_names:
            data_dict[name_part] = np.array([])
            console.print(f"    [dim]Optional file {fpath} not found, {name_part} set to empty array.[/dim]")
        else:
            console.print(f"    [bold red]❌ Error: Expected CSV file {fpath} not found.[/bold red]")
            return None

    le_classes_path = os.path.join(csv_dir, f"{dataset_name_prefix}_label_encoder_classes.txt")
    if os.path.exists(le_classes_path):
        # --- 关键修正 5：为 label encoder classes 文件读取添加 encoding ---
        with open(le_classes_path, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f]
        data_dict['label_encoder_classes'] = np.array(classes)
        console.print(f"    [green]✓ Loaded label_encoder_classes[/green] from [dim]{le_classes_path}[/dim]")
    else:
        data_dict['label_encoder_classes'] = None
        console.print(f"    [dim]Label encoder classes file not found at {le_classes_path}.[/dim]")
    return data_dict

# ... (load_scaler 和 load_label_encoder 是二进制文件，不需要 encoding)
def load_scaler(scaler_filepath, console=None):
    if console is None: console = default_console
    if scaler_filepath and os.path.exists(scaler_filepath): 
        scaler = joblib.load(scaler_filepath)
        console.print(f"  [green]✓ Loaded scaler[/green] from [dim]{scaler_filepath}[/dim]")
        return scaler
    console.print(f"  [yellow]Scaler file not found at {scaler_filepath}.[/yellow]")
    return None

def load_label_encoder(le_filepath, console=None):
    if console is None: console = default_console
    if le_filepath and os.path.exists(le_filepath): 
        label_encoder = joblib.load(le_filepath)
        console.print(f"  [green]✓ Loaded label encoder[/green] from [dim]{le_filepath}[/dim]")
        return label_encoder
    console.print(f"  [yellow]Label encoder file not found at {le_filepath}.[/yellow]")
    return None

