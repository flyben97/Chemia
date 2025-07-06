# utils/io_handler.py
import os
import torch
import joblib
from datetime import datetime
import numpy as np 
import pandas as pd 
import json 
from rich.console import Console
import textwrap
import re
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

console = Console(width=120, highlight=False)

# --- 全局控制台 ---
default_console = Console(width=120)

# --- 横幅 ---
CHEMIA_BANNER = r"""
  ███████╗██╗  ██╗███████╗███╗   ███╗██╗ █████╗ 
  ██╔════╝██║  ██║██╔════╝████╗ ████║██║██╔══██╗
  ██║     ███████║█████╗  ██╔████╔██║██║███████║
  ██║     ██╔══██║██╔══╝  ██║╚██╔╝██║██║██╔══██║
  ╚██████╗██║  ██║███████╗██║ ╚═╝ ██║██║██║  ██║
   ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═╝
"""

# --- 内部辅助函数 ---
def _format_config_to_str(config_dict: dict) -> str:
    if not isinstance(config_dict, dict):
        return ""
    
    clean_config = config_dict.copy()
    if '_internal_feature_names' in clean_config:
        del clean_config['_internal_feature_names']

    try:
        flat_config = pd.json_normalize(clean_config, sep='.').to_dict(orient='records')[0]
    except (IndexError, TypeError):
        flat_config = clean_config

    parts = []
    for key, value in flat_config.items():
        if value is None:
            continue
        if isinstance(value, list):
            parts.append(f"{key}=[{len(value)} items]" if len(value) > 3 else f"{key}={value}")
        else:
            parts.append(f"{key}={value}")
    
    return ", ".join(parts)

def _summarize_feature_order(feature_names: list) -> str:
    if not feature_names:
        return "N/A"
    
    try:
        prefixes = []
        for name in feature_names:
            match = re.match(r'^([a-zA-Z0-9]+)(_|\d|$)', name)
            if match:
                prefixes.append(match.group(1))
            else:
                # Fallback for names that don't match the pattern
                prefixes.append(name.split('_')[0] if '_' in name else name)
    except (AttributeError, TypeError):
        return "Unable to summarize feature order."

    summary_parts, current_prefix, count = [], None, 0
    descriptor_names_set = {name.lower() for name, func in Descriptors._descList}

    for prefix in prefixes:
        prefix_normalized = "rdkit_descriptors" if prefix.lower() in descriptor_names_set or prefix == "Num" else prefix
        if prefix_normalized != current_prefix:
            if current_prefix is not None:
                summary_parts.append(f"{current_prefix} ({count} features)")
            current_prefix, count = prefix_normalized, 1
        else:
            count += 1
            
    if current_prefix is not None:
        summary_parts.append(f"{current_prefix} ({count} features)")
        
    return " -> ".join(summary_parts)

# --- 目录管理 ---
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

# --- Artifact 保存函数 (用于训练流程) ---
def save_config(config: dict, run_dir: str, console=None):
    """Saves the full configuration dictionary to the run directory as a JSON file."""
    if console is None: console = default_console
    config_path = os.path.join(run_dir, "run_config.json")
    clean_config = config.copy()
    if '_internal_feature_names' in clean_config:
        del clean_config['_internal_feature_names']
        
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(clean_config, f, indent=4)
    console.print(f"[green]✓ Saved run configuration to [dim]{config_path}[/dim]")

def save_model_artifact(model_object, model_artifact_name, model_specific_output_dir, 
                        model_name: str, is_pytorch_model: bool = False, console=None):
    """Saves a model artifact, handling different model types."""
    if console is None: console = default_console
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'catboost':
        model_path = os.path.join(model_specific_output_dir, f"{model_artifact_name}.cbm")
        model_object.save_model(model_path, format='cbm')
    elif model_name_lower == 'xgboost':
        model_path = os.path.join(model_specific_output_dir, f"{model_artifact_name}.json")
        model_object.save_model(model_path)
    elif is_pytorch_model:
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
        serializable_params = {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v.tolist() if isinstance(v, np.ndarray) else v for k, v in hyperparameters.items()}
        json.dump(serializable_params, f, indent=4)
    console.print(f"  [green]✓ Saved hyperparameters[/green] for {model_name_short} to [dim]{params_path}[/dim]")
    return params_path

def save_predictions(metrics_dict, model_name_short, model_specific_output_dir, y_train_true, y_test_true, y_val_true=None, console=None):
    if console is None: console = default_console
    predictions_dir = os.path.join(model_specific_output_dir, "predictions_final_model"); os.makedirs(predictions_dir, exist_ok=True)
    def save_df(df, filename):
        path = os.path.join(predictions_dir, filename); df.to_csv(path, index=False, encoding='utf-8'); console.print(f"  [green]✓ Saved predictions[/green] for {model_name_short} to [dim]{path}[/dim]")
    for name, y_true, y_pred_key, y_proba_key in [('train', y_train_true, 'y_train_pred', 'y_train_pred_proba'), ('test', y_test_true, 'y_test_pred', 'y_test_pred_proba'), ('validation', y_val_true, 'y_val_pred', 'y_val_pred_proba')]:
        if y_true is not None and y_true.size > 0 and y_pred_key in metrics_dict:
            df = pd.DataFrame({'y_true': y_true.ravel(), 'y_pred': metrics_dict[y_pred_key].ravel()})
            if y_proba_key in metrics_dict and metrics_dict[y_proba_key] is not None:
                proba_df = pd.DataFrame(metrics_dict[y_proba_key], columns=[f'proba_class_{i}' for i in range(metrics_dict[y_proba_key].shape[1])])
                df = pd.concat([df, proba_df], axis=1)
            save_df(df, f"{model_name_short}_{name}_predictions.csv")

def save_cv_fold_predictions(cv_predictions_dict, model_name_short, model_specific_output_dir, console=None):
    if console is None: console = default_console
    if cv_predictions_dict is None: return
    predictions_dir = os.path.join(model_specific_output_dir, "predictions_cv_oof"); os.makedirs(predictions_dir, exist_ok=True)
    cv_df = pd.DataFrame({'y_true_oof': cv_predictions_dict['y_true_oof'].ravel(), 'y_pred_oof': cv_predictions_dict['y_pred_oof'].ravel()})
    if cv_predictions_dict.get('y_proba_oof') is not None:
        proba_df_cv = pd.DataFrame(cv_predictions_dict['y_proba_oof'], columns=[f'proba_class_{i}' for i in range(cv_predictions_dict['y_proba_oof'].shape[1])])
        cv_df = pd.concat([cv_df, proba_df_cv], axis=1)
    path = os.path.join(predictions_dir, f"{model_name_short}_cv_oof_predictions.csv")
    cv_df.to_csv(path, index=False, encoding='utf-8'); console.print(f"  [green]✓ Saved CV OOF predictions[/green] to [dim]{path}[/dim]")

def save_data_splits_csv(data_splits_dir, dataset_name_prefix, X_train, y_train, X_val, y_val, X_test, y_test, scaler=None, label_encoder=None, console=None):
    if console is None: console = default_console
    datasets_to_save = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    if X_val is not None and X_val.size > 0: datasets_to_save['X_val'] = X_val
    if y_val is not None and y_val.size > 0: datasets_to_save['y_val'] = y_val
    for name, data in datasets_to_save.items():
        if data is None or data.size == 0: continue
        df = pd.DataFrame(data)
        if data.ndim == 1 and name.startswith('y_'): df.columns = ['target']
        path = os.path.join(data_splits_dir, f"{dataset_name_prefix}_{name}.csv")
        df.to_csv(path, index=False, encoding='utf-8')
        console.print(f"  [green]✓ Saved {name}[/green] ({dataset_name_prefix}) to [dim]{path}[/dim]")
    if label_encoder is not None:
        le_path = os.path.join(data_splits_dir, f"{dataset_name_prefix}_label_encoder.joblib")
        joblib.dump(label_encoder, le_path); console.print(f"  [green]✓ Saved label encoder[/green] to [dim]{le_path}[/dim]")
    if scaler is not None:
        scaler_path = os.path.join(data_splits_dir, f"{dataset_name_prefix}_scaler.joblib")
        joblib.dump(scaler, scaler_path); console.print(f"  [green]✓ Saved scaler[/green] to [dim]{scaler_path}[/dim]")

# --- Artifact 加载函数 (用于预测流程) ---
def load_config_from_run(run_dir: str, console=None) -> dict:
    """Loads the run configuration from the JSON file in the run directory."""
    if console is None: console = default_console
    config_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file 'run_config.json' not found in {run_dir}. "
                              "This experiment may be from an older version of CHEMIA.")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

# --- MODIFICATION: More robust model loading ---
def load_model_artifact(model_dir: str, model_name: str, console=None):
    """
    Loads a model artifact, trying different extensions and common name variations.
    """
    if console is None: console = default_console
    
    # Check for two possible base filenames: 'xgb' and 'xgb_model'
    paths_to_check = [
        os.path.join(model_dir, model_name),
        os.path.join(model_dir, f"{model_name}_model")
    ]
    extensions = ['.cbm', '.json', '.joblib', '.pth']
    
    for path_prefix in paths_to_check:
        for ext in extensions:
            model_path = path_prefix + ext
            if os.path.exists(model_path):
                console.print(f"  - Found model artifact: [dim]{model_path}[/dim]")
                if ext == '.cbm':
                    from catboost import CatBoostRegressor
                    model = CatBoostRegressor()
                    model.load_model(model_path)
                    return model
                elif ext == '.json':
                    from xgboost import XGBRegressor
                    model = XGBRegressor()
                    model.load_model(model_path)
                    return model
                elif ext == '.joblib':
                    return joblib.load(model_path)
                elif ext == '.pth':
                    raise NotImplementedError("PyTorch model loading requires the model class definition.")
    
    raise FileNotFoundError(f"No model artifact found for '{model_name}' in '{model_dir}' with extensions {extensions}")

def log_prediction_summary(log_path, run_dir, model_name, input_file, output_file, num_predictions, duration, config, console=None):
    """Logs the summary of a prediction run to a file."""
    if console is None: console = default_console
    
    # --- MODIFICATION: Added version number and cleaned up format ---
    log_content = [
        CHEMIA_BANNER,
        "CHEMIA - v1.4.0 - Prediction Run Log\n",
        "="*88,
        f"{'Prediction Time:':<28} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'Total Duration:':<28} {duration:.2f} seconds",
        "-"*88,
        f"{'Source Experiment Run:':<28} {os.path.basename(run_dir)}",
        f"{'Model Used:':<28} {model_name.upper()}",
        f"{'Task Type from Config:':<28} {config.get('task_type', 'N/A')}",
        "-"*88,
        f"{'Input Data File:':<28} {input_file}",
        f"{'Number of Predictions Made:':<28} {num_predictions}",
        f"{'Output Predictions File:':<28} {output_file}",
        "="*88,
        "\n*** End of Prediction Log ***"
    ]
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_content))
    
    console.print(f"  - Prediction summary logged to: [dim]{log_path}[/dim]")

def load_scaler(data_splits_dir: str, console=None):
    """Loads the StandardScaler object if it exists."""
    if console is None: console = default_console
    scaler_path = os.path.join(data_splits_dir, "processed_dataset_scaler.joblib")
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    console.print("  - [dim]No scaler found, proceeding without it.[/dim]")
    return None

def load_label_encoder(data_splits_dir: str, console=None):
    """Loads the LabelEncoder object if it exists."""
    if console is None: console = default_console
    le_path = os.path.join(data_splits_dir, "processed_dataset_label_encoder.joblib")
    if os.path.exists(le_path):
        return joblib.load(le_path)
    console.print("  - [dim]No label encoder found.[/dim]")
    return None

# --- 日志记录函数 ---
def log_results(model_name_short, best_params, best_score, metrics_dict, 
                model_specific_output_dir, task_type='regression', 
                best_trial_fold_scores=None, console=None,
                experiment_run_name=None, model_runtime_seconds=None, config=None,
                data_shapes=None, cv_fold_metrics=None):
    if console is None: console = default_console
    log_path = os.path.join(model_specific_output_dir, f"{model_name_short}_results.log") 
    main_metric_name = "R²" if task_type == 'regression' else "F1 (Weighted)"
    if task_type == 'binary_classification': main_metric_name = "F1 (Binary)"
    log_content = [CHEMIA_BANNER, "CHEMIA - v1.3.0\n", "="*88]
    log_content.extend([f"{'Log Generated:':<25} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"{'Experiment Run Name:':<25} {experiment_run_name}",
                        f"{'Run Configuration:':<25} {textwrap.fill(_format_config_to_str(config or {}), 86, initial_indent=' '*27, subsequent_indent=' '*27).strip()}",
                        "-"*88,
                        f"{'Model Trained:':<25} {model_name_short.upper()}",
                        f"{'Task Type:':<25} {task_type}",
                        f"{'Feature Concatenation:':<25} {textwrap.fill(_summarize_feature_order((config or {}).get('_internal_feature_names',[])), 86, initial_indent=' '*27, subsequent_indent=' '*27).strip()}"])
    if data_shapes:
        log_content.extend(["\n" + "--- Data & Split Information ---",
                            f"{'Training Set Dimensions:':<25} {data_shapes.get('train', ('N/A', 'N/A'))[0]} samples x {data_shapes.get('train', ('N/A', 'N/A'))[1]} features",
                            f"{'HPO Method:':<25} {data_shapes.get('hpo_method', 'N/A')}"])
        if 'CV' in data_shapes.get('hpo_method', ''): log_content.append(f"{'CV Folds:':<25} {data_shapes.get('cv_folds', 'N/A')}")
        else: log_content.append(f"{'Validation Set Dimensions:':<25} {data_shapes.get('val', ('N/A', 'N/A'))[0]} samples x {data_shapes.get('val', ('N/A', 'N/A'))[1]} features")
        log_content.append(f"{'Test Set Dimensions:':<25} {data_shapes.get('test', ('N/A', 'N/A'))[0]} samples x {data_shapes.get('test', ('N/A', 'N/A'))[1]} features")
    log_content.extend(["="*88 + "\n", "--- Hyperparameter Optimization (Optuna) ---", f"Best HPO Score ({main_metric_name}): {best_score:.4f}"])
    if best_trial_fold_scores: log_content.append(f"  Scores from individual HPO folds: [{', '.join([f'{s:.4f}' for s in best_trial_fold_scores])}]")
    log_content.extend(["\nBest Hyperparameters Found:", json.dumps(best_params, indent=4)])
    if cv_fold_metrics:
        log_content.extend(["\n" + "="*88 + "\n", "--- Cross-Validation Fold Performance (on Training Set) ---"])
        df_cv = pd.DataFrame(cv_fold_metrics).set_index('fold')
        metric_cols = ['r2', 'rmse', 'mae'] if task_type == 'regression' else ['accuracy', 'f1', 'precision', 'recall']
        header_map = {'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE', 'accuracy': 'Accuracy', 'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall'}
        header = f"| {'Fold':<6} |" + "".join([f" {header_map.get(m, m.upper()):^12} |" for m in metric_cols])
        separator = f"|{'-'*8}|" + "".join([f"{'-'*14}|" for _ in metric_cols])
        log_content.extend([header, separator])
        for index, row in df_cv.iterrows(): log_content.append(f"| {index:<6} |" + "".join([f" {row.get(m, 0):.4f}{' ':<6} |" for m in metric_cols]))
        log_content.append(separator)
        for stat in ['mean', 'std']:
            stat_row = df_cv.agg([stat]).iloc[0]; log_content.append(f"| {stat.upper():<6} |" + "".join([f" {stat_row.get(m, 0):.4f}{' ':<6} |" for m in metric_cols]))
        log_content.append(separator)
    log_content.extend(["\n" + "="*88 + "\n", "--- Final Model Performance (on pre-defined splits) ---"])
    header = f"| {'Metric':<25} | {'Train Set':^15} | {'Validation Set':^15} | {'Test Set':^15} |"; separator = f"|{'-'*27}|{'-'*17}|{'-'*17}|{'-'*17}|"
    log_content.extend([header, separator])
    metrics_map = {'regression': {'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE'}, 'binary_classification': {'accuracy': 'Accuracy', 'f1': 'F1 (Binary)', 'precision': 'Precision (B)', 'recall': 'Recall (B)', 'auc': 'AUC'}, 'multiclass_classification': {'accuracy': 'Accuracy', 'f1': 'F1 (Weighted)', 'precision': 'Precision (W)', 'recall': 'Recall (W)', 'auc_ovr_weighted': 'AUC (OvR W)'}}
    for stem, name in metrics_map.get(task_type, {}).items():
        if f'train_{stem}' in metrics_dict or f'test_{stem}' in metrics_dict:
            train_v, val_v, test_v = metrics_dict.get(f'train_{stem}'), metrics_dict.get(f'val_{stem}'), metrics_dict.get(f'test_{stem}')
            train_s, val_s, test_s = (f"{v:.4f}" if v is not None else "N/A" for v in [train_v, val_v, test_v])
            log_content.append(f"| {name:<25} | {train_s:^15} | {val_s:^15} | {test_s:^15} |")
    log_content.extend([separator, "\n" + "="*88 + "\n"])
    if model_runtime_seconds: log_content.append(f"Model-Specific Runtime (HPO, Fit, Eval): {model_runtime_seconds:.2f} seconds")
    log_content.append("\n*** End of CHEMIA Log for this Model ***")
    with open(log_path, 'w', encoding='utf-8') as f: f.write("\n".join(log_content))
    console.print(f"  [green]✓ Logged results[/green] for {model_name_short} ({task_type}) to [dim]{log_path}[/dim]")

def log_experiment_summary(run_base_dir, experiment_run_name, config, total_duration_seconds, start_timestamp, all_results, console=None):
    if console is None: console = default_console
    summary_log_path = os.path.join(run_base_dir, "_experiment_summary.log")
    days, rem = divmod(total_duration_seconds, 86400); hours, rem = divmod(rem, 3600); mins, secs = divmod(rem, 60)
    duration_str = f"{int(days)}d {int(hours)}h {int(mins)}m {int(secs)}s"
    log_content = [CHEMIA_BANNER, "CHEMIA - v1.3.0\n", "=" * 88, "  EXPERIMENT RUN SUMMARY", "=" * 88]
    log_content.extend([f"{'Experiment Name:':<25} {experiment_run_name}", f"{'Run Started:':<25} {datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')}", f"{'Total Runtime:':<25} {duration_str}"])
    if config and isinstance(config, dict):
        log_content.append(f"{'Run Configuration:':<25} {textwrap.fill(_format_config_to_str(config), 86, initial_indent=' ' * 27, subsequent_indent=' ' * 27).strip()}")
        if '_internal_feature_names' in config: log_content.append(f"{'Feature Concatenation:':<25} {textwrap.fill(_summarize_feature_order(config['_internal_feature_names']), 86, initial_indent=' ' * 27, subsequent_indent=' ' * 27).strip()}")
    log_content.extend(["\n" + "="*88, "  MODEL PERFORMANCE SUMMARY", "="*88])
    if not all_results: log_content.append("\nNo models were successfully run to summarize.")
    else:
        for result in all_results:
            model, task = result.get('model_name', 'N/A'), result.get('task_type', 'N/A')
            main_metric = "R²" if task == 'regression' else "F1 (W)" if task == 'multiclass_classification' else "F1 (B)"
            log_content.extend([f"\n----- Model: {model.upper()} -----", f"Task Type: {task}"])
            if 'data_shapes' in result:
                shapes = result['data_shapes']; train_s, test_s = shapes.get('train', ('-','-')), shapes.get('test', ('-','-'))
                dim_str = f"Data Dimensions: Train=({train_s[0]},{train_s[1]}), Test=({test_s[0]},{test_s[1]})"
                if 'CV' in shapes.get('hpo_method',''): dim_str += f", HPO on {shapes.get('cv_folds', 'N/A')}-Folds CV"
                else: val_s = shapes.get('val', ('-','-')); dim_str += f", Val=({val_s[0]},{val_s[1]})"
                log_content.append(dim_str)
            log_content.append(f"Best HPO Score ({main_metric}): {result.get('best_optuna_score', float('nan')):.4f}")
            if result.get('best_trial_fold_scores'): log_content.append(f"  - HPO Fold Scores: [{', '.join([f'{s:.4f}' for s in result['best_trial_fold_scores']])}]")
            log_content.extend(["Best Hyperparameters:", json.dumps(result.get('best_params', {}), indent=2)])
            if result.get('cv_fold_metrics'):
                df_cv = pd.DataFrame(result['cv_fold_metrics']); cv_summary = df_cv.agg(['mean', 'std']).to_dict()
                log_content.append("\nCV Performance (on Training Set):")
                if task == 'regression': log_content.append(f"  - Avg Fold R²: {cv_summary.get('r2', {}).get('mean', 0):.4f} (± {cv_summary.get('r2', {}).get('std', 0):.4f})")
                else: log_content.append(f"  - Avg Fold F1: {cv_summary.get('f1', {}).get('mean', 0):.4f} (± {cv_summary.get('f1', {}).get('std', 0):.4f})")
            log_content.extend(["\nFinal Metrics (on pre-defined splits):", f"| {'Metric':<25} | {'Train Set':^15} | {'Validation Set':^15} | {'Test Set':^15} |", f"|{'-'*27}|{'-'*17}|{'-'*17}|{'-'*17}|"])
            metrics_map = {'regression': {'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE'}, 'binary_classification': {'accuracy': 'Accuracy', 'f1': 'F1 (Binary)', 'precision': 'Precision (B)', 'recall': 'Recall (B)', 'auc': 'AUC'}, 'multiclass_classification': {'accuracy': 'Accuracy', 'f1': 'F1 (Weighted)', 'precision': 'Precision (W)', 'recall': 'Recall (W)', 'auc_ovr_weighted': 'AUC (OvR W)'}}
            for stem, name in metrics_map.get(task, {}).items():
                if f'train_{stem}' in result or f'test_{stem}' in result:
                    train_v, val_v, test_v = result.get(f'train_{stem}'), result.get(f'val_{stem}'), result.get(f'test_{stem}')
                    log_content.append(f"| {name:<25} | {(f'{train_v:.4f}' if train_v is not None else 'N/A'):^15} | {(f'{val_v:.4f}' if val_v is not None else 'N/A'):^15} | {(f'{test_v:.4f}' if test_v is not None else 'N/A'):^15} |")
            log_content.append(f"|{'-'*27}|{'-'*17}|{'-'*17}|{'-'*17}|")
    log_content.extend(["\n" + "="*88, "--- Notes & Known Behaviors ---", "Note on Warnings: 'UserWarning' about 'X does not have valid feature names' is expected when using models like LightGBM.", "This is safely handled by the pipeline and warnings are suppressed for cleaner output."])
    
    # Add simple acknowledgment to main experiment log only
    log_content.extend([
        "\n" + "="*88,
        "ACKNOWLEDGMENTS: Wuhan University • Shanghai AI Lab • Tsinghua University • SIOC, CAS • UCAS",
        "Contributors: Gao, Ben • Wan, Haiyuan • Huang, Huaihai",
        "="*88
    ])
    
    log_content.extend(["\n" + "="*88, "*** End of Summary Log ***"])
    with open(summary_log_path, 'w', encoding='utf-8') as f: f.write("\n".join(log_content))
    console.print(f"[bold green]✓ Experiment summary log created at:[/bold green] [dim]{summary_log_path}[/dim]")

# --- START OF MODIFICATION: Move the function here ---
def get_full_model_name(short_name: str) -> str:
    """Converts a model alias (e.g., 'xgb') to its full name ('xgboost')."""
    aliases = {
        'rf': 'randomforest', 'dt': 'decisiontree', 'knn': 'kneighbors',
        'lr': 'logisticregression', 'svm': 'svc', 'krr':'kernelridge',
        'xgb': 'xgboost', 'lgbm': 'lgbm', 'cat': 'catboost', 'hgb':'histgradientboosting',
        'ann': 'ann', 'ridge': 'ridge', 'svr': 'svr', 'adab': 'adaboost'
    }
    return aliases.get(short_name.lower(), short_name.lower())
# --- END OF MODIFICATION ---

def find_model_file(model_dir: str, model_name: str) -> str:
    """Finds the model file in a directory, trying different extensions."""
    base_path = os.path.join(model_dir, model_name)
    paths_to_check = [f"{base_path}_model", base_path]
    extensions = ['.json', '.cbm', '.joblib']
    for path in paths_to_check:
        for ext in extensions:
            if os.path.exists(path + ext):
                return path + ext
    raise FileNotFoundError(f"Could not find model file for '{model_name}' in '{model_dir}'")


def load_config_from_path(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    console.print(f"  • Loaded configuration from '[dim]{config_path}[/dim]'.")
    return config

def load_model_from_path(model_path: str, task_type: str):
    """
    Loads a model artifact from a direct file path, considering the task type.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    ext = os.path.splitext(model_path)[1]
    model = None
    
    # 根据任务类型选择正确的模型类
    if ext == '.json': # XGBoost
        model = XGBClassifier() if task_type != 'regression' else XGBRegressor()
        model.load_model(model_path)
    elif ext == '.cbm': # CatBoost
        model = CatBoostClassifier() if task_type != 'regression' else CatBoostRegressor()
        model.load_model(model_path)
    elif ext == '.joblib': # Scikit-learn models
        model = joblib.load(model_path)
    elif ext == '.pth':
        raise NotImplementedError("PyTorch model loading from path requires the model class definition.")
    else:
        raise ValueError(f"Unsupported model file extension: {ext}")
    
    console.print(f"  • Loaded [cyan]{model.__class__.__name__}[/cyan] model from '[dim]{model_path}[/dim]'.")
    return model

def load_scaler_from_path(scaler_path: str):
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        console.print(f"  • Loaded scaler from '[dim]{scaler_path}[/dim]'.")
        return scaler
    if scaler_path:
        console.print(f"  • [yellow]Scaler not found at '{scaler_path}', proceeding without it.[/yellow]")
    return None

def load_label_encoder_from_path(encoder_path: str):
    """Loads the LabelEncoder object from a direct path, if it exists."""
    if encoder_path and os.path.exists(encoder_path):
        label_encoder = joblib.load(encoder_path)
        console.print(f"  • Loaded label encoder from '[dim]{encoder_path}[/dim]'.")
        return label_encoder
    if encoder_path:
        console.print(f"  • [yellow]Label encoder not found at '{encoder_path}'.[/yellow]")
    return None

# Note: Some function aliases removed due to signature incompatibilities
