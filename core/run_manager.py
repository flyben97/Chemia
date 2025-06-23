# core/run_manager.py
import time
import os
from datetime import datetime
import numpy as np
import pandas as pd
from rich.console import Console

console = Console(width=120)

def parse_feature_columns(df: pd.DataFrame, col_spec: str | list) -> list:
    # ... (此函数不变)
    if isinstance(col_spec, list):
        missing = [col for col in col_spec if col not in df.columns]
        if missing:
            raise ValueError(f"Specified feature columns not found in dataframe: {missing}")
        return col_spec
    if isinstance(col_spec, str):
        parts = col_spec.split(':')
        if len(parts) == 2:
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else df.shape[1]
            return df.columns[start:end].tolist()
    raise ValueError(f"Invalid format for feature_columns: {col_spec}. Use 'start:end' string or a list of column names.")

def process_dataframe(df: pd.DataFrame, common_cfg: dict, feature_gen_cfg: dict, output_dir: str):
    # ... (此函数不变)
    from utils.feature_generator import generate_features

    smiles_col = common_cfg['smiles_col']
    target_col = common_cfg['target_col']
    df_clean = df.dropna(subset=[smiles_col, target_col]).copy()
    df_clean.drop_duplicates(subset=[smiles_col], keep='first', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    all_feature_dfs = []
    precomputed_cfg = common_cfg.get('precomputed_features')
    if precomputed_cfg and precomputed_cfg.get('feature_columns'):
        feature_col_names = parse_feature_columns(df_clean, precomputed_cfg['feature_columns'])
        all_feature_dfs.append(df_clean[[smiles_col] + feature_col_names])
    if feature_gen_cfg.get('generators'):
        smiles_list = df_clean[smiles_col].tolist()
        generated_features_df = generate_features(smiles_list, feature_gen_cfg['generators'], output_dir=output_dir)
        generated_features_df[smiles_col] = smiles_list
        all_feature_dfs.append(generated_features_df)
    if not all_feature_dfs:
        raise ValueError("No features were loaded or generated. Check your configuration.")
    final_df = df_clean[[smiles_col, target_col]].copy()
    for feat_df in all_feature_dfs:
        final_df = pd.merge(final_df, feat_df, on=smiles_col, how='inner')
    feature_cols = [col for col in final_df.columns if col not in [smiles_col, target_col]]
    final_df.dropna(subset=feature_cols, inplace=True)
    X = final_df[feature_cols].values
    y = final_df[target_col].values
    return X, y


def load_and_prepare_data(config: dict, output_dir: str):
    """
    Loads, processes, and splits data based on the source_mode in the config.
    """
    from sklearn.model_selection import train_test_split
    from charset_normalizer import detect

    data_cfg = config['data']
    source_mode = data_cfg['source_mode']
    feature_gen_cfg = config.get('features', {})

    def read_csv(path):
        try:
            with open(path, 'rb') as f:
                encoding = detect(f.read(10000)).get('encoding', 'utf-8')
            return pd.read_csv(path, encoding=encoding)
        except Exception as e:
            raise FileNotFoundError(f"Could not read data file: {path}. Error: {e}")

    # --- 新增：处理 features_only 模式 ---
    if source_mode == 'features_only':
        console.print("[bold cyan]Step 1: Loading from features-only file...[/bold cyan]")
        cfg = data_cfg['features_only_config']
        df = read_csv(cfg['file_path'])
        
        target_col = cfg['target_col']
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the file.")
            
        feature_cols = parse_feature_columns(df, cfg['feature_columns'])
        
        # 确保目标列不在特征列中
        if target_col in feature_cols:
            feature_cols.remove(target_col)
            
        df.dropna(subset=[target_col] + feature_cols, inplace=True)
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # 在这种模式下，数据已经准备好，直接进入拆分阶段
        console.print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
        return split_data(X, y, config) # 调用拆分函数
        
    # --- 已有模式的逻辑保持不变 ---
    elif source_mode == 'single_file':
        console.print("[bold cyan]Step 1: Loading from single file and splitting...[/bold cyan]")
        common_cfg = data_cfg['single_file_config']
        df_full = read_csv(common_cfg['main_file_path'])
        X, y = process_dataframe(df_full, common_cfg, feature_gen_cfg, output_dir)
        return split_data(X, y, config)
        
    elif source_mode == 'pre_split_cv':
        console.print("[bold cyan]Step 1: Loading pre-split train/test files...[/bold cyan]")
        common_cfg = data_cfg['pre_split_cv_config']
        df_train = read_csv(common_cfg['train_path'])
        df_test = read_csv(common_cfg['test_path'])
        
        X_train, y_train = process_dataframe(df_train, common_cfg, feature_gen_cfg, output_dir)
        X_test, y_test = process_dataframe(df_test, common_cfg, feature_gen_cfg, output_dir)
        return X_train, y_train, None, None, X_test, y_test

    elif source_mode == 'pre_split_t_v_t':
        console.print("[bold cyan]Step 1: Loading pre-split train/valid/test files...[/bold cyan]")
        common_cfg = data_cfg['pre_split_t_v_t_config']
        df_train = read_csv(common_cfg['train_path'])
        df_valid = read_csv(common_cfg['valid_path'])
        df_test = read_csv(common_cfg['test_path'])

        X_train, y_train = process_dataframe(df_train, common_cfg, feature_gen_cfg, output_dir)
        X_val, y_val = process_dataframe(df_valid, common_cfg, feature_gen_cfg, output_dir)
        X_test, y_test = process_dataframe(df_test, common_cfg, feature_gen_cfg, output_dir)
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    else:
        raise ValueError(f"Invalid `data.source_mode` in config: {source_mode}")


def split_data(X, y, config: dict):
    # ... (此函数现在只负责拆分，逻辑不变) ...
    from sklearn.model_selection import train_test_split
    
    split_mode = config['split_mode']
    task_type = config['task_type']
    stratify = y if task_type != 'regression' else None

    if split_mode == 'train_valid_test':
        cfg = config['split_config']['train_valid_test']
        test_size = cfg['test_size']
        if not (0 < test_size < 1 and 0 <= cfg['valid_size'] < 1 and (test_size + cfg['valid_size']) < 1):
             raise ValueError("Invalid train/valid/test split sizes. They must be between 0 and 1 and sum to less than 1.")
        
        valid_size_of_remainder = cfg['valid_size'] / (1 - test_size)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=cfg['random_state'], stratify=stratify)
        stratify_train_val = y_train_val if task_type != 'regression' else None
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=valid_size_of_remainder, random_state=cfg['random_state'], stratify=stratify_train_val)
        return X_train, y_train, X_val, y_val, X_test, y_test

    elif split_mode == 'cross_validation':
        cfg = config['split_config']['cross_validation']
        test_size = cfg.get('test_size_for_cv', 0.2) # Provide default
        if not (0 < test_size < 1):
            raise ValueError("Invalid test_size_for_cv. It must be between 0 and 1.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=cfg['random_state'], stratify=stratify)
        return X_train, y_train, None, None, X_test, y_test
    else:
        raise ValueError(f"Invalid split_mode: {split_mode}")

def start_experiment_run(config):
    # --- start_experiment_run 现在调用新的 load_and_prepare_data ---
    # ... (此函数内部逻辑大部分不变, 只是现在调用的是新的数据加载函数)
    from utils.data import encode_labels
    from utils.io import ensure_experiment_directories, save_data_splits_csv, log_experiment_summary
    from core.trainer_setup import run_all_models_on_data
    from sklearn.preprocessing import StandardScaler
    import optuna

    script_start_time = time.time()
    
    if config['training'].get('quiet_optuna', False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['experiment_name']}_{config['task_type']}_{run_timestamp}"
    run_dir, models_dir, data_splits_dir = ensure_experiment_directories('output', exp_name, console)

    # 1. Load, Prepare and Split Data
    data_tuple = load_and_prepare_data(config, run_dir)
    if data_tuple is None: return
    X_train, y_train, X_val, y_val, X_test, y_test = data_tuple

    # 2. Preprocessing
    console.print("[bold cyan]Step 2: Preprocessing Data[/bold cyan]")
    # ... (这部分逻辑不变)
    label_encoder = None
    if config['task_type'] != 'regression':
        all_y_to_encode = [d for d in [y_train, y_val, y_test] if d is not None and d.size > 0]
        if all_y_to_encode:
            all_y_combined = np.concatenate(all_y_to_encode)
            y_processed, label_encoder = encode_labels(all_y_combined, task_type=config['task_type'], console=console)
            
            start_idx = 0
            y_train = y_processed[start_idx:start_idx+len(y_train)]; start_idx += len(y_train)
            if y_val is not None and y_val.size > 0:
                y_val = y_processed[start_idx:start_idx+len(y_val)]; start_idx += len(y_val)
            if y_test is not None and y_test.size > 0:
                y_test = y_processed[start_idx:start_idx+len(y_test)]
    
    scaler = None
    if config.get('features', {}).get('scaling', False):
        console.print("Applying StandardScaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_val is not None and X_val.size > 0: X_val = scaler.transform(X_val)
        if X_test is not None and X_test.size > 0: X_test = scaler.transform(X_test)

    # 3. Save artifacts and run training
    save_data_splits_csv(data_splits_dir, "dataset", X_train, y_train, X_val, y_val, X_test, y_test, scaler, label_encoder, console=console)
    
    console.print("[bold cyan]Step 3: Starting Model Training and HPO[/bold cyan]")
    all_results = run_all_models_on_data(
        X_train, y_train, X_val, y_val, X_test, y_test,
        models_dir, exp_name, config
    )
    
    total_runtime = time.time() - script_start_time
    log_experiment_summary(run_dir, exp_name, config, total_runtime, script_start_time, all_results, console)



    