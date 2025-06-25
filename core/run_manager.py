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
    # --- MODIFICATION: Rewritten for robust feature merging ---
    from utils.feature_generator import generate_features

    # 1. Normalize smiles_col to a list and identify all key columns
    smiles_col_spec = common_cfg['smiles_col']
    smiles_cols_list = [smiles_col_spec] if isinstance(smiles_col_spec, str) else smiles_col_spec
    target_col = common_cfg['target_col']

    # 2. Start with a clean base DataFrame containing only essential columns
    # This also performs the initial cleaning (dropping NaNs in key identifier columns)
    base_cols = smiles_cols_list + [target_col]
    df_base = df[base_cols].dropna().copy()
    df_base.drop_duplicates(subset=smiles_cols_list, keep='first', inplace=True)
    df_base.reset_index(drop=True, inplace=True)
    
    console.print(f"  Initial clean data has {df_base.shape[0]} unique samples.")

    # This will be our final features DataFrame, starting empty with the correct index
    final_features_df = pd.DataFrame(index=df_base.index)

    # 3. Handle precomputed features first
    precomputed_cfg = common_cfg.get('precomputed_features')
    if precomputed_cfg and precomputed_cfg.get('feature_columns'):
        console.print("  Loading pre-computed features...")
        # Get precomputed columns from the original (un-cleaned) dataframe using the clean index
        feature_col_names = parse_feature_columns(df, precomputed_cfg['feature_columns'])
        # Use the index from df_base to select the correct rows from the original df
        precomputed_df = df.iloc[df_base.index][feature_col_names].copy()
        # Reset index to align with final_features_df for concatenation
        precomputed_df.reset_index(drop=True, inplace=True)
        final_features_df = pd.concat([final_features_df, precomputed_df], axis=1)

    # 4. Handle generated features based on config mode
    use_per_col_config = 'per_smiles_col_generators' in feature_gen_cfg
    use_global_config = 'generators' in feature_gen_cfg

    if use_per_col_config:
        console.print("[bold cyan]Using per-SMILES-column feature configuration.[/bold cyan]")
        per_col_configs = feature_gen_cfg['per_smiles_col_generators']
        
        for s_col in smiles_cols_list:
            if s_col in per_col_configs:
                current_gen_list = per_col_configs[s_col]
                console.print(f"  Column [bold magenta]'{s_col}'[/bold magenta]: Generating features...")
                smiles_list_for_gen = df_base[s_col].tolist()
                # Generate features for this column
                generated_df = generate_features(smiles_list_for_gen, current_gen_list, output_dir=output_dir)
                # Align index and merge
                generated_df.set_index(final_features_df.index, inplace=True)
                final_features_df = pd.concat([final_features_df, generated_df], axis=1)
            else:
                console.print(f"  Column [bold magenta]'{s_col}'[/bold magenta]: No specific generators found. Skipping.")

    elif use_global_config:
        console.print("[bold cyan]Using global feature configuration for all SMILES columns.[/bold cyan]")
        global_gen_list = feature_gen_cfg['generators']
        for s_col in smiles_cols_list:
            console.print(f"  Column [bold magenta]'{s_col}'[/bold magenta]: Applying global generators...")
            smiles_list_for_gen = df_base[s_col].tolist()
            generated_df = generate_features(smiles_list_for_gen, global_gen_list, output_dir=output_dir)
            generated_df.set_index(final_features_df.index, inplace=True)
            final_features_df = pd.concat([final_features_df, generated_df], axis=1)
    
    if final_features_df.empty:
        raise ValueError("No features were loaded or generated. Check your configuration.")

    # 5. Finalize X, y, and feature names by cleaning NaNs from the FEATURE matrix
    console.print(f"  Combined feature matrix shape before final NaN drop: {final_features_df.shape}")
    
    # Drop rows in the feature matrix that have NaNs (e.g., from failed calculations)
    final_features_df.dropna(inplace=True)
    console.print(f"  Combined feature matrix shape after final NaN drop: {final_features_df.shape}")

    # Use the cleaned index from the feature matrix to select the corresponding y values
    # This guarantees X and y have the same length.
    final_y = df_base.loc[final_features_df.index, target_col]

    X = final_features_df.values
    y = final_y.values
    feature_cols = final_features_df.columns.tolist()

    console.print(f"  Final data shapes: X={X.shape}, y={y.shape}")
    
    # This is the final check that will prevent the error
    if X.shape[0] != y.shape[0]:
        raise RuntimeError(f"FATAL: X and y shape mismatch after processing! X:{X.shape[0]}, y:{y.shape[0]}")

    return X, y, feature_cols


def load_and_prepare_data(config: dict, output_dir: str):
    """
    Loads, processes, and splits data.
    --- MODIFICATION: Now returns feature_columns as well. ---
    """
    from sklearn.model_selection import train_test_split
    from charset_normalizer import detect

    data_cfg = config['data']
    source_mode = data_cfg['source_mode']
    feature_gen_cfg = config.get('features', {})
    feature_columns = None # Initialize feature_columns

    def read_csv(path):
        try:
            with open(path, 'rb') as f:
                encoding = detect(f.read(10000)).get('encoding', 'utf-8')
            return pd.read_csv(path, encoding=encoding)
        except Exception as e:
            raise FileNotFoundError(f"Could not read data file: {path}. Error: {e}")

    if source_mode == 'features_only':
        console.print("[bold cyan]Step 1: Loading from features-only file...[/bold cyan]")
        cfg = data_cfg['features_only_config']
        df = read_csv(cfg['file_path'])
        target_col = cfg['target_col']
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the file.")
        
        # --- MODIFICATION: Capture feature columns ---
        feature_columns = parse_feature_columns(df, cfg['feature_columns'])
        if target_col in feature_columns:
            feature_columns.remove(target_col)
            
        df.dropna(subset=[target_col] + feature_columns, inplace=True)
        X = df[feature_columns].values
        y = df[target_col].values
        
        console.print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, config)
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns
        
    elif source_mode == 'single_file':
        console.print("[bold cyan]Step 1: Loading from single file and splitting...[/bold cyan]")
        common_cfg = data_cfg['single_file_config']
        df_full = read_csv(common_cfg['main_file_path'])
        
        # --- MODIFICATION: Capture feature columns ---
        X, y, feature_columns = process_dataframe(df_full, common_cfg, feature_gen_cfg, output_dir)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, config)
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns
        
    elif source_mode == 'pre_split_cv':
        console.print("[bold cyan]Step 1: Loading pre-split train/test files...[/bold cyan]")
        common_cfg = data_cfg['pre_split_cv_config']
        df_train = read_csv(common_cfg['train_path'])
        df_test = read_csv(common_cfg['test_path'])
        
        # --- MODIFICATION: Capture feature columns from the training set ---
        X_train, y_train, feature_columns = process_dataframe(df_train, common_cfg, feature_gen_cfg, output_dir)
        # Assume test set has the same features
        X_test, y_test, _ = process_dataframe(df_test, common_cfg, feature_gen_cfg, output_dir)
        return X_train, y_train, None, None, X_test, y_test, feature_columns

    elif source_mode == 'pre_split_t_v_t':
        console.print("[bold cyan]Step 1: Loading pre-split train/valid/test files...[/bold cyan]")
        common_cfg = data_cfg['pre_split_t_v_t_config']
        df_train = read_csv(common_cfg['train_path'])
        df_valid = read_csv(common_cfg['valid_path'])
        df_test = read_csv(common_cfg['test_path'])

        # --- MODIFICATION: Capture feature columns from the training set ---
        X_train, y_train, feature_columns = process_dataframe(df_train, common_cfg, feature_gen_cfg, output_dir)
        # Assume valid and test sets have same features
        X_val, y_val, _ = process_dataframe(df_valid, common_cfg, feature_gen_cfg, output_dir)
        X_test, y_test, _ = process_dataframe(df_test, common_cfg, feature_gen_cfg, output_dir)
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns
    
    else:
        raise ValueError(f"Invalid `data.source_mode` in config: {source_mode}")


def split_data(X, y, config: dict):
    # ... (此函数现在只负责拆分，逻辑不变, 返回值也变了) ...
    from sklearn.model_selection import train_test_split
    
    split_mode = config['split_mode']
    task_type = config['task_type']
    stratify = y if task_type != 'regression' else None

    if split_mode == 'train_valid_test':
        cfg = config['split_config']['train_valid_test']
        test_size = cfg['test_size']
        if not (0 < test_size < 1 and 0 <= cfg['valid_size'] < 1 and (test_size + cfg['valid_size']) < 1):
             raise ValueError("Invalid train/valid/test split sizes. They must be between 0 and 1 and sum to less than 1.")
        
        valid_size_of_remainder = cfg['valid_size'] / (1 - test_size) if (1 - test_size) > 0 else 0
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=cfg['random_state'], stratify=stratify)
        stratify_train_val = y_train_val if task_type != 'regression' else None
        
        # Handle case where no validation set is needed
        if valid_size_of_remainder > 0:
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=valid_size_of_remainder, random_state=cfg['random_state'], stratify=stratify_train_val)
        else:
            X_train, y_train = X_train_val, y_train_val
            X_val, y_val = np.array([]).reshape(0, X.shape[1]), np.array([])
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    elif split_mode == 'cross_validation':
        cfg = config['split_config']['cross_validation']
        test_size = cfg.get('test_size_for_cv', 0.2) # Provide default
        if not (0 <= test_size < 1):
            raise ValueError("Invalid test_size_for_cv. It must be between 0 and 1.")
        
        # Handle case where there is no test set
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=cfg['random_state'], stratify=stratify)
        else:
            X_train, y_train = X, y
            X_test, y_test = np.array([]).reshape(0, X.shape[1]), np.array([])

        return X_train, y_train, None, None, X_test, y_test
    else:
        raise ValueError(f"Invalid split_mode: {split_mode}")

def start_experiment_run(config):
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
    # --- MODIFICATION: Capture feature_columns ---
    data_tuple = load_and_prepare_data(config, run_dir)
    if data_tuple is None: return
    X_train, y_train, X_val, y_val, X_test, y_test, feature_columns = data_tuple
    
    # --- MODIFICATION: Add feature_columns to the config to pass it down ---
    config['_internal_feature_names'] = feature_columns

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