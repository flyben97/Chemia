# core/run_manager.py
import time
import os
from datetime import datetime
import numpy as np
import pandas as pd
from rich.console import Console

console = Console(width=120)

def parse_feature_columns(df: pd.DataFrame, col_spec) -> list:
    if isinstance(col_spec, list):
        missing = [col for col in col_spec if col not in df.columns]
        if missing: raise ValueError(f"Specified feature columns not found: {missing}")
        return col_spec
    if isinstance(col_spec, str):
        parts = col_spec.split(':'); start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else df.shape[1]
        return df.columns[start:end].tolist()
    raise ValueError(f"Invalid format for feature_columns: {col_spec}")

def save_original_data_splits(original_df: pd.DataFrame, indices_dict: dict, output_dir: str, console):
    """
    Save the original data as separate CSV files based on train/val/test split indices.
    
    Args:
        original_df: The original DataFrame before any processing
        indices_dict: Dictionary containing 'train', 'val', 'test' indices
        output_dir: Directory to save the split files
        console: Rich console for logging
    """
    console.print("\n[bold cyan]Saving original data splits (before feature generation)...[/bold cyan]")
    
    splits_dir = os.path.join(output_dir, 'original_data_splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    for split_name, indices in indices_dict.items():
        if indices is not None and len(indices) > 0:
            split_df = original_df.iloc[indices].copy()
            split_file = os.path.join(splits_dir, f'{split_name}_original_data.csv')
            split_df.to_csv(split_file, index=False)
            console.print(f"  - Saved {split_name} set: {len(split_df)} samples → [dim]{split_file}[/dim]")
    
    # Also save a summary file
    summary_data = []
    for split_name, indices in indices_dict.items():
        if indices is not None:
            summary_data.append({
                'split': split_name,
                'count': len(indices),
                'percentage': f"{len(indices)/len(original_df)*100:.1f}%"
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(splits_dir, 'data_split_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    console.print(f"  - Split summary saved → [dim]{summary_file}[/dim]")

def process_dataframe(df: pd.DataFrame, common_cfg: dict, feature_gen_cfg: dict, output_dir: str):
    """
    Processes a raw DataFrame to generate a final feature matrix (X) and target vector (y).
    This is the definitive, robust version that handles all edge cases.
    Returns X, y, feature_cols, and the cleaned DataFrame with surviving indices.
    """
    from utils.feature_generator import generate_features
    
    # 1. Work on a copy and perform initial cleaning.
    df_processed = df.copy()
    smiles_col_spec = common_cfg.get('smiles_col', [])
    smiles_cols_list = [smiles_col_spec] if isinstance(smiles_col_spec, str) else smiles_col_spec
    existing_smiles_cols = [col for col in smiles_cols_list if col in df_processed.columns]
    
    if existing_smiles_cols:
        df_processed.dropna(subset=existing_smiles_cols, inplace=True)
    
    df_processed.reset_index(drop=True, inplace=True)

    console.print(f"  - Initial clean data has {df_processed.shape[0]} samples.")
    
    # This will hold all feature parts
    all_feature_dfs = []

    # 2. Load Pre-computed Features
    precomputed_cfg = common_cfg.get('precomputed_features')
    if precomputed_cfg and precomputed_cfg.get('feature_columns'):
        console.print("  - Loading pre-computed features...")
        feature_col_names = parse_feature_columns(df_processed, precomputed_cfg['feature_columns'])
        
        # --- START OF FIX (Based on your suggestion) ---
        # Ensure we only select numerical columns to prevent type errors
        numerical_feature_cols = []
        for col in feature_col_names:
            try:
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    numerical_feature_cols.append(col)
            except Exception:
                continue
        
        if len(numerical_feature_cols) < len(feature_col_names):
            dropped_cols = set(feature_col_names) - set(numerical_feature_cols)
            # Log detailed info but show concise terminal message  
            import logging
            logging.info(f"Dropping non-numerical columns from pre-computed features: {list(dropped_cols)}")
            console.print(f"    - [yellow]Warning:[/yellow] Dropped {len(dropped_cols)} non-numerical feature columns (see log for details)")
        
        if not numerical_feature_cols:
            console.print("    - [yellow]Warning:[/yellow] No numerical pre-computed feature columns found. Skipping.")
            precomputed_df = pd.DataFrame()
        else:
            precomputed_df = df_processed[numerical_feature_cols]
            console.print(f"    - ✓ Loaded {len(numerical_feature_cols)} pre-computed features")
        # --- END OF FIX ---

        all_feature_dfs.append(precomputed_df)

    # 3. Generate Features from SMILES columns.
    use_per_col_config = 'per_smiles_col_generators' in feature_gen_cfg
    if use_per_col_config and existing_smiles_cols:
        console.print("  - Using per-SMILES-column feature configuration...")
        per_col_configs = feature_gen_cfg['per_smiles_col_generators']
        
        for s_col in smiles_cols_list:
            if s_col in existing_smiles_cols and s_col in per_col_configs:
                console.print(f"    - Processing column: [magenta]{s_col}[/magenta]")
                smiles_list_for_gen = df_processed[s_col].tolist()
                generated_df = generate_features(smiles_list_for_gen, per_col_configs[s_col], output_dir=output_dir)
                all_feature_dfs.append(generated_df.reset_index(drop=True))

    # 4. Concatenate all feature parts.
    if not all_feature_dfs:
        raise ValueError("No features were loaded or generated. Check your configuration.")
    
    final_features_df = pd.concat(all_feature_dfs, axis=1)
    
    console.print(f"  - Combined feature matrix shape before NaN drop: {final_features_df.shape}")
    
    final_features_df.dropna(inplace=True)
    surviving_indices = final_features_df.index
    console.print(f"  - Combined feature matrix shape after final NaN drop: {final_features_df.shape}")
    
    target_col = common_cfg.get('target_col')
    has_target = target_col and target_col in df_processed.columns
    
    y = df_processed.loc[surviving_indices, target_col].values if has_target and not final_features_df.empty else np.array([])
    X = final_features_df.values
    feature_cols = final_features_df.columns.tolist()
    
    console.print(f"  - Final data shapes: X={X.shape}, y={y.shape}")
    if has_target and X.shape[0] != y.shape[0]:
        raise RuntimeError(f"FATAL: X and y shape mismatch! X:{X.shape[0]}, y:{y.shape[0]}")
        
    # Return the cleaned dataframe and surviving indices for saving original data splits
    cleaned_df = df_processed.iloc[surviving_indices].reset_index(drop=True)
    return X, y, feature_cols, cleaned_df

def split_data_with_indices(X, y, config: dict):
    """
    Splits X and y into train, validation, and test sets based on config.
    Returns the split data and the indices used for splitting.
    """
    from sklearn.model_selection import train_test_split
    
    split_mode = config['split_mode']
    task_type = config['task_type']
    stratify = y if task_type != 'regression' and len(np.unique(y)) > 1 else None
    
    n_samples = len(X)
    all_indices = np.arange(n_samples)

    if split_mode == 'train_valid_test':
        cfg = config['split_config']['train_valid_test']
        test_size = cfg.get('test_size', 0.2)
        valid_size = cfg.get('valid_size', 0.1)
        
        # Adjust valid_size to be a fraction of the training/validation set
        if (1 - test_size) <= 0:
            valid_size_of_remainder = 0
        else:
            valid_size_of_remainder = valid_size / (1 - test_size)
        
        indices_train_val, indices_test, X_train_val, X_test, y_train_val, y_test = train_test_split(
            all_indices, X, y, test_size=test_size, random_state=cfg['random_state'], stratify=stratify
        )
        
        stratify_train_val = y_train_val if task_type != 'regression' and len(np.unique(y_train_val)) > 1 else None

        if valid_size_of_remainder > 0:
            indices_train, indices_val, X_train, X_val, y_train, y_val = train_test_split(
                indices_train_val, X_train_val, y_train_val, test_size=valid_size_of_remainder, 
                random_state=cfg['random_state'], stratify=stratify_train_val
            )
        else:
            indices_train, y_train = indices_train_val, y_train_val
            X_train = X_train_val
            indices_val, X_val, y_val = np.array([]), np.array([]).reshape(0, X.shape[1]), np.array([])
        
        split_indices = {
            'train': indices_train,
            'val': indices_val if len(indices_val) > 0 else None,
            'test': indices_test
        }
        
        return X_train, y_train, X_val, y_val, X_test, y_test, split_indices

    elif split_mode == 'cross_validation':
        cfg = config['split_config']['cross_validation']
        test_size = cfg.get('test_size_for_cv', 0.2)
        
        if test_size > 0:
            indices_train, indices_test, X_train, X_test, y_train, y_test = train_test_split(
                all_indices, X, y, test_size=test_size, random_state=cfg['random_state'], stratify=stratify
            )
        else:
            indices_train, y_train = all_indices, y
            X_train = X
            indices_test, X_test, y_test = np.array([]), np.array([]).reshape(0, X.shape[1]), np.array([])

        split_indices = {
            'train': indices_train,
            'val': None,  # No validation set in cross-validation mode
            'test': indices_test if len(indices_test) > 0 else None
        }
        
        return X_train, y_train, None, None, X_test, y_test, split_indices
    else:
        raise ValueError(f"Invalid split_mode: {split_mode}")

def split_data(X, y, config: dict):
    """Splits X and y into train, validation, and test sets based on config."""
    result = split_data_with_indices(X, y, config)
    return result[:6]  # Return only the data splits, not the indices

def load_and_prepare_data(config: dict, output_dir: str):
    """
    Loads data based on the source_mode in the config, processes it to
    generate features, and splits it into train/val/test sets.
    """
    from charset_normalizer import detect

    data_cfg = config['data']
    source_mode = data_cfg['source_mode']
    feature_gen_cfg = config.get('features', {})
    feature_columns = None 

    def read_csv(path):
        try:
            with open(path, 'rb') as f:
                encoding = detect(f.read(20000)).get('encoding', 'utf-8')
            return pd.read_csv(path, encoding=encoding)
        except Exception as e:
            raise FileNotFoundError(f"Could not read data file: {path}. Error: {e}")

    if source_mode == 'single_file':
        console.print("[bold cyan]Step 1: Loading from single file and splitting...[/bold cyan]")
        common_cfg = data_cfg['single_file_config']
        df_full = read_csv(common_cfg['main_file_path'])
        
        # Store original dataframe for saving splits later
        original_df = df_full.copy()
        
        X, y, feature_columns, cleaned_df = process_dataframe(df_full, common_cfg, feature_gen_cfg, output_dir)
        X_train, y_train, X_val, y_val, X_test, y_test, split_indices = split_data_with_indices(X, y, config)
        
        # Save original data splits
        save_original_data_splits(cleaned_df, split_indices, output_dir, console)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns
        
    # Add other source modes here if needed (e.g., pre_split_cv)
    else:
        raise ValueError(f"Invalid `data.source_mode` in config: {source_mode}")

def start_experiment_run(config):
    """Orchestrates a complete training experiment run."""
    from utils.data import encode_labels
    from utils.io_handler import ensure_experiment_directories, save_data_splits_csv, log_experiment_summary, save_config
    from core.trainer_setup import run_all_models_on_data
    from sklearn.preprocessing import StandardScaler
    import optuna

    script_start_time = time.time()
    
    if config['training'].get('quiet_optuna', False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.get('experiment_name', 'INTERNCRANE_run')}_{config.get('task_type', 'task')}_{run_timestamp}"
    run_dir, models_dir, data_splits_dir = ensure_experiment_directories('output', exp_name, console)
    
    save_config(config, run_dir, console=console)
    
    # This call is now correct
    data_tuple = load_and_prepare_data(config, run_dir)
    if not data_tuple: return
    X_train, y_train, X_val, y_val, X_test, y_test, feature_columns = data_tuple
    config['_internal_feature_names'] = feature_columns

    console.print("\n[bold cyan]Saving raw (un-processed) data splits...[/bold cyan]")
    save_data_splits_csv(
        data_splits_dir, "raw_dataset",
        X_train, y_train, X_val, y_val, X_test, y_test,
        scaler=None, label_encoder=None, console=console
    )

    console.print("\n[bold cyan]Step 2: Preprocessing Data[/bold cyan]")
    label_encoder = None
    if config['task_type'] != 'regression':
        all_y_to_encode = [d for d in [y_train, y_val, y_test] if d is not None and len(d) > 0]
        if all_y_to_encode:
            all_y_combined = np.concatenate(all_y_to_encode)
            y_processed, label_encoder = encode_labels(all_y_combined, task_type=config['task_type'], console=console)
            
            # Check if y_processed is valid before using it
            if y_processed is not None:
                start_idx = 0
                y_train_len = len(y_train)
                y_train = y_processed[start_idx:start_idx+y_train_len]; start_idx += y_train_len
                
                if y_val is not None and len(y_val) > 0:
                    y_val_len = len(y_val)
                    y_val = y_processed[start_idx:start_idx+y_val_len]; start_idx += y_val_len
                
                if y_test is not None and len(y_test) > 0:
                    y_test = y_processed[start_idx:]

    scaler = None
    if config.get('features', {}).get('scaling', False):
        console.print("  - Applying StandardScaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_val is not None and len(X_val) > 0: 
            X_val = scaler.transform(X_val)
        if X_test is not None and len(X_test) > 0: 
            X_test = scaler.transform(X_test)

    console.print("\n[bold cyan]Saving processed data splits and artifacts...[/bold cyan]")
    save_data_splits_csv(
        data_splits_dir, "processed_dataset",
        X_train, y_train, X_val, y_val, X_test, y_test,
        scaler, label_encoder, console=console
    )
    
    console.print("\n[bold cyan]Step 3: Starting Model Training and HPO[/bold cyan]")
    all_results = run_all_models_on_data(
        X_train, y_train, X_val, y_val, X_test, y_test,
        models_dir, exp_name, config
    )
    
    total_runtime = time.time() - script_start_time
    log_experiment_summary(run_dir, exp_name, config, total_runtime, script_start_time, all_results, console)
    
    return {
        "run_directory": run_dir,
        "results": all_results
    }