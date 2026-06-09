# core/run_manager.py
import time
import os
from datetime import datetime
import numpy as np
import pandas as pd
from rich.console import Console

console = Console(width=120)

def parse_feature_columns(df: pd.DataFrame, col_spec, target_col=None) -> list:
    """
    Parse feature column specification and ensure target column is excluded.

    Args:
        df: DataFrame to parse columns from
        col_spec: Column specification (list of names or string slice like "5:")
        target_col: Target column name to exclude from features

    Returns:
        List of feature column names (excluding target column)
    """
    if isinstance(col_spec, list):
        missing = [col for col in col_spec if col not in df.columns]
        if missing: raise ValueError(f"Specified feature columns not found: {missing}")
        # Explicitly exclude target column
        result = [col for col in col_spec if col != target_col] if target_col else col_spec
        return result
    if isinstance(col_spec, str):
        parts = col_spec.split(':'); start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else df.shape[1]
        columns = df.columns[start:end].tolist()
        # Explicitly exclude target column
        result = [col for col in columns if col != target_col] if target_col else columns
        return result
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

def save_raw_original_data_splits(raw_original_df: pd.DataFrame, cleaned_indices: np.ndarray,
                                 split_indices: dict, output_dir: str, console):
    """
    Save the truly original raw data (before any cleaning or processing) as separate CSV files.

    Args:
        raw_original_df: The completely original DataFrame as loaded from file
        cleaned_indices: Indices of rows that survived the cleaning process
        split_indices: Dictionary containing 'train', 'val', 'test' indices (relative to cleaned data)
        output_dir: Directory to save the split files
        console: Rich console for logging
    """
    console.print("\n[bold cyan]Saving raw original data splits (completely unprocessed)...[/bold cyan]")

    raw_splits_dir = os.path.join(output_dir, 'raw_original_data_splits')
    os.makedirs(raw_splits_dir, exist_ok=True)

    # Map cleaned indices back to original indices
    for split_name, cleaned_split_indices in split_indices.items():
        if cleaned_split_indices is not None and len(cleaned_split_indices) > 0:
            # Map from cleaned data indices to original data indices
            original_split_indices = cleaned_indices[cleaned_split_indices]
            split_df = raw_original_df.iloc[original_split_indices].copy()

            split_file = os.path.join(raw_splits_dir, f'{split_name}_raw_original_data.csv')
            split_df.to_csv(split_file, index=False)
            console.print(f"  - Saved {split_name} raw set: {len(split_df)} samples → [dim]{split_file}[/dim]")

    # Save complete original dataset for reference
    complete_file = os.path.join(raw_splits_dir, 'complete_raw_original_data.csv')
    raw_original_df.to_csv(complete_file, index=False)
    console.print(f"  - Saved complete raw dataset: {len(raw_original_df)} samples → [dim]{complete_file}[/dim]")

    # Create detailed summary
    summary_data = []
    total_cleaned = len(cleaned_indices)

    summary_data.append({
        'dataset': 'original_raw',
        'count': len(raw_original_df),
        'percentage': '100.0%',
        'description': 'Complete original dataset as loaded from file'
    })

    summary_data.append({
        'dataset': 'cleaned_data',
        'count': total_cleaned,
        'percentage': f"{total_cleaned/len(raw_original_df)*100:.1f}%",
        'description': 'Data after cleaning (NaN removal, SMILES validation, etc.)'
    })

    for split_name, cleaned_split_indices in split_indices.items():
        if cleaned_split_indices is not None:
            count = len(cleaned_split_indices)
            summary_data.append({
                'dataset': f'{split_name}_split',
                'count': count,
                'percentage': f"{count/len(raw_original_df)*100:.1f}%",
                'description': f'{split_name.capitalize()} split from cleaned data'
            })

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(raw_splits_dir, 'raw_data_split_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    console.print(f"  - Raw data summary saved → [dim]{summary_file}[/dim]")

    # Save index mapping for reference
    index_mapping = pd.DataFrame({
        'cleaned_index': range(len(cleaned_indices)),
        'original_index': cleaned_indices
    })
    mapping_file = os.path.join(raw_splits_dir, 'index_mapping.csv')
    index_mapping.to_csv(mapping_file, index=False)
    console.print(f"  - Index mapping saved → [dim]{mapping_file}[/dim]")

def process_dataframe(df: pd.DataFrame, common_cfg: dict, feature_gen_cfg: dict, output_dir: str):
    """
    Processes a raw DataFrame to generate a final feature matrix (X) and target vector (y).
    This is the definitive, robust version that handles all edge cases.
    Returns X, y, feature_cols, cleaned DataFrame, and original indices of surviving rows.
    """
    from utils.feature_generator import generate_features
    from utils.smiles_validator import validate_smiles_columns

    # 1. Work on a copy and perform initial cleaning.
    df_processed = df.copy()
    # Add original index tracking
    df_processed['_original_index'] = df_processed.index
    original_indices = df_processed['_original_index'].values.copy()

    smiles_col_spec = common_cfg.get('smiles_col', [])
    smiles_cols_list = [smiles_col_spec] if isinstance(smiles_col_spec, str) else smiles_col_spec

    # 1.1 CRITICAL: Validate that all specified SMILES columns exist in the data
    if smiles_cols_list:
        missing_smiles_cols = [col for col in smiles_cols_list if col not in df_processed.columns]
        if missing_smiles_cols:
            console.print(f"\n[bold red]❌ FATAL ERROR: Missing SMILES columns in data file![/bold red]")
            console.print(f"[red]Configured SMILES columns: {smiles_cols_list}[/red]")
            console.print(f"[red]Missing columns: {missing_smiles_cols}[/red]")
            console.print(f"[yellow]Available columns in data: {list(df_processed.columns)}[/yellow]")

            console.print(f"\n[bold blue]💡 Troubleshooting Tips:[/bold blue]")
            console.print(f"  1. Check your data file column names (case-sensitive)")
            console.print(f"  2. Update your config file 'smiles_col' to match actual column names")
            console.print(f"  3. Ensure the data file is correctly formatted")
            console.print(f"  4. Check for typos in column names")

            # Show potential SMILES-like columns if any
            potential_smiles_cols = [col for col in df_processed.columns
                                   if any(keyword in col.lower() for keyword in ['smiles', 'smi', 'mol', 'structure'])]
            if potential_smiles_cols:
                console.print(f"\n[green]🔍 Potential SMILES columns found in data:[/green]")
                for col in potential_smiles_cols:
                    console.print(f"  • {col}")
                console.print(f"Consider updating your config to use these column names.")

            raise ValueError(
                f"Missing required SMILES columns: {missing_smiles_cols}. "
                f"Please ensure all columns specified in 'smiles_col' exist in your data file. "
                f"Available columns: {list(df_processed.columns)}"
            )

    existing_smiles_cols = [col for col in smiles_cols_list if col in df_processed.columns]

    # 2. SMILES Validation - NEW FEATURE
    if existing_smiles_cols:
        console.print(f"\n[bold cyan]🔍 SMILES Validation:[/bold cyan]")
        console.print(f"  - Checking {len(existing_smiles_cols)} SMILES columns: {existing_smiles_cols}")

        # Get validation settings from config
        validation_cfg = common_cfg.get('smiles_validation', {})
        enabled = validation_cfg.get('enabled', True)  # Default: enabled
        min_valid_ratio = validation_cfg.get('min_valid_ratio', 0.8)  # Default: 80%
        sample_size = validation_cfg.get('sample_size', 200)  # Default: 200 samples
        strict_mode = validation_cfg.get('strict_mode', True)  # Default: strict (raise error)

        if not enabled:
            console.print(f"  [yellow]⚠️  SMILES validation is disabled in configuration[/yellow]")
        else:
            # Validate SMILES columns
            all_valid, validation_results = validate_smiles_columns(
                df_processed,
                existing_smiles_cols,
                sample_size=min(sample_size, len(df_processed)),
                min_valid_ratio=min_valid_ratio,
                show_details=True
            )

            if not all_valid:
                console.print(f"\n[bold red]❌ SMILES Validation Failed![/bold red]")
                console.print(f"[yellow]One or more specified SMILES columns contain invalid SMILES strings.[/yellow]")

                # Create detailed error message
                error_details = []
                for col, results in validation_results.items():
                    if not results['is_valid_column']:
                        if results['error_message']:
                            error_details.append(f"Column '{col}': {results['error_message']}")
                        else:
                            error_details.append(
                                f"Column '{col}': Only {results['valid_count']}/{results['sample_size_checked']} "
                                f"samples are valid SMILES ({results['valid_ratio']:.1%})"
                            )

                # Suggest potential solutions
                console.print(f"\n[bold blue]💡 Suggested Solutions:[/bold blue]")
                console.print(f"  1. Check your configuration file - make sure 'smiles_col' points to actual SMILES columns")
                console.print(f"  2. Clean your data - remove or fix invalid SMILES strings")
                console.print(f"  3. Use SMILES standardization tools (e.g., RDKit canonicalization)")
                console.print(f"  4. Consider using different column names if these are not SMILES columns")
                console.print(f"  5. Disable SMILES validation by setting 'smiles_validation.enabled: false' in config")
                console.print(f"  6. Lower the validation threshold by setting 'smiles_validation.min_valid_ratio: 0.5' in config")

                # Check if there might be other SMILES columns
                from utils.smiles_validator import suggest_potential_smiles_columns
                potential_cols = suggest_potential_smiles_columns(df_processed)
                if potential_cols:
                    console.print(f"\n[bold green]🔍 Potential SMILES columns detected:[/bold green]")
                    for col in potential_cols:
                        console.print(f"  • {col}")
                    console.print(f"Consider updating your configuration to use these columns instead.")

                # Handle based on strict mode
                if strict_mode:
                    # Raise error with detailed information
                    raise ValueError(
                        f"SMILES validation failed for columns: {[col for col in validation_results.keys() if not validation_results[col]['is_valid_column']]}. "
                        f"Details: {'; '.join(error_details)}. "
                        f"Please check your data and configuration, or set 'smiles_validation.strict_mode: false' to continue with warnings."
                    )
                else:
                    # Continue with warning
                    console.print(f"\n[yellow]⚠️  Continuing with invalid SMILES data (strict_mode: false)[/yellow]")
                    console.print(f"[dim]Note: This may cause errors during feature generation.[/dim]")
            else:
                console.print(f"[green]✅ All SMILES columns validated successfully![/green]")
                console.print(f"[dim]  • Sample size: {sample_size}, Min valid ratio: {min_valid_ratio:.1%}[/dim]")

    # 3. Continue with existing processing
    if existing_smiles_cols:
        df_processed.dropna(subset=existing_smiles_cols, inplace=True)

    # 3.1 SMILES Standardization - NEW FEATURE (默认启用)
    if existing_smiles_cols:
        smiles_standardization_cfg = common_cfg.get('smiles_standardization', {})
        standardize_enabled = smiles_standardization_cfg.get('enabled', True)  # Default: enabled
        
        if standardize_enabled:
            console.print(f"\n[bold cyan]🔄 SMILES Standardization:[/bold cyan]")
            console.print(f"  - Standardizing SMILES using RDKit canonicalization...")
            
            from utils.smiles_validator import standardize_smiles as _standardize_smiles
            
            for col in existing_smiles_cols:
                original_values = df_processed[col].copy()
                df_processed[col] = df_processed[col].apply(_standardize_smiles)
                
                # 统计标准化带来的变化
                changed_count = (original_values != df_processed[col]).sum()
                if changed_count > 0:
                    console.print(f"    - Column '{col}': {changed_count} SMILES standardized")
            
            console.print(f"[green]✅ SMILES standardization completed![/green]")
        else:
            console.print(f"\n[yellow]⚠️  SMILES standardization is disabled in configuration[/yellow]")
            console.print(f"[dim]  To enable, set 'smiles_standardization.enabled: true' in config[/dim]")

    # Update original indices after dropna
    current_original_indices = df_processed['_original_index'].values
    df_processed.reset_index(drop=True, inplace=True)

    console.print(f"  - Initial clean data has {df_processed.shape[0]} samples.")

    # Get target column early - needed throughout the function
    target_col = common_cfg.get('target_col')

    # This will hold all feature parts
    all_feature_dfs = []

    # 4. Load Pre-computed Features
    precomputed_cfg = common_cfg.get('precomputed_features')
    if precomputed_cfg and precomputed_cfg.get('feature_columns'):
        console.print("  - Loading pre-computed features...")
        feature_col_names = parse_feature_columns(df_processed, precomputed_cfg['feature_columns'], target_col)

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

        all_feature_dfs.append(precomputed_df)

    # 5. Generate Features from SMILES columns.
    use_per_col_config = 'per_smiles_col_generators' in feature_gen_cfg
    use_global_config = 'generators' in feature_gen_cfg

    # Handle global generators (same features for all SMILES columns)
    if use_global_config and existing_smiles_cols:
        console.print("  - Using global feature configuration for all SMILES columns...")
        global_configs = feature_gen_cfg['generators']

        for s_col in existing_smiles_cols:
            console.print(f"    - Processing column: [magenta]{s_col}[/magenta]")
            smiles_list_for_gen = df_processed[s_col].tolist()
            generated_df = generate_features(smiles_list_for_gen, global_configs, output_dir=output_dir)

            # Add prefix to column names to avoid clashes
            generated_df.columns = [f"{s_col}_{col}" for col in generated_df.columns]
            console.print(f"    - Generated {generated_df.shape[1]} features with prefix '{s_col}_'")

            all_feature_dfs.append(generated_df.reset_index(drop=True))

    # Handle per-column generators (different features for different SMILES columns)
    elif use_per_col_config and existing_smiles_cols:
        console.print("  - Using per-SMILES-column feature configuration...")
        per_col_configs = feature_gen_cfg['per_smiles_col_generators']

        for s_col in smiles_cols_list:
            if s_col in existing_smiles_cols and s_col in per_col_configs:
                console.print(f"    - Processing column: [magenta]{s_col}[/magenta]")
                smiles_list_for_gen = df_processed[s_col].tolist()
                generated_df = generate_features(smiles_list_for_gen, per_col_configs[s_col], output_dir=output_dir)

                # Add prefix to column names to avoid clashes and match optimization phase naming
                generated_df.columns = [f"{s_col}_{col}" for col in generated_df.columns]
                console.print(f"    - Generated {generated_df.shape[1]} features with prefix '{s_col}_'")

                all_feature_dfs.append(generated_df.reset_index(drop=True))

    # 6. Concatenate all feature parts.
    if not all_feature_dfs:
        raise ValueError("No features were loaded or generated. Check your configuration.")

    final_features_df = pd.concat(all_feature_dfs, axis=1)

    console.print(f"  - Combined feature matrix shape before NaN drop: {final_features_df.shape}")

    final_features_df.dropna(inplace=True)
    surviving_indices = final_features_df.index
    console.print(f"  - Combined feature matrix shape after final NaN drop: {final_features_df.shape}")

    # Get the original indices of surviving rows
    final_original_indices = current_original_indices[surviving_indices]

    has_target = target_col and target_col in df_processed.columns

    y = df_processed.loc[surviving_indices, target_col].values if has_target and not final_features_df.empty else np.array([])
    X = final_features_df.values
    feature_cols = final_features_df.columns.tolist()

    console.print(f"  - Final data shapes: X={X.shape}, y={y.shape}")
    if has_target and X.shape[0] != y.shape[0]:
        raise RuntimeError(f"FATAL: X and y shape mismatch! X:{X.shape[0]}, y:{y.shape[0]}")

    # Return the cleaned dataframe and surviving indices for saving original data splits
    cleaned_df = df_processed.iloc[surviving_indices].copy()
    # Remove the temporary index column
    if '_original_index' in cleaned_df.columns:
        cleaned_df = cleaned_df.drop('_original_index', axis=1)
    cleaned_df.reset_index(drop=True, inplace=True)

    return X, y, feature_cols, cleaned_df, final_original_indices

def split_data_with_indices(X, y, config: dict, smiles_list=None):
    """
    Splits X and y into train, validation, and test sets based on config.
    Returns the split data and the indices used for splitting.

    Args:
        X: Feature matrix
        y: Target values
        config: Configuration dict
        smiles_list: Optional list of SMILES strings for scaffold-based splitting
    """
    from sklearn.model_selection import train_test_split

    split_cfg = config.get('split_config', {})
    split_mode = split_cfg.get('split_mode', 'train_valid_test')
    task_type = config['task_type']
    stratify = y if task_type != 'regression' and len(np.unique(y)) > 1 else None

    n_samples = len(X)
    all_indices = np.arange(n_samples)

    # hold_out is an alias for train_valid_test with valid_size=0
    if split_mode == 'hold_out':
        split_mode = 'train_valid_test'

    # Scaffold-based split: group molecules by Bemis-Murcko scaffold
    if split_mode == 'scaffold_split':
        cfg = split_cfg.get('scaffold_split', {})
        test_size = cfg.get('test_size', 0.2)
        random_state = cfg.get('random_state', 42)

        if smiles_list is None or len(smiles_list) != n_samples:
            console.print("[yellow]Warning: scaffold_split requires SMILES list. Falling back to random split.[/yellow]")
            split_mode = 'train_valid_test'
        else:
            from collections import defaultdict
            scaffold_to_indices = defaultdict(list)
            failed_scaffold_indices = []
            try:
                from rdkit import Chem
                from rdkit.Chem.Scaffolds import MurckoScaffold
                for i, smi in enumerate(smiles_list):
                    try:
                        mol = Chem.MolFromSmiles(str(smi))
                        if mol is not None:
                            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                            scaffold_to_indices[scaffold].append(i)
                        else:
                            failed_scaffold_indices.append(i)
                    except Exception:
                        failed_scaffold_indices.append(i)
            except ImportError:
                console.print("[yellow]Warning: RDKit scaffold requires rdkit. Falling back to random split.[/yellow]")
                split_mode = 'train_valid_test'

            if split_mode == 'scaffold_split':
                scaffolds = list(scaffold_to_indices.keys())
                console.print(f"  - Unique scaffolds found: {len(scaffolds)}")
                console.print(f"  - Molecules with failed scaffold: {len(failed_scaffold_indices)}")

                # Split scaffolds (not molecules) into train/test
                scaffold_train, scaffold_test = train_test_split(
                    scaffolds, test_size=test_size, random_state=random_state
                )

                indices_train = np.array([i for s in scaffold_train for i in scaffold_to_indices[s]])
                indices_test = np.array([i for s in scaffold_test for i in scaffold_to_indices[s]])

                # Add failed scaffold molecules to training set
                if failed_scaffold_indices:
                    indices_train = np.concatenate([indices_train, np.array(failed_scaffold_indices)])

                console.print(f"  - Train scaffolds: {len(scaffold_train)}, Test scaffolds: {len(scaffold_test)}")
                console.print(f"  - Train samples: {len(indices_train)}, Test samples: {len(indices_test)}")

                X_train, X_test = X[indices_train], X[indices_test]
                y_train, y_test = y[indices_train], y[indices_test]
                X_val, y_val = np.array([]).reshape(0, X.shape[1]), np.array([])
                indices_val = np.array([])

                split_indices = {
                    'train': indices_train,
                    'val': None,
                    'test': indices_test
                }
                return X_train, y_train, X_val, y_val, X_test, y_test, split_indices

    if split_mode == 'train_valid_test':
        cfg = split_cfg.get('train_valid_test') or split_cfg.get('hold_out', {})
        test_size = cfg.get('test_size', 0.2)
        valid_size = cfg.get('valid_size', 0.0)

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
        cfg = split_cfg.get('cross_validation', {})
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

        X, y, feature_columns, cleaned_df, final_original_indices = process_dataframe(df_full, common_cfg, feature_gen_cfg, output_dir)
        # Extract SMILES for potential scaffold split
        smiles_col = common_cfg.get('smiles_col')
        if isinstance(smiles_col, list):
            smiles_for_split = cleaned_df[smiles_col[0]].tolist() if smiles_col else None
        else:
            smiles_for_split = cleaned_df[smiles_col].tolist() if smiles_col else None
        X_train, y_train, X_val, y_val, X_test, y_test, split_indices = split_data_with_indices(X, y, config, smiles_for_split)

        # Save cleaned data splits (processed but before feature generation)
        save_original_data_splits(cleaned_df, split_indices, output_dir, console)

        # Save raw original data splits (completely unprocessed)
        save_raw_original_data_splits(original_df, final_original_indices, split_indices, output_dir, console)

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns

    elif source_mode == 'pre_split_cv':
        console.print("[bold cyan]Step 1: Loading pre-split train and test files...[/bold cyan]")
        pre_split_cfg = data_cfg['pre_split_cv_config']

        # Load training data
        console.print(f"  - Loading training data from: {pre_split_cfg['train_path']}")
        df_train = read_csv(pre_split_cfg['train_path'])
        console.print(f"    Training set shape: {df_train.shape}")

        # Load test data
        console.print(f"  - Loading test data from: {pre_split_cfg['test_path']}")
        df_test = read_csv(pre_split_cfg['test_path'])
        console.print(f"    Test set shape: {df_test.shape}")

        # Process training data (for cross-validation)
        X_train, y_train, feature_columns, cleaned_df_train, train_original_indices = process_dataframe(df_train, pre_split_cfg, feature_gen_cfg, output_dir)
        console.print(f"    Processed training data shape: X={X_train.shape}, y={y_train.shape}")

        # Process test data (for final evaluation)
        X_test, y_test, _, cleaned_df_test, test_original_indices = process_dataframe(df_test, pre_split_cfg, feature_gen_cfg, output_dir)
        console.print(f"    Processed test data shape: X={X_test.shape}, y={y_test.shape}")

        # In pre_split_cv mode, we don't use a separate validation set
        # Cross-validation will be used on the training set for hyperparameter optimization
        X_val, y_val = None, None

        # Create split indices for saving original data
        split_indices = {
            'train': np.arange(len(cleaned_df_train)),
            'val': None,
            'test': np.arange(len(cleaned_df_test))
        }

        # Save cleaned data splits
        combined_cleaned_df = pd.concat([cleaned_df_train, cleaned_df_test], ignore_index=True)
        save_original_data_splits(combined_cleaned_df, {
            'train': np.arange(len(cleaned_df_train)),
            'val': None,
            'test': np.arange(len(cleaned_df_train), len(cleaned_df_train) + len(cleaned_df_test))
        }, output_dir, console)

        # Save raw original data splits for pre-split mode
        # Combine original dataframes and indices
        combined_original_df = pd.concat([df_train, df_test], ignore_index=True)
        combined_original_indices = np.concatenate([
            train_original_indices,
            test_original_indices + len(df_train)  # Adjust test indices for combined dataframe
        ])

        save_raw_original_data_splits(combined_original_df, combined_original_indices, {
            'train': np.arange(len(train_original_indices)),
            'val': None,
            'test': np.arange(len(train_original_indices), len(train_original_indices) + len(test_original_indices))
        }, output_dir, console)

        console.print("[bold green]✓ Pre-split data loaded successfully![/bold green]")
        console.print(f"  - Training set will be used for cross-validation (CV)")
        console.print(f"  - Test set will be used for final evaluation")

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns

    elif source_mode == 'pre_split_train_test':
        console.print("[bold cyan]Step 1: Loading pre-split train and test files (train/test mode)...[/bold cyan]")
        pre_split_cfg = data_cfg['pre_split_train_test_config']

        # Load training data
        console.print(f"  - Loading training data from: {pre_split_cfg['train_path']}")
        df_train = read_csv(pre_split_cfg['train_path'])
        console.print(f"    Training set shape: {df_train.shape}")

        # Load test data
        console.print(f"  - Loading test data from: {pre_split_cfg['test_path']}")
        df_test = read_csv(pre_split_cfg['test_path'])
        console.print(f"    Test set shape: {df_test.shape}")

        # Process training data
        X_train, y_train, feature_columns, cleaned_df_train, train_original_indices = process_dataframe(df_train, pre_split_cfg, feature_gen_cfg, output_dir)
        console.print(f"    Processed training data shape: X={X_train.shape}, y={y_train.shape}")

        # Process test data (for final evaluation)
        X_test, y_test, _, cleaned_df_test, test_original_indices = process_dataframe(df_test, pre_split_cfg, feature_gen_cfg, output_dir)
        console.print(f"    Processed test data shape: X={X_test.shape}, y={y_test.shape}")

        # Split training data into train/val for hyperparameter optimization
        split_config = config.get('split_config', {}).get('train_valid_test', {})
        train_size = split_config.get('train_size', 0.8)
        val_size = split_config.get('valid_size', 0.2)
        random_state = split_config.get('random_state', 42)

        # Split training data
        from sklearn.model_selection import train_test_split

        # Create indices for tracking splits
        train_indices_local = np.arange(len(X_train))

        # Perform split on data and indices
        X_train_split, X_val, y_train_split, y_val, idx_train_split, idx_val_split = train_test_split(
            X_train, y_train, train_indices_local,
            train_size=train_size,
            test_size=val_size,
            random_state=random_state
        )

        console.print(f"    Training data split: train={X_train_split.shape[0]}, val={X_val.shape[0]}")

        # Create split indices for saving original data
        split_indices = {
            'train': idx_train_split,
            'val': idx_val_split,
            'test': np.arange(len(cleaned_df_train), len(cleaned_df_train) + len(cleaned_df_test))
        }

        # Save cleaned data splits
        combined_cleaned_df = pd.concat([cleaned_df_train, cleaned_df_test], ignore_index=True)
        save_original_data_splits(combined_cleaned_df, {
            'train': np.arange(len(cleaned_df_train)),
            'val': None,
            'test': np.arange(len(cleaned_df_train), len(cleaned_df_train) + len(cleaned_df_test))
        }, output_dir, console)

        # Save raw original data splits
        combined_original_df = pd.concat([df_train, df_test], ignore_index=True)
        combined_original_indices = np.concatenate([
            train_original_indices,
            test_original_indices + len(df_train)
        ])

        save_raw_original_data_splits(combined_original_df, combined_original_indices, {
            'train': np.arange(len(train_original_indices)),
            'val': None,
            'test': np.arange(len(train_original_indices), len(train_original_indices) + len(test_original_indices))
        }, output_dir, console)

        console.print("[bold green]✓ Pre-split train/test data loaded successfully![/bold green]")
        console.print(f"  - Training set split: {X_train_split.shape[0]} for training, {X_val.shape[0]} for validation")
        console.print(f"  - Test set: {X_test.shape[0]} samples for final evaluation")

        return X_train_split, y_train_split, X_val, y_val, X_test, y_test, feature_columns

    elif source_mode == 'pre_split_t_v_t':
        console.print("[bold cyan]Step 1: Loading pre-split train/val/test files...[/bold cyan]")
        pre_split_cfg = data_cfg['pre_split_t_v_t_config']

        # Load training data
        console.print(f"  - Loading training data from: {pre_split_cfg['train_path']}")
        df_train = read_csv(pre_split_cfg['train_path'])
        console.print(f"    Training set shape: {df_train.shape}")

        # Load validation data (optional)
        df_val = None
        if 'valid_path' in pre_split_cfg and pre_split_cfg['valid_path']:
            console.print(f"  - Loading validation data from: {pre_split_cfg['valid_path']}")
            df_val = read_csv(pre_split_cfg['valid_path'])
            console.print(f"    Validation set shape: {df_val.shape}")

        # Load test data
        console.print(f"  - Loading test data from: {pre_split_cfg['test_path']}")
        df_test = read_csv(pre_split_cfg['test_path'])
        console.print(f"    Test set shape: {df_test.shape}")

        # Process training data
        X_train, y_train, feature_columns, cleaned_df_train, train_original_indices = process_dataframe(df_train, pre_split_cfg, feature_gen_cfg, output_dir)
        console.print(f"    Processed training data shape: X={X_train.shape}, y={y_train.shape}")

        # Process validation data if provided
        if df_val is not None:
            X_val, y_val, _, cleaned_df_val, val_original_indices = process_dataframe(df_val, pre_split_cfg, feature_gen_cfg, output_dir)
            console.print(f"    Processed validation data shape: X={X_val.shape}, y={y_val.shape}")
        else:
            X_val, y_val = None, None
            cleaned_df_val = None
            val_original_indices = None

        # Process test data
        X_test, y_test, _, cleaned_df_test, test_original_indices = process_dataframe(df_test, pre_split_cfg, feature_gen_cfg, output_dir)
        console.print(f"    Processed test data shape: X={X_test.shape}, y={y_test.shape}")

        # Save cleaned data splits
        if df_val is not None:
            combined_cleaned_df = pd.concat([cleaned_df_train, cleaned_df_val, cleaned_df_test], ignore_index=True)
            save_original_data_splits(combined_cleaned_df, {
                'train': np.arange(len(cleaned_df_train)),
                'val': np.arange(len(cleaned_df_train), len(cleaned_df_train) + len(cleaned_df_val)),
                'test': np.arange(len(cleaned_df_train) + len(cleaned_df_val), len(combined_cleaned_df))
            }, output_dir, console)
        else:
            combined_cleaned_df = pd.concat([cleaned_df_train, cleaned_df_test], ignore_index=True)
            save_original_data_splits(combined_cleaned_df, {
                'train': np.arange(len(cleaned_df_train)),
                'val': None,
                'test': np.arange(len(cleaned_df_train), len(combined_cleaned_df))
            }, output_dir, console)

        console.print("[bold green]✓ Pre-split train/val/test data loaded successfully![/bold green]")

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns

    else:
        raise ValueError(f"Invalid `data.source_mode` in config: {source_mode}. Supported modes: single_file, pre_split_train_test, pre_split_t_v_t, pre_split_cv")

class RunManager:
    """
    实验运行管理器 - 为批量工作流等场景提供面向对象的接口。
    内部封装了 start_experiment_run 的完整逻辑。
    """

    def __init__(self, config: dict, output_dir: str = 'output', training_only: bool = False):
        self.config = config
        self.output_dir = output_dir
        self.training_only = training_only

    def run_training_only(self):
        """运行仅训练模式（与 run_training_only.py 等价）"""
        return start_experiment_run(self.config)

    def run_full_workflow(self):
        """运行完整工作流程（训练 + 预测/优化）"""
        # 目前完整工作流程的核心也是 start_experiment_run
        # 如需额外预测/优化步骤，可在此处扩展
        return start_experiment_run(self.config)


def _load_features_only(config, output_dir):
    """
    Load data and generate features WITHOUT target extraction or splitting.
    For multi-target scenarios: features are computed once, then per-target
    y-extraction, splitting, and training are done in a loop.
    Returns (X, feature_columns, cleaned_df, smiles_for_split, original_df).
    """
    from charset_normalizer import detect
    data_cfg = config['data']
    common_cfg = data_cfg['single_file_config']

    def read_csv(path):
        try:
            with open(path, 'rb') as f:
                encoding = detect(f.read(20000)).get('encoding', 'utf-8')
            return pd.read_csv(path, encoding=encoding)
        except Exception as e:
            raise FileNotFoundError(f"Could not read data file: {path}. Error: {e}")

    console.print("[bold cyan]Step 1: Loading data and generating features (multi-target)...[/bold cyan]")
    df_full = read_csv(common_cfg['main_file_path'])
    original_df = df_full.copy()

    feature_gen_cfg = config.get('features', {})
    # Temporarily remove target_col so process_dataframe doesn't extract y
    saved_target = common_cfg.pop('target_col', None)
    X, _, feature_columns, cleaned_df, _ = process_dataframe(
        df_full, common_cfg, feature_gen_cfg, output_dir
    )
    if saved_target:
        common_cfg['target_col'] = saved_target

    smiles_col = common_cfg.get('smiles_col')
    if isinstance(smiles_col, list):
        smiles_for_split = cleaned_df[smiles_col[0]].tolist() if smiles_col else None
    else:
        smiles_for_split = cleaned_df[smiles_col].tolist() if smiles_col else None

    return X, feature_columns, cleaned_df, smiles_for_split, original_df


def _train_single_target(X_train, y_train, X_val, y_val, X_test, y_test,
                         config, run_dir, models_dir, data_splits_dir, exp_name, console):
    """Train all models for a single target. Shared by single-target and multi-target paths."""
    from utils.data import encode_labels
    from utils.io_handler import save_data_splits_csv, log_experiment_summary
    from core.trainer_setup import run_all_models_on_data
    from sklearn.preprocessing import StandardScaler

    console.print("\n[bold cyan]Saving raw (un-processed) data splits...[/bold cyan]")
    save_data_splits_csv(
        data_splits_dir, "raw_dataset",
        X_train, y_train, X_test, y_test,
        X_val=X_val, y_val=y_val,
        scaler=None, label_encoder=None, console=console
    )

    console.print("\n[bold cyan]Step 2: Preprocessing Data[/bold cyan]")
    label_encoder = None
    if config['task_type'] != 'regression':
        all_y_to_encode = [d for d in [y_train, y_val, y_test] if d is not None and len(d) > 0]
        if all_y_to_encode:
            all_y_combined = np.concatenate(all_y_to_encode)
            y_processed, label_encoder = encode_labels(all_y_combined, task_type=config['task_type'], console=console)
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
        X_train, y_train, X_test, y_test,
        X_val=X_val, y_val=y_val,
        scaler=scaler, label_encoder=label_encoder, console=console
    )

    console.print("\n[bold cyan]Step 3: Starting Model Training and HPO[/bold cyan]")
    all_results = run_all_models_on_data(
        X_train, y_train, X_val, y_val, X_test, y_test,
        models_dir, exp_name, config
    )
    return all_results


def start_experiment_run(config):
    from utils.data import encode_labels
    from utils.io_handler import ensure_experiment_directories, save_data_splits_csv, log_experiment_summary, save_config
    from core.trainer_setup import run_all_models_on_data
    from sklearn.preprocessing import StandardScaler
    import optuna

    script_start_time = time.time()

    if config['training'].get('quiet_optuna', False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.get('experiment_name', 'CHEMIA_run')}_{config.get('task_type', 'task')}_{run_timestamp}"
    run_dir, models_dir, data_splits_dir = ensure_experiment_directories('output', exp_name, console)
    save_config(config, run_dir, console=console)

    # Check for multi-target config
    data_cfg = config.get('data', {})
    source_mode = data_cfg.get('source_mode', 'single_file')
    common_cfg = data_cfg.get(f'{source_mode}_config') or data_cfg.get('single_file_config', {})
    target_cols = common_cfg.get('target_cols')

    if target_cols and isinstance(target_cols, list) and len(target_cols) > 0 and source_mode == 'single_file':
        # === MULTI-TARGET PATH ===
        console.rule(f"[bold]Multi-Target Run: {len(target_cols)} targets[/bold]")

        # 1. Generate features once
        X_all, feature_columns, cleaned_df, smiles_for_split, original_df = \
            _load_features_only(config, run_dir)
        config['_internal_feature_names'] = feature_columns

        all_target_results = {}
        all_target_rows = []
        for tgt_idx, target_col in enumerate(target_cols):
            console.rule(f"[bold yellow]Target {tgt_idx+1}/{len(target_cols)}: {target_col}[/bold yellow]")

            if target_col not in cleaned_df.columns:
                console.print(f"[yellow]Column '{target_col}' not found. Skipping.[/yellow]")
                continue

            # 2. Extract y and split
            y_all = cleaned_df[target_col].values
            X_train, y_train, X_val, y_val, X_test, y_test, split_indices = \
                split_data_with_indices(X_all, y_all, config, smiles_for_split)

            # 3. Per-target output directory
            safe_name = target_col.lower().replace(' ', '_')[:40]
            tgt_run_dir = os.path.join(run_dir, f"target_{safe_name}")
            tgt_models_dir = os.path.join(tgt_run_dir, "models")
            tgt_data_dir = os.path.join(tgt_run_dir, "data_splits")
            os.makedirs(tgt_models_dir, exist_ok=True)
            os.makedirs(tgt_data_dir, exist_ok=True)

            # 4. Save original splits for this target
            save_original_data_splits(cleaned_df, split_indices, tgt_run_dir, console)
            save_raw_original_data_splits(original_df, np.arange(len(cleaned_df)), split_indices, tgt_run_dir, console)

            # 5. Train
            tgt_results = _train_single_target(
                X_train, y_train, X_val, y_val, X_test, y_test,
                config, tgt_run_dir, tgt_models_dir, tgt_data_dir, exp_name, console
            )

            for r in tgt_results:
                r['target_col'] = target_col
            all_target_results[target_col] = tgt_results
            all_target_rows.extend(tgt_results)

        # 6. Combined results
        if all_target_rows:
            import pandas as pd
            pd.DataFrame(all_target_rows).to_csv(
                os.path.join(run_dir, "combined_results.csv"), index=False
            )

        total_runtime = time.time() - script_start_time
        log_experiment_summary(run_dir, exp_name, config, total_runtime, script_start_time, all_target_rows, console)

        # Clean up embedding cache
        from utils.embedding_cache import clear_cache
        clear_cache()

        return {
            "run_directory": run_dir,
            "results": all_target_results
        }

    # === SINGLE-TARGET PATH (original behavior) ===
    data_tuple = load_and_prepare_data(config, run_dir)
    if not data_tuple: return
    X_train, y_train, X_val, y_val, X_test, y_test, feature_columns = data_tuple
    config['_internal_feature_names'] = feature_columns

    all_results = _train_single_target(
        X_train, y_train, X_val, y_val, X_test, y_test,
        config, run_dir, models_dir, data_splits_dir, exp_name, console
    )

    total_runtime = time.time() - script_start_time
    log_experiment_summary(run_dir, exp_name, config, total_runtime, script_start_time, all_results, console)

    # Clean up embedding cache
    from utils.embedding_cache import clear_cache
    clear_cache()

    return {
        "run_directory": run_dir,
        "results": all_results
    }
