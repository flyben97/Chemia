# core/config_loader.py
import yaml
from rich.console import Console
from typing import Any, Dict, List

console = Console()

def _validate_smiles_col(cfg: Dict[str, Any], mode_name: str):
    """Helper function to validate the smiles_col configuration."""
    if 'smiles_col' not in cfg:
        raise ValueError(f"Config error: `data.{mode_name}.smiles_col` is required.")
    
    smiles_col_spec = cfg.get('smiles_col')
    if not isinstance(smiles_col_spec, (str, list)):
        raise ValueError(f"Config error: `data.{mode_name}.smiles_col` must be a string or a list of strings.")
    if isinstance(smiles_col_spec, list) and not all(isinstance(item, str) for item in smiles_col_spec):
        raise ValueError(f"Config error: If `smiles_col` is a list, all its items must be strings.")

def _validate_feature_generators(features_cfg: Dict[str, Any], smiles_cols: List[str]):
    """Validates the feature generator configuration."""
    has_global = 'generators' in features_cfg
    has_per_col = 'per_smiles_col_generators' in features_cfg

    if has_global and has_per_col:
        raise ValueError("Config error: Cannot specify both `features.generators` (global) and `features.per_smiles_col_generators` (per-column). Please choose one.")

    if has_per_col:
        per_col_cfg = features_cfg['per_smiles_col_generators']
        if not isinstance(per_col_cfg, dict):
            raise ValueError("Config error: `features.per_smiles_col_generators` must be a dictionary.")
        
        for col_name, generators in per_col_cfg.items():
            if col_name not in smiles_cols:
                raise ValueError(f"Config error: The column '{col_name}' specified in `per_smiles_col_generators` is not found in the `data.smiles_col` list: {smiles_cols}")
            if not isinstance(generators, list):
                raise ValueError(f"Config error: The value for '{col_name}' in `per_smiles_col_generators` must be a list of generator configs.")
    elif not has_global:
        # This is a valid case if only precomputed features are used.
        # The check for no features at all is in run_manager.py
        pass


def load_config(config_path: str) -> dict:
    """
    Loads and validates the YAML configuration file.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Error: Configuration file not found at '{config_path}'[/bold red]")
        raise
    except yaml.YAMLError as e:
        console.print(f"[bold red]Error parsing YAML file: {e}[/bold red]")
        raise
    
    # --- Data Source Validation ---
    data_cfg = config.get('data', {})
    source_mode = data_cfg.get('source_mode')
    
    if not source_mode:
        raise ValueError("Config error: `data.source_mode` must be specified.")
    
    smiles_cols_list = []
    
    if source_mode == 'single_file':
        cfg = data_cfg.get('single_file_config', {})
        if not cfg.get('main_file_path'):
            raise ValueError("Config error: For 'single_file' mode, `data.single_file_config.main_file_path` is required.")
        _validate_smiles_col(cfg, 'single_file_config')
        smiles_col_spec = cfg['smiles_col']
    elif source_mode == 'pre_split_cv':
        cfg = data_cfg.get('pre_split_cv_config', {})
        if not cfg.get('train_path') or not cfg.get('test_path'):
            raise ValueError("Config error: For 'pre_split_cv' mode, `train_path` and `test_path` are required.")
        _validate_smiles_col(cfg, 'pre_split_cv_config')
        smiles_col_spec = cfg['smiles_col']
    elif source_mode == 'pre_split_t_v_t':
        cfg = data_cfg.get('pre_split_t_v_t_config', {})
        if not cfg.get('train_path') or not cfg.get('valid_path') or not cfg.get('test_path'):
            raise ValueError("Config error: For 'pre_split_t_v_t' mode, `train_path`, `valid_path`, and `test_path` are required.")
        _validate_smiles_col(cfg, 'pre_split_t_v_t_config')
        smiles_col_spec = cfg['smiles_col']
    elif source_mode == 'features_only':
        cfg = data_cfg.get('features_only_config', {})
        if not cfg.get('file_path') or not cfg.get('target_col') or not cfg.get('feature_columns'):
             raise ValueError("Config error: For 'features_only' mode, `file_path`, `target_col`, and `feature_columns` are required.")
        smiles_col_spec = [] # No smiles columns in this mode
    else:
        raise ValueError(f"Invalid `data.source_mode`: {source_mode}. Must be 'single_file', 'pre_split_cv', 'pre_split_t_v_t', or 'features_only'.")

    smiles_cols_list = [smiles_col_spec] if isinstance(smiles_col_spec, str) else smiles_col_spec

    # --- Feature Generation Validation ---
    features_cfg = config.get('features', {})
    _validate_feature_generators(features_cfg, smiles_cols_list)

    console.print(f"[green]✓ Configuration loaded successfully from '{config_path}'[/green]")
    return config
