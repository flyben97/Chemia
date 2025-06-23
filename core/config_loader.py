# core/config_loader.py
import yaml
from rich.console import Console

console = Console()

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
    
    # --- Validation ---
    data_cfg = config.get('data', {})
    source_mode = data_cfg.get('source_mode')
    
    if not source_mode:
        raise ValueError("Config error: `data.source_mode` must be specified.")
    
    if source_mode == 'single_file':
        if not data_cfg.get('single_file_config', {}).get('main_file_path'):
            raise ValueError("Config error: For 'single_file' mode, `data.single_file_config.main_file_path` is required.")
    elif source_mode == 'pre_split_cv':
        if not data_cfg.get('pre_split_cv_config', {}).get('train_path') or not data_cfg.get('pre_split_cv_config', {}).get('test_path'):
            raise ValueError("Config error: For 'pre_split_cv' mode, `train_path` and `test_path` are required.")
    elif source_mode == 'pre_split_t_v_t':
        if not data_cfg.get('pre_split_t_v_t_config', {}).get('train_path') or not data_cfg.get('pre_split_t_v_t_config', {}).get('valid_path') or not data_cfg.get('pre_split_t_v_t_config', {}).get('test_path'):
            raise ValueError("Config error: For 'pre_split_t_v_t' mode, `train_path`, `valid_path`, and `test_path` are required.")
    elif source_mode == 'features_only':
        cfg = data_cfg.get('features_only_config', {})
        if not cfg.get('file_path') or not cfg.get('target_col') or not cfg.get('feature_columns'):
             raise ValueError("Config error: For 'features_only' mode, `file_path`, `target_col`, and `feature_columns` are required.")
    else:
        raise ValueError(f"Invalid `data.source_mode`: {source_mode}. Must be 'single_file', 'pre_split_cv', 'pre_split_t_v_t', or 'features_only'.")

    console.print(f"[green]✓ Configuration loaded successfully from '{config_path}'[/green]")
    return config


