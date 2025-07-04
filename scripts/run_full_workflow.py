# run_full_workflow.py

import os
import sys
try:
    import yaml
    yaml_loader = yaml.safe_load
except ImportError:
    # Use ruamel.yaml as fallback with proper typing
    import ruamel.yaml  # type: ignore
    yaml_obj = ruamel.yaml.YAML(typ='safe', pure=True)
    yaml_loader = yaml_obj.load
import logging
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
import argparse

# --- SUPPRESS DEBUG LOGS AND WARNINGS ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress graphviz deprecation warnings
warnings.filterwarnings("ignore", message=".*positional args.*")

# Configure logging to suppress DEBUG messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('graphviz').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('rdkit').setLevel(logging.WARNING)

# Set root logger to INFO level to avoid DEBUG spam
logging.basicConfig(level=logging.INFO)

# --- START OF FIX: Ensure the project root is in the path ---
# This makes imports like 'from optimization.optimizer ...' work reliably.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END OF FIX ---

# Now, the imports should work correctly.
from core.run_manager import start_experiment_run
from utils.predictor_api import Predictor
from optimization.space_loader import SearchSpaceLoader
from optimization.optimizer import BayesianReactionOptimizer
from utils.io_handler import (
    load_model_from_path, 
    load_scaler_from_path, 
    load_label_encoder_from_path,
    load_config_from_path,
    get_full_model_name,
    find_model_file
)

# Setup console
console = Console(width=120)

# ... (The rest of the file remains the same as the previous correct version) ...
def select_best_model(results: list, metric: str, rank_mode: str) -> dict:
    if not results:
        raise ValueError("Training produced no results to select from.")
    console.print(f"\nSelecting best model based on [bold yellow]{metric}[/bold yellow] ({rank_mode})...")
    valid_results = [res for res in results if metric in res and res[metric] is not None]
    if not valid_results:
        raise ValueError(f"No models found with the specified metric '{metric}'.")
    reverse = True if rank_mode == 'higher_is_better' else False
    best_model_result = sorted(valid_results, key=lambda x: x[metric], reverse=reverse)[0]
    console.print(f"  [green]✓ Best model selected:[/green] [bold magenta]{best_model_result['model_name'].upper()}[/bold magenta] "
                  f"with {metric} = {best_model_result[metric]:.4f}")
    return best_model_result

def setup_optimization_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "optimization_run.log")
    opt_logger = logging.getLogger("optimization_logger")
    opt_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if opt_logger.hasHandlers():
        opt_logger.handlers.clear()
    
    # Prevent propagation to parent loggers (this is key!)
    opt_logger.propagate = False
    
    # Only add file handler - no terminal output for detailed logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')  # Simple format for log file
    file_handler.setFormatter(formatter)
    opt_logger.addHandler(file_handler)
    
    return opt_logger

def main(config_path: str):
    console.rule("[bold]INTERNCRANE End-to-End Workflow[/bold]")
    console.print(f"Loading workflow configuration from: [cyan]{config_path}[/cyan]")
    with open(config_path, 'r') as f:
        workflow_config = yaml_loader(f)

    # PHASE 1: MODEL TRAINING
    console.rule("[bold blue]Phase 1: Model Training[/bold blue]")
    training_config = workflow_config['training_config']
    training_output = start_experiment_run(training_config)
    if not training_output or not training_output.get("results"):
        console.print("[bold red]Training phase failed. Workflow aborted.[/bold red]")
        return
    console.print(f"\n[green]✓ Training Phase Complete.[/green] Results saved in: [dim]{training_output['run_directory']}[/dim]")

    # PHASE 2: MODEL SELECTION
    console.rule("[bold blue]Phase 2: Best Model Selection[/bold blue]")
    selection_cfg = workflow_config['model_selection']
    best_model_info = select_best_model(
        results=training_output['results'],
        metric=selection_cfg['metric'],
        rank_mode=selection_cfg['rank_mode']
    )

    # PHASE 3: REACTION OPTIMIZATION
    console.rule("[bold blue]Phase 3: Reaction Optimization[/bold blue]")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt_output_dir = f"output/E2E_optimization_{timestamp}"
    opt_logger = setup_optimization_logging(opt_output_dir)
    opt_logger.info("INTERNCRANE Reaction Optimizer Started (as part of E2E workflow)")
    opt_logger.info(f"Output will be saved to: {opt_output_dir}")
    opt_logger.info(f"Using best model from training run: {best_model_info['model_name'].upper()}")

    try:
        source_dir = training_output['run_directory']
        full_model_name = get_full_model_name(best_model_info['model_name'])
        model_dir = os.path.join(source_dir, 'models', full_model_name)
        data_splits_dir = os.path.join(source_dir, 'data_splits')
        
        train_config_path = os.path.join(source_dir, 'run_config.json')
        model_path = find_model_file(model_dir, full_model_name)
        scaler_path = os.path.join(data_splits_dir, 'processed_dataset_scaler.joblib')
        encoder_path = os.path.join(data_splits_dir, 'processed_dataset_label_encoder.joblib')

        train_config = load_config_from_path(train_config_path)
        task_type = train_config.get('task_type', 'regression')
        model = load_model_from_path(model_path, task_type)
        scaler = load_scaler_from_path(scaler_path) if os.path.exists(scaler_path) else None
        label_encoder = load_label_encoder_from_path(encoder_path) if (task_type != 'regression' and os.path.exists(encoder_path)) else None

        opt_logger.info("-" * 50)
        opt_logger.info("Initializing optimization components...")
        
        optimization_cfg = workflow_config['optimization_config']
        predictor = Predictor(model, scaler, label_encoder, train_config, output_dir=opt_output_dir)
        space_loader = SearchSpaceLoader(optimization_cfg['reaction_components'])
        
        # Extract fixed components and feature generation config
        fixed_components = optimization_cfg.get('fixed_components', {})
        feature_gen_config = train_config.get('features', {})
        
        optimizer = BayesianReactionOptimizer(
            predictor=predictor,
            space_loader=space_loader,
            opt_config=optimization_cfg['bayesian_optimization'],
            fixed_components=fixed_components,
            feature_gen_config=feature_gen_config,
            output_dir=opt_output_dir
        )
        
        opt_logger.info("-" * 50)
        top_results = optimizer.run()
        
        opt_logger.info("=" * 50)
        opt_logger.info("Top Optimized Conditions:")
        opt_logger.info("\n" + top_results.to_string())
        
        # Get top_k from config for dynamic filename
        top_k = optimization_cfg['bayesian_optimization'].get('top_k_results', 10)
        output_csv_path = os.path.join(opt_output_dir, f"top_{top_k}_optimized_conditions.csv")
        top_results.to_csv(output_csv_path, index=False)
        opt_logger.info(f"\nReport saved to: {output_csv_path}")

    except Exception as e:
        opt_logger.error("A critical error occurred during the optimization phase.", exc_info=True)
    
    console.rule("[bold green]✓ End-to-End Workflow Finished[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full CRAFT workflow: Train -> Select -> Optimize.")
    parser.add_argument(
        '--config', 
        type=str, 
        default="config_full_workflow.yaml",
        help="Path to the full workflow configuration YAML file."
    )
    args = parser.parse_args()
    main(args.config)