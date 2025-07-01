# run_optimization_only.py

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

# --- START OF FIX: Ensure the project root is in the path ---
# This makes imports like 'from optimization.optimizer ...' work reliably.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END OF FIX ---

# Now, the imports should work correctly.
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

# Setup console
console = Console(width=120)

def setup_optimization_logging(output_dir):
    """Setup dedicated logging for optimization process"""
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
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    opt_logger.addHandler(file_handler)
    
    return opt_logger

def load_model_components(model_source_config):
    """Load model, scaler, encoder and training config from specified sources"""
    
    # Check if we're using source_run_dir or direct_paths
    if 'source_run_dir' in model_source_config and 'model_to_use' in model_source_config:
        # Load from training run directory
        source_dir = model_source_config['source_run_dir']
        model_name = model_source_config['model_to_use']
        
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source run directory not found: {source_dir}")
            
        full_model_name = get_full_model_name(model_name)
        model_dir = os.path.join(source_dir, 'models', full_model_name)
        data_splits_dir = os.path.join(source_dir, 'data_splits')
        
        train_config_path = os.path.join(source_dir, 'run_config.json')
        model_path = find_model_file(model_dir, full_model_name)
        scaler_path = os.path.join(data_splits_dir, 'processed_dataset_scaler.joblib')
        encoder_path = os.path.join(data_splits_dir, 'processed_dataset_label_encoder.joblib')
        
        console.print(f"Loading model from training run: [cyan]{source_dir}[/cyan]")
        console.print(f"Using model: [bold magenta]{model_name.upper()}[/bold magenta]")
        
    elif 'direct_paths' in model_source_config:
        # Load from direct file paths
        paths = model_source_config['direct_paths']
        train_config_path = paths['config_file']
        model_path = paths['model_file']
        scaler_path = paths['scaler_file']
        encoder_path = paths.get('encoder_file', None)  # Optional for regression
        
        console.print(f"Loading model from direct paths:")
        console.print(f"  Model: [cyan]{model_path}[/cyan]")
        console.print(f"  Config: [cyan]{train_config_path}[/cyan]")
        
    else:
        raise ValueError("model_source must contain either 'source_run_dir' or 'direct_paths'")
    
    # Load all components
    train_config = load_config_from_path(train_config_path)
    task_type = train_config.get('task_type', 'regression')
    
    console.print(f"Task type: [yellow]{task_type}[/yellow]")
    
    model = load_model_from_path(model_path, task_type)
    scaler = load_scaler_from_path(scaler_path)
    
    # Load label encoder only for classification tasks
    if task_type != 'regression' and encoder_path and os.path.exists(encoder_path):
        label_encoder = load_label_encoder_from_path(encoder_path)
    else:
        label_encoder = None
    
    console.print("[green]✓ Model components loaded successfully[/green]")
    
    return model, scaler, label_encoder, train_config

def main(config_path: str):
    console.rule("[bold]CRAFT Optimization Only Workflow[/bold]")
    console.print(f"Loading optimization configuration from: [cyan]{config_path}[/cyan]")
    
    with open(config_path, 'r') as f:
        config = yaml_loader(f)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = config.get('output', {}).get('output_dir_prefix', 'optimization')
    opt_output_dir = f"output/{output_prefix}_{timestamp}"
    
    # Setup logging
    opt_logger = setup_optimization_logging(opt_output_dir)
    opt_logger.info("CRAFT Reaction Optimizer Started (Optimization Only)")
    opt_logger.info(f"Output will be saved to: {opt_output_dir}")
    
    try:
        # PHASE 1: LOAD PRE-TRAINED MODEL
        console.rule("[bold blue]Phase 1: Loading Pre-trained Model[/bold blue]")
        
        model, scaler, label_encoder, train_config = load_model_components(config['model_source'])
        
        # PHASE 2: REACTION OPTIMIZATION
        console.rule("[bold blue]Phase 2: Reaction Optimization[/bold blue]")
        
        opt_logger.info("-" * 50)
        opt_logger.info("Initializing optimization components...")
        
        optimization_cfg = config['optimization_config']
        
        # Initialize predictor
        predictor = Predictor(model, scaler, label_encoder, train_config, output_dir=opt_output_dir)
        
        # Initialize search space loader
        space_loader = SearchSpaceLoader(optimization_cfg['reaction_components'])
        
        # Extract fixed components and feature generation config
        fixed_components = optimization_cfg.get('fixed_components', {})
        feature_gen_config = train_config.get('features', {})
        
        # Initialize optimizer
        optimizer = BayesianReactionOptimizer(
            predictor=predictor,
            space_loader=space_loader,
            opt_config=optimization_cfg['bayesian_optimization'],
            fixed_components=fixed_components,
            feature_gen_config=feature_gen_config,
            output_dir=opt_output_dir
        )
        
        console.print("[green]✓ Optimization components initialized[/green]")
        console.print("Starting Bayesian optimization...")
        
        opt_logger.info("-" * 50)
        opt_logger.info("Starting Bayesian optimization...")
        
        # Run optimization
        top_results = optimizer.run()
        
        # Save results
        opt_logger.info("=" * 50)
        opt_logger.info("Top Optimized Conditions:")
        opt_logger.info("\n" + top_results.to_string())
        
        # Get top_k from config for dynamic filename
        top_k = optimization_cfg['bayesian_optimization'].get('top_k_results', 10)
        output_csv_path = os.path.join(opt_output_dir, f"top_{top_k}_optimized_conditions.csv")
        top_results.to_csv(output_csv_path, index=False)
        
        opt_logger.info(f"\nResults saved to: {output_csv_path}")
        
        console.print(f"\n[green]✓ Optimization complete![/green]")
        console.print(f"Results saved to: [cyan]{output_csv_path}[/cyan]")
        console.print(f"Full log available at: [dim]{os.path.join(opt_output_dir, 'optimization_run.log')}[/dim]")

    except Exception as e:
        opt_logger.error("A critical error occurred during the optimization phase.", exc_info=True)
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        console.print(f"Check the log file for details: [dim]{os.path.join(opt_output_dir, 'optimization_run.log')}[/dim]")
        raise
    
    console.rule("[bold green]✓ Optimization Workflow Finished[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CRAFT Bayesian optimization on a pre-trained model.")
    parser.add_argument(
        '--config', 
        type=str, 
        default="config_optimization_only.yaml",
        help="Path to the optimization-only configuration YAML file."
    )
    args = parser.parse_args()
    main(args.config) 