# run_optimization.py

import os
import yaml
import logging
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
import argparse
import json

# Import from our modules
from utils.predictor_api import Predictor
from optimization.space_loader import SearchSpaceLoader
from optimization.optimizer import BayesianReactionOptimizer
from utils.io_handler import (
    load_model_from_path, 
    load_scaler_from_path,
    load_label_encoder_from_path, # <-- Import the new loader
    load_config_from_path,
    get_full_model_name,
    find_model_file
)

def setup_logging(output_dir):
    """Creates output directory and sets up logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "optimization_run.log")
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), RichHandler(show_path=False, console=Console(stderr=True))]
    )

def main(config_path: str):
    """Main function to orchestrate the reaction optimization."""
    # 1. Load Optimization Configuration
    with open(config_path, 'r') as f:
        opt_config = yaml.safe_load(f)

    # 2. Setup Output & Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/optimization_{timestamp}"
    setup_logging(output_dir)
    logging.info("CRAFT Reaction Optimizer Started")
    logging.info(f"Output will be saved to: {output_dir}")

    try:
        # --- Artifact loading logic ---
        mode = opt_config.get('mode', 'run_directory')
        logging.info("-" * 50)
        logging.info(f"Operating in [bold cyan]{mode.replace('_', ' ').title()}[/bold cyan] mode.")

        if mode == 'run_directory':
            cfg = opt_config['run_directory_mode']
            source_dir = cfg['source_run_dir']
            model_name_alias = cfg['model_to_use']
            full_model_name = get_full_model_name(model_name_alias)
            logging.info(f"Resolving model alias: '{model_name_alias}' -> '{full_model_name}'")

            model_dir = os.path.join(source_dir, 'models', full_model_name)
            data_splits_dir = os.path.join(source_dir, 'data_splits')
            
            train_config_path = os.path.join(source_dir, 'run_config.json')
            model_path = find_model_file(model_dir, full_model_name)
            scaler_path = os.path.join(data_splits_dir, 'processed_dataset_scaler.joblib')
            encoder_path = os.path.join(data_splits_dir, 'processed_dataset_label_encoder.joblib')
            
        elif mode == 'custom_artifacts':
            cfg = opt_config['custom_artifacts_mode']
            base_dir = cfg['base_dir']
            
            train_config_path = os.path.join(base_dir, cfg['training_config_filename'])
            model_path = os.path.join(base_dir, cfg['model_filename'])
            scaler_path = os.path.join(base_dir, cfg['scaler_filename']) if cfg.get('scaler_filename') else None
            encoder_path = os.path.join(base_dir, cfg['encoder_filename']) if cfg.get('encoder_filename') else None
        
        else:
            raise ValueError(f"Invalid mode '{mode}' in optimization config. Choose 'run_directory' or 'custom_artifacts'.")

        logging.info("Loading artifacts...")
        train_config = load_config_from_path(train_config_path)
        task_type = train_config.get('task_type', 'regression')
        
        model = load_model_from_path(model_path, task_type)
        scaler = load_scaler_from_path(scaler_path)
        label_encoder = load_label_encoder_from_path(encoder_path) if task_type != 'regression' else None
        
        # 3. Initialize the Core Components
        logging.info("-" * 50)
        logging.info("Initializing components...")
        
        predictor = Predictor(
            model=model, 
            scaler=scaler, 
            label_encoder=label_encoder, 
            run_config=train_config, 
            output_dir=output_dir
        )
        
        # --- MODIFICATION: Pass the entire components block to the loader ---
        space_loader = SearchSpaceLoader(opt_config['reaction_components'])
        
        optimizer = BayesianReactionOptimizer(
            predictor=predictor,
            space_loader=space_loader, # The loader now knows about both fixed and search spaces
            opt_config=opt_config['bayesian_optimization']
        )
        
        # 4. Run the Optimization
        logging.info("-" * 50)
        top_results = optimizer.run()
        
        # 5. Save Final Report
        logging.info("=" * 50)
        logging.info("Top Optimized Conditions:")
        logging.info("\n" + top_results.to_string())
        
        # Get top_k from config for dynamic filename
        top_k = opt_config['bayesian_optimization'].get('top_k_results', 10)
        output_csv_path = os.path.join(output_dir, f"top_{top_k}_optimized_conditions.csv")
        top_results.to_csv(output_csv_path, index=False)
        logging.info(f"\nReport saved to: {output_csv_path}")
        logging.info("Optimization process completed successfully.")

    except Exception as e:
        logging.error("A critical error occurred during the optimization process.", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization for chemical reactions using a CRAFT model.")
    parser.add_argument(
        '--config', 
        type=str, 
        default="config_optimization.yaml",
        help="Path to the optimization configuration YAML file."
    )
    args = parser.parse_args()
    main(args.config)