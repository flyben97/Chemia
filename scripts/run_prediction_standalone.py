#!/usr/bin/env python3
"""
CRAFT Standalone Prediction Runner

This script provides a user-friendly interface for making predictions with trained CRAFT models.
It supports both YAML configuration files and direct command-line usage, giving users maximum
flexibility to use trained models without complex setup procedures.

Features:
- YAML configuration support for reproducible predictions
- Direct command-line interface for quick one-off predictions
- Interactive mode for guided prediction setup
- Batch processing for large datasets
- Memory-efficient processing
- Comprehensive error handling and logging
- Support for both regression and classification tasks
"""

import os
import sys
import yaml
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import contextlib
from typing import Optional, Dict, Any, List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- PATH SETUP ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.io_handler import (
    load_model_from_path, load_scaler_from_path, load_label_encoder_from_path,
    load_config_from_path, get_full_model_name, find_model_file
)
from core.run_manager import process_dataframe

# Setup console
console = Console(width=120)

class PredictionRunner:
    """
    A comprehensive prediction runner that supports multiple modes of operation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.training_config = None
        self.task_type = 'regression'
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.INFO
        if self.config.get('logging', {}).get('verbose', False):
            log_level = logging.DEBUG
            
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        self.logger = logging.getLogger(__name__)
    
    @contextlib.contextmanager
    def suppress_output(self):
        """Context manager to suppress stdout/stderr during feature generation"""
        if self.config.get('logging', {}).get('verbose', False):
            yield  # Don't suppress if verbose mode is enabled
        else:
            original_stdout, original_stderr = sys.stdout, sys.stderr
            devnull = open(os.devnull, 'w')
            sys.stdout, sys.stderr = devnull, devnull
            try:
                yield
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                devnull.close()
    
    def display_banner(self):
        """Display application banner"""
        banner_text = """
    ██████ ██████   █████  ███████ ████████ 
   ██      ██   ██ ██   ██ ██         ██    
   ██      ██████  ███████ █████      ██    
   ██      ██   ██ ██   ██ ██         ██    
    ██████ ██   ██ ██   ██ ██         ██    
        
        Prediction Runner v2.0
        """
        console.print(Panel(banner_text, style="bold blue", title="CRAFT Framework"))
    
    def load_artifacts_from_experiment(self, run_dir: str, model_name: str) -> bool:
        """Load model artifacts from an experiment directory"""
        try:
            console.print(f"[bold cyan]Loading from experiment directory:[/bold cyan] {run_dir}")
            
            # Load training configuration
            config_path = os.path.join(run_dir, "run_config.json")
            if not os.path.exists(config_path):
                console.print(f"[bold red]❌ Configuration file not found:[/bold red] {config_path}")
                return False
            
            self.training_config = load_config_from_path(config_path)
            self.task_type = self.training_config.get('task_type', 'regression')
            console.print(f"   • Task type: [magenta]{self.task_type}[/magenta]")
            
            # Load model
            full_model_name = get_full_model_name(model_name)
            model_dir = os.path.join(run_dir, 'models', full_model_name)
            
            if not os.path.exists(model_dir):
                console.print(f"[bold red]❌ Model directory not found:[/bold red] {model_dir}")
                return False
            
            model_path = find_model_file(model_dir, full_model_name)
            self.model = load_model_from_path(model_path, self.task_type)
            console.print(f"   • Model loaded: [green]{type(self.model).__name__}[/green]")
            
            # Load preprocessing artifacts
            data_splits_dir = os.path.join(run_dir, 'data_splits')
            
            # Load scaler
            scaler_path = os.path.join(data_splits_dir, "processed_dataset_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = load_scaler_from_path(scaler_path)
                console.print("   • Scaler loaded: [green]✓[/green]")
            else:
                console.print("   • Scaler: [yellow]Not found (will proceed without scaling)[/yellow]")
            
            # Load label encoder (for classification)
            if self.task_type != 'regression':
                encoder_path = os.path.join(data_splits_dir, "processed_dataset_label_encoder.joblib")
                if os.path.exists(encoder_path):
                    self.label_encoder = load_label_encoder_from_path(encoder_path)
                    console.print("   • Label encoder loaded: [green]✓[/green]")
                else:
                    console.print("   • Label encoder: [yellow]Not found[/yellow]")
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Error loading artifacts:[/bold red] {str(e)}")
            return False
    
    def load_artifacts_from_files(self, model_path: str, config_path: str, 
                                 scaler_path: Optional[str] = None, 
                                 encoder_path: Optional[str] = None) -> bool:
        """Load model artifacts from direct file paths"""
        try:
            console.print("[bold cyan]Loading from direct file paths[/bold cyan]")
            
            # Load training configuration
            self.training_config = load_config_from_path(config_path)
            self.task_type = self.training_config.get('task_type', 'regression')
            console.print(f"   • Task type: [magenta]{self.task_type}[/magenta]")
            
            # Load model
            self.model = load_model_from_path(model_path, self.task_type)
            console.print(f"   • Model loaded: [green]{type(self.model).__name__}[/green]")
            
            # Load scaler
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = load_scaler_from_path(scaler_path)
                console.print("   • Scaler loaded: [green]✓[/green]")
            else:
                console.print("   • Scaler: [yellow]Not provided or not found[/yellow]")
            
            # Load label encoder
            if encoder_path and os.path.exists(encoder_path):
                self.label_encoder = load_label_encoder_from_path(encoder_path)
                console.print("   • Label encoder loaded: [green]✓[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Error loading artifacts:[/bold red] {str(e)}")
            return False
    
    def load_and_validate_input_data(self, input_path: str) -> Optional[pd.DataFrame]:
        """Load and validate input data"""
        try:
            console.print(f"[bold cyan]Loading input data:[/bold cyan] {input_path}")
            
            if not os.path.exists(input_path):
                console.print(f"[bold red]❌ Input file not found:[/bold red] {input_path}")
                return None
            
            # Try to read the CSV file
            df = pd.read_csv(input_path)
            console.print(f"   • Data shape: [green]{df.shape}[/green]")
            console.print(f"   • Columns: {list(df.columns)}")
            
            # Apply column mapping if specified
            column_mapping = self.config.get('data', {}).get('column_mapping', {})
            if column_mapping:
                df = df.rename(columns=column_mapping)
                console.print("   • Applied column mapping")
            
            return df
            
        except Exception as e:
            console.print(f"[bold red]❌ Error loading input data:[/bold red] {str(e)}")
            return None
    
    def process_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Process input data to generate features"""
        console.print("[bold cyan]Generating features...[/bold cyan]")
        
        try:
            # Prepare configuration for feature processing
            if not self.training_config:
                console.print("[bold red]❌ Training configuration not loaded[/bold red]")
                return None
            
            common_cfg = self.training_config.get('data', {}).get('single_file_config', {})
            feature_gen_cfg = self.training_config.get('features', {})
            
            # Add dummy target column if not present
            target_col = common_cfg.get('target_col', 'target')
            if target_col not in df.columns:
                df[target_col] = 0  # Dummy values for prediction
            
            # Generate features with optional output suppression
            with self.suppress_output():
                X, _, feature_names, _ = process_dataframe(
                    df=df.copy(),
                    common_cfg=common_cfg,
                    feature_gen_cfg=feature_gen_cfg,
                    output_dir="."
                )
            
            console.print(f"   • Feature matrix shape: [green]{X.shape}[/green]")
            
            # Apply scaling if available
            if self.scaler is not None:
                X = self.scaler.transform(X)
                console.print("   • Applied feature scaling: [green]✓[/green]")
            
            return X
            
        except Exception as e:
            console.print(f"[bold red]❌ Error processing features:[/bold red] {str(e)}")
            return None
    
    def make_predictions(self, X: np.ndarray):
        """Make predictions using the loaded model"""
        console.print("[bold cyan]Making predictions...[/bold cyan]")
        
        try:
            if self.model is None:
                console.print("[bold red]❌ Model not loaded[/bold red]")
                return None, None
            
            # Make predictions
            predictions = self.model.predict(X)
            # Ensure predictions is a numpy array
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            console.print(f"   • Generated [green]{len(predictions)}[/green] predictions")
            
            # Get probabilities for classification tasks
            probabilities = None
            if self.task_type != 'regression':
                try:
                    if hasattr(self.model, 'predict_proba'):
                        predict_proba_method = getattr(self.model, 'predict_proba')
                        if callable(predict_proba_method):
                            prob_result = predict_proba_method(X)
                            # Ensure probabilities is a numpy array
                            if prob_result is not None:
                                probabilities = np.array(prob_result) if not isinstance(prob_result, np.ndarray) else prob_result
                                console.print("   • Generated class probabilities: [green]✓[/green]")
                except Exception as e:
                    console.print(f"   • [yellow]Warning: Could not generate probabilities: {e}[/yellow]")
            
            return predictions, probabilities
            
        except Exception as e:
            console.print(f"[bold red]❌ Error making predictions:[/bold red] {str(e)}")
            return None, None
    
    def save_results(self, df_original: pd.DataFrame, predictions: np.ndarray, 
                    probabilities: Optional[np.ndarray], output_path: str) -> bool:
        """Save prediction results to file"""
        try:
            console.print(f"[bold cyan]Saving results to:[/bold cyan] {output_path}")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Prepare output dataframe
            output_config = self.config.get('prediction', {}).get('output_format', {})
            include_input = output_config.get('include_input_data', True)
            add_metadata = output_config.get('add_prediction_metadata', True)
            precision = output_config.get('precision', 4)
            
            if include_input:
                result_df = df_original.head(len(predictions)).copy()
            else:
                result_df = pd.DataFrame()
            
            # Add predictions
            if self.task_type != 'regression' and self.label_encoder:
                # For classification, add both encoded and decoded predictions
                result_df['prediction_label'] = self.label_encoder.inverse_transform(predictions)
                result_df['prediction_encoded'] = predictions
            else:
                # For regression
                result_df['prediction'] = np.round(predictions, precision)
            
            # Add probabilities for classification
            if probabilities is not None:
                save_probabilities = self.config.get('prediction', {}).get('save_probabilities', True)
                if save_probabilities:
                    if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
                        class_names = list(self.label_encoder.classes_)
                    else:
                        class_names = [f'class_{i}' for i in range(probabilities.shape[1])]
                    
                    for i, class_name in enumerate(class_names):
                        result_df[f'proba_{class_name}'] = np.round(probabilities[:, i], precision)
            
            # Add metadata
            if add_metadata:
                result_df['prediction_timestamp'] = datetime.now().isoformat()
                result_df['model_type'] = type(self.model).__name__
                result_df['task_type'] = self.task_type
            
            # Save to file
            result_df.to_csv(output_path, index=False)
            console.print(f"   • Saved [green]{len(result_df)}[/green] predictions")
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Error saving results:[/bold red] {str(e)}")
            return False
    
    def run_prediction_pipeline(self) -> bool:
        """Run the complete prediction pipeline"""
        self.display_banner()
        
        # Determine prediction mode
        prediction_mode = self.config.get('prediction_mode', 'experiment_directory')
        
        console.print(f"\n[bold]📋 Prediction Mode:[/bold] [cyan]{prediction_mode}[/cyan]")
        
        # Load model artifacts
        if prediction_mode == 'experiment_directory':
            exp_config = self.config.get('experiment_directory_mode', {})
            run_dir = exp_config.get('run_directory', '')
            model_name = exp_config.get('model_name', '')
            
            if not self.load_artifacts_from_experiment(run_dir, model_name):
                return False
        
        elif prediction_mode == 'direct_files':
            file_config = self.config.get('direct_files_mode', {})
            model_path = file_config.get('model_path', '')
            config_path = file_config.get('config_path', '')
            scaler_path = file_config.get('scaler_path')
            encoder_path = file_config.get('label_encoder_path')
            
            if not self.load_artifacts_from_files(model_path, config_path, scaler_path, encoder_path):
                return False
        
        else:
            console.print(f"[bold red]❌ Invalid prediction mode:[/bold red] {prediction_mode}")
            return False
        
        # Load input data
        input_file = self.config.get('data', {}).get('input_file', '')
        df_input = self.load_and_validate_input_data(input_file)
        if df_input is None:
            return False
        
        # Process features
        X = self.process_features(df_input)
        if X is None:
            return False
        
        # Make predictions
        predictions, probabilities = self.make_predictions(X)
        if predictions is None:
            return False
        
        # Save results
        output_file = self.config.get('data', {}).get('output_file', 'predictions.csv')
        if not self.save_results(df_input, predictions, probabilities, output_file):
            return False
        
        console.print("\n[bold green]🎉 Prediction pipeline completed successfully![/bold green]")
        return True

def interactive_mode():
    """Interactive mode for guided prediction setup"""
    console.print(Panel("Welcome to CRAFT Interactive Prediction Mode", style="bold blue"))
    
    # Get prediction mode
    mode = Prompt.ask(
        "Choose prediction mode",
        choices=["experiment", "files"],
        default="experiment"
    )
    
    config = {
        'prediction_mode': 'experiment_directory' if mode == 'experiment' else 'direct_files',
        'logging': {'verbose': False},
        'prediction': {'save_probabilities': True, 'output_format': {'include_input_data': True}},
        'data': {}
    }
    
    if mode == 'experiment':
        run_dir = Prompt.ask("Enter experiment run directory path")
        model_name = Prompt.ask("Enter model name", default="xgb")
        config['experiment_directory_mode'] = {
            'run_directory': run_dir,
            'model_name': model_name
        }
    else:
        model_path = Prompt.ask("Enter model file path")
        config_path = Prompt.ask("Enter training config path")
        scaler_path = Prompt.ask("Enter scaler path (optional)", default="")
        encoder_path = Prompt.ask("Enter label encoder path (optional)", default="")
        
        config['direct_files_mode'] = {
            'model_path': model_path,
            'config_path': config_path,
            'scaler_path': scaler_path if scaler_path else None,
            'label_encoder_path': encoder_path if encoder_path else None
        }
    
    # Get input/output files
    input_file = Prompt.ask("Enter input CSV file path")
    output_file = Prompt.ask("Enter output file path", default="predictions.csv")
    
    config['data']['input_file'] = input_file
    config['data']['output_file'] = output_file
    
    # Ask for verbose mode
    verbose = Confirm.ask("Enable verbose logging?", default=False)
    config['logging']['verbose'] = verbose
    
    return config

def load_config_from_yaml(config_path: str) -> Optional[Dict[str, Any]]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        console.print(f"[green]✓ Configuration loaded from:[/green] {config_path}")
        return config
    except Exception as e:
        console.print(f"[bold red]❌ Error loading config:[/bold red] {str(e)}")
        return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CRAFT Standalone Prediction Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use YAML configuration file
  python run_prediction_standalone.py --config config_prediction.yaml
  
  # Interactive mode
  python run_prediction_standalone.py --interactive
  
  # Direct command line (experiment mode)
  python run_prediction_standalone.py \\
    --run_dir output/my_experiment \\
    --model_name xgb \\
    --input data/new_data.csv \\
    --output predictions.csv
  
  # Direct command line (file mode)
  python run_prediction_standalone.py \\
    --model_path models/model.json \\
    --config_path config/run_config.json \\
    --input data/new_data.csv \\
    --output predictions.csv
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--config', type=str, help='Path to YAML configuration file')
    mode_group.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    mode_group.add_argument('--run_dir', type=str, help='Experiment run directory (experiment mode)')
    mode_group.add_argument('--model_path', type=str, help='Direct model file path (file mode)')
    
    # Additional arguments for direct mode
    parser.add_argument('--model_name', type=str, help='Model name (for experiment mode)')
    parser.add_argument('--config_path', type=str, help='Training config path (for file mode)')
    parser.add_argument('--scaler_path', type=str, help='Scaler file path (optional)')
    parser.add_argument('--encoder_path', type=str, help='Label encoder path (optional)')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Get configuration
    config = None
    
    if args.config:
        # YAML configuration mode
        config = load_config_from_yaml(args.config)
        if config is None:
            sys.exit(1)
    
    elif args.interactive:
        # Interactive mode
        config = interactive_mode()
    
    elif args.run_dir:
        # Direct experiment mode
        if not args.model_name or not args.input or not args.output:
            console.print("[bold red]❌ For experiment mode, --model_name, --input, and --output are required[/bold red]")
            sys.exit(1)
        
        config = {
            'prediction_mode': 'experiment_directory',
            'experiment_directory_mode': {
                'run_directory': args.run_dir,
                'model_name': args.model_name
            },
            'data': {
                'input_file': args.input,
                'output_file': args.output
            },
            'logging': {'verbose': args.verbose},
            'prediction': {'save_probabilities': True, 'output_format': {'include_input_data': True}}
        }
    
    elif args.model_path:
        # Direct file mode
        if not args.config_path or not args.input or not args.output:
            console.print("[bold red]❌ For file mode, --config_path, --input, and --output are required[/bold red]")
            sys.exit(1)
        
        config = {
            'prediction_mode': 'direct_files',
            'direct_files_mode': {
                'model_path': args.model_path,
                'config_path': args.config_path,
                'scaler_path': args.scaler_path,
                'label_encoder_path': args.encoder_path
            },
            'data': {
                'input_file': args.input,
                'output_file': args.output
            },
            'logging': {'verbose': args.verbose},
            'prediction': {'save_probabilities': True, 'output_format': {'include_input_data': True}}
        }
    
    if config is None:
        console.print("[bold red]❌ Invalid configuration[/bold red]")
        sys.exit(1)
    
    # Run prediction pipeline
    runner = PredictionRunner(config)
    success = runner.run_prediction_pipeline()
    
    if not success:
        console.print("\n[bold red]❌ Prediction pipeline failed[/bold red]")
        sys.exit(1)
    
    console.print(f"\n[bold green]✅ Results saved to:[/bold green] {config['data']['output_file']}")

if __name__ == "__main__":
    main() 