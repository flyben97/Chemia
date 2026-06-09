#!/usr/bin/env python3
"""
SHAP Analysis Runner

Performs SHAP-based model interpretability analysis on trained models.
Supports all tree-based and compatible models from the training pipeline.
"""

import os
import sys
import argparse
import json
import numpy as np


from rich.console import Console

from rich.table import Table

# --- 标准 CLI 初始化 ---
from utils.cli_setup import standard_cli_setup
from utils.constants import DEFAULT_CONSOLE_WIDTH
standard_cli_setup()
# --- END ---

from utils.shap_analyzer import SHAPAnalyzer, SHAP_AVAILABLE
from utils.io_handler import (
    load_config_from_path, load_model_from_path, get_full_model_name, find_model_file
)


console = Console(width=DEFAULT_CONSOLE_WIDTH, highlight=False)

def get_available_models(run_dir: str) -> list:
    """Get list of available trained models in the run directory."""
    models_dir = os.path.join(run_dir, 'models')
    if not os.path.exists(models_dir):
        return []

    available_models = []
    for model_folder in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_folder)
        if os.path.isdir(model_path):
            available_models.append(model_folder)

    return sorted(available_models)

def load_data_for_analysis(run_dir: str, config: dict) -> tuple:
    """Load training and test data for SHAP analysis."""
    data_splits_dir = os.path.join(run_dir, 'data_splits')

    # Load processed data
    X_train_path = os.path.join(data_splits_dir, 'processed_dataset_X_train.npy')
    X_test_path = os.path.join(data_splits_dir, 'processed_dataset_X_test.npy')
    y_train_path = os.path.join(data_splits_dir, 'processed_dataset_y_train.npy')
    y_test_path = os.path.join(data_splits_dir, 'processed_dataset_y_test.npy')

    if not all(os.path.exists(p) for p in [X_train_path, X_test_path]):
        console.print("[bold red]✗ Processed data files not found![/bold red]")
        return None

    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path) if os.path.exists(y_train_path) else None
    y_test = np.load(y_test_path) if os.path.exists(y_test_path) else None

    return X_train, X_test, y_train, y_test

def get_feature_names(run_dir: str, config: dict) -> list:
    """Get feature names from the experiment."""
    # Try to load from config
    if '_internal_feature_names' in config:
        return config['_internal_feature_names']

    # Try to load from saved feature names file
    feature_names_path = os.path.join(run_dir, 'data_splits', 'feature_names.json')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            return json.load(f)

    # Generate default feature names
    data_splits_dir = os.path.join(run_dir, 'data_splits')
    X_train_path = os.path.join(data_splits_dir, 'processed_dataset_X_train.npy')
    if os.path.exists(X_train_path):
        X_train = np.load(X_train_path)
        return [f"Feature_{i}" for i in range(X_train.shape[1])]

    return []

def analyze_single_model(
    run_dir: str,
    model_name: str,
    config: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    max_samples: int = 100
) -> bool:
    """Perform SHAP analysis on a single model."""

    console.print(f"\n[bold cyan]Analyzing model: {model_name.upper()}[/bold cyan]")

    # Load model
    full_model_name = get_full_model_name(model_name)
    model_dir = os.path.join(run_dir, 'models', full_model_name)
    task_type = config.get('task_type', 'regression')

    try:
        model_path_to_load = find_model_file(model_dir, full_model_name)
        model = load_model_from_path(model_path_to_load, task_type)
        console.print(f"  ✓ Loaded model from {os.path.basename(model_path_to_load)}")
    except Exception as e:
        console.print(f"  [bold red]✗ Failed to load model: {e}[/bold red]")
        return False

    # Create SHAP analyzer
    try:
        analyzer = SHAPAnalyzer(model, X_train, feature_names)
        console.print(f"  ✓ Created SHAP analyzer (model type: {analyzer.model_type})")
    except Exception as e:
        console.print(f"  [bold red]✗ Failed to create analyzer: {e}[/bold red]")
        return False

    # Generate SHAP analysis report
    output_dir = os.path.join(run_dir, 'shap_analysis', full_model_name)
    try:
        analyzer.generate_report(X_test[:max_samples], output_dir)
        console.print(f"  ✓ SHAP analysis complete!")
        console.print(f"    Results saved to: [dim]{output_dir}[/dim]")
        return True
    except Exception as e:
        console.print(f"  [bold red]✗ Failed to generate report: {e}[/bold red]")
        return False

def main(args):
    """Main SHAP analysis function."""

    console.rule("[bold]🔍 SHAP Model Interpretability Analysis[/bold]")

    # Validate run directory
    if not os.path.exists(args.run_dir):
        console.print(f"[bold red]✗ Run directory not found: {args.run_dir}[/bold red]")
        sys.exit(1)

    # Load configuration
    config_path = os.path.join(args.run_dir, 'run_config.json')
    if not os.path.exists(config_path):
        console.print(f"[bold red]✗ Configuration file not found: {config_path}[/bold red]")
        sys.exit(1)

    try:
        config = load_config_from_path(config_path)
        console.print(f"✓ Loaded configuration from {os.path.basename(args.run_dir)}")
    except Exception as e:
        console.print(f"[bold red]✗ Failed to load config: {e}[/bold red]")
        sys.exit(1)

    # Check SHAP availability
    if not SHAP_AVAILABLE:
        console.print("[bold red]✗ SHAP is not installed![/bold red]")
        console.print("   Install with: pip install shap")
        sys.exit(1)

    # Load data
    console.print("\n[bold cyan]Loading data...[/bold cyan]")
    data_result = load_data_for_analysis(args.run_dir, config)
    if data_result is None:
        sys.exit(1)

    X_train, X_test, y_train, y_test = data_result
    console.print(f"  ✓ Loaded training data: {X_train.shape}")
    console.print(f"  ✓ Loaded test data: {X_test.shape}")

    # Get feature names
    feature_names = get_feature_names(args.run_dir, config)
    if not feature_names:
        console.print("[bold yellow]⚠ Could not determine feature names, using defaults[/bold yellow]")
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    # Get available models
    available_models = get_available_models(args.run_dir)
    if not available_models:
        console.print("[bold red]✗ No trained models found in run directory![/bold red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]Available models: {len(available_models)}[/bold cyan]")
    for model in available_models:
        console.print(f"  • {model}")

    # Determine which models to analyze
    if args.models:
        models_to_analyze = [m.lower() for m in args.models.split(',')]
        models_to_analyze = [m for m in models_to_analyze if m in [x.lower() for x in available_models]]
        if not models_to_analyze:
            console.print("[bold red]✗ No matching models found![/bold red]")
            sys.exit(1)
    else:
        models_to_analyze = available_models

    console.print(f"\n[bold cyan]Analyzing {len(models_to_analyze)} model(s)...[/bold cyan]")

    # Analyze each model
    results = []
    for model_name in models_to_analyze:
        success = analyze_single_model(
            args.run_dir,
            model_name,
            config,
            X_train,
            X_test,
            feature_names,
            max_samples=args.max_samples
        )
        results.append((model_name, success))

    # Summary
    console.rule("[bold green]📊 Analysis Summary[/bold green]")

    table = Table(title="SHAP Analysis Results", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", width=20)
    table.add_column("Status", style="white", width=15)
    table.add_column("Output Directory", style="dim", width=50)

    successful = 0
    for model_name, success in results:
        status = "✓ Success" if success else "✗ Failed"
        status_style = "green" if success else "red"
        output_dir = os.path.join(args.run_dir, 'shap_analysis', get_full_model_name(model_name))
        table.add_row(
            model_name,
            f"[{status_style}]{status}[/{status_style}]",
            output_dir if success else "N/A"
        )
        if success:
            successful += 1

    console.print(table)

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  • Successful: [bold green]{successful}/{len(results)}[/bold green]")
    console.print(f"  • Failed: [bold red]{len(results) - successful}/{len(results)}[/bold red]")

    if successful > 0:
        console.print(f"\n[bold cyan]SHAP analysis results saved to:[/bold cyan]")
        console.print(f"  [dim]{os.path.join(args.run_dir, 'shap_analysis')}[/dim]")

    console.rule("[bold green]✓ SHAP Analysis Complete[/bold green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform SHAP-based model interpretability analysis on trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all models in a run
  python run_shap_analysis.py --run-dir output/CHEMIA_run_regression_20240101_120000

  # Analyze specific models
  python run_shap_analysis.py --run-dir output/CHEMIA_run_regression_20240101_120000 --models xgboost,lgbm

  # Limit samples for faster analysis
  python run_shap_analysis.py --run-dir output/CHEMIA_run_regression_20240101_120000 --max-samples 50
        """
    )

    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to the completed training run directory'
    )

    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of models to analyze (e.g., "xgboost,lgbm,catboost")'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum number of test samples to analyze (default: 100)'
    )

    args = parser.parse_args()
    main(args)
