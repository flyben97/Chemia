#!/usr/bin/env python3
"""
CRAFT Training-Only Runner

This script focuses exclusively on the model training phase with comprehensive
algorithm selection and hyperparameter optimization. It's designed for maximum
flexibility and performance comparison across different ML algorithms.
"""

import os
import sys
import yaml
import logging
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import argparse
import pandas as pd
import numpy as np

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
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END OF FIX ---

from core.run_manager import start_experiment_run

# Setup console
console = Console(width=120)

def display_algorithm_summary(config):
    """Display a summary of algorithms to be trained"""
    
    models = config['training']['models_to_run']
    n_trials = config['training']['n_trials']
    
    # Categorize algorithms
    categories = {
        'Gradient Boosting': ['xgb', 'lgbm', 'catboost', 'gbdt', 'histgradientboosting'],
        'Tree Ensembles': ['rf', 'extratrees', 'adaboost'],
        'Linear Regularized': ['ridge', 'elasticnet', 'lasso', 'bayesianridge'],
        'Advanced/Kernel': ['gpr', 'krr', 'svr'],
        'Neural Networks': ['ann'],  # PyTorch-based ANN
        'Graph Neural Networks': ['gcn', 'gat', 'mpnn', 'afp', 'gtn', 'graph_transformer', 'ensemble_gnn'],  # GNN models
        'Simple Methods': ['kneighbors', 'sgd']
    }
    
    table = Table(title="🤖 Algorithm Training Plan", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", width=20)
    table.add_column("Algorithms", style="white", width=40)
    table.add_column("Count", style="green", justify="center", width=10)
    table.add_column("Trials Each", style="yellow", justify="center", width=12)
    table.add_column("Total Trials", style="red", justify="center", width=12)
    
    total_algorithms = 0
    total_trials = 0
    
    for category, category_models in categories.items():
        selected_models = [m for m in models if m in category_models]
        if selected_models:
            count = len(selected_models)
            trials_for_category = count * n_trials
            total_algorithms += count
            total_trials += trials_for_category
            
            table.add_row(
                category,
                ", ".join(selected_models),
                str(count),
                str(n_trials),
                str(trials_for_category)
            )
    
    # Add summary row
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_algorithms} algorithms[/bold]",
        f"[bold]{total_algorithms}[/bold]",
        f"[bold]{n_trials}[/bold]",
        f"[bold]{total_trials}[/bold]"
    )
    
    console.print(table)
    console.print(f"\n[bold green]📊 Training Summary:[/bold green]")
    console.print(f"   • Total algorithms: [bold]{total_algorithms}[/bold]")
    console.print(f"   • Hyperparameter trials per algorithm: [bold]{n_trials}[/bold]")
    console.print(f"   • Total hyperparameter optimization trials: [bold]{total_trials}[/bold]")

def analyze_training_results(results, config):
    """Analyze and display detailed training results"""
    
    console.rule("[bold green]📈 Training Results Analysis[/bold green]")
    
    if not results:
        console.print("[bold red]❌ No training results to analyze![/bold red]")
        return
    
    # Create results summary table
    table = Table(title="🏆 Model Performance Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Algorithm", style="cyan", width=15)
    table.add_column("R² Score", style="green", justify="center", width=12)
    table.add_column("RMSE", style="yellow", justify="center", width=12)
    table.add_column("MAE", style="blue", justify="center", width=12)
    table.add_column("CV Std", style="red", justify="center", width=12)
    table.add_column("Status", style="white", justify="center", width=15)
    
    # Sort results by R² score
    sorted_results = sorted(results, key=lambda x: x.get('test_r2', -999), reverse=True)
    
    best_r2 = -999
    for i, result in enumerate(sorted_results):
        model_name = result['model_name'].upper()
        r2 = result.get('test_r2', None)
        rmse = result.get('test_rmse', None)
        mae = result.get('test_mae', None)
        cv_std = result.get('cv_std_r2', None)
        
        # Format metrics
        r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
        rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
        mae_str = f"{mae:.4f}" if mae is not None else "N/A"
        cv_std_str = f"{cv_std:.4f}" if cv_std is not None else "N/A"
        
        # Determine status
        if i == 0 and r2 is not None:
            status = "🥇 BEST"
            best_r2 = r2
        elif r2 is not None and r2 > 0:
            status = "✅ Good"
        elif r2 is not None and r2 > -0.5:
            status = "⚠️ Fair"
        else:
            status = "❌ Poor"
        
        table.add_row(model_name, r2_str, rmse_str, mae_str, cv_std_str, status)
    
    console.print(table)
    
    # Performance insights
    console.print(f"\n[bold blue]📊 Performance Insights:[/bold blue]")
    
    # Count good performers
    good_performers = [r for r in results if r.get('test_r2', -999) > 0]
    poor_performers = [r for r in results if r.get('test_r2', -999) <= -0.5]
    
    console.print(f"   • Best R² Score: [bold green]{best_r2:.4f}[/bold green]")
    console.print(f"   • Models with R² > 0: [bold]{len(good_performers)}/{len(results)}[/bold]")
    
    if poor_performers:
        console.print(f"   • Poor performers (R² ≤ -0.5): [bold red]{len(poor_performers)}[/bold red]")
        poor_names = [r['model_name'].upper() for r in poor_performers]
        console.print(f"     {', '.join(poor_names)}")
    
    # Overfitting analysis
    overfit_models = []
    for result in results:
        train_r2 = result.get('train_r2', None)
        test_r2 = result.get('test_r2', None)
        if train_r2 is not None and test_r2 is not None:
            if train_r2 - test_r2 > 0.3:  # Large gap indicates overfitting
                overfit_models.append((result['model_name'], train_r2 - test_r2))
    
    if overfit_models:
        console.print(f"\n[bold yellow]⚠️  Potential Overfitting Detected:[/bold yellow]")
        for model_name, gap in overfit_models:
            console.print(f"   • {model_name.upper()}: Train-Test gap = {gap:.4f}")
    
    # Recommendations
    console.print(f"\n[bold cyan]💡 Recommendations:[/bold cyan]")
    
    if best_r2 < 0:
        console.print("   • [red]All models show poor performance (R² < 0)[/red]")
        console.print("   • Consider: More data, feature engineering, different target encoding")
    elif best_r2 < 0.3:
        console.print("   • [yellow]Moderate performance. Room for improvement.[/yellow]")
        console.print("   • Consider: Feature selection, ensemble methods, data augmentation")
    else:
        console.print("   • [green]Good performance achieved![/green]")
        console.print("   • Consider: Fine-tuning best model, ensemble of top performers")

def main(config_path: str, dry_run: bool = False):
    """Main training function"""
    
    console.rule("[bold]🚀 CRAFT Training-Only Pipeline[/bold]")
    console.print(f"Loading training configuration from: [cyan]{config_path}[/cyan]")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Display algorithm summary
    display_algorithm_summary(config)
    
    if dry_run:
        console.print(f"\n[bold yellow]🔍 DRY RUN MODE - No actual training will be performed[/bold yellow]")
        return
    
    # Start training
    console.rule("[bold blue]🎯 Starting Model Training[/bold blue]")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        training_output = start_experiment_run(config)
        
        if not training_output or not training_output.get("results"):
            console.print("[bold red]❌ Training failed or produced no results![/bold red]")
            return
        
        console.print(f"\n[green]✅ Training completed successfully![/green]")
        console.print(f"Results saved in: [dim]{training_output['run_directory']}[/dim]")
        
        # Analyze results
        analyze_training_results(training_output['results'], config)
        
        # Save additional analysis
        results_df = pd.DataFrame(training_output['results'])
        analysis_path = os.path.join(training_output['run_directory'], 'model_comparison.csv')
        results_df.to_csv(analysis_path, index=False)
        console.print(f"\n[dim]Detailed results saved to: {analysis_path}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Training failed with error:[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        raise
    
    console.rule("[bold green]🎉 Training Pipeline Complete[/bold green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive model training with CRAFT framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training_only.py                          # Use default config
  python run_training_only.py --config my_config.yaml  # Use custom config
  python run_training_only.py --dry-run                # Preview without training
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default="config_training_only.yaml",
        help="Path to the training configuration YAML file (default: config_training_only.yaml)"
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Preview the training plan without actually running it"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        console.print(f"[bold red]❌ Configuration file not found: {args.config}[/bold red]")
        sys.exit(1)
    
    main(args.config, args.dry_run) 