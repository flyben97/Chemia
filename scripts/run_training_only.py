#!/usr/bin/env python3
"""
CHEMIA Training-Only Runner

This script focuses exclusively on the model training phase with comprehensive
algorithm selection and hyperparameter optimization. It's designed for maximum
flexibility and performance comparison across different ML algorithms.
"""

import os
import sys
import yaml

from datetime import datetime

from rich.console import Console

from rich.table import Table
import argparse
import pandas as pd


# --- 标准 CLI 初始化 ---
from utils.cli_setup import standard_cli_setup
from utils.constants import DEFAULT_CONSOLE_WIDTH
project_root = standard_cli_setup()
# --- END ---

from core.run_manager import start_experiment_run
from utils.io_handler import get_full_model_name

console = Console(width=DEFAULT_CONSOLE_WIDTH)

def display_algorithm_summary(config):
    """Display a summary of algorithms to be trained"""

    models = config['training']['models_to_run']
    # Convert model aliases to full names
    models = [get_full_model_name(m) for m in models]
    n_trials = config['training']['n_trials']

    # --- MODIFICATION START: Remove local alias dictionary and use a simplified category list ---
    # The full name conversion is now handled elsewhere.
    categories = {
        'Gradient Boosting': ['xgboost', 'lgbm', 'catboost', 'gbdt', 'histgradientboosting'],
        'Tree Ensembles': ['randomforest', 'extratrees', 'adaboost'],
        'Linear Regularized': ['ridge', 'elasticnet', 'lasso', 'bayesianridge', 'logisticregression'],
        'Advanced/Kernel': ['gpr', 'kernelridge', 'svr', 'svc'],
        'Neural Networks': ['ann', 'tabnet'],
        'Graph Neural Networks': ['gcn', 'gat', 'mpnn', 'afp', 'graph_transformer', 'ensemble_gnn'],
        'Simple Methods': ['kneighbors', 'sgd', 'decisiontree']
    }

    # Map all aliases in categories to their full names for matching
    categorized_models = {}
    for cat, model_list in categories.items():
        # Also check original short names for matching against the config
        all_names = set(model_list)
        for m in model_list:
            all_names.add(get_full_model_name(m))
        categorized_models[cat] = all_names
    # --- MODIFICATION END ---

    table = Table(title="🤖 Algorithm Training Plan", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", width=20)
    table.add_column("Algorithms", style="white", width=40)
    table.add_column("Count", style="green", justify="center", width=10)
    table.add_column("Trials Each", style="yellow", justify="center", width=12)
    table.add_column("Total Trials", style="red", justify="center", width=12)

    total_algorithms = 0
    total_trials = 0

    # --- MODIFICATION START: Update matching logic ---
    models_lower = {m.lower() for m in models}
    for category, category_model_names in categorized_models.items():
        selected_models = [m for m in models_lower if m in category_model_names]

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
    # --- MODIFICATION END ---

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

# --- MODIFICATION START: Make analyze_training_results task-aware ---
def analyze_training_results(results, config):
    """Analyze and display detailed training results based on the task type."""

    console.rule("[bold green]📈 Training Results Analysis[/bold green]")

    if not results:
        console.print("[bold red]❌ No training results to analyze![/bold red]")
        return

    task_type = config.get('task_type', 'regression')
    is_regression = task_type == 'regression'

    # Determine primary metric for sorting and display
    if is_regression:
        primary_metric = 'test_r2'
        primary_metric_name = 'R² Score'
        sort_reverse = True
        default_sort_val = -999.0
    else:
        primary_metric = 'test_f1' # F1 is a good general metric for classification
        primary_metric_name = 'F1 Score (Test)'
        sort_reverse = True
        default_sort_val = -1.0

    # Create results summary table
    table = Table(title="🏆 Model Performance Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Algorithm", style="cyan", width=15)

    if is_regression:
        table.add_column("R² Score", style="green", justify="center", width=12)
        table.add_column("RMSE", style="yellow", justify="center", width=12)
        table.add_column("MAE", style="blue", justify="center", width=12)
    else:
        table.add_column("F1 Score", style="green", justify="center", width=12)
        table.add_column("Accuracy", style="yellow", justify="center", width=12)
        table.add_column("AUC", style="blue", justify="center", width=12)

    table.add_column("CV Score Std", style="red", justify="center", width=12)
    table.add_column("Status", style="white", justify="center", width=15)

    # Sort results by the primary metric
    sorted_results = sorted(results, key=lambda x: x.get(primary_metric, default_sort_val), reverse=sort_reverse)

    best_score = default_sort_val
    for i, result in enumerate(sorted_results):
        model_name = result['model_name'].upper()

        # Determine CV score standard deviation key
        cv_std_key = 'cv_std_r2' if is_regression else 'cv_std_f1'
        cv_std = result.get(cv_std_key, None)
        cv_std_str = f"{cv_std:.4f}" if cv_std is not None else "N/A"

        # Status determination
        current_score = result.get(primary_metric)
        if i == 0 and current_score is not None:
            status = "🥇 BEST"
            best_score = current_score
        elif current_score is not None and ((is_regression and current_score > 0) or (not is_regression and current_score > 0.5)):
            status = "✅ Good"
        elif current_score is not None:
            status = "⚠️ Fair"
        else:
            status = "❌ Poor"

        if is_regression:
            r2 = result.get('test_r2', None)
            rmse = result.get('test_rmse', None)
            mae = result.get('test_mae', None)
            r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
            rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
            mae_str = f"{mae:.4f}" if mae is not None else "N/A"
            table.add_row(model_name, r2_str, rmse_str, mae_str, cv_std_str, status)
        else: # Classification
            f1 = result.get('test_f1', None)
            acc = result.get('test_accuracy', None)
            auc = result.get('test_auc', None) # Assumes 'test_auc' is the key
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            table.add_row(model_name, f1_str, acc_str, auc_str, cv_std_str, status)

    console.print(table)

    # Performance insights
    console.print(f"\n[bold blue]📊 Performance Insights:[/bold blue]")

    if is_regression:
        good_performers = [r for r in results if r.get('test_r2', -999) > 0]
        poor_performers = [r for r in results if r.get('test_r2', -999) <= -0.5]
        console.print(f"   • Best R² Score: [bold green]{best_score:.4f}[/bold green]")
        console.print(f"   • Models with R² > 0: [bold]{len(good_performers)}/{len(results)}[/bold]")
        if poor_performers:
            console.print(f"   • Poor performers (R² ≤ -0.5): [bold red]{len(poor_performers)}[/bold red]")
            poor_names = [r['model_name'].upper() for r in poor_performers]
            console.print(f"     {', '.join(poor_names)}")
    else: # Classification
        good_performers = [r for r in results if r.get('test_f1', 0) > 0.5]
        console.print(f"   • Best F1 Score: [bold green]{best_score:.4f}[/bold green]")
        console.print(f"   • Models with F1 > 0.5: [bold]{len(good_performers)}/{len(results)}[/bold]")

    # Overfitting analysis (using primary metric)
    overfit_models = []
    for result in results:
        train_metric = result.get(f'train_{primary_metric.split("_")[-1]}', None)
        test_metric = result.get(primary_metric, None)
        if train_metric is not None and test_metric is not None:
            gap = train_metric - test_metric
            if (is_regression and gap > 0.3) or (not is_regression and gap > 0.2):
                overfit_models.append((result['model_name'], gap))

    if overfit_models:
        console.print(f"\n[bold yellow]⚠️  Potential Overfitting Detected:[/bold yellow]")
        for model_name, gap in overfit_models:
            console.print(f"   • {model_name.upper()}: Train-Test gap = {gap:.4f}")

    # Recommendations
    console.print(f"\n[bold cyan]💡 Recommendations:[/bold cyan]")

    if (is_regression and best_score < 0) or (not is_regression and best_score < 0.3):
        console.print(f"   • [red]All models show poor performance ({primary_metric_name} is low)[/red]")
        console.print("   • Consider: More data, feature engineering, different target encoding")
    elif (is_regression and best_score < 0.3) or (not is_regression and best_score < 0.6):
        console.print("   • [yellow]Moderate performance. Room for improvement.[/yellow]")
        console.print("   • Consider: Feature selection, ensemble methods, data augmentation")
    else:
        console.print("   • [green]Good performance achieved![/green]")
# --- MODIFICATION END ---


def main(config_path: str, dry_run: bool = False):
    """Main training function"""

    console.rule("[bold]🚀 CHEMIA Training-Only Pipeline[/bold]")

    # Handle config file path resolution
    if not os.path.isabs(config_path):
        # Try relative to current working directory first
        if os.path.exists(config_path):
            config_path = os.path.abspath(config_path)
        # If not found, try relative to project root
        elif os.path.exists(os.path.join(project_root, config_path)):
            config_path = os.path.join(project_root, config_path)
        # If still not found, try relative to script directory
        elif os.path.exists(os.path.join(os.path.dirname(__file__), config_path)):
            config_path = os.path.join(os.path.dirname(__file__), config_path)

    console.print(f"Loading training configuration from: [cyan]{config_path}[/cyan]")

    if not os.path.exists(config_path):
        console.print(f"[bold red]❌ Configuration file not found: {config_path}[/bold red]")
        console.print(f"[dim]Searched in:")
        console.print(f"  - Current directory: {os.getcwd()}")
        console.print(f"  - Project root: {project_root}")
        console.print(f"  - Script directory: {os.path.dirname(__file__)}[/dim]")
        sys.exit(1)

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
        description="Run comprehensive model training with CHEMIA framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training_only.py                          # Use default config
  python run_training_only.py --config my_config.yaml  # Use custom config
  python run_training_only.py my_config.yaml           # Use custom config (positional)
  python run_training_only.py --dry-run                # Preview without training
  python run_training_only.py my_config.yaml --dry-run # Preview with custom config
        """
    )

    parser.add_argument(
        'config_file',
        nargs='?',
        help="Path to the training configuration YAML file (can be relative or absolute)"
    )

    parser.add_argument(
        '--config',
        type=str,
        help="Path to the training configuration YAML file (alternative to positional argument)"
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Preview the training plan without actually running it"
    )

    args = parser.parse_args()

    # Determine config file path
    config_path = None
    if args.config_file:
        config_path = args.config_file
    elif args.config:
        config_path = args.config
    else:
        # Try default config in multiple locations
        default_configs = [
            "config_training_only.yaml",
            os.path.join(project_root, "config_training_only.yaml"),
            os.path.join(project_root, "examples", "configs", "config_training_only.yaml")
        ]

        for default_config in default_configs:
            if os.path.exists(default_config):
                config_path = default_config
                break

        if not config_path:
            console.print(f"[bold red]❌ No configuration file specified and no default found![/bold red]")
            console.print(f"[dim]Searched for defaults in:")
            for dc in default_configs:
                console.print(f"  - {dc}")
            console.print(f"[/dim]")
            console.print(f"[bold]Usage:[/bold] python {sys.argv[0]} <config_file> [--dry-run]")
            sys.exit(1)

    main(config_path, args.dry_run)
