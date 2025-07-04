#!/usr/bin/env python3
"""
Demo script to show the new original data splits feature.

This script demonstrates how to use CRAFT's new feature that saves
the original data as separate CSV files based on train/val/test splits.

Usage:
    python demo_original_splits.py
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.run_manager import start_experiment_run
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(width=120)

def demo_original_splits():
    """
    Demonstrates the original data splits feature.
    """
    console.rule("[bold]CRAFT Original Data Splits Demo[/bold]")
    
    # Create a simple demo configuration
    demo_config = {
        'experiment_name': 'demo_original_splits',
        'task_type': 'regression',
        'data': {
            'source_mode': 'single_file',
            'single_file_config': {
                'main_file_path': 'data/agent5.csv',  # Change this to your data file
                'smiles_col': ['Reactant1', 'Reactant2', 'Ligand'],
                'target_col': 'ee',
                'precomputed_features': {
                    'feature_columns': '5:'
                }
            }
        },
        'training': {
            'models_to_run': ['xgb'],  # Just one model for demo
            'n_trials': 2,  # Quick demo
            'quiet_optuna': True
        },
        'features': {
            'per_smiles_col_generators': {
                'Reactant1': [{'type': 'maccs'}],
                'Reactant2': [{'type': 'maccs'}],
                'Ligand': [{'type': 'rdkit_descriptors'}]
            },
            'scaling': True
        },
        'split_mode': 'cross_validation',
        'split_config': {
            'cross_validation': {
                'n_splits': 3,
                'test_size_for_cv': 0.2,
                'random_state': 42
            }
        }
    }
    
    console.print(Panel.fit(
        "[bold cyan]This demo will:[/bold cyan]\n"
        "1. Load your original data file\n"
        "2. Split it into train/test sets\n"
        "3. Save the original data splits as separate CSV files\n"
        "4. Show you where to find them",
        title="Demo Overview"
    ))
    
    # Check if data file exists
    data_file = demo_config['data']['single_file_config']['main_file_path']
    if not os.path.exists(data_file):
        console.print(f"[red]Error:[/red] Data file not found: {data_file}")
        console.print("Please make sure your data file exists or update the path in this demo script.")
        return
    
    console.print(f"[green]✓[/green] Found data file: {data_file}")
    
    # Quick peek at the original data
    try:
        df_original = pd.read_csv(data_file)
        console.print(f"[green]✓[/green] Original data shape: {df_original.shape}")
        
        # Show first few rows
        table = Table(title="Original Data Sample (first 3 rows)")
        for col in df_original.columns[:6]:  # Show first 6 columns
            table.add_column(col, style="cyan")
        
        for i in range(min(3, len(df_original))):
            row_data = [str(df_original.iloc[i, j])[:20] + "..." if len(str(df_original.iloc[i, j])) > 20 
                       else str(df_original.iloc[i, j]) for j in range(min(6, len(df_original.columns)))]
            table.add_row(*row_data)
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error reading data file:[/red] {e}")
        return
    
    # Run the training (this will also save original splits)
    console.print("\n[bold yellow]Starting training with original data splits...[/bold yellow]")
    
    try:
        result = start_experiment_run(demo_config)
        
        if result and 'run_directory' in result:
            run_dir = result['run_directory']
            splits_dir = os.path.join(run_dir, 'original_data_splits')
            
            console.rule("[bold green]✓ Demo Complete![/bold green]")
            
            console.print(f"\n[bold cyan]Original data splits saved to:[/bold cyan]")
            console.print(f"  📁 {splits_dir}")
            
            # Show what files were created
            if os.path.exists(splits_dir):
                files = os.listdir(splits_dir)
                for file in sorted(files):
                    file_path = os.path.join(splits_dir, file)
                    if file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        console.print(f"  📄 [green]{file}[/green] - {len(df)} samples")
            
            console.print(f"\n[bold yellow]You can now inspect:[/bold yellow]")
            console.print(f"• Training data: [cyan]{splits_dir}/train_original_data.csv[/cyan]")
            console.print(f"• Test data: [cyan]{splits_dir}/test_original_data.csv[/cyan]")
            console.print(f"• Split summary: [cyan]{splits_dir}/data_split_summary.csv[/cyan]")
            
        else:
            console.print("[red]Training failed or returned no results.[/red]")
    
    except Exception as e:
        console.print(f"[red]Error during training:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    demo_original_splits() 