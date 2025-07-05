#!/usr/bin/env python3
"""
CHEMIA Quick Prediction Script

This is a simplified prediction script that provides the most basic model prediction functionality.
Suitable for quick prediction tasks without complex configuration.

Usage example:
    python quick_predict.py /path/to/experiment_dir xgb input.csv output.csv
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from run_prediction_standalone import PredictionRunner, console

def quick_predict(experiment_dir: str, model_name: str, input_file: str, output_file: str, verbose: bool = False):
    """
    Quick prediction function
    
    Args:
        experiment_dir: Training experiment directory path
        model_name: Model name (e.g., xgb, lgbm, catboost)
        input_file: Input CSV file path
        output_file: Output CSV file path
        verbose: Whether to show detailed information
    """
    
    console.print(f"[bold blue]🚀 CHEMIA Quick Prediction[/bold blue]")
    console.print(f"[cyan]Experiment directory:[/cyan] {experiment_dir}")
    console.print(f"[cyan]Model name:[/cyan] {model_name}")
    console.print(f"[cyan]Input file:[/cyan] {input_file}")
    console.print(f"[cyan]Output file:[/cyan] {output_file}")
    console.print("-" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        console.print(f"[bold red]❌ Input file does not exist:[/bold red] {input_file}")
        return False
    
    # Check if experiment directory exists
    if not os.path.exists(experiment_dir):
        console.print(f"[bold red]❌ Experiment directory does not exist:[/bold red] {experiment_dir}")
        return False
    
    # Create configuration
    config = {
        'prediction_mode': 'experiment_directory',
        'experiment_directory_mode': {
            'run_directory': experiment_dir,
            'model_name': model_name
        },
        'data': {
            'input_file': input_file,
            'output_file': output_file
        },
        'logging': {
            'verbose': verbose
        },
        'prediction': {
            'save_probabilities': True,
            'output_format': {
                'include_input_data': True,
                'add_prediction_metadata': True,
                'precision': 4
            }
        },
        'advanced': {
            'memory_efficient': True,
            'skip_invalid_rows': True
        }
    }
    
    # Create prediction runner and execute
    runner = PredictionRunner(config)
    success = runner.run_prediction_pipeline()
    
    if success:
        console.print(f"\n[bold green]✅ Prediction completed![/bold green]")
        console.print(f"[green]Results saved to:[/green] {output_file}")
        
        # Show simple result statistics
        try:
            import pandas as pd
            result_df = pd.read_csv(output_file)
            console.print(f"[cyan]Number of predicted samples:[/cyan] {len(result_df)}")
            
            # For regression tasks, show prediction value range
            if 'prediction' in result_df.columns:
                pred_min = result_df['prediction'].min()
                pred_max = result_df['prediction'].max()
                pred_mean = result_df['prediction'].mean()
                console.print(f"[cyan]Prediction range:[/cyan] {pred_min:.4f} ~ {pred_max:.4f} (mean: {pred_mean:.4f})")
            
            # For classification tasks, show class distribution
            elif 'prediction_label' in result_df.columns:
                class_counts = result_df['prediction_label'].value_counts()
                console.print(f"[cyan]Class distribution:[/cyan]")
                for class_name, count in class_counts.items():
                    console.print(f"  - {class_name}: {count} samples")
                    
        except Exception as e:
            console.print(f"[yellow]Note: Unable to read result statistics: {e}[/yellow]")
    else:
        console.print(f"\n[bold red]❌ Prediction failed![/bold red]")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="CHEMIA Quick Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Basic usage
  python quick_predict.py output/my_experiment xgb input.csv output.csv
  
  # Show detailed information
  python quick_predict.py output/my_experiment xgb input.csv output.csv --verbose
  
  # Use other models
  python quick_predict.py output/my_experiment lgbm input.csv output.csv
  python quick_predict.py output/my_experiment catboost input.csv output.csv

Supported model names:
  - xgb (XGBoost)
  - lgbm (LightGBM) 
  - catboost (CatBoost)
  - rf (Random Forest)
  - ann (Artificial Neural Network)
  - and other models used during training
        """
    )
    
    parser.add_argument('experiment_dir', type=str, 
                       help='Training experiment directory path')
    parser.add_argument('model_name', type=str,
                       help='Model name (e.g., xgb, lgbm, catboost)')
    parser.add_argument('input_file', type=str,
                       help='Input CSV file path')
    parser.add_argument('output_file', type=str,
                       help='Output CSV file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information')
    
    args = parser.parse_args()
    
    # Execute prediction
    success = quick_predict(
        experiment_dir=args.experiment_dir,
        model_name=args.model_name,
        input_file=args.input_file,
        output_file=args.output_file,
        verbose=args.verbose
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 