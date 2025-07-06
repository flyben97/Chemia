# utils/predictor.py

import os
import sys
import time
import contextlib
from datetime import datetime
import json
import argparse
import pandas as pd
from rich.console import Console
from rich.text import Text

# --- Path fix (ensures script can find core/utils modules) ---
current_script_path = os.path.abspath(__file__)
utils_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(utils_dir)
sys.path.insert(0, project_root)

from utils.io_handler import (
    load_config_from_path, log_prediction_summary, CHEMIA_BANNER,
    load_trained_model_from_run, load_model_from_path, load_scaler_from_path, load_label_encoder_from_path, 
    suppress_output, is_run_directory_path, get_full_model_name, find_model_file
)

from core.run_manager import process_dataframe

console = Console(width=120, highlight=False)

@contextlib.contextmanager
def suppress_output():
    """A context manager to temporarily suppress stdout and stderr."""
    original_stdout, original_stderr = sys.stdout, sys.stderr
    devnull = open(os.devnull, 'w')
    sys.stdout, sys.stderr = devnull, devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        devnull.close()

def predict(args):
    """Main prediction function with a minimalist UI."""
    prediction_start_time = time.time()
    
    # --- UI Enhancement: Final Minimalist Header ---
    console.print(Text(CHEMIA_BANNER, justify="center", style="bold blue"))
    console.print(f"[bold cyan]CHEMIA Predictor v1.5.7[/bold cyan]")
    console.print(f"[dim]Starting Prediction Pipeline...[/dim]")
    console.print("-" * 80)

    # --- 1. Load Artifacts ---
    console.print("\n[bold green]▶ Step 1/4: Loading Artifacts[/bold green]")
    try:
        config, model, scaler, label_encoder = None, None, None, None
        
        if args.run_dir:
            console.print(f"  [dim]Mode: Experiment Directory[/dim]")
            full_model_name = get_full_model_name(args.model_name)
            model_dir = os.path.join(args.run_dir, 'models', full_model_name)
            data_splits_dir = os.path.join(args.run_dir, 'data_splits')
            
            config = load_config_from_path(os.path.join(args.run_dir, "run_config.json"))
            task_type = config.get('task_type', 'regression')
            
            model_path_to_load = find_model_file(model_dir, full_model_name)
            model = load_model_from_path(model_path_to_load, task_type)
            scaler = load_scaler_from_path(os.path.join(data_splits_dir, "processed_dataset_scaler.joblib"))
            if task_type != 'regression':
                label_encoder = load_label_encoder_from_path(os.path.join(data_splits_dir, "processed_dataset_label_encoder.joblib"))
        
        else: # File Mode
            console.print(f"  [dim]Mode: Direct File Paths[/dim]")
            config = load_config_from_path(args.config_path)
            task_type = config.get('task_type', 'regression')
            
            model = load_model_from_path(args.model_path, task_type)
            scaler = load_scaler_from_path(args.scaler_path)
            if task_type != 'regression':
                 label_encoder = load_label_encoder_from_path(args.encoder_path)

    except Exception as e:
        console.print(f"[bold red]  ✗ Error loading artifacts: {e}[/bold red]"); return

    # --- 2. Process Input Data ---
    console.print("\n[bold green]▶ Step 2/4: Processing Input Data[/bold green]")
    df_new = pd.read_csv(args.input_file)
    console.print(f"  • Loaded input from: [cyan]{os.path.basename(args.input_file)}[/cyan] (Shape: {df_new.shape})")
    
    if config is None:
        console.print("[bold red]  ✗ Error: Configuration not loaded properly.[/bold red]")
        return
        
    common_cfg = config.get('data', {}).get('single_file_config', {})
    target_col = common_cfg.get('target_col', 'dummy_target')
    if target_col not in df_new.columns: df_new[target_col] = 0
    
    console.print("  • Generating features... ([dim]Use --verbose for details[/dim])")
    
    process_context = suppress_output() if not args.verbose else contextlib.nullcontext()
    with process_context:
        X_new, _, _, _ = process_dataframe(
            df=df_new, common_cfg=common_cfg, feature_gen_cfg=config.get('features', {}), output_dir="."
        )
    
    console.print(f"  • Final feature matrix shape: [bold cyan]{X_new.shape}[/bold cyan]")
    if scaler:
        console.print("  • Applying StandardScaler.")
        X_new = scaler.transform(X_new)

    # --- 3. Make Predictions ---
    console.print("\n[bold green]▶ Step 3/4: Making Predictions[/bold green]")
    if model is None:
        console.print("[bold red]  ✗ Error: Model not loaded properly.[/bold red]")
        return
        
    predictions = model.predict(X_new)
    console.print(f"  • Generated [bold magenta]{len(predictions)}[/bold magenta] predictions.")
    
    probabilities = None
    if (config and config.get('task_type') != 'regression' and 
        hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba'))):
        try:
            probabilities = model.predict_proba(X_new)  # type: ignore
            if probabilities is not None:
                console.print("  • Generated class probabilities.")
        except Exception as e:
            console.print(f"  [yellow]• Warning: Could not generate probabilities: {e}[/yellow]")

    # --- 4. Save Results & Log ---
    console.print("\n[bold green]▶ Step 4/4: Saving Results[/bold green]")
    output_df = df_new.head(len(predictions)).copy()
    
    if config and config.get('task_type') != 'regression' and label_encoder:
        output_df['prediction_label'] = label_encoder.inverse_transform(predictions)
        output_df['prediction_encoded'] = predictions
    else:
        output_df['prediction'] = predictions

    if probabilities is not None:
        if label_encoder and hasattr(label_encoder, 'classes_'):
            class_names = list(label_encoder.classes_)
        else:
            class_names = [f'class_{i}' for i in range(probabilities.shape[1])]
            
        for i, class_name in enumerate(class_names):
            output_df[f'proba_{class_name}'] = probabilities[:, i]

    final_output_path = args.output_file
    output_dir = os.path.dirname(final_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    output_df.to_csv(final_output_path, index=False)
    console.print(f"  • Predictions saved to: [cyan]{final_output_path}[/cyan]")

    prediction_duration = time.time() - prediction_start_time
    
    log_path = os.path.join(output_dir, f"prediction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_prediction_summary(
        log_path=log_path, run_dir=args.run_dir if args.run_dir else "N/A (File Mode)",
        model_name=get_full_model_name(args.model_name) if args.run_dir else os.path.basename(args.model_path),
        input_file=args.input_file, output_file=final_output_path, num_predictions=len(predictions),
        duration=prediction_duration, config=config, console=console
    )
    console.print(f"  • Prediction summary logged to: [dim]{log_path}[/dim]")
    
    console.print("-" * 80)
    console.print("[bold green]✓ Prediction Complete[/bold green]")
    console.print(f"  [bold]Output File:[/bold] [cyan]{final_output_path}[/cyan]")
    console.print(f"  [bold]Duration:[/bold]    [yellow]{prediction_duration:.2f}s[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="CHEMIA Predictor: Use a trained model on new data.", formatter_class=argparse.RawTextHelpFormatter)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--run_dir', type=str, help='(Experiment Mode) Path to a completed experiment run directory.')
    mode_group.add_argument('--model_path', type=str, help='(File Mode) Direct path to the model file (e.g., model.json).')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_file', type=str, required=True, help='Full path to save the output CSV file.')
    parser.add_argument('--verbose', action='store_true', help='Show detailed feature generation logs.')
    parser.add_argument('--model_name', type=str, help='(Experiment Mode) Name of the model (e.g., "xgb").')
    parser.add_argument('--config_path', type=str, help='(File Mode) Direct path to the run_config.json file.')
    parser.add_argument('--scaler_path', type=str, help='(File Mode, Optional) Direct path to the scaler.joblib file.')
    parser.add_argument('--encoder_path', type=str, help='(File Mode, Optional) Direct path to the label_encoder.joblib file.')
    
    args = parser.parse_args()

    if args.run_dir and not args.model_name: parser.error("--model_name is required when using --run_dir.")
    if args.model_path and not args.config_path: parser.error("--config_path is required when using --model_path.")

    predict(args)

if __name__ == "__main__":
    main()