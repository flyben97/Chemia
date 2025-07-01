# examples/predict_example.py

import subprocess
import os
import sys
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_prediction(run_dir: str, model_name: str, input_file: str, output_file_name: str, verbose: bool = False):
    """
    A Python wrapper to call the command-line predictor script with styled output.
    """
    command = [
        "python",
        "utils/predictor.py",
        "--run_dir", run_dir,
        "--model_name", model_name,
        "--input_file", input_file,
        "--output_file", output_file_name,
    ]
    if verbose:
        command.append("--verbose")

    console.print(Panel(f"[yellow]Executing Command:[/yellow]\n{' '.join(command)}", 
                        title="[bold blue]Prediction Runner[/bold blue]", 
                        border_style="blue"))
    
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set environment to force color output from the child process
        process_env = os.environ.copy()
        process_env["FORCE_COLOR"] = "1"
        
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            cwd=project_root,
            env=process_env
        )
        
        # --- FIX: Use low-level write to prevent double rendering ---
        if result.stdout:
            sys.stdout.write(result.stdout)
        
        if result.stderr:
            console.print(Panel(result.stderr.strip(), title="[bold red]Errors[/bold red]", border_style="red"))

    except subprocess.CalledProcessError as e:
        console.print(Panel(f"Prediction script failed with exit code {e.returncode}", 
                            title="[bold red]FATAL ERROR[/bold red]", 
                            border_style="red"))
        if e.stdout:
            sys.stdout.write(e.stdout)
        if e.stderr:
            sys.stderr.write(e.stderr)

# ... (create_dummy_input_file_with_features is unchanged) ...
def create_dummy_input_file_with_features(num_samples: int, num_precomputed_features: int) -> str:
    console.print("[dim]Creating a dummy input file...[/dim]")
    smiles_data = {'Reactant1': ['CCO', 'c1ccccc1'] * (num_samples // 2 + 1),'Reactant2': ['CC(=O)O', 'CNC(=O)N'] * (num_samples // 2 + 1),'Ligand': ['O=C(C)Oc1ccccc1C(=O)O', 'CCN(CC)CC'] * (num_samples // 2 + 1)}
    df_smiles = pd.DataFrame(smiles_data).head(num_samples)
    dummy_cols_before_features = {'id': [f'pred_{i+1}' for i in range(num_samples)],'rxn_id': [f"rxn_pred_{i}" for i in range(num_samples)],'catalyst': [f"cat_pred_{i}" for i in range(num_samples)],'solvent': [f"sol_pred_{i}" for i in range(num_samples)]}
    df_before = pd.DataFrame(dummy_cols_before_features)
    feature_matrix = np.random.rand(num_samples, num_precomputed_features)
    feature_names = [f'precomp_feat_{i+1}' for i in range(num_precomputed_features)]
    df_features = pd.DataFrame(feature_matrix, columns=feature_names)
    df_final = pd.concat([df_before, df_features, df_smiles], axis=1)
    os.makedirs("data", exist_ok=True)
    dummy_path = "data/dummy_prediction_input_with_features.csv"
    df_final.to_csv(dummy_path, index=False)
    console.print(f"[dim]Dummy input file created at: '{dummy_path}'[/dim]")
    return dummy_path

if __name__ == "__main__":
    COMPLETED_RUN_DIR = "output/S17_MultiColumn_Demo_regression_20250626_165527"
    if not os.path.isdir(COMPLETED_RUN_DIR):
        console.print(Panel("...Error message...", title="[bold red]Configuration Error[/bold red]", border_style="red"))
    else:
        dummy_input_path = create_dummy_input_file_with_features(num_samples=2, num_precomputed_features=35)
        MODEL_TO_USE = "xgboost"
        OUTPUT_FILENAME = f"predictions_from_{MODEL_TO_USE}.csv"
        run_prediction(
            run_dir=COMPLETED_RUN_DIR,
            model_name=MODEL_TO_USE,
            input_file=dummy_input_path,
            output_file_name=OUTPUT_FILENAME,
            verbose=False # Set to True to see detailed feature generation logs
        )