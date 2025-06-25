# predict_with_saved_model.py
import pandas as pd
import joblib
from catboost import CatBoostRegressor, CatBoostClassifier
from rich.console import Console
from rich.panel import Panel

# Import the feature calculation API
from utils.feature_generator import calculate_features_from_smiles

def predict_from_file(
    input_csv_path: str,
    smiles_col: str,
    model_path: str,
    scaler_path: str = None, # Optional: Path to the saved StandardScaler
    output_csv_path: str = "output/predictions.csv"
):
    """
    Loads a trained model, calculates features for new SMILES data,
    and makes predictions.

    Args:
        input_csv_path (str): Path to the input CSV with SMILES.
        smiles_col (str): Name of the SMILES column.
        model_path (str): Path to the saved model file (.cbm, .json, or .joblib).
        scaler_path (str, optional): Path to the saved scaler.joblib file.
                                     Required if the model was trained on scaled data.
        output_csv_path (str): Path to save the predictions.
    """
    console = Console()
    console.print(Panel("CRAFT Inference Pipeline", style="bold magenta", expand=False))

    # --- 1. Load New SMILES Data ---
    try:
        console.print(f"Loading data from '[cyan]{input_csv_path}[/cyan]'...")
        df_new_data = pd.read_csv(input_csv_path)
        if smiles_col not in df_new_data.columns:
            console.print(f"[bold red]Error: SMILES column '{smiles_col}' not found.[/bold red]")
            return
        smiles_to_predict = df_new_data[smiles_col].dropna().tolist()
        console.print(f"Found {len(smiles_to_predict)} SMILES to predict.")
    except FileNotFoundError:
        console.print(f"[bold red]Error: Input file not found at '{input_csv_path}'.[/bold red]")
        return
    
    # --- 2. Calculate Features (Matching the Training Configuration) ---
    console.print("\nCalculating features exactly as configured during training...")
    
    # Part 1: Morgan fingerprints
    morgan_features_df = calculate_features_from_smiles(
        smiles_list=smiles_to_predict,
        feature_type="morgan",
        nBits=1024,
        radius=2
    )
    
    # Part 2: RDKit descriptors
    descriptor_features_df = calculate_features_from_smiles(
        smiles_list=smiles_to_predict,
        feature_type="rdkit_descriptors"
    )

    if morgan_features_df is None or descriptor_features_df is None:
        console.print("[bold red]Error during feature calculation. Aborting.[/bold red]")
        return
        
    # --- 3. Concatenate and Preprocess Features ---
    # IMPORTANT: The concatenation order MUST match the training order.
    # The framework internally concatenates in the order specified in the config.
    console.print("\nConcatenating features in the correct order (Morgan -> RDKit Descriptors)...")
    X_new_features = pd.concat([morgan_features_df, descriptor_features_df], axis=1)
    
    # Verify that the final feature count matches what the model expects.
    # This is a crucial sanity check.
    console.print(f"Final feature matrix shape: {X_new_features.shape}")

    # Apply scaling if a scaler is provided
    if scaler_path:
        try:
            console.print(f"Loading and applying scaler from '[cyan]{scaler_path}[/cyan]'...")
            scaler = joblib.load(scaler_path)
            X_new_scaled = scaler.transform(X_new_features)
        except FileNotFoundError:
            console.print(f"[bold red]Error: Scaler file not found at '{scaler_path}'.[/bold red]")
            console.print("[yellow]If the model was trained with scaling, this file is required.[/yellow]")
            return
    else:
        console.print("No scaler provided. Using unscaled features.")
        X_new_scaled = X_new_features.values
        
    # --- 4. Load the Trained Model ---
    console.print(f"\nLoading trained model from '[cyan]{model_path}[/cyan]'...")
    try:
        # Assuming a regression task as per the original config
        model = CatBoostRegressor()
        model.load_model(model_path)
        console.print("CatBoost model loaded successfully.")
    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        return

    # --- 5. Make Predictions ---
    console.print("Making predictions on new data...")
    predictions = model.predict(X_new_scaled)

    # --- 6. Save Results ---
    output_df = pd.DataFrame({
        'SMILES': smiles_to_predict,
        'Predicted_Value': predictions
    })

    try:
        # Create output directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        output_df.to_csv(output_csv_path, index=False)
        console.print(f"\n[bold green]✓ Success! Predictions saved to '[cyan]{output_csv_path}[/cyan]'[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error saving output file: {e}[/bold red]")


if __name__ == '__main__':
    # --- USER-CONFIGURABLE SECTION ---
    
    # 1. Path to your new data file
    INPUT_FILE = "data/test.csv" # Or any other CSV file with SMILES
    SMILES_COLUMN_NAME = "SMILES"

    # 2. Path to the folder of your saved experiment
    #    (Change 'Your_Exp_Name_xxxx' to your actual experiment folder name)
    EXPERIMENT_FOLDER = "output/Prediction_Demo_EN_20231027_100000" # <-- IMPORTANT: CHANGE THIS

    # 3. Construct paths to the model and scaler artifacts
    MODEL_FILE_PATH = f"{EXPERIMENT_FOLDER}/models/cat/cat_model.cbm"
    
    # Set SCALER_FILE_PATH to None if your model was trained without scaling
    SCALER_FILE_PATH = f"{EXPERIMENT_FOLDER}/data_splits/dataset_scaler.joblib"
    
    # 4. Define where to save the final predictions
    OUTPUT_FILE = f"{EXPERIMENT_FOLDER}/predictions_on_new_data.csv"
    
    # --- END OF CONFIGURABLE SECTION ---

    predict_from_file(
        input_csv_path=INPUT_FILE,
        smiles_col=SMILES_COLUMN_NAME,
        model_path=MODEL_FILE_PATH,
        scaler_path=SCALER_FILE_PATH,
        output_csv_path=OUTPUT_FILE
    )