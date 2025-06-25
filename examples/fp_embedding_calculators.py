# my_analysis_script.py
import pandas as pd
from utils.feature_generator import calculate_features_from_smiles

def run_my_analysis():
    """
    An example script demonstrating how to use the feature calculation API.
    """
    # 1. Define your list of SMILES strings
    my_molecules = [
        'CCO',                      # Ethanol
        'c1ccccc1',                 # Benzene
        'O=C(C)Oc1ccccc1C(=O)O',     # Aspirin
        'InvalidSMILES'             # An invalid SMILES to test robustness
    ]

    print("--- Starting feature calculation examples ---")

    # --- Example 1: Calculate MACCS keys (default parameters) ---
    print("\n1. Calculating MACCS keys...")
    maccs_df = calculate_features_from_smiles(
        smiles_list=my_molecules,
        feature_type="maccs"
    )
    if maccs_df is not None:
        print("MACCS keys calculated successfully!")
        print("Shape:", maccs_df.shape)
        # Display the features for Aspirin
        print("Aspirin's MACCS keys (first 10):")
        print(maccs_df.loc['O=C(C)Oc1ccccc1C(=O)O'].head(10))

    # --- Example 2: Calculate Morgan fingerprints with custom parameters ---
    print("\n2. Calculating Morgan fingerprints with nBits=512...")
    morgan_df = calculate_features_from_smiles(
        smiles_list=my_molecules,
        feature_type="morgan",
        nBits=512,  # Pass custom parameters as keyword arguments
        radius=2
    )
    if morgan_df is not None:
        print("Custom Morgan fingerprints calculated successfully!")
        print("Shape:", morgan_df.shape)

    # --- Example 3: Calculate RDKit 2D descriptors ---
    print("\n3. Calculating RDKit 2D descriptors...")
    descriptors_df = calculate_features_from_smiles(
        smiles_list=my_molecules,
        feature_type="rdkit_descriptors"
    )
    if descriptors_df is not None:
        print("RDKit descriptors calculated successfully!")
        print("Shape:", descriptors_df.shape)
        # Show some example descriptor values for Benzene
        print("Benzene's descriptors (MolWt, TPSA, MolLogP):")
        print(descriptors_df.loc['c1ccccc1'][['MolWt', 'TPSA', 'MolLogP']])

    # --- Example 4: Calculate Uni-Mol embeddings ---
    print("\n4. Calculating Uni-Mol embeddings...")
    unimol_df = calculate_features_from_smiles(
        smiles_list=my_molecules,
        feature_type="unimol",
        model_version="v2",
        model_size="84m",
        output_dir_for_logs="output/unimol_api_logs" # Specify where to save Uni-Mol's internal logs
    )
    if unimol_df is not None:
        print("Uni-Mol embeddings calculated successfully!")
        print("Shape:", unimol_df.shape)

    # You can now use these DataFrames for any downstream tasks like PCA, clustering, etc.

if __name__ == "__main__":
    run_my_analysis()