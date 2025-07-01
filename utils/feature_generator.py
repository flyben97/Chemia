# utils/feature_generator.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import track

# Internal console for logging within the module
_console = Console()

# --- Internal Helper Functions (prefixed with _) ---
# These functions contain the core logic but are not meant for direct user interaction.

def _get_rdkit_features(smiles_list: List[str], config: Dict[str, Any]) -> pd.DataFrame:
    """Helper to calculate RDKit fingerprints/descriptors."""
    from .mol_fp_features import calculate_molecular_features

    fp_type = config.get('type')
    descriptors = "all" if config.get('descriptors', False) else False
    radius = config.get('radius', 2)
    nBits = config.get('nBits', 2048)

    desc_name_parts = []
    if fp_type:
        desc_name_parts.append(f"{str(fp_type).upper()} Fingerprints")
    if descriptors:
        desc_name_parts.append("RDKit Descriptors")
    desc_name = " & ".join(desc_name_parts)

    # Create appropriate log message based on fingerprint type
    if fp_type == 'maccs':
        _console.log(f"Calculating RDKit features: type='{fp_type}', descriptors={descriptors} (MACCS: fixed 166 bits)...")
    elif fp_type == 'morgan':
        _console.log(f"Calculating RDKit features: type='{fp_type}', descriptors={descriptors}, nBits={nBits}, radius={radius}...")
    elif fp_type in ['rdkit', 'atompair', 'torsion']:
        _console.log(f"Calculating RDKit features: type='{fp_type}', descriptors={descriptors}, nBits={nBits}...")
    elif fp_type is None and descriptors:
        _console.log(f"Calculating RDKit descriptors only...")
    else:
        _console.log(f"Calculating RDKit features: type='{fp_type}', descriptors={descriptors}...")

    all_features_rows = []
    feature_columns = None
    feature_length = 0
    failed_smiles = []

    # Determine feature columns and length from the first valid SMILES
    for smiles in smiles_list:
        if pd.notna(smiles) and smiles:
            first_valid_df = calculate_molecular_features(smiles, fp_type=fp_type, descriptors=descriptors, radius=radius, nBits=nBits)  # type: ignore
            if first_valid_df is not None:
                feature_columns = first_valid_df.columns
                feature_length = len(feature_columns)
                break
    
    if feature_columns is None:
        _console.log(f"[bold red]Error:[/bold red] Could not determine feature columns for {desc_name}. All SMILES may be invalid. Skipping this feature set.")
        return pd.DataFrame(index=smiles_list)

    zero_row = [0] * feature_length
    
    for smiles in track(smiles_list, description=f"Processing {desc_name}..."):
        if pd.isna(smiles) or not smiles:
            all_features_rows.append(zero_row)
            failed_smiles.append(str(smiles)) 
            continue
        
        features_df = calculate_molecular_features(smiles, fp_type=fp_type, descriptors=descriptors, radius=radius, nBits=nBits)  # type: ignore
        
        if features_df is None or features_df.shape[1] != feature_length:
            all_features_rows.append(zero_row)
            failed_smiles.append(smiles)
        else:
            all_features_rows.append(features_df.iloc[0].tolist())
    
    if failed_smiles:
        _console.log(f"[yellow]Info ({desc_name}):[/yellow] Calculation failed for {len(failed_smiles)} molecules. Their features have been filled with zeros.")

    return pd.DataFrame(all_features_rows, index=smiles_list, columns=feature_columns)


def _get_embedding_features(smiles_list: List[str], config: Dict[str, Any], embedding_func, output_dir: Optional[str] = None) -> pd.DataFrame:
    """Helper to calculate embeddings."""
    feature_type = config.get('type')
    _console.log(f"Calculating embeddings for '{feature_type}'...")

    embedding_func_args = {k: v for k, v in config.items() if k != 'type'}
    
    if feature_type == 'unimol' and output_dir:
        embedding_func_args['log_dir'] = output_dir

    embeddings = embedding_func(smiles_list, **embedding_func_args)
    
    if embeddings is None:
        _console.log(f"[bold red]Fatal error:[/bold red] Cannot calculate embeddings for {feature_type}. Skipping.")
        return pd.DataFrame(index=smiles_list)

    nan_rows_mask = np.isnan(embeddings).all(axis=1)
    if np.any(nan_rows_mask):
        num_failed = np.sum(nan_rows_mask)
        _console.log(f"[yellow]Info ({feature_type}):[/yellow] The underlying library failed to generate embeddings for {num_failed} molecules. Their features have been filled with zeros.")
        embeddings[nan_rows_mask] = 0

    columns = [f"{feature_type}_{i}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, index=smiles_list, columns=columns)

# This function remains for internal use by the main training pipeline
def generate_features(smiles_list: List[str], feature_configs: List[Dict[str, Any]], output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Internal function to generate a combined feature DataFrame based on a list of configs.
    This is used by the main CRAFT pipeline.
    """
    # This function body remains the same as your current version
    from .transformer_embeddings import get_chemberta_embedding, get_molt5_embedding, get_chemroberta_embedding
    from .unimol_embedding import get_unimol_embedding

    if not isinstance(smiles_list, list):
        smiles_list = list(smiles_list)
        
    feature_dispatch = {
        'chemberta': lambda sm, cfg, out_dir: _get_embedding_features(sm, cfg, get_chemberta_embedding, out_dir),
        'molt5': lambda sm, cfg, out_dir: _get_embedding_features(sm, cfg, get_molt5_embedding, out_dir),
        'chemroberta': lambda sm, cfg, out_dir: _get_embedding_features(sm, cfg, get_chemroberta_embedding, out_dir),
        'unimol': lambda sm, cfg, out_dir: _get_embedding_features(sm, cfg, get_unimol_embedding, out_dir),
    }

    rdkit_fp_types = ["maccs", "morgan", "rdkit", "atompair", "torsion"]
    all_feature_dfs = []

    for config in feature_configs:
        feature_type = config.get('type')
        if not feature_type:
            _console.log("[yellow]Warning: Skipping a config because it lacks a 'type' key.[/yellow]")
            continue
        
        df = None
        if feature_type in rdkit_fp_types:
            rdkit_config = config.copy()
            rdkit_config['descriptors'] = False # Ensure only FP is calculated
            df = _get_rdkit_features(smiles_list, rdkit_config)
        elif feature_type == 'rdkit_descriptors':
            # This type implies only descriptors are needed
            rdkit_config = {'type': None, 'descriptors': True}
            df = _get_rdkit_features(smiles_list, rdkit_config)
        elif feature_type in feature_dispatch:
            handler = feature_dispatch[feature_type]
            df = handler(smiles_list, config, output_dir)
        else:
            _console.log(f"[red]Error: Unknown feature type '{feature_type}'. Skipping.[/red]")
            continue

        if df is not None and not df.empty:
            all_feature_dfs.append(df)

    if not all_feature_dfs:
        _console.log("[red]Error: No features were generated. Returning empty DataFrame.[/red]")
        return pd.DataFrame()

    import logging
    logging.info("Concatenating all feature sets...")
    
    final_df = pd.concat(all_feature_dfs, axis=1)

    logging.info(f"Generated final feature matrix with shape: {final_df.shape}")
    return final_df


# --- NEW Public API Function ---
# This is the new, user-friendly function for external scripts.
def calculate_features_from_smiles(
    smiles_list: List[str],
    feature_type: str,
    output_dir_for_logs: str = './temp_feature_logs',
    **kwargs
) -> Optional[pd.DataFrame]:
    """
    Calculates and returns a single type of molecular feature for a list of SMILES.

    This is the primary API for using feature calculation in external scripts.

    Args:
        smiles_list (List[str]): A list of SMILES strings.
        feature_type (str): The type of feature to calculate.
            Available options:
            - Fingerprints: "maccs", "morgan", "rdkit", "atompair", "torsion"
            - Descriptors: "rdkit_descriptors"
            - Embeddings: "chemberta", "molt5", "chemroberta", "unimol"
        output_dir_for_logs (str): Directory to save logs, especially for Uni-Mol.
                                   Defaults to './temp_feature_logs'.
        **kwargs: Additional parameters specific to the feature type.
            - For "morgan", "rdkit", "atompair", "torsion": nBits (int), radius (int, for Morgan only)
            - For "unimol": model_version (str), model_size (str)
            - For "chemberta", "molt5", etc.: model_name (str) to use a different checkpoint.

    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame where the index is the SMILES string
                                and columns are the calculated features. Returns None if the
                                feature type is invalid.
    """
    config = {'type': feature_type, **kwargs}
    
    # We call the internal 'generate_features' function, which is already robust.
    # We wrap the config in a list because that's what generate_features expects.
    
    # Ensure the directory for logs exists.
    import os
    os.makedirs(output_dir_for_logs, exist_ok=True)
    
    # Reduce console output for internal use
    import logging
    logging.info(f"Calculating features of type: '{feature_type}'")
    
    features_df = generate_features(smiles_list, [config], output_dir=output_dir_for_logs)
    
    if features_df.empty:
        logging.error(f"Calculation failed for feature type '{feature_type}'")
        return None
        
    return features_df
