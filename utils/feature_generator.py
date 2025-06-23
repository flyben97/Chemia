# utils/feature_generator.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import track

# ... (_get_rdkit_features 函数保持不变) ...
def _get_rdkit_features(smiles_list: List[str], config: Dict[str, Any]) -> pd.DataFrame:
    """Helper to calculate RDKit fingerprints/descriptors, explicitly logging and filling failures with zeros."""
    from .mol_fp_features import calculate_molecular_features as calculate_rdkit_features_single

    fp_type = config.get('type')
    descriptors = "all" if config.get('descriptors', False) else False
    radius = config.get('radius', 2)
    nBits = config.get('nBits', 2048)

    desc_name = "RDKit Descriptors" if descriptors and not fp_type else f"{str(fp_type).upper()} Fingerprints"
    console.log(f"Calculating RDKit features: type='{fp_type}', descriptors={descriptors}...")

    all_features_rows = []
    feature_columns = None
    feature_length = 0
    failed_smiles = [] 

    for smiles in smiles_list:
        if pd.notna(smiles) and smiles:
            first_valid_df = calculate_rdkit_features_single(smiles, fp_type, descriptors, radius, nBits)
            if first_valid_df is not None:
                feature_columns = first_valid_df.columns
                feature_length = len(feature_columns)
                break
    
    if feature_columns is None:
        console.log(f"[red]Could not determine feature columns for {desc_name}. Skipping this feature set.[/red]")
        return pd.DataFrame(index=smiles_list)

    zero_row = [0] * feature_length
    
    for smiles in track(smiles_list, description=f"Processing {desc_name}..."):
        if pd.isna(smiles) or not smiles:
            all_features_rows.append(zero_row)
            failed_smiles.append(str(smiles)) 
            continue
        
        features_df = calculate_rdkit_features_single(smiles, fp_type, descriptors, radius, nBits)
        
        if features_df is None or features_df.shape[1] != feature_length:
            all_features_rows.append(zero_row)
            failed_smiles.append(smiles)
        else:
            all_features_rows.append(features_df.iloc[0].tolist())
    
    if failed_smiles:
        console.log(f"[bold yellow]Info ({desc_name}):[/bold yellow] 计算失败 {len(failed_smiles)} 个分子，其特征已用 0 填充。")
        for i, s in enumerate(failed_smiles[:5]):
            console.log(f"  - 失败样本 (示例 {i+1}): {s}")
        if len(failed_smiles) > 5:
            console.log(f"  - ...以及其他 {len(failed_smiles) - 5} 个。")

    return pd.DataFrame(all_features_rows, index=smiles_list, columns=feature_columns)


console = Console()

def _get_embedding_features(smiles_list: List[str], config: Dict[str, Any], embedding_func, output_dir: Optional[str] = None) -> pd.DataFrame:
    """Helper to calculate embeddings, explicitly logging and filling failures with zeros."""
    feature_type = config.get('type')
    console.log(f"Calculating embeddings for '{feature_type}'...")

    embedding_func_args = {k: v for k, v in config.items() if k != 'type'}
    
    # 如果是 unimol，传递 log_dir
    if feature_type == 'unimol' and output_dir:
        embedding_func_args['log_dir'] = output_dir

    embeddings = embedding_func(smiles_list, **embedding_func_args)
    
    if embeddings is None:
        console.log(f"[red]Fatal error calculating embeddings for {feature_type}. Cannot determine dimension. Skipping.[/red]")
        return pd.DataFrame(index=smiles_list)

    nan_rows_mask = np.isnan(embeddings).all(axis=1)
    if np.any(nan_rows_mask):
        num_failed = np.sum(nan_rows_mask)
        failed_indices = np.where(nan_rows_mask)[0]
        
        console.log(f"[bold yellow]Info ({feature_type}):[/bold yellow] 底层库报告 {num_failed} 个分子生成失败，其特征已用 0 填充。")
        for i, idx in enumerate(failed_indices[:5]):
             console.log(f"  - 失败样本 (索引 {idx}): {smiles_list[idx]}")
        if num_failed > 5:
            console.log(f"  - ...以及其他 {num_failed - 5} 个。")
            
        embeddings[nan_rows_mask] = 0

    columns = [f"{feature_type}_{i}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, index=smiles_list, columns=columns)

# --- 修改 generate_features 函数签名 ---
def generate_features(smiles_list: List[str], feature_configs: List[Dict[str, Any]], output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Generates a combined feature DataFrame, filling any calculation failures with zeros.
    """
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
            console.log("[yellow]Warning: Skipping a config because it lacks a 'type' key.[/yellow]")
            continue
        
        df = None
        if feature_type in rdkit_fp_types:
            rdkit_config = config.copy()
            rdkit_config['descriptors'] = False
            df = _get_rdkit_features(smiles_list, rdkit_config)
        elif feature_type == 'rdkit_descriptors':
            rdkit_config = {'type': None, 'descriptors': True}
            df = _get_rdkit_features(smiles_list, rdkit_config)
        elif feature_type in feature_dispatch:
            handler = feature_dispatch[feature_type]
            # --- 传递 output_dir ---
            df = handler(smiles_list, config, output_dir)
        else:
            console.log(f"[red]Error: Unknown feature type '{feature_type}'. Skipping.[/red]")
            continue

        if df is not None and not df.empty:
            all_feature_dfs.append(df)

    if not all_feature_dfs:
        console.log("[red]Error: No features were generated. Returning empty DataFrame.[/red]")
        return pd.DataFrame()

    console.log("Concatenating all feature sets...")
    
    final_df = pd.concat(all_feature_dfs, axis=1)

    console.log(f"Generated final feature matrix with shape: {final_df.shape}")
    return final_df