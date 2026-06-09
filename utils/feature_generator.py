# utils/feature_generator.py
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import track

# Explicitly suppress numba debug output when using UniMol
from .suppress_logs import suppress_debug_logs
suppress_debug_logs()

# Internal console for logging within the module
_console = Console()

# --- Internal Helper Functions (prefixed with _) ---
# These functions contain the core logic but are not meant for direct user interaction.

def _get_rdkit_features(smiles_list: List[str], config: Dict[str, Any]) -> pd.DataFrame:
    """Helper to calculate RDKit fingerprints/descriptors with optional parallel processing."""
    from .mol_fp_features import calculate_molecular_features

    fp_type = config.get('type')
    descriptors = "all" if config.get('descriptors', False) else False
    radius = config.get('radius', 2)
    nBits = config.get('nBits', 2048)
    use_parallel = config.get('parallel', True)  # Enable parallel by default
    n_jobs = config.get('n_jobs', None)
    batch_size = config.get('batch_size', None)

    desc_name_parts = []
    if fp_type:
        desc_name_parts.append(f"{str(fp_type).upper()} Fingerprints")
    if descriptors:
        desc_name_parts.append("RDKit Descriptors")
    desc_name = " & ".join(desc_name_parts)

    # Create appropriate log message based on fingerprint type
    parallel_status = " (Parallel)" if use_parallel and not descriptors else " (Sequential)"
    if fp_type == 'maccs':
        _console.log(f"Calculating RDKit features: type='{fp_type}', descriptors={descriptors} (MACCS: fixed 167 bits){parallel_status}...")
    elif fp_type == 'morgan':
        _console.log(f"Calculating RDKit features: type='{fp_type}', descriptors={descriptors}, nBits={nBits}, radius={radius}{parallel_status}...")
    elif fp_type in ['rdkit', 'atompair', 'torsion']:
        _console.log(f"Calculating RDKit features: type='{fp_type}', descriptors={descriptors}, nBits={nBits}{parallel_status}...")
    elif fp_type is None and descriptors:
        _console.log(f"Calculating RDKit descriptors only...")
    else:
        _console.log(f"Calculating RDKit features: type='{fp_type}', descriptors={descriptors}{parallel_status}...")

    # Use parallel processing for fingerprints only (not descriptors) and when enabled
    if use_parallel and fp_type and not descriptors and len(smiles_list) > 10:
        try:
            from .parallel_fingerprint import calculate_fingerprints_parallel

            # Clean SMILES list (remove None/NaN values)
            clean_smiles = []
            smiles_mapping = {}  # Map clean index to original index

            for i, smiles in enumerate(smiles_list):
                if pd.notna(smiles) and smiles:
                    clean_smiles.append(smiles)
                    smiles_mapping[len(clean_smiles) - 1] = i

            if clean_smiles:
                _console.log(f"[cyan]Using parallel processing for {len(clean_smiles)} valid SMILES...[/cyan]")
                parallel_df = calculate_fingerprints_parallel(
                    clean_smiles,
                    fp_type,
                    n_jobs=n_jobs,
                    batch_size=batch_size,
                    radius=radius,
                    nBits=nBits
                )

                if not parallel_df.empty:
                    # Create full DataFrame with original SMILES order
                    feature_columns = parallel_df.columns
                    zero_row = [0] * len(feature_columns)
                    all_features_rows = []

                    clean_idx = 0
                    for i, smiles in enumerate(smiles_list):
                        if pd.notna(smiles) and smiles and clean_idx < len(parallel_df):
                            all_features_rows.append(parallel_df.iloc[clean_idx].tolist())
                            clean_idx += 1
                        else:
                            all_features_rows.append(zero_row)

                    return pd.DataFrame(all_features_rows, index=pd.Index(smiles_list), columns=pd.Index(feature_columns))

        except ImportError:
            _console.log("[yellow]Warning: Parallel fingerprint module not available. Falling back to sequential processing.[/yellow]")
        except Exception as e:
            _console.log(f"[yellow]Warning: Parallel processing failed ({str(e)}). Falling back to sequential processing.[/yellow]")

    # Fallback to sequential processing
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
        return pd.DataFrame(index=pd.Index(smiles_list))

    zero_row = [0] * feature_length
    failed_smiles_info = []  # 记录失败的 SMILES 和对应索引，便于排查

    for idx, smiles in enumerate(track(smiles_list, description=f"Processing {desc_name}...")):
        if pd.isna(smiles) or not smiles:
            all_features_rows.append(zero_row)
            failed_smiles_info.append((idx, str(smiles) if pd.notna(smiles) else "NaN/Empty"))
            continue

        features_df = calculate_molecular_features(smiles, fp_type=fp_type, descriptors=descriptors, radius=radius, nBits=nBits)  # type: ignore

        if features_df is None or features_df.shape[1] != feature_length:
            all_features_rows.append(zero_row)
            failed_smiles_info.append((idx, smiles))
        else:
            all_features_rows.append(features_df.iloc[0].tolist())

    if failed_smiles_info:
        failed_count = len(failed_smiles_info)
        _console.log(f"[yellow]Warning ({desc_name}):[/yellow] Calculation failed for {failed_count} molecules. Their features have been filled with zeros.")
        if failed_count <= 10:
            for idx, sm in failed_smiles_info:
                _console.log(f"  [dim]- Index {idx}: {sm}[/dim]")
        else:
            for idx, sm in failed_smiles_info[:5]:
                _console.log(f"  [dim]- Index {idx}: {sm}[/dim]")
            _console.log(f"  [dim]... and {failed_count - 5} more[/dim]")

    return pd.DataFrame(all_features_rows, index=pd.Index(smiles_list), columns=pd.Index(feature_columns))


def _get_embedding_features(smiles_list: List[str], config: Dict[str, Any], embedding_func, output_dir: Optional[str] = None) -> pd.DataFrame:
    """Helper to calculate embeddings, with optional disk caching."""
    from .embedding_cache import get_cached, put_cache, build_model_id, init_cache_dir

    feature_type = config.get('type')
    model_id = build_model_id(feature_type, config)

    # Optional cache: try load from cache first
    cache_enabled = not os.environ.get('CHEMIA_NO_CACHE')
    cache_dir = None
    if cache_enabled:
        try:
            cache_dir = init_cache_dir()
            cached = get_cached(smiles_list, model_id, cache_dir)
            if cached is not None:
                _console.log(f"[dim]Embedding cache hit for '{feature_type}' ({len(smiles_list)} molecules)[/dim]")
                nan_rows_mask = np.isnan(cached).all(axis=1)
                if np.any(nan_rows_mask):
                    cached[nan_rows_mask] = 0
                columns = [f"{feature_type}_{i}" for i in range(cached.shape[1])]
                return pd.DataFrame(cached, index=pd.Index(smiles_list), columns=pd.Index(columns))
        except Exception:
            cache_dir = None  # Cache failure is non-fatal

    _console.log(f"Calculating embeddings for '{feature_type}'...")

    embedding_func_args = {k: v for k, v in config.items() if k not in ('type', 'model_type')}

    if feature_type == 'unimol' and output_dir:
        unimol_log_dir = os.path.join(output_dir, 'unimol_logs')
        embedding_func_args['log_dir'] = unimol_log_dir

    embeddings = embedding_func(smiles_list, **embedding_func_args)

    if embeddings is None:
        _console.log(f"[bold red]Fatal error:[/bold red] Cannot calculate embeddings for {feature_type}. Skipping.")
        return pd.DataFrame(index=pd.Index(smiles_list))

    # Cache miss: store result
    if cache_dir is not None:
        try:
            put_cache(smiles_list, embeddings, model_id, cache_dir)
        except Exception:
            pass

    nan_rows_mask = np.isnan(embeddings).all(axis=1)
    if np.any(nan_rows_mask):
        num_failed = np.sum(nan_rows_mask)
        _console.log(f"[yellow]Info ({feature_type}):[/yellow] The underlying library failed to generate embeddings for {num_failed} molecules. Their features have been filled with zeros.")
        embeddings[nan_rows_mask] = 0

    columns = [f"{feature_type}_{i}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, index=pd.Index(smiles_list), columns=pd.Index(columns))

# This function remains for internal use by the main training pipeline
def generate_features(smiles_list: List[str], feature_configs: List[Dict[str, Any]], output_dir: Optional[str] = None, standardize_smiles: bool = True) -> pd.DataFrame:
    """
    Internal function to generate a combined feature DataFrame based on a list of configs.
    This is used by the main CHEMIA pipeline.
    
    Args:
        smiles_list: SMILES字符串列表
        feature_configs: 特征配置列表
        output_dir: 日志输出目录
        standardize_smiles: 是否对SMILES进行标准化（默认True，使用RDKit canonical SMILES）
    """
    # This function body remains the same as your current version
    from .transformer_embeddings import (
        get_chemberta_embedding, get_molt5_embedding, get_chemroberta_embedding,
        get_roberta_embedding
    )
    from .unimol_embedding import get_unimol_embedding
    from .smiles_validator import standardize_smiles as _standardize_smiles

    if not isinstance(smiles_list, list):
        smiles_list = list(smiles_list)
    
    # SMILES标准化（默认启用）
    if standardize_smiles:
        _console.log("[cyan]Standardizing SMILES using RDKit canonicalization...[/cyan]")
        original_count = len(smiles_list)
        processed_smiles_list = [_standardize_smiles(smi) for smi in smiles_list]
        _console.log(f"[green]✓ Standardized {original_count} SMILES strings[/green]")
    else:
        processed_smiles_list = smiles_list

    feature_dispatch = {
        'chemberta': lambda sm, cfg, out_dir: _get_embedding_features(sm, cfg, get_chemberta_embedding, out_dir),
        'molt5': lambda sm, cfg, out_dir: _get_embedding_features(sm, cfg, get_molt5_embedding, out_dir),
        'chemroberta': lambda sm, cfg, out_dir: _get_embedding_features(sm, cfg, get_chemroberta_embedding, out_dir),
        'roberta': lambda sm, cfg, out_dir: _get_embedding_features(sm, cfg, get_roberta_embedding, out_dir),
        'unimol': lambda sm, cfg, out_dir: _get_embedding_features(sm, {**cfg, 'standardize_smiles': False}, get_unimol_embedding, out_dir),  # 已标准化
        'unimolv2_84m': lambda sm, cfg, out_dir: _get_embedding_features(sm, {**cfg, 'model_version': 'v2', 'model_size': '84m', 'standardize_smiles': False}, get_unimol_embedding, out_dir),  # 已标准化
        'unimolv2_164m': lambda sm, cfg, out_dir: _get_embedding_features(sm, {**cfg, 'model_version': 'v2', 'model_size': '164m', 'standardize_smiles': False}, get_unimol_embedding, out_dir),  # 已标准化
        'unimolv2_310m': lambda sm, cfg, out_dir: _get_embedding_features(sm, {**cfg, 'model_version': 'v2', 'model_size': '310m', 'standardize_smiles': False}, get_unimol_embedding, out_dir),  # 已标准化
        'unimolv2_570m': lambda sm, cfg, out_dir: _get_embedding_features(sm, {**cfg, 'model_version': 'v2', 'model_size': '570m', 'standardize_smiles': False}, get_unimol_embedding, out_dir),  # 已标准化
        'unimolv2_1b': lambda sm, cfg, out_dir: _get_embedding_features(sm, {**cfg, 'model_version': 'v2', 'model_size': '1.1B', 'standardize_smiles': False}, get_unimol_embedding, out_dir),  # 已标准化
    }

    rdkit_fp_types = ["maccs", "morgan", "rdkit", "atompair", "torsion"]
    all_feature_dfs = []

    for config in feature_configs:
        if not isinstance(config, dict):
            _console.log(f"[red]Error: config is not a dict! It's a {type(config)}: {config}[/red]")
            continue

        # 兼容两种配置格式：
        # 1. 扁平格式（实际工作格式）: {'type': 'morgan', 'radius': 2, ...}
        # 2. 嵌套格式（README/TUTORIAL 示例）: {'name': 'transformer_embedding', 'config': {'model_type': 'unimol', ...}}
        if 'type' in config:
            feature_type = config['type']
            working_config = config
        elif 'name' in config and 'config' in config:
            name = config['name']
            inner_config = config['config']
            if name == 'transformer_embedding':
                feature_type = inner_config.get('model_type')
                working_config = {'type': feature_type, **inner_config}
            elif name == 'rdkit_fingerprint':
                feature_type = inner_config.get('type')
                working_config = {'type': feature_type, **inner_config}
            elif name == 'rdkit_descriptors':
                feature_type = 'rdkit_descriptors'
                working_config = {'type': feature_type, **inner_config}
            else:
                _console.log(f"[yellow]Warning: Unknown generator name '{name}'. Skipping.[/yellow]")
                continue
        else:
            _console.log("[yellow]Warning: Skipping a config because it lacks a 'type' or 'name' key.[/yellow]")
            continue

        df = None
        if feature_type in rdkit_fp_types:
            rdkit_config = working_config.copy()
            rdkit_config['descriptors'] = False # Ensure only FP is calculated
            df = _get_rdkit_features(processed_smiles_list, rdkit_config)
        elif feature_type == 'rdkit_descriptors':
            # This type implies only descriptors are needed
            rdkit_config = {'type': None, 'descriptors': True}
            df = _get_rdkit_features(processed_smiles_list, rdkit_config)
        elif feature_type in feature_dispatch:
            handler = feature_dispatch[feature_type]
            df = handler(processed_smiles_list, working_config, output_dir)
        else:
            _console.log(f"[red]Error: Unknown feature type '{feature_type}'. Skipping.[/red]")
            continue

        if df is not None and not df.empty:
            # Validate feature dimensions consistency
            expected_rows = len(smiles_list)
            if df.shape[0] != expected_rows:
                _console.log(f"[yellow]Warning: Feature dimension mismatch for '{feature_type}'. Expected {expected_rows} rows, got {df.shape[0]}. Adjusting...[/yellow]")
                # Pad or truncate to match expected dimensions
                if df.shape[0] < expected_rows:
                    # Pad with zeros
                    missing_rows = expected_rows - df.shape[0]
                    zero_df = pd.DataFrame(0, index=range(df.shape[0], expected_rows), columns=df.columns)
                    df = pd.concat([df, zero_df], axis=0)
                else:
                    # Truncate
                    df = df.iloc[:expected_rows]
                _console.log(f"[green]✓ Adjusted '{feature_type}' features to shape: {df.shape}[/green]")

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
    output_dir_for_logs: str = './output/feature_logs',
    parallel: bool = True,
    n_jobs: Optional[int] = None,
    batch_size: Optional[int] = None,
    standardize_smiles: bool = True,
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
                                   Defaults to './output/feature_logs'.
        parallel (bool): Whether to use parallel processing for fingerprints (default: True).
                        Only applies to fingerprint calculations, not descriptors or embeddings.
        n_jobs (Optional[int]): Number of parallel processes to use. If None, uses CPU count - 1.
        batch_size (Optional[int]): Batch size for parallel processing. If None, automatically determined.
        standardize_smiles (bool): Whether to standardize SMILES using RDKit canonicalization (default: True).
        **kwargs: Additional parameters specific to the feature type.
            - For "morgan", "rdkit", "atompair", "torsion": nBits (int), radius (int, for Morgan only)
            - For "unimol": model_version (str), model_size (str)
            - For "chemberta", "molt5", etc.: model_name (str) to use a different checkpoint.

    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame where the index is the SMILES string
                                and columns are the calculated features. Returns None if the
                                feature type is invalid.
    """
    config = {
        'type': feature_type,
        'parallel': parallel,
        'n_jobs': n_jobs,
        'batch_size': batch_size,
        **kwargs
    }

    # We call the internal 'generate_features' function, which is already robust.
    # We wrap the config in a list because that's what generate_features expects.

    # Ensure the directory for logs exists.
    import os
    os.makedirs(output_dir_for_logs, exist_ok=True)

    # Reduce console output for internal use
    import logging
    logging.info(f"Calculating features of type: '{feature_type}' (parallel={parallel}, standardize={standardize_smiles})")

    features_df = generate_features(smiles_list, [config], output_dir=output_dir_for_logs, standardize_smiles=standardize_smiles)

    if features_df.empty:
        logging.error(f"Calculation failed for feature type '{feature_type}'")
        return None

    return features_df
