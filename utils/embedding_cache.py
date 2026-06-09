# utils/embedding_cache.py
"""Disk cache for molecular embeddings. Shared within one batch run, deleted after."""

import os, json, hashlib, shutil
import numpy as np


def build_model_id(feature_type, config):
    """Deterministic model identifier from feature type and config params."""
    stable_items = sorted(
        (k, str(v)) for k, v in config.items()
        if k not in ('type', 'model_type', 'standardize_smiles', 'batch_size')
    )
    return f"{feature_type}_" + "_".join(f"{k}={v}" for k, v in stable_items)


def _file_cache_key(smiles_list, model_id):
    """Hash sorted SMILES + model_id into a filename-safe key."""
    payload = json.dumps({"smiles": sorted(smiles_list), "model_id": model_id}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def init_cache_dir():
    """Create and return the cache directory path under output/."""
    cache_dir = os.path.join(os.getcwd(), "output", ".cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def clear_cache():
    """Remove the entire embedding cache directory."""
    cache_dir = os.path.join(os.getcwd(), "output", ".cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


def get_cached(smiles_list, model_id, cache_dir):
    """
    Check cache for pre-computed embeddings.
    Returns embeddings array (reordered to match smiles_list) or None on miss.
    """
    key = _file_cache_key(smiles_list, model_id)
    npy_path = os.path.join(cache_dir, f"{key}_data.npy")
    meta_path = os.path.join(cache_dir, f"{key}_meta.json")

    if not (os.path.exists(npy_path) and os.path.exists(meta_path)):
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        cached_smiles = meta["smiles"]
        if len(cached_smiles) != len(smiles_list):
            return None

        embeddings = np.load(npy_path)
        if embeddings.shape[0] != len(smiles_list):
            return None

        # Reorder: cached order (sorted) → caller order
        smi_to_idx = {s: i for i, s in enumerate(cached_smiles)}
        order = [smi_to_idx[s] for s in smiles_list]
        return embeddings[order]
    except Exception:
        return None


def put_cache(smiles_list, embeddings, model_id, cache_dir):
    """Store embeddings to cache (in sorted SMILES order). Non-fatal on failure."""
    try:
        key = _file_cache_key(smiles_list, model_id)
        npy_path = os.path.join(cache_dir, f"{key}_data.npy")
        meta_path = os.path.join(cache_dir, f"{key}_meta.json")

        sorted_smiles = sorted(smiles_list)
        smi_to_idx = {s: i for i, s in enumerate(smiles_list)}
        order = [smi_to_idx[s] for s in sorted_smiles]

        np.save(npy_path, embeddings[order])
        with open(meta_path, "w") as f:
            json.dump({"smiles": sorted_smiles, "model_id": model_id, "n": len(smiles_list)}, f)
    except Exception:
        pass
