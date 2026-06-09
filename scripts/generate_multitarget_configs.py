#!/usr/bin/env python3
"""Auto-generate YAML configs for all target columns in multi-target datasets."""

import os, yaml
import pandas as pd

CONFIG_DIR = "examples/configs"

DATASET_SPECS = {
    "ClinTox": {
        "file": "data/ClinTox.csv",
        "smiles_col": "smiles",
        "task_type": "binary_classification",
        "exclude_cols": ["smiles"],
    },
    "Tox21": {
        "file": "data/tox21.csv",
        "smiles_col": "smiles",
        "task_type": "binary_classification",
        "exclude_cols": ["smiles", "mol_id"],
    },
    "SIDER": {
        "file": "data/sider.csv",
        "smiles_col": "smiles",
        "task_type": "binary_classification",
        "exclude_cols": ["smiles"],
    },
}

BASE_CONFIG = {
    "data": {
        "source_mode": "single_file",
        "single_file_config": {
            "main_file_path": None,  # filled per dataset
            "smiles_col": None,
            "target_col": None,
        },
    },
    "task_type": "binary_classification",
    "features": {
        "generators": [
            {"name": "transformer_embedding", "config": {"model_type": "unimolv2_310m"}},
            {"name": "rdkit_fingerprint", "config": {"type": "morgan", "nBits": 2048, "radius": 2}},
        ],
    },
    "split_config": {
        "split_mode": "hold_out",
        "hold_out": {"test_size": 0.2, "random_state": 42},
    },
    "training": {
        "models_to_run": ["xgboost", "ann", "tabnet", "lgbm", "catboost", "randomforest"],
        "n_trials": 50,
        "random_state": 42,
    },
}

os.makedirs(CONFIG_DIR, exist_ok=True)

for ds_name, spec in DATASET_SPECS.items():
    df = pd.read_csv(spec["file"])
    target_cols = [c for c in df.columns if c not in spec["exclude_cols"]]

    # Create subdirectory for multi-target datasets
    ds_dir = os.path.join(CONFIG_DIR, ds_name.lower())
    os.makedirs(ds_dir, exist_ok=True)

    for target in target_cols:
        # Sanitize filename
        safe_name = target.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        safe_name = safe_name.replace("/", "_").replace("\\", "_")
        safe_name = safe_name[:60]  # Truncate

        config = yaml.safe_load(yaml.dump(BASE_CONFIG))  # Deep copy
        config["data"]["single_file_config"]["main_file_path"] = spec["file"]
        config["data"]["single_file_config"]["smiles_col"] = spec["smiles_col"]
        config["data"]["single_file_config"]["target_col"] = target
        config["task_type"] = spec["task_type"]

        yaml_path = os.path.join(ds_dir, f"{safe_name}.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"# {ds_name} — {target}\n")
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"{ds_name}: {len(target_cols)} configs → {ds_dir}/")

print("\nDone. See examples/configs/{clintox,tox21,sider}/")
