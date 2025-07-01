#!/usr/bin/env python3
"""
Debug script to check GNN model detection
"""

import yaml
from core.trainer_setup import _has_smiles_columns, _get_smiles_columns

# Load the configuration
with open('config_gnn_demo.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Configuration loaded:")
print("Features config:", config.get('features', {}))
print("Molecular config:", config.get('features', {}).get('molecular', {}))

# Test the detection functions
has_smiles = _has_smiles_columns(config)
smiles_cols = _get_smiles_columns(config)

print(f"\nDetection results:")
print(f"Has SMILES columns: {has_smiles}")
print(f"SMILES columns: {smiles_cols}")

# Check availability of GNN models
try:
    from optimizers.gnn_optimizer import GNNOptimizer
    gnn_available = True
    print(f"GNN optimizer available: {gnn_available}")
except (ImportError, AttributeError) as e:  # 用元组捕获多个异常
    gnn_available = False
    print(f"GNN optimizer NOT available: 发生错误: {type(e).__name__} - {e}")

print(f"\nShould GNN models be available? {gnn_available and has_smiles}") 