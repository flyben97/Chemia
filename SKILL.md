---
name: chemia
description: Chemia is a comprehensive, standardized machine learning framework for molecular property prediction, chemical reaction optimization, and materials design. This skill should be used whenever the user mentions cheminformatics, molecular prediction, QSAR/QSPR, reaction yield optimization, molecular fingerprinting, SMILES-based ML, drug discovery ML, materials design with ML, or needs to train/evaluate/deploy models for molecular properties. Also use when the user talks about "chemy", "chemistry ML", "molecule AI", "molecular embeddings", "UniMol", "ChemBERTa", or any task involving predicting molecular properties from SMILES strings. Trigger even if the user doesn't explicitly say "Chemia" — any cheminformatics ML workflow should default to using this skill.
---

# Chemia: Molecular ML Framework

A comprehensive machine learning framework for molecular property prediction, chemical reaction optimization, and materials design. Supports multiple molecular representation methods, pretrained transformer models, and diverse ML algorithms — all configurable via YAML without writing Python code.

## Project Structure

```
chemia/
├── SKILL.md              # This file
├── core/                 # Core pipeline (config loading, run management, training)
│   ├── config_loader.py  # YAML config validation
│   ├── run_manager.py    # End-to-end experiment orchestration
│   └── trainer_setup.py  # Model training setup
├── utils/                # Feature generation, prediction, SHAP, I/O
│   ├── feature_generator.py        # Orchestrates all feature generation
│   ├── molecular_representation_api.py  # Unified API for fingerprints & embeddings
│   ├── mol_fp_features.py          # RDKit fingerprints & descriptors
│   ├── transformer_embeddings.py   # ChemBERTa, MolT5, ChemRoBERTa, RoBERTa
│   ├── unimol_embedding.py         # UniMol/UniMolv2 embeddings
│   ├── predictor.py / predictor_api.py  # Model prediction pipeline
│   ├── shap_analyzer.py / shap_integration.py  # SHAP interpretability
│   ├── data.py / data_split_manager.py  # Data loading and splitting
│   ├── io_handler.py      # Model/config persistence
│   ├── metrics.py         # Evaluation metrics
│   └── ...
├── models/               # Model implementations (ANN, sklearn, transformers)
├── optimizers/           # Optuna-based hyperparameter optimization
├── optimization/         # Bayesian reaction optimization
│   ├── optimizer.py      # BayesianReactionOptimizer
│   └── space_loader.py   # Search space definition
├── scripts/              # CLI entry points for all workflows
│   ├── run_full_workflow.py        # Train → Select → Optimize (end-to-end)
│   ├── run_training_only.py        # Model training with hyperparameter optimization
│   ├── run_optimization.py         # Bayesian reaction optimization
│   ├── run_prediction_standalone.py # Prediction with trained models
│   ├── run_shap_analysis.py        # SHAP model interpretability
│   ├── run_batch_workflow.py       # Batch execution of multiple configs
│   ├── prediction_api.py           # Python API for predictions
│   └── download_pretrained_models.py  # Download pretrained transformer models
├── references/           # Detailed reference docs (load as needed)
│   ├── configuration.md  # Complete YAML config reference
│   └── api.md           # Python API reference
├── docs/                 # Additional documentation
├── data/                 # Example datasets (BACE, BBBP, ClinTox, etc.)
├── examples/configs/     # Example YAML configuration files
└── requirements.txt      # Python dependencies
```

## When to Use This Skill

Invoke this skill when the user needs to:

1. **Set up molecular ML workflows** — creating YAML configs, preparing data
2. **Train models on molecular data** — regression, classification, multi-task
3. **Make predictions** with trained models on new molecules
4. **Optimize chemical reactions** — Bayesian optimization for reaction conditions
5. **Run SHAP analysis** — interpret model predictions with chemical insights
6. **Debug cheminformatics ML issues** — feature generation errors, model failures, data problems
7. **Batch process** multiple datasets or configurations

## Core Workflows

### 1. Training Workflow

Run model training with hyperparameter optimization:

```bash
python scripts/run_training_only.py config.yaml
```

Or with a specific config path:

```bash
python scripts/run_training_only.py --config path/to/config.yaml
```

### 2. Prediction Workflow

Make predictions with trained models:

```bash
python scripts/run_prediction_standalone.py --model-dir output/my_run --data new_molecules.csv
```

### 3. Full End-to-End Workflow

Train → Select Best Model → Bayesian Optimization:

```bash
python scripts/run_full_workflow.py config_full_workflow.yaml
```

### 4. Bayesian Reaction Optimization

Optimize reaction conditions using a trained model as surrogate:

```bash
python scripts/run_optimization.py optimization_config.yaml
```

### 5. SHAP Analysis

Interpret model predictions:

```bash
python scripts/run_shap_analysis.py --run-dir output/my_run
```

### 6. Batch Workflow

Run multiple configs sequentially:

```bash
python scripts/run_batch_workflow.py --config-dir configs/
```

### 7. Download Pretrained Models

```bash
python download_pretrained_models.py
```

## Quick Start: Creating a Configuration

Chemia uses YAML configuration files. Here is the minimal configuration for molecular property prediction:

```yaml
data:
  source_mode: single_file
  single_file_config:
    main_file_path: data/molecules.csv
    smiles_col: SMILES
    target_col: property

task_type: regression  # or: binary_classification, multiclass_classification

features:
  generators:
    - name: transformer_embedding
      config:
        model_type: unimolv2_310m  # Recommended; also: unimol, chemberta, molt5

split_config:
  split_mode: hold_out
  hold_out:
    test_size: 0.2

training:
  models_to_run: [xgboost, lgbm]
  n_trials: 50  # Optuna hyperparameter optimization trials
```

Save as `config.yaml` and run:

```bash
python scripts/run_training_only.py config.yaml
```

## Configuration Patterns

### Data Source Modes

- **single_file**: One CSV with SMILES + target → auto split
- **pre_split**: Separate train/val/test CSV files (see `references/configuration.md`)
- **features_only**: Pre-computed feature matrix (no SMILES processing)
- **multi_smiles**: Multiple SMILES columns (for reactions, formulations)

### Feature Generators

- `transformer_embedding` — UniMol, UniMolv2, ChemBERTa, MolT5, ChemRoBERTa, RoBERTa
- `rdkit_fingerprint` — Morgan (ECFP), MACCS, RDKit, AtomPair, Torsion fingerprints + 2D descriptors
- `precomputed_features` — Use pre-computed features from CSV columns

### Supported ML Models

| Category | Models |
|---|---|
| Gradient Boosting | xgboost, lgbm, catboost, gbdt |
| Tree Ensembles | randomforest, extratrees |
| Linear | ridge, lasso, elasticnet |
| Neural Networks | ann, tabnet |
| Advanced/Kernel | svr, gpr, kernelridge |
| Graph Neural Nets | gcn, gat, mpnn, afp |

### Data Split Modes

- `hold_out` — Single train/test split
- `cross_validation` — K-fold CV for robust evaluation
- `pre_split` — Use pre-defined splits from separate files

## Python API

Chemia also provides a Python API for programmatic use:

```python
from scripts.prediction_api import load_model, predict, predict_single

# Load a trained model
predictor = load_model("output/my_experiment", "xgb")

# Predict single molecule
result = predict_single(predictor, {"SMILES": "CCO"})

# Batch predict
samples = [{"SMILES": "CCO"}, {"SMILES": "c1ccccc1"}]
results = predict(predictor, samples)
```

For molecular representation directly:

```python
from utils.molecular_representation_api import MolecularRepresentationAPI

api = MolecularRepresentationAPI()
fp = api.get_fingerprint("CCO", fingerprint_type="morgan", radius=2, nBits=2048)
embedding = api.get_embedding("CCO", model_type="unimolv2_310m")
```

## Key Design Points

### Project Root Resolution
All scripts use `utils.cli_setup.standard_cli_setup()` to locate the project root. This must run before other imports from the project. Scripts should be run from the project root or with `PYTHONPATH` set correctly.

### Pretrained Models
Transformer embeddings (UniMol, ChemBERTa, etc.) download models automatically on first use and cache them in `pretrained_models/`. Run `download_pretrained_models.py` to pre-download all models.

### Output Structure
Each training run creates `output/<experiment_name>_<timestamp>/` containing:
- `models/` — Trained model files
- `data_splits/` — Processed data and scalers
- `run_config.json` — Saved configuration for reproducibility
- `results.csv` — Performance metrics for all trained models

## When NOT to Use

Skip this skill when:
- The user is working with non-molecular ML tasks (general tabular data, images, NLP)
- The user is doing pure quantum chemistry (DFT, molecular dynamics)
- The user needs a custom deep learning architecture not supported by the framework
- The task is about cheminformatics tools not related to ML (e.g., SMILES parsing, structure drawing)

## Troubleshooting Quick Reference

- **Import errors**: Ensure `pip install -r requirements.txt` is complete
- **CUDA/GPU issues**: The framework auto-detects GPU; set `CUDA_VISIBLE_DEVICES=""` to force CPU
- **Pretrained model download failures**: Run `python download_pretrained_models.py` manually
- **SMILES parsing errors**: Check for invalid SMILES with `utils/smiles_validator.py`
- **Memory issues with large datasets**: Use `rdkit_fingerprint` with `batch_size` or reduce `nBits`
