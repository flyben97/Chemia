# CRAFT: Chemical Representation & Analysis for Functional Targets

**CRAFT** is a flexible, configurable, and automated machine learning pipeline framework designed for cheminformatics and drug discovery. It can also be easily extended as a general-purpose machine learning platform.

This project aims to streamline the entire workflow from molecular structures (SMILES) to model performance evaluation. With a simple YAML configuration file, users can effortlessly define all steps, including data sources, feature engineering, data splitting, model training, and hyperparameter optimization.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Scikit--learn-1.2+-f89931.svg" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Optuna-3.1+-8d3cbf.svg" alt="Optuna">
  <img src="https://img.shields.io/badge/RDKit-2022.09+-026464.svg" alt="RDKit">
</p>

<p align="center">
  <a href="README_zh-CN.md">简体中文</a>
</p>

---

## :sparkles: Core Features

*   **Configuration-Driven**: The entire experiment workflow is controlled by a single `config.yaml` file, no code modification required.
*   **Flexible Data Sources**:
    *   Supports automatic dataset splitting from a single file.
    *   Supports user-provided, pre-split training/validation/test sets.
    *   Supports feature-only data (no SMILES required), enabling its use as a general ML framework.
*   **Powerful Feature Engineering**:
    *   Seamless integration of various molecular feature generation methods (RDKit fingerprints, descriptors).
    *   Built-in support for generating molecular embeddings from state-of-the-art pre-trained models (Uni-Mol, ChemBERTa, MolT5, etc.).
    *   Ability to combine dynamically generated features with user-provided pre-computed features.
*   **Automated Model Training**:
    *   Supports a wide range of classic machine learning models (XGBoost, LightGBM, RandomForest, SVR, ANN, etc. - 14 models in total).
    *   Utilizes [Optuna](https://optuna.org/) for efficient and automated hyperparameter optimization (HPO).
*   **Comprehensive Evaluation & Logging**:
    *   Automatically calculates and logs multiple evaluation metrics (R², RMSE, F1, Accuracy, etc.).
    *   Generates detailed log files and performance plots for each experiment and model.
    *   Saves trained models, best hyperparameters, prediction results, and out-of-fold (OOF) predictions for cross-validation.

---

## :rocket: Quick Start

### 1. Environment Setup

We highly recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage the project environment for compatibility.

**Step 1: Clone the repository**
```bash
git clone https://github.com/flyben97/craft.git
cd craft
```

**Step 2: Create and activate the Conda environment**
```bash
# Recommended: create from the environment.yml file (to be provided)
# conda env create -f environment.yml
# conda activate craft
# Or, create the environment manually
conda create -n craft python=3.10 -y
conda activate craft
```

**Step 3: Install core dependencies**
```bash
# Install ML and data processing libraries
pip install numpy pandas scikit-learn pyyaml rich

# Install model libraries
pip install xgboost lightgbm catboost

# Install hyperparameter optimization library
pip install optuna

# Install cheminformatics library
pip install rdkit-pypi

# Install deep learning library (choose the command that matches your CUDA version)
# CPU version:
pip install torch torchvision torchaudio

# GPU version (e.g., for CUDA 11.8), check PyTorch official website for the latest command:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For generating embeddings with pre-trained models
pip install huggingface_hub

# (Recommended) Set Hugging Face mirror before downloading models
export HF_ENDPOINT=https://hf-mirror.com

# Install molecular embedding libraries
pip install transformers sentencepiece
pip install unimol-tools

# Note: All pre-trained models will be downloaded on first use.
```

### 2. Prepare Your Data

Prepare your data file(s) and place them in the `data/` directory (or any location specified in `config.yaml`).

### 3. Configure Your Experiment

Open `config.yaml` in the project root and modify it according to your needs. See `config.en.yaml` for a fully commented template.

### 4. Run the Experiment

In your terminal with the `craft` environment activated, run:
```bash
python main.py --config config.yaml
```
Or, if your config file is named `config.yaml` and is in the root directory:
```bash
python main.py
```

### 5. Check the Results

All experiment artifacts (logs, models, plots, predictions) will be saved in the `output/` directory, organized by `experiment_name` and a timestamp.

```
output/
└── My_First_Experiment_regression_20231027_103000/
    ├── _experiment_summary.log       # Summary report for the entire experiment
    ├── data_splits/                  # Processed and split data
    │   ├── dataset_X_train.csv
    │   └── ...
    ├── models/
    │   ├── xgb/
    │   │   ├── xgb_hyperparameters.json
    │   │   ├── xgb_model.joblib
    │   │   ├── xgb_results.log       # Detailed report for the XGBoost model
    │   │   └── predictions_final_model/
    │   │       └── ...
    │   └── rf/
    │       └── ...
    └── unimol_tools.log              # Log for Uni-Mol if used
```

---

## ​Framework Structure

*   **`main.py`**: :classical_building: The sole entry point. It parses the config path and starts the core workflow.
*   **`config.yaml`**: :gear: The "blueprint" for your experiment. Defines all parameters, separating configuration from code.
*   **`core/`**: :brain: The project's brain.
    *   `config_loader.py`: Loads and validates `config.yaml`.
    *   `run_manager.py`: The main conductor. Orchestrates data loading, splitting, preprocessing, and model training.
    *   `trainer_setup.py`: Manages the training loop for each specified model.
*   **`optimizers/`**: :wrench: Model optimizers.
    *   `base_optimizer.py`: Defines the abstract base class for all optimizers, unifying interfaces like `optimize`, `fit`, and `predict`.
    *   `sklearn_optimizer.py`: Implements HPO for all Scikit-learn compatible models.
    *   `ann_optimizer.py`: A dedicated optimizer for the PyTorch-based Artificial Neural Network (ANN).
*   **`utils/`**: :toolbox: The utility toolkit.
    *   `feature_generator.py`: A unified interface for generating features from various backends.
    *   `mol_fp_features.py`: Backend for RDKit feature calculation.
    *   `transformer_embeddings.py`, `unimol_embedding.py`: Backends for molecular embedding models.
    *   `io.py`: Handles all file I/O (saving models, logs, predictions).
    *   `data.py`, `metrics.py`: Helper functions for data processing and performance evaluation.
*   **`models/`**: :bricks: Model definitions.
    *   `ann.py`: Defines the PyTorch ANN architecture.
    *   `sklearn_models.py`: Centralizes imports for all Scikit-learn ecosystem models (XGBoost, LightGBM, etc.).

---

## Customization and Extension

To add a new model or feature:

*   **Add a new model**: Simply add the model's name and hyperparameter search space to the `param_grids` dictionary in `optimizers/sklearn_optimizer.py`.
*   **Add a new feature generator**:
    1.  Create a new backend file in `utils/` (e.g., `my_new_feature.py`).
    2.  Import your new function in `utils/feature_generator.py` and register it in the `feature_dispatch` dictionary.
    3.  You can now call it in `config.yaml` using `type: "my_new_feature"`.

---

