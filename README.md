<div align="center">
  <img src="images/chemia.png" alt="CHEMIA Logo" width="200"/>
  <h1>CHEMIA</h1>
  <p><strong>A Comprehensive Machine Learning Framework for Predicting and Optimizing Chemical Properties and Reactions</strong></p>
  <p>
    <a href="https://img.shields.io/badge/python-3.8+-blue.svg"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python"></a>
    <a href="https://img.shields.io/badge/license-MIT-green.svg"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
    <a href="https://img.shields.io/badge/status-active-brightgreen.svg"><img src="https://img.shields.io/badge/status-active-brightgreen.svg" alt="Status"></a>
  </p>
  <p>
    <strong>English</strong> | <a href="README_zh-CN.md">简体中文</a>
  </p>
</div>



---

**CHEMIA** is a powerful tool designed for machine learning researchers in chemistry. Through simple YAML configuration files, it provides a "one-stop" workflow—from data processing and feature engineering to model training, hyperparameter tuning, and Bayesian optimization. This allows you to focus on the chemistry problem itself, rather than the tedious engineering details.

## 📚 Table of Contents

*   [**Chapter 1: Getting to Know CRANE**](#chapter-1-getting-to-know-crane)
    *   [1.1 Core Features](#11-core-features)
    *   [1.2 Project Structure](#12-project-structure)
*   [**Chapter 2: Five-Minute Quick Start**](#chapter-2-five-minute-quick-start)
    *   [2.1 Prepare Your Data](#21-prepare-your-data)
    *   [2.2 Configure Your Experiment](#22-configure-your-experiment)
    *   [2.3 Launch the Training](#23-launch-the-training)
*   [**Chapter 3: Installation and Deployment**](#chapter-3-installation-and-deployment)
    *   [3.1 Environment Setup](#31-environment-setup)
    *   [3.2 Script Guide](#32-script-guide)
*   [**Chapter 4: In-Depth Usage Guide**](#chapter-4-in-depth-usage-guide)
    *   [4.1 Detailed Training Modes](#41-detailed-training-modes)
    *   [4.2 Bayesian Optimization](#42-bayesian-optimization)
    *   [4.3 End-to-End Workflow](#43-end-to-end-workflow)
*   [**Chapter 5: Configuration File Deep Dive**](#chapter-5-configuration-file-deep-dive)
    *   [5.1 Training Configuration Templates](#51-training-configuration-templates)
    *   [5.2 Optimization Configuration Templates](#52-optimization-configuration-templates)
*   [**Chapter 6: Core Capabilities Analysis**](#chapter-6-core-capabilities-analysis)
    *   [6.1 Supported Algorithm Library](#61-supported-algorithm-library)
    *   [6.2 Automated Feature Engineering](#62-automated-feature-engineering)
    *   [6.3 Data Splitting Strategies](#63-data-splitting-strategies)
    *   [6.4 Data Format Requirements](#64-data-format-requirements)
*   [**Chapter 7: Advanced Techniques and Programmatic API**](#chapter-7-advanced-techniques-and-programmatic-api)
    *   [7.1 Custom Configurations](#71-custom-configurations)
    *   [7.2 Programmatic Usage](#72-programmatic-usage)
*   [**Chapter 8: Interpreting Results and Outputs**](#chapter-8-interpreting-results-and-outputs)
*   [**Appendix: License**](#appendix-license)

---

## Chapter 1: Getting to Know CRANE

### 1.1 Core Features

*   **🤖 Diverse Machine Learning Algorithms**: Built-in support for 15+ classic algorithms, including `XGBoost`, `LightGBM`, `CatBoost`, `Random Forest`, `Gaussian Process Regression`, as well as `Neural Networks` and `Graph Neural Networks`.
*   **✨ Automated Feature Engineering**: Automatically generates molecular fingerprints (`Morgan`, `MACCS`) and `RDKit` descriptors from SMILES, and supports embeddings from pre-trained models like `UniMol`.
*   **🧩 Flexible Data Splitting**: Supports train/test split, train/validation/test split, and k-fold cross-validation to meet various evaluation needs.
*   **🔎 Hyperparameter Optimization**: Deeply integrated with `Optuna` for fully automated and efficient hyperparameter searching.
*   **🎯 Bayesian Optimization**: Utilizes trained surrogate models to intelligently explore and find optimal reaction conditions.
*   **⚙️ End-to-End Workflow**: Provides a complete, automated pipeline from model training to reaction optimization.
*   **📜 Rich Configuration Options**: A clear and intuitive YAML-based configuration system that offers high flexibility for customization.

### 1.2 Project Structure

```
chemia/
├── 📂 core/                    # Core Framework Components
│   ├── run_manager.py      # Experiment management
│   ├── config_loader.py    # Configuration loading
│   └── trainer_setup.py    # Model training setup
├── 📂 models/                  # Model Implementations
│   ├── sklearn_models.py   # Traditional ML models
│   ├── ann.py              # Neural Networks
│   └── gnn_models.py       # Graph Neural Networks
├── 📂 optimization/            # Bayesian Optimization
│   ├── optimizer.py        # Main optimization engine
│   └── space_loader.py     # Search space management
├── 📂 utils/                   # Utility functions
├── 📂 examples/                # Example configurations and scripts
│   └── configs/            # Configuration files
├── 📂 data/                    # Data directory
└── 📂 output/                  # Results and trained models
```

---

## Chapter 2: Five-Minute Quick Start

Let's quickly experience the power of CRANE with a simple example.

### 2.1 Prepare Your Data

First, place your dataset (e.g., `CPA.csv`) in the `data/` directory. The file format should be as follows:

```csv
# data/CPA.csv
Catalyst,Imine,Thiol,Output
O=P1(O)OC2=C(C3=CC=CC=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=CC=C6)=CC7=C5C=CC=C7,O=C(C1=CC=CC=C1)/N=C/C2=CC=C(C(F)(F)F)C=C2,CCS,0.501758501
O=P1(O)OC2=C(C3=C(F)C=C(OC)C=C3F)C=C4C(C=CC=C4)=[C@]2[C@]5=C(O1)C(C6=C(F)C=C(OC)C=C6F)=CC7=C5C=CC=C7,O=C(C1=CC=CC=C1)/N=C/C2=CC=C(C(F)(F)F)C=C2,CCS,1.074990526
...
```

### 2.2 Configure Your Experiment

Next, create a YAML configuration file (or use an existing template) to define your experiment.

```yaml
# Example config file: config_quick_start.yaml

# ===================================================================
# 1. Basic Experiment Information
# ===================================================================
experiment_name: "CPA_Quick_Start" # Experiment name; all results will be saved to output/CPA_Quick_Start/
task_type: "regression"            # Task type: 'regression' or 'classification'

# ===================================================================
# 2. Data Source Configuration
# ===================================================================
data:
  source_mode: "single_file"
  single_file_config:
    main_file_path: "data/CPA.csv"
    smiles_col: ["Catalyst", "Imine", "Thiol"] # Define columns containing SMILES
    target_col: "Output"                     # Define the target value column
    precomputed_features: null               # No pre-computed features in this example

# ===================================================================
# 3. Feature Engineering Strategy
# ===================================================================
features:
  per_smiles_col_generators:
    Catalyst: 
      - type: "unimol"       # Generate features using UniMol v2
        model_version: "v2"
        model_size: "310m"
      - type: "morgan"
        nbits: 2048
        radius: 2
    Imine: 
      - type: "rdkit_descriptors"
    Thiol: 
      - type: "rdkit_descriptors"
  scaling: true # Standardize features

# ===================================================================
# 4. Training and Evaluation Configuration
# ===================================================================
training:
  models_to_run: # Select models to train
    - "lgbm"
    - "catboost"
    - "xgb"
  n_trials: 30 # Number of Optuna hyperparameter search trials

split_mode: "train_valid_test" # Dataset splitting mode
split_config:
  train_valid_test:
    valid_size: 0.1
    test_size: 0.1
    random_state: 42

evaluation:
  primary_metric: "r2" # Primary metric for model selection
  additional_metrics: ["rmse", "mae"]

# ===================================================================
# 5. Other Settings (can be left as default)
# ===================================================================
output:
  save_predictions: true
  save_feature_importance: true

computational:
  n_jobs: -1 # Use all available CPU cores
  random_state: 42
```

> **💡 Tip:** GNN algorithms are currently in the testing phase, so there's no need to install the CUDA version of PyTorch for now.

### 2.3 Launch the Training

Once configured, run the following command in your terminal:

```bash
python scripts/run_training_only.py --config examples/configs/quick_start.yaml
```

Congratulations! You have successfully launched a complete machine learning training pipeline. The results, models, and analysis reports will be automatically saved in the `output/CPA_Quick_Start/` directory.

---

## Chapter 3: Installation and Deployment

### 3.1 Environment Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/flyben97/Chemia.git
    cd crane
    ```

2.  **Install dependencies**:
    
    ```bash
    pip install -r requirements.txt
    ```

### 3.2 Script Guide

CHEMIA provides three core execution scripts to suit different needs:

| Script                     | Function                                                     | Recommended Use Case                                         |
| :------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **`run_training_only.py`** | Focuses on model training, evaluation, and comparison.       | **Model development**, algorithm comparison, and for beginners. |
| **`run_optimization.py`**  | Performs standalone Bayesian optimization with a pre-trained model. | When you have a reliable model and need to perform **condition optimization**. |
| **`run_full_workflow.py`** | An automated end-to-end pipeline (Train → Optimize).         | For fully automated **production environments** or complete experiments. |

---

## Chapter 4: In-Depth Usage Guide

### 4.1 Detailed Training Modes

Using `run_training_only.py` with different configuration files allows for various training modes.

*   **Basic Regression Training**:
    ```bash
    python scripts/run_training_only.py --config examples/configs/regression_training_simple.yaml
    ```
*   **Basic Classification Training**:
    ```bash
    python scripts/run_training_only.py --config examples/configs/classification_training_simple.yaml
    ```
*   **5-Fold Cross-Validation Training** (for more robust evaluation):
    ```bash
    python scripts/run_training_only.py --config examples/configs/regression_training_kfold.yaml
    ```

### 4.2 Bayesian Optimization

If you have already trained and saved a model, you can use `run_optimization.py` to find the optimal reaction conditions.

```bash
python scripts/run_optimization.py --config examples/configs/bayesian_optimization_only.yaml
```

### 4.3 End-to-End Workflow

For fully automated needs, `run_full_workflow.py` will first train models, then automatically select the best one for Bayesian optimization.

```bash
python scripts/run_full_workflow.py --config examples/configs/end_to_end_workflow.yaml
```

---

## Chapter 5: Configuration File Deep Dive

The core of CHEMIA is its powerful YAML configuration system. The `examples/configs/` directory provides a rich set of templates.

### 5.1 Training Configuration Templates

| Config File                           | Description                                                  |
| :------------------------------------ | :----------------------------------------------------------- |
| `quick_start.yaml`                    | A minimal setup for quick testing and first-time use.        |
| `regression_training_simple.yaml`     | Basic configuration for standard regression tasks.           |
| `regression_training_kfold.yaml`      | Regression training using k-fold cross-validation.           |
| `regression_training_split.yaml`      | Regression training with a train/validation/test split.      |
| `classification_training_simple.yaml` | Basic configuration for standard classification tasks.       |
| `classification_training_kfold.yaml`  | Classification training using k-fold cross-validation.       |
| `training_with_features.yaml`         | A configuration with extensive feature engineering options.  |
| `gnn_training.yaml`                   | A configuration specifically for training Graph Neural Networks. |

### 5.2 Optimization Configuration Templates

| Config File                       | Description                                                  |
| :-------------------------------- | :----------------------------------------------------------- |
| `bayesian_optimization_only.yaml` | Runs standalone Bayesian optimization, requiring a pre-trained model. |
| `end_to_end_workflow.yaml`        | Configuration for the complete train-and-optimize pipeline.  |

---

## Chapter 6: Core Capabilities Analysis

### 6.1 Supported Algorithm Library

*   **Gradient Boosting**: XGBoost, LightGBM, CatBoost, Histogram-based Gradient Boosting
*   **Tree Ensembles**: Random Forest, Extra Trees, AdaBoost
*   **Linear Models**: Ridge, LASSO, ElasticNet, Bayesian Ridge
*   **Kernel Methods**: Gaussian Process Regression, Kernel Ridge, Support Vector Regression
*   **Instance-based Methods**: k-Nearest Neighbors
*   **Neural Networks**: PyTorch-based fully connected Artificial Neural Networks (ANN)
*   **Graph Neural Networks**: GCN, GAT, MPNN, Graph Transformer, etc.

### 6.2 Automated Feature Engineering

CRANE automatically generates high-quality molecular features from SMILES strings:

*   **Morgan Fingerprints**: Circular fingerprints with customizable radius and bit length.
*   **MACCS Keys**: 166-bit structural keys.
*   **RDKit Descriptors**: 200+ physicochemical properties and topological descriptors.
*   **Pre-trained Model Embeddings**: Supports pre-computed features from models like `UniMol`, `ChemBerta`, and `Molt5`.

### 6.3 Data Splitting Strategies

1.  **Train/Test Split**: A simple split by a user-defined ratio.
2.  **Train/Validation/Test Split**: Used for model development and tuning, with customizable ratios.
3.  **K-Fold Cross-Validation**: Provides a more robust assessment of model performance, with a customizable number of folds.

### 6.4 Data Format Requirements

Your input data should be in CSV format, containing columns with SMILES strings and a target value column. Other numerical or categorical features are also handled automatically.

```csv
Catalyst,Reactant1,Reactant2,Temperature,Solvent,yield
CC(C)P(c1ccccc1)c1ccccc1,CC(=O)c1ccccc1,NCc1ccccc1,80,toluene,95.2
CCc1ccc(P(CCc2ccccc2)CCc2ccccc2)cc1,CC(=O)c1ccccc1,NCc1ccccc1,60,THF,87.5
...
```

---

## Chapter 7: Advanced Techniques and Programmatic API

### 7.1 Custom Configurations

You can create your own `my_config.yaml` file based on any of the provided templates to meet specific experimental needs.

```yaml
experiment_name: "My_New_Experiment"
task_type: "regression"

data:
  single_file_config:
    main_file_path: "data/my_reactions.csv"
    smiles_col: ["Catalyst", "Reactant1", "Reactant2"]
    target_col: "yield"
    # ...

training:
  models_to_run: ["xgb", "lgbm", "rf"]
  n_trials: 50
# ... other configurations
```

### 7.2 Programmatic Usage

Besides the command line, you can also call CHEMIA's core functions within your Python scripts.

```python
from core.run_manager import start_experiment_run
from core.config_loader import load_config

# 1. Load the configuration file
config = load_config("my_config.yaml")

# 2. Run the experiment
results_summary = start_experiment_run(config)

# 3. Access and analyze the results
best_model_info = max(results_summary['results'], key=lambda x: x['test_r2'])
print(f"Best Model: {best_model_info['model_name']} (R² = {best_model_info['test_r2']:.4f})")
```

---

## Chapter 8: Interpreting Results and Outputs

After each experiment, CHEMIA generates a well-structured folder in the `output/` directory, containing:

*   **📈 Comprehensive Report**: A `_model_comparison.csv` file summarizing the performance metrics of all models.
*   **📦 Trained Models**: Serialized model files (e.g., `.pkl` or `.pt`), along with the associated scaler and feature names.
*   **📝 Prediction Results**: CSV files with predictions for the validation and test sets.
*   **📊 Visualizations**: Feature importance plots, learning curves, and more.
*   **🎯 Optimization Results**: The best conditions found by Bayesian optimization and their predicted values.
*   **⚙️ Experiment Records**: A saved copy of the hyperparameters and configuration file to ensure reproducibility.

---

## Appendix: License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Happy modeling on your chemistry tasks! 🧪✨</strong>
</div>
