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

## Example

```python
from catboost import CatBoostRegressor, CatBoostClassifier

# 定义模型文件路径
model_path = "output/Your_Experiment_Name_xxxx/models/cat/cat_model.cbm"

# 1. 创建一个空的模型实例
# 如果是回归任务:
model_to_load = CatBoostRegressor()
# 如果是分类任务:
# model_to_load = CatBoostClassifier()

# 2. 使用 .load_model() 方法加载模型
model_to_load.load_model(model_path)

# 现在模型已经准备好进行预测
# predictions = model_to_load.predict(your_new_data)

print(f"成功加载 CatBoost 模型。")
```

```python
import joblib

# 定义模型文件路径
# 例如，加载 RandomForest 模型, 这适用于大多数模型，如 lgbm, rf, dt, knn, svr, ridge, krr, adab, lr, svc 等
model_path = "output/Your_Experiment_Name_xxxx/models/rf/rf_model.joblib"

# 使用 joblib.load 加载模型
loaded_model = joblib.load(model_path)

# 现在模型已经准备好进行预测
# predictions = loaded_model.predict(your_new_data)

print(f"成功加载模型: {type(loaded_model)}")
```

```python
import xgboost as xgb

# 定义模型文件路径
model_path = "output/Your_Experiment_Name_xxxx/models/xgb/xgb_model.json"

# 1. 创建一个空的模型实例
# 如果是回归任务:
model_to_load = xgb.XGBRegressor()
# 如果是分类任务:
# model_to_load = xgb.XGBClassifier()

# 2. 使用 .load_model() 方法加载模型
model_to_load.load_model(model_path)

# 现在模型已经准备好进行预测
# predictions = model_to_load.predict(your_new_data)

print(f"成功加载 XGBoost 模型。")
```

```python
import torch
# 确保 ComplexANN 类的定义在当前作用域内可用
# 您需要从 models/ann.py 导入它
from models.ann import ComplexANN 

# --- 关键步骤: 重新创建模型实例 ---
# 您必须使用与训练时完全相同的参数来实例化模型
# 这些参数（input_size, hidden_sizes, etc.）可以在日志文件或超参数json文件中找到

# 示例参数 (请根据您的实际情况修改!)
INPUT_SIZE = 1232  # 特征数量
HIDDEN_SIZES = [1024, 512, 64] # 隐藏层结构
OUTPUT_SIZE = 1 # 回归任务为1, 分类任务为类别数
TASK_TYPE = 'regression'
DROPOUT_RATE = 0.25 # 保存时使用的dropout率

# 1. 实例化模型结构
loaded_model = ComplexANN(
    input_size=INPUT_SIZE,
    hidden_sizes=HIDDEN_SIZES,
    output_size=OUTPUT_SIZE,
    task_type=TASK_TYPE,
    dropout_rate=DROPOUT_RATE
)

# 定义模型文件路径
model_path = "output/Your_Experiment_Name_xxxx/models/ann/ann_model.pth"

# 2. 加载权重 (state dictionary)
# 如果您在CPU上加载，即使模型是在GPU上训练的，也要使用 map_location
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model.load_state_dict(torch.load(model_path, map_location=device))
loaded_model.to(device) # 将模型移动到适当的设备

# 3. 设置为评估模式
# 这对于关闭 dropout 和 batch normalization 的训练行为非常重要
loaded_model.eval()

# 现在模型已经准备好进行预测
# with torch.no_grad():
#     input_tensor = torch.tensor(your_new_data, dtype=torch.float32).to(device)
#     predictions = loaded_model(input_tensor)

print(f"成功加载 ANN (PyTorch) 模型。")
```



## Framework Structure

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

