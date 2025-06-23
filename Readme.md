# CRAFT: Chemical Representation & Analysis for Functional Targets

**CRAFT** 是一个灵活、可配置、自动化的机器学习流程框架，专为化学信息学和药物发现设计，同时也可轻松扩展为通用的机器学习平台。

本项目旨在简化从分子结构（SMILES）到模型性能评估的全过程，通过一个简单的 YAML 配置文件，用户可以轻松定义数据源、特征工程、数据拆分、模型训练和超参数优化等所有步骤。

<p align="center">   <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">   <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">   <img src="https://img.shields.io/badge/Scikit--learn-1.2+-f89931.svg" alt="Scikit-learn">   <img src="https://img.shields.io/badge/Optuna-3.1+-8d3cbf.svg" alt="Optuna">   <img src="https://img.shields.io/badge/RDKit-2022.09+-026464.svg" alt="RDKit"> </p>

---

## :sparkles: 核心功能

*   **配置驱动**：整个实验流程由一个独立的 `config.yaml` 文件控制，无需修改任何代码。
*   **灵活的数据源**：
    *   支持从单个文件自动拆分数据集。
    *   支持使用用户预先划分好的训练/验证/测试集。
    *   支持纯特征数据，无需SMILES，可作为通用机器学习框架使用。
*   **集成的特征工程**：
    *   无缝集成多种分子特征生成方法（RDKit指纹、描述符）。
    *   内置支持先进的预训练模型（Uni-Mol, ChemBERTa, MolT5等）生成分子嵌入。
    *   能够组合动态生成的特征和用户提供的预计算特征。
*   **自动化模型训练**：
    *   支持多种经典机器学习模型（XGBoost, LightGBM, RandomForest, SVR, ANN等14个机器学习模型）。
    *   使用 [Optuna](https://optuna.org/) 进行高效的自动化超参数优化（HPO）。
*   **全面的评估与日志**：
    *   自动计算并记录多种评估指标（R², RMSE, F1, Accuracy等）。
    *   为每个实验和每个模型生成详细的日志文件和性能图表。
    *   保存训练好的模型、最佳超参数、预测结果和交叉验证（OOF）预测。

---

## :rocket: 快速开始

### 1. 环境安装

我们强烈建议使用 [Conda](https://docs.conda.io/en/latest/miniconda.html) 来管理项目环境，以确保所有依赖包的版本兼容性。

**步骤一：克隆项目**
```bash
git clone https://github.com/flyben97/craft.git 
cd craft
```

**步骤二：创建并激活 Conda 环境**
```bash
# 从 environment.yml 文件创建环境（推荐），以后会提供
conda env create -f environment.yml
conda activate craft
```
现在可以手动下载包并创建环境

```bash
conda create -n craft python=3.10 -y
conda activate craft
```

**步骤三：安装核心依赖包**
```bash
# 安装机器学习和数据处理库
pip install numpy pandas scikit-learn pyyaml rich

# 安装模型库
pip install xgboost lightgbm catboost

# 安装超参数优化库
pip install optuna

# 安装化学信息学库
pip install rdkit-pypi

# 安装深度学习库 (根据您的CUDA版本选择合适的命令)
# CPU 版本:
pip install torch torchvision torchaudio

# GPU 版本 (以CUDA 11.8为例)，建议去pytorch官网寻找安装命令，测试时候安装的是
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 使用预训练模型生成embedding时需要用huggingface下载和加载模型
pip install huggingface_hub

# 下载预训练模型前，推荐设置huggingface镜像
# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 安装分子嵌入模型库
pip install transformers sentencepiece
pip install unimol-tools

# 所有预训练模型都会在首次运行时下载，之后运行会直接加载不会重复下载
```
### 2. 准备数据

根据您的需求，准备好您的数据文件，并将其放入 `data/` 目录（或您指定的任何位置，具体请参考`config.yaml`）。

### 3. 配置实验

打开项目根目录下的 `config.yaml` 文件，根据您的实验需求进行修改。这是一个最小化的示例：

**`config.yaml`**
```yaml
# ===================================================================
#           CRAFT (Chemical Representation and Analysis)
#                 实验总配置文件 (v0.9.4)
# ===================================================================
#
# 使用说明:
# 1. 修改 "experiment_name" 和 "task_type" 以符合您的实验目的。
# 2. 在 "data" 部分，选择四种 "source_mode" 中的一种，并填写对应的配置。
#    确保只保留一种模式的配置是激活的，其他模式的配置可以注释掉或忽略。
# 3. 在 "features" 部分，配置您需要动态生成的分子特征和是否进行标准化。
# 4. 在 "split_mode" 和 "split_config" 部分，配置数据拆分方式。
#    (注意: 此部分仅在 data.source_mode = 'single_file' 或 'features_only' 时有效)。
# 5. 在 "training" 部分，选择要训练的模型和超参数搜索的次数。
#
# ===================================================================


# --- 1. 实验基本设置 ---
experiment_name: "Prediction_Demo" # 实验名称，将用于创建输出文件夹
task_type: "regression"            # 任务类型: "regression", "binary_classification", or "multiclass_classification"

# --- 2. 数据源配置 ---
data:
  # 选择一种数据源模式 (四选一):
  # - "single_file":       提供一个含SMILES的文件，由程序自动拆分。
  # - "pre_split_t_v_t":   提供预拆分的含SMILES的训练/验证/测试集。
  # - "pre_split_cv":      提供预拆分的含SMILES的训练/测试集，用于交叉验证。
  # - "features_only":     提供一个纯特征+目标的文件，不含SMILES。
  # 注意: 只能选择一种模式，其他模式的配置将被忽略。
  # 如果需要使用其他模式，请注释掉不需要的配置。
  # 当选择 "features_only" 时，特征生成配置将被忽略，但数据集划分方式，以及是否提供划分好的数据集等功能仍然支持

  source_mode: "single_file" # <--- 在这里选择你的模式

  # --- 模式1: single_file ---
  # (当 source_mode: "single_file" 时使用此配置)
  single_file_config:
    main_file_path: "data/0610-collected-data.csv" # 包含SMILES和目标值的主数据文件
    smiles_col: "SMILES"                 # SMILES列的名称
    target_col: "emission_energy(eV)"    # 目标列的名称，在化学任务中，通常可以是Yield, energy, Activity等

    # (可选) 如果预计算的特征与SMILES在同一个文件里
    precomputed_features:
      path: null                         # null 表示使用 main_file_path
      # 指定特征所在的列。格式: "start:end" (不含end), "start:", ":end", 或列名列表
      feature_columns: "8:"              # 示例: 从第8列到最后一列都是特征，如果是第8列到第10列，则使用 "8:10"
  
  # 默认选择模式1，当要启用其他模式时，对相应模式取消注释（在VScode里面选中Ctrl+/快速取消注释）
  # --- 模式2: pre_split_t_v_t ---
  # (当 source_mode: "pre_split_t_v_t" 时使用此配置)
  # pre_split_t_v_t_config:
  #   train_path: "data/user_split/train.csv"
  #   valid_path: "data/user_split/valid.csv"
  #   test_path: "data/user_split/test.csv"
  #   smiles_col: "SMILES"
  #   target_col: "TARGET"
  #   precomputed_features:
  #     feature_columns: null # 假设没有预计算特征

  # --- 模式3: pre_split_cv ---
  # (当 source_mode: "pre_split_cv" 时使用此配置)
  # pre_split_cv_config:
  #   train_path: "data/user_split/train_for_cv.csv"
  #   test_path: "data/user_split/test_for_cv.csv"
  #   smiles_col: "SMILES"
  #   target_col: "TARGET"
  #   precomputed_features:
  #     feature_columns: null # 假设没有预计算特征

  # --- 模式4: features_only ---
  # (当 source_mode: "features_only" 时使用此配置)
  # features_only_config:
  #   file_path: "data/features_and_target.csv"
  #   target_col: "activity"
  #   feature_columns: "1:1001" # 假设特征在第1列到第1000列

# --- 3. 动态特征生成配置 (在 features_only 模式下将被忽略) ---
# 注意: 只有在 source_mode 为模式1，2，3时，此部分才会被使用。因为计算的特征需要SMILES列。
features:
  # 要动态生成的特征列表。如果不需要，设为 []
  # 可用的特征类型包括: 
  # "maccs", "morgan", "rdkit", "atompair", "torsion"，"rdkit_descriptors"
  # 'chemberta', 'molt5', 'chemroberta', 'unimol'
  # 注意'chemberta', 'molt5', 'chemroberta', 'unimol'等模型需要额外的模型文件下载
  # 这些模型会在初次使用的时候进行下载，如果需要全部下载预计需要6GB的空间

  generators:
    - type: "morgan"
      nBits: 1024
      radius: 2
    - type: "rdkit_descriptors"
    - type: "unimol"
      model_version: "v2"
      model_size: "84m"
      
  # 是否对最终的特征矩阵进行标准化处理 (StandardScaler)
  scaling: true

# --- 4. 数据拆分与验证策略 (仅当 source_mode 为 'single_file' 或 'features_only' 时有效) ---
# 选择一种模式: "train_valid_test" 或 "cross_validation"
# 通常情况下，我们在数据量较大时使用 "train_valid_test"，在数据量较小时使用 "cross_validation"。
split_mode: "train_valid_test"

split_config:
  # 当 split_mode 为 "train_valid_test" 时使用
  train_valid_test:
    train_size: 0.8
    valid_size: 0.1
    test_size: 0.1
    random_state: 0

  # 当 split_mode 为 "cross_validation" 时使用
  cross_validation:
    n_splits: 5           # 交叉验证的折数
    test_size_for_cv: 0.2 # 从总数据中分出20%作为最终的、独立的测试集
    random_state: 42

# --- 5. 模型训练配置 ---
training:
  # 要训练的模型列表。
  # 可用: xgb, lgbm, rf, cat, hgb, dt, knn, svr, ridge, ann, adab, lr, svc, krr
  models_to_run:
    - "xgb"
    # - "lgbm"
    # - "rf"
    # - "ann"
  
  # Optuna 超参数优化的试验次数
  # 当数据量较大时，建议设置为较少的实验轮数
  # 当数据量较小时，可以增加试验轮数以获得更好的超参数。
  n_trials: 50
  
  # HPO策略:
  # - 如果 split_mode 是 cross_validation，程序会自动使用 CV 进行 HPO。
  # - 如果 split_mode 是 train_valid_test，程序会使用 validation set 进行 HPO。
  
  # (可选) 减少Optuna的日志输出
  quiet_optuna: true
```

### 4. 运行实验

在激活 `craft` 环境的终端中，运行以下命令：

```bash
python main.py --config config.yaml
```

或者，如果您的配置文件就叫 `config.yaml` 并位于根目录，可以直接运行：
```bash
python main.py
```

实验开始后，您将在终端看到实时的进度和日志。

### 5. 查看结果

所有实验结果，包括日志、模型、图表和预测文件，都将保存在 `output/` 目录下，并以 `experiment_name` 和时间戳命名。

```
output/
└── My_First_Experiment_regression_20231027_103000/
    ├── _experiment_summary.log       # 整个实验的总结报告
    ├── data_splits/                  # 保存处理和拆分后的数据
    │   ├── dataset_X_train.csv
    │   └── ...
    ├── models/
    │   ├── xgb/
    │   │   ├── xgb_hyperparameters.json
    │   │   ├── xgb_model.joblib
    │   │   ├── xgb_results.log       # XGBoost模型的详细报告
    │   │   └── predictions_final_model/
    │   │       └── ...
    │   └── rf/
    │       └── ...
    └── unimol_tools.log              # 如果使用Uni-Mol，其日志会在这里
```

---

## :memo: 框架结构介绍

本项目的代码结构清晰，遵循模块化的设计原则：

*   **`main.py`**: 项目的唯一入口。它负责解析配置文件路径并启动核心流程。

*   **`config.yaml`**: 实验的“设计蓝图”。用户在这里定义所有实验参数，实现代码与配置的分离。

*   **`core/`**: 项目的核心大脑。
    *   `config_loader.py`: 负责加载和验证 `config.yaml` 文件。
    *   `run_manager.py`: 实验的总调度中心。它根据配置，依次调用数据处理、拆分、预处理和模型训练等模块。
    *   `trainer_setup.py`: 负责具体的模型训练循环。它为每个指定的模型设置优化器，并调用它们进行训练和评估。

*   **`optimizers/`**: 模型优化器。
    *   `base_optimizer.py`: 定义了所有优化器的抽象基类，统一了 `optimize`, `fit`, `predict` 等接口。
    *   `sklearn_optimizer.py`: 实现了对所有 Scikit-learn 兼容模型的超参数优化。
    *   `ann_optimizer.py`: 专门为基于 PyTorch 的人工神经网络（ANN）模型实现的优化器。

*   **`utils/`**: 通用工具箱。
    *   `feature_generator.py`: 强大的特征生成统一接口，可以调用各种后端模块生成特征。
    *   `mol_fp_features.py`: RDKit 特征计算的后端。
    *   `transformer_embeddings.py`, `unimol_embedding.py`: 分子嵌入模型的后端。
    *   `io.py`: 负责所有文件I/O操作，如保存模型、记录日志、写入预测结果。
    *   `data.py`, `metrics.py`: 提供数据处理和性能评估的辅助函数。

*   **`models/`**: 模型定义。
    *   `ann.py`: 定义了 PyTorch ANN 模型的网络结构。
    *   `sklearn_models.py`: 集中管理所有从 Scikit-learn 及其生态（XGBoost, LightGBM等）导入的模型类。

---

##  自定义与扩展

本框架具有良好的扩展性。例如，要添加一个新模型或新特征：

*   **添加新模型**：只需在 `optimizers/sklearn_optimizer.py` 的 `param_grids` 字典中添加新模型的名称和超参数搜索空间。
*   **添加新特征生成器**：
    1.  在 `utils/` 目录下创建一个新的特征计算后端文件（如 `my_new_feature.py`）。
    2.  在 `utils/feature_generator.py` 中导入你的新函数，并在 `feature_dispatch` 字典中注册它。
    3.  现在你就可以在 `config.yaml` 中通过 `type: "my_new_feature"` 来调用它了。

---

希望这份文档能帮助您快速、高效地使用和扩展 CRAFT 框架！
