---

<div align="center">
  <img src="images/crane.png" alt="CRAFT Logo" width="200"/>
  <h1>INTERNCRANE</h1>
  <p><strong>一个专为化学性质与反应预测与优化而生的综合机器学习框架</strong></p>
  <p>
    <a href="https://img.shields.io/badge/python-3.8+-blue.svg"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python"></a>
    <a href="https://img.shields.io/badge/license-MIT-green.svg"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
    <a href="https://img.shields.io/badge/status-active-brightgreen.svg"><img src="https://img.shields.io/badge/status-active-brightgreen.svg" alt="Status"></a>
  </p>
  <p>
    <a href="README.md">English</a> | <strong>简体中文</strong>
  </p>
</div>


---

**CRANE** — **C**hemical **R**epresentation and **A**nalysis using **N**eural and **E**nsemble models

**CRANE** 是一个面向化学方向机器学习研究人员的强大工具。它通过简单的 YAML 配置文件，实现了从数据处理、特征工程到模型训练、超参数优化和贝叶斯优化的“一站式”工作流，让您可以专注于化学问题本身，而非繁琐的工程细节。

## 📚 目录

*   [**第一章：初识 CRANE**](#第一章初识-crane)
    *   [1.1 核心特性](#11-核心特性)
    *   [1.2 项目结构](#12-项目结构)
*   [**第二章：五分钟快速上手**](#第二章五分钟快速上手)
    *   [2.1 准备数据](#21-准备数据)
    *   [2.2 配置实验](#22-配置实验)
    *   [2.3 启动训练](#23-启动训练)
*   [**第三章：安装与部署**](#第三章安装与部署)
    *   [3.1 环境安装](#31-环境安装)
    *   [3.2 运行脚本说明](#32-运行脚本说明)
*   [**第四章：深入使用指南**](#第四章深入使用指南)
    *   [4.1 训练模式详解](#41-训练模式详解)
    *   [4.2 贝叶斯优化](#42-贝叶斯优化)
    *   [4.3 端到端工作流](#43-端到端工作流)
*   [**第五章：配置文件详解**](#第五章配置文件详解)
    *   [5.1 训练配置模板](#51-训练配置模板)
    *   [5.2 优化配置模板](#52-优化配置模板)
*   [**第六章：核心能力剖析**](#第六章核心能力剖析)
    *   [6.1 支持的算法库](#61-支持的算法库)
    *   [6.2 自动化特征工程](#62-自动化特征工程)
    *   [6.3 数据分割策略](#63-数据分割策略)
    *   [6.4 数据格式要求](#64-数据格式要求)
*   [**第七章：高级技巧与编程接口**](#第七章高级技巧与编程接口)
    *   [7.1 自定义配置](#71-自定义配置)
    *   [7.2 程序化使用](#72-程序化使用)
*   [**第八章：结果解读与输出**](#第八章结果解读与输出)
*   [**附录：许可证**](#附录许可证)

---

## 第一章：初识 CRANE

### 1.1 核心特性

*   **🤖 多种机器学习算法**：内置支持 15+ 种经典算法，如 `XGBoost`, `LightGBM`, `CatBoost`, `随机森林`, `高斯过程回归`，以及 `神经网络` 和 `图神经网络`。
*   **✨ 自动特征工程**：从 SMILES 自动生成分子指纹（`Morgan`, `MACCS`）、`RDKit` 描述符，并支持 `UniMol` 等预训练模型嵌入。
*   **🧩 灵活的数据分割**：支持训练/测试分割、训练/验证/测试分割和 k-折交叉验证，满足不同评估需求。
*   **🔎 超参数优化**：深度集成 `Optuna`，实现全自动、高效的超参数搜索。
*   **🎯 贝叶斯优化**：利用训练好的代理模型，智能探索并寻找最优反应条件。
*   **⚙️ 端到端工作流程**：提供从模型训练到反应优化的完整、自动化的流水线。
*   **📜 丰富的配置选项**：基于 YAML 的配置系统，清晰直观，提供极高的自定义灵活性。

### 1.2 项目结构

```
craft/
├── 📂 core/                    # 核心框架组件
│   ├── run_manager.py      # 实验管理
│   ├── config_loader.py    # 配置加载
│   └── trainer_setup.py    # 模型训练设置
├── 📂 models/                  # 模型实现
│   ├── sklearn_models.py   # 传统机器学习模型
│   ├── ann.py              # 神经网络
│   └── gnn_models.py       # 图神经网络
├── 📂 optimization/            # 贝叶斯优化
│   ├── optimizer.py        # 主优化引擎
│   └── space_loader.py     # 搜索空间管理
├── 📂 utils/                   # 工具函数
├── 📂 examples/                # 示例配置和脚本
│   └── configs/            # 配置文件
├── 📂 data/                    # 数据目录
└── 📂 output/                  # 结果和训练模型
```

---

## 第二章：五分钟快速上手

让我们通过一个简单的例子，快速体验 CRANE 的强大功能。

### 2.1 准备数据

首先，将您的数据集（例如 `CPA.csv`）放置在 `data/` 目录下。文件格式如下：

```csv
# data/CPA.csv
Catalyst,Imine,Thiol,Output
O=P1(O)OC2=C(C3=CC=CC=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=CC=C6)=CC7=C5C=CC=C7,O=C(C1=CC=CC=C1)/N=C/C2=CC=C(C(F)(F)F)C=C2,CCS,0.501758501
O=P1(O)OC2=C(C3=C(F)C=C(OC)C=C3F)C=C4C(C=CC=C4)=[C@]2[C@]5=C(O1)C(C6=C(F)C=C(OC)C=C6F)=CC7=C5C=CC=C7,O=C(C1=CC=CC=C1)/N=C/C2=CC=C(C(F)(F)F)C=C2,CCS,1.074990526
...
```

### 2.2 配置实验

接着，创建一个 YAML 配置文件（或使用已有模板），定义您的实验。

```yaml
# 示例配置文件：config_quick_start.yaml

# ===================================================================
# 1. 实验基本信息
# ===================================================================
experiment_name: "CPA_Quick_Start" # 实验名称，所有结果将保存在 output/CPA_Quick_Start/
task_type: "regression"            # 任务类型: 'regression' 或 'classification'

# ===================================================================
# 2. 数据源配置
# ===================================================================
data:
  source_mode: "single_file"
  single_file_config:
    main_file_path: "data/CPA.csv"
    smiles_col: ["Catalyst", "Imine", "Thiol"] # 定义包含SMILES的列
    target_col: "Output"                     # 定义目标值列
    precomputed_features: null               # 本例中无预计算特征

# ===================================================================
# 3. 特征工程策略
# ===================================================================
features:
  per_smiles_col_generators:
    Catalyst: 
      - type: "unimol"       # 使用 UniMol v2 生成特征
        model_version: "v2"
        model_size: "310m"
      - type: "morgan"
        nbits: 2048
        radius: 2
    Imine: 
      - type: "rdkit_descriptors"
    Thiol: 
      - type: "rdkit_descriptors"
  scaling: true # 对特征进行标准化

# ===================================================================
# 4. 训练与评估配置
# ===================================================================
training:
  models_to_run: # 选择要训练的模型
    - "lgbm"
    - "catboost"
    - "xgb"
  n_trials: 30 # Optuna超参数搜索次数

split_mode: "train_valid_test" # 数据集划分模式
split_config:
  train_valid_test:
    valid_size: 0.1
    test_size: 0.1
    random_state: 42

evaluation:
  primary_metric: "r2" # 用于模型选择的主要指标
  additional_metrics: ["rmse", "mae"]

# ===================================================================
# 5. 其他设置 (可保持默认)
# ===================================================================
output:
  save_predictions: true
  save_feature_importance: true

computational:
  n_jobs: -1 # 使用所有CPU核心
  random_state: 42
```

> **💡 提示：** GNN 等算法目前仍在测试中，因此暂时无需安装 CUDA 版本的 PyTorch。

### 2.3 启动训练

配置完成后，在终端中运行以下命令：

```bash
python run_training_only.py --config examples/configs/quick_start.yaml
```

恭喜！您已经成功启动了一次完整的机器学习训练流程。结果、模型和分析报告将自动保存在 `output/CPA_Quick_Start/` 目录下。

---

## 第三章：安装与部署

### 3.1 环境安装

1.  **克隆仓库**：
    ```bash
    git clone https://github.com/flyben97/interncrane.git
    cd crane
    ```

2.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
    ```

### 3.2 运行脚本说明

CRANE 提供了三个核心运行脚本，以适应不同需求：

| 脚本                       | 功能                                 | 推荐场景                                 |
| :------------------------- | :----------------------------------- | :--------------------------------------- |
| **`run_training_only.py`** | 专注于模型训练、评估和比较。         | **模型开发**、算法对比、初学者入门。     |
| **`run_optimization.py`**  | 使用预训练模型进行独立的贝叶斯优化。 | 已有可靠模型，需要进行**条件优化**。     |
| **`run_full_workflow.py`** | 自动化的端到端流程（训练 → 优化）。  | 需要全自动处理的**生产环境**或完整实验。 |

---

## 第四章：深入使用指南

### 4.1 训练模式详解

使用 `run_training_only.py` 配合不同的配置文件，可以实现多种训练模式。

*   **基础回归训练**：
    ```bash
    python run_training_only.py --config examples/configs/regression_training_simple.yaml
    ```
*   **基础分类训练**：
    ```bash
    python run_training_only.py --config examples/configs/classification_training_simple.yaml
    ```
*   **5-折交叉验证训练** (更稳健的评估):
    ```bash
    python run_training_only.py --config examples/configs/regression_training_kfold.yaml
    ```

### 4.2 贝叶斯优化

如果您已经训练并保存了一个模型，可以使用 `run_optimization.py` 来寻找最佳反应条件。

```bash
python run_optimization.py --config examples/configs/bayesian_optimization_only.yaml
```

### 4.3 端到端工作流程

对于全自动化需求，`run_full_workflow.py` 会先训练模型，然后自动选择最优模型进行贝叶斯优化。

```bash
python run_full_workflow.py --config examples/configs/end_to_end_workflow.yaml
```

---

## 第五章：配置文件详解

CRANE 的核心在于其强大的 YAML 配置系统。`examples/configs/` 目录下提供了丰富的模板。

### 5.1 训练配置模板

| 配置文件                              | 描述                                 |
| :------------------------------------ | :----------------------------------- |
| `quick_start.yaml`                    | 最小化配置，用于快速测试和初次体验。 |
| `regression_training_simple.yaml`     | 标准回归任务的基础配置。             |
| `regression_training_kfold.yaml`      | 使用 k-折交叉验证的回归训练。        |
| `regression_training_split.yaml`      | 使用训练/验证/测试集划分的回归训练。 |
| `classification_training_simple.yaml` | 标准分类任务的基础配置。             |
| `classification_training_kfold.yaml`  | 使用 k-折交叉验证的分类训练。        |
| `training_with_features.yaml`         | 包含丰富特征工程选项的配置。         |
| `gnn_training.yaml`                   | 专门用于图神经网络训练的配置。       |

### 5.2 优化配置模板

| 配置文件                          | 描述                                   |
| :-------------------------------- | :------------------------------------- |
| `bayesian_optimization_only.yaml` | 独立运行贝叶斯优化，需指定预训练模型。 |
| `end_to_end_workflow.yaml`        | 完整的训练+优化流水线配置。            |

---

## 第六章：核心能力剖析

### 6.1 支持的算法库

*   **梯度提升方法**：XGBoost, LightGBM, CatBoost, 直方图梯度提升
*   **树集成方法**：随机森林, 极端随机树, AdaBoost
*   **线性模型**：岭回归, LASSO, ElasticNet, 贝叶斯岭回归
*   **核方法**：高斯过程回归, 核岭回归, 支持向量回归
*   **基于实例的方法**：k-近邻
*   **神经网络**：基于 PyTorch 的全连接神经网络 (ANN)
*   **图神经网络**：GCN, GAT, MPNN, 图 Transformer 等

### 6.2 自动化特征工程

CRANE 自动从 SMILES 字符串生成高质量分子特征：

*   **Morgan 指纹**：可定制半径和位数的圆形指纹。
*   **MACCS 密钥**：166 位结构密钥。
*   **RDKit 描述符**：200+ 种物理化学性质和拓扑描述符。
*   **预训练模型嵌入**：支持 `UniMol`, `ChemBerta`, `Molt5` 等模型的预计算特征。

### 6.3 数据分割策略

1.  **训练/测试分割**：简单按比例划分，可自定义。
2.  **训练/验证/测试分割**：用于模型开发和调优，可自定义比例。
3.  **K-折交叉验证**：提供更稳健的模型性能评估，折数可自定义。

### 6.4 数据格式要求

您的输入数据应为 CSV 格式，包含 SMILES 字符串列和目标值列。其他数值型或类别型特征也能被自动处理。

```csv
Catalyst,Reactant1,Reactant2,Temperature,Solvent,yield
CC(C)P(c1ccccc1)c1ccccc1,CC(=O)c1ccccc1,NCc1ccccc1,80,toluene,95.2
CCc1ccc(P(CCc2ccccc2)CCc2ccccc2)cc1,CC(=O)c1ccccc1,NCc1ccccc1,60,THF,87.5
...
```

---

## 第七章：高级技巧与编程接口

### 7.1 自定义配置

您可以基于任意示例模板创建自己的 `my_config.yaml` 文件，以满足特定的实验需求。

```yaml
experiment_name: "我的新实验"
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
# ... 其他配置
```

### 7.2 程序化使用

除了命令行，您也可以在 Python 脚本中调用 CRANE 的核心功能。

```python
from core.run_manager import start_experiment_run
from core.config_loader import load_config

# 1. 加载配置文件
config = load_config("my_config.yaml")

# 2. 运行实验
results_summary = start_experiment_run(config)

# 3. 访问和分析结果
best_model_info = max(results_summary['results'], key=lambda x: x['test_r2'])
print(f"最佳模型: {best_model_info['model_name']} (R² = {best_model_info['test_r2']:.4f})")
```

---

## 第八章：结果解读与输出

每次实验运行后，CRANE 会在 `output/` 目录下生成一个结构清晰的文件夹，包含：

*   **📈 综合报告**：一个 `_model_comparison.csv` 文件，汇总所有模型的性能指标。
*   **📦 训练好的模型**：序列化的模型文件（如 `.pkl` 或 `.pt`），以及相关的 scaler 和特征名称。
*   **📝 预测结果**：包含验证集和测试集预测值的 CSV 文件。
*   **📊 可视化图表**：特征重要性图、学习曲线等。
*   **🎯 优化结果**：贝叶斯优化找到的最佳条件及其预测值。
*   **⚙️ 实验记录**：保存的超参数和配置文件副本，确保实验可复现。

---

## 附录：许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

---

<div align="center">
  <strong>祝您化学任务建模愉快！ 🧪✨</strong>
</div>