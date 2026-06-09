<div align="center">
  <img src="images/chemia_logo.png" alt="Chemia Logo" width="300"/>
  <p><strong>一个专为化学性质与反应预测与优化而生的综合机器学习框架</strong></p>
  <p>
    <a href="https://github.com/flyben97/Chemia">github.com/flyben97/Chemia</a>
  </p>
  <p>
    <a href="README.md">English</a> | <strong>简体中文</strong>
  </p>
</div>

---

**Chemia** 是一个面向化学方向机器学习研究人员的强大工具。通过简单的 YAML 配置文件，实现从数据处理、特征工程到模型训练、超参数优化的"一站式"工作流，让您专注于化学问题本身。

## 📚 目录

*   [**第一章：初识 Chemia**](#第一章初识-chemia)
    *   [1.1 核心特性](#11-核心特性)
    *   [1.2 项目结构](#12-项目结构)
*   [**第二章：五分钟快速上手**](#第二章五分钟快速上手)
*   [**第三章：安装与部署**](#第三章安装与部署)
*   [**第四章：深入使用指南**](#第四章深入使用指南)
*   [**第五章：配置文件详解**](#第五章配置文件详解)
*   [**第六章：核心能力剖析**](#第六章核心能力剖析)
*   [**附录：许可证**](#附录许可证)

---

## 第一章：初识 Chemia

### 1.1 核心特性

*   **🤖 多种机器学习算法**：内置支持 XGBoost、LightGBM、CatBoost、随机森林、高斯过程回归、SVM，以及神经网络 TabNet/ANN。
*   **✨ 自动特征工程**：从 SMILES 自动生成分子指纹（Morgan、MACCS）、RDKit 描述符，并支持 UniMolv2、ChemBERTa、MolT5 等预训练模型嵌入。
*   **🧩 灵活的数据分割**：支持 Hold-Out、骨架划分（Scaffold Split）、K-折交叉验证和预分割数据。
*   **🔎 超参数优化**：深度集成 Optuna，实现全自动、高效的超参数搜索。
*   **💾 智能缓存**：Embedding 自动缓存复用，运行结束后自动清理。
*   **🔄 断点续跑**：批量任务支持 `--resume`，崩溃后跳过已完成项。
*   **🐳 Docker 支持**：一行 `docker-compose run` 即可运行。

### 1.2 项目结构

```
Chemia/
├── core/                     # 核心框架 (config_loader, run_manager, trainer_setup)
├── utils/                    # 工具 (embedding_cache, feature_generator, shap_analyzer, ...)
├── models/                   # 模型定义 (sklearn_models, ann, transformer_models)
├── optimizers/               # Optuna 超参数优化
├── optimization/             # 贝叶斯反应优化
├── scripts/                  # CLI 入口
│   ├── run_training_only.py
│   ├── batch_run_all.py      # 批量实验 + 断点续跑
│   └── run_prediction_standalone.py
├── examples/configs/         # YAML 配置示例
├── data/                     # 示例数据集
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 第二章：五分钟快速上手

### 2.1 准备数据

将数据集（CSV 格式）放置在 `data/` 目录下：

```csv
SMILES,target
CCO,0.5
c1ccccc1,0.8
```

### 2.2 配置实验

创建 YAML 配置文件：

```yaml
data:
  source_mode: single_file
  single_file_config:
    main_file_path: data/molecules.csv
    smiles_col: SMILES
    target_col: target

task_type: regression

features:
  generators:
    - name: transformer_embedding
      config:
        model_type: unimolv2_310m
    - name: rdkit_fingerprint
      config:
        type: morgan
        nBits: 2048
        radius: 2

split_config:
  split_mode: hold_out
  hold_out:
    test_size: 0.2
    random_state: 42

training:
  models_to_run: [xgboost, ann, tabnet, lgbm, catboost, randomforest]
  n_trials: 50
```

### 2.3 启动训练

```bash
python scripts/run_training_only.py --config config.yaml
```

结果将自动保存在 `output/` 目录下。

---

## 第三章：安装与部署

### 3.1 环境安装

```bash
git clone https://github.com/flyben97/Chemia.git && cd Chemia

# Conda
conda create -n chemia python=3.10 && conda activate chemia
pip install -r requirements.txt

# Docker
docker-compose build
```

### 3.2 运行脚本

| 脚本 | 功能 |
|------|------|
| `run_training_only.py` | 模型训练 + 超参数优化 |
| `run_full_workflow.py` | 训练 → 选最优 → 贝叶斯优化 |
| `run_prediction_standalone.py` | 使用已训练模型预测 |
| `run_shap_analysis.py` | SHAP 模型可解释性 |
| `batch_run_all.py` | 批量实验 + 误差棒 + 断点续跑 |

---

## 第四章：深入使用指南

### 单目标训练

```bash
python scripts/run_training_only.py --config examples/configs/freesolv_regression.yaml
```

### 多目标预测

一个 YAML 同时预测多个目标：

```yaml
data:
  single_file_config:
    main_file_path: data/sider.csv
    smiles_col: smiles
    target_cols:
      - Hepatobiliary disorders
      - Cardiac disorders
```

### 批量实验

```bash
python scripts/batch_run_all.py --datasets BACE BBBP FreeSolv --trials 50
python scripts/batch_run_all.py --resume --retry-failed
```

### 预测

```bash
python scripts/run_prediction_standalone.py \
  --run-dir output/experiment_dir --model-name xgboost \
  --input-file data/test.csv --output-file predictions.csv
```

---

## 第五章：配置文件详解

### 预训练模型

| 模型 | 维度 | 速度 | 推荐 |
|------|------|------|------|
| UniMolv2 (310M) | 1024 | ⭐⭐⭐⭐ | ⭐ 推荐 |
| UniMolv2 (164M) | 768 | ⭐⭐⭐⭐ | 快速 |
| UniMolv2 (570M) | 1536 | ⭐⭐⭐ | 高性能 |
| UniMol | 512 | ⭐⭐⭐⭐ | 经典 |
| ChemBERTa | 768 | ⭐⭐⭐⭐ | 专业 |

### 数据分割

| 模式 | 描述 |
|------|------|
| `hold_out` | 随机训练/测试划分 |
| `scaffold_split` | 按 Bemis-Murcko 骨架分组（更严格） |
| `cross_validation` | K-折交叉验证 |
| `pre_split` | 预定义 train/val/test 文件 |

---

## 第六章：核心能力剖析

### 支持的模型

*   **树模型**：XGBoost、LightGBM、CatBoost、Random Forest
*   **线性模型**：Ridge、Lasso、ElasticNet
*   **神经网络**：TabNet、ANN
*   **其他**：SVM、KNN、Gaussian Process

### 分子表征

*   **预训练模型**：UniMolv2、UniMol、ChemBERTa、MolT5
*   **传统指纹**：Morgan、MACCS、RDKit、AtomPair、Torsion
*   **RDKit 描述符**：200+ 种物理化学性质

---

## 附录：许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

---

<div align="center">
  <strong>祝您化学任务建模愉快！ 🧪✨</strong>
</div>
