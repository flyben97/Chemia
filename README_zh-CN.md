<div align="center">
  <img src="images/chemia_logo.png" alt="Chemia Logo" width="300"/>
  <p>一个全面的机器学习框架，用于分子性质预测、化学反应优化与材料设计。支持多种分子表示方法、预训练模型及多种机器学习算法。</p>
  <a href="https://github.com/flyben97/Chemia">github.com/flyben97/Chemia</a>
  <br/>
  <sub><a href="README.md">English</a> | <a href="README_zh-CN.md">中文</a></sub>
</div>

## ✨ 核心特性

### 🧬 分子表征

- **预训练模型**（自动下载与缓存）：UniMol ⭐、UniMolv2 (84M~1.1B)、ChemBERTa、MolT5、ChemRoBERTa、RoBERTa
- **传统特征**：Morgan/MACCS/RDKit/AtomPair/Torsion 指纹、RDKit 2D 描述符

### 🤖 支持的模型

- **树模型**：XGBoost、LightGBM、CatBoost、Random Forest
- **线性模型**：Ridge、Lasso、ElasticNet
- **神经网络**：TabNet、ANN
- **其他**：SVM、KNN、Gaussian Process

### 🎯 多目标预测

- 一个 YAML 配置 `target_cols: [col1, col2, ...]` 同时训练所有目标
- 特征仅计算一次，每个目标独立训练，自动汇总 `combined_results.csv`

### 💾 智能 Embedding 缓存

- UniMol/ChemBERTa 等 embedding 在批次内自动缓存，无需重复计算
- 运行结束后自动清理，不占用磁盘空间

### 🔄 断点续跑

- `--resume` 跳过崩溃后已完成的任务
- 进度原子写入 `batch_progress.json`

### 🐳 Docker 支持

- `docker-compose run` — 一键启动，无需手动配置环境

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/flyben97/Chemia.git && cd Chemia

# Conda
conda create -n chemia python=3.10 && conda activate chemia
pip install -r requirements.txt

# Docker（无需配置 Python 环境）
docker-compose build
```

### 2. 准备数据

```csv
SMILES,target
CCO,0.5
c1ccccc1,0.8
```

### 3. 配置文件

**单目标：**
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
  random_state: 42
```

**多目标（一个 YAML 预测所有列）：**
```yaml
data:
  single_file_config:
    main_file_path: data/sider.csv
    smiles_col: smiles
    target_cols:
      - Hepatobiliary disorders
      - Cardiac disorders
      # ... 全部 27 个副作用
```

### 4. 运行

```bash
# 单目标
python scripts/run_training_only.py --config config.yaml

# 多目标
python scripts/run_training_only.py --config config_multi.yaml

# 批量实验 + 3 seeds 误差棒
python scripts/batch_run_all.py --datasets BACE BBBP FreeSolv --trials 50

# 断点续跑
python scripts/batch_run_all.py --resume --retry-failed

# Docker
docker-compose run chemia scripts/run_training_only.py --config examples/configs/bace_classification.yaml
```

### 5. 预测

```bash
python scripts/run_prediction_standalone.py \
  --run-dir output/experiment_dir --model-name xgboost \
  --input-file data/test.csv --output-file predictions.csv
```

```python
from scripts.prediction_api import load_model, predict_single
predictor = load_model('output/experiment_dir', 'xgboost')
result = predict_single(predictor, {'SMILES': 'CCO'})
```

---

## 📖 分子表征

| 模型 | 维度 | 速度 | 说明 |
|------|------|------|------|
| UniMolv2 (310M) | 1024 | ⭐⭐⭐⭐ | ⭐ 推荐 |
| UniMolv2 (164M) | 768 | ⭐⭐⭐⭐ | 快速 |
| UniMolv2 (570M) | 1536 | ⭐⭐⭐ | 高性能 |
| UniMol | 512 | ⭐⭐⭐⭐ | 经典 |
| ChemBERTa | 768 | ⭐⭐⭐⭐ | 专业 |

```yaml
features:
  generators:
    - name: transformer_embedding
      config: {model_type: unimolv2_310m}
    - name: rdkit_fingerprint
      config: {type: morgan, nBits: 2048}
```

---

## 📊 数据分割

| 模式 | 描述 |
|------|------|
| `hold_out` | 随机训练/测试划分 |
| `scaffold_split` | 按 Bemis-Murcko 骨架分组划分（评估更严格） |
| `cross_validation` | K-折交叉验证 |
| `pre_split` | 预定义的 train/val/test 文件 |

---

## 📜 可用脚本

| 脚本 | 功能 |
|------|------|
| `run_training_only.py` | 模型训练 + 超参数优化 |
| `run_full_workflow.py` | 训练 → 选最优 → 贝叶斯优化 |
| `run_prediction_standalone.py` | 使用已训练模型预测 |
| `run_shap_analysis.py` | SHAP 模型可解释性 |
| `batch_run_all.py` | 批量实验 + 误差棒 + 断点续跑 |
| `prediction_api.py` | Python API |

---

## 📁 项目结构

```
Chemia/
├── core/                 # 核心管线
├── utils/                # 工具 (embedding 缓存、特征生成、unimol 等)
├── models/               # 模型定义
├── optimizers/           # Optuna 超参数优化
├── scripts/              # CLI 入口
│   ├── run_training_only.py
│   └── batch_run_all.py
├── examples/configs/     # YAML 配置示例
│   ├── sider/            # 27 个 SIDER 副作用目标
│   ├── tox21/            # 12 个 Tox21 毒性目标
│   └── clintox/          # 2 个 ClinTox 临床毒性目标
├── data/                 # 示例数据集
├── references/           # API/配置参考文档
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🐳 Docker

```bash
docker-compose build
docker-compose run chemia scripts/run_training_only.py --config examples/configs/bace_classification.yaml
docker-compose --profile cpu run chemia-cpu scripts/run_training_only.py --config ...
```

---

## 🔍 常见问题

**Q: 如何选择模型？** A: XGBoost/LightGBM 通用，ANN 精度更高但需调参，TabNet 需要较大数据集

**Q: GPU 显存不足？** A: 减小 batch_size，或使用 `CUDA_VISIBLE_DEVICES=""` 强制 CPU

**Q: 如何处理多目标？** A: 使用 `target_cols: [...]`

**Q: 训练中断了？** A: `python scripts/batch_run_all.py --resume`

---

## 📄 许可证

MIT License
