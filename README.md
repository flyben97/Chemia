

<div align="center">
  <img src="images/chemia_logo.png" alt="Chemia Logo" width="300"/>
  <p>A comprehensive ML framework for molecular property prediction, reaction optimization, and materials design. Supports multiple molecular representations, pretrained models, and diverse ML algorithms.</p>
  <a href="https://github.com/flyben97/Chemia">github.com/flyben97/Chemia</a>
  <br/>
  <sub><a href="README.md">English</a> | <a href="README_zh-CN.md">中文</a></sub>
</div>

## ✨ Key Features

### 🧬 Molecular Representations

- **Pretrained Models** (auto-download & cache): UniMol ⭐, UniMolv2 (84M~1.1B), ChemBERTa, MolT5, ChemRoBERTa, RoBERTa
- **Traditional Features**: Morgan/MACCS/RDKit/AtomPair/Torsion fingerprints, RDKit 2D descriptors

### 🤖 Supported Models

- **Tree**: XGBoost, LightGBM, CatBoost, Random Forest
- **Linear**: Ridge, Lasso, ElasticNet
- **Neural**: TabNet, ANN
- **Others**: SVM, KNN, Gaussian Process

### 🎯 Multi-Target Prediction

- One YAML with `target_cols: [col1, col2, ...]` to train all targets in a single run
- Features computed once, per-target training, aggregated `combined_results.csv`

### 💾 Smart Embedding Cache

- UniMol/ChemBERTa embeddings auto-cached within a batch run, no recomputation
- Auto-cleaned after completion, zero disk waste

### 🔄 Checkpoint/Resume

- `--resume` skips completed tasks after a crash
- Atomic progress writes to `batch_results/batch_progress.json` by default

### 🐳 Docker

- `docker-compose run` — zero-config reproducible environment

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/flyben97/Chemia.git && cd Chemia

# Conda
conda create -n chemia python=3.10 && conda activate chemia
pip install -r requirements.txt

# Docker (no Python setup needed)
docker-compose build
```

### 2. Prepare Data

```csv
SMILES,target
CCO,0.5
c1ccccc1,0.8
```

### 3. Config

**Single target:**
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

**Multi-target (one YAML for all columns):**
```yaml
data:
  single_file_config:
    main_file_path: data/sider.csv
    smiles_col: smiles
    target_cols:
      - Hepatobiliary disorders
      - Cardiac disorders
      # ... all 27 side effects
```

### 4. Run

```bash
# Single target
python scripts/run_training_only.py --config config.yaml

# Multi-target
python scripts/run_training_only.py --config config_multi.yaml

# Batch with 3 seeds + error bars
python scripts/batch_run_all.py --datasets BACE BBBP FreeSolv --trials 50

# Resume from checkpoint
python scripts/batch_run_all.py --resume --retry-failed

# Docker
docker-compose run chemia scripts/run_training_only.py --config examples/configs/bace_classification.yaml
```

### 5. Predict

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

## 📖 Molecular Representations

| Model | Dims | Speed | Note |
|-------|------|-------|------|
| UniMolv2 (310M) | 1024 | ⭐⭐⭐⭐ | ⭐ Recommended |
| UniMolv2 (164M) | 768 | ⭐⭐⭐⭐ | Fast |
| UniMolv2 (570M) | 1536 | ⭐⭐⭐ | High performance |
| UniMol | 512 | ⭐⭐⭐⭐ | Classic |
| ChemBERTa | 768 | ⭐⭐⭐⭐ | Specialized |

```yaml
features:
  generators:
    - name: transformer_embedding
      config: {model_type: unimolv2_310m}
    - name: rdkit_fingerprint
      config: {type: morgan, nBits: 2048}
```

---

## 📊 Data Splits

| Mode | Description |
|------|-------------|
| `hold_out` | Random train/test split |
| `scaffold_split` | Split by Bemis-Murcko scaffold (stricter) |
| `cross_validation` | K-fold CV |
| `pre_split` | Pre-defined train/val/test files |

---

## 📜 Scripts

| Script | Purpose |
|--------|---------|
| `run_training_only.py` | Train + HPO |
| `run_full_workflow.py` | Train → Select → Bayesian optimize |
| `run_prediction_standalone.py` | Predict with trained models |
| `run_shap_analysis.py` | SHAP interpretability |
| `batch_run_all.py` | Batch experiments + error bars + checkpoint |
| `prediction_api.py` | Python API |

---

## 📁 Project Structure

```
Chemia/
├── core/                 # Core pipeline
├── utils/                # Embedding cache, feature gen, unimol, ...
├── models/               # Model definitions
├── optimizers/           # Optuna HPO
├── scripts/              # CLI entry points
│   ├── run_training_only.py
│   └── batch_run_all.py
├── examples/configs/     # YAML examples
│   ├── sider/            # 27 SIDER targets
│   ├── tox21/            # 12 Tox21 targets
│   └── clintox/          # 2 ClinTox targets
├── data/                 # Example datasets
├── references/           # API/Config reference docs
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

## 🔍 FAQ

**Q: Which model?** A: XGBoost/LightGBM general, ANN for max accuracy, TabNet needs larger datasets

**Q: GPU OOM?** A: Reduce batch_size, or `CUDA_VISIBLE_DEVICES=""`

**Q: Multiple targets?** A: Use `target_cols: [...]`

**Q: Training interrupted?** A: `python scripts/batch_run_all.py --resume`

---

## 📄 License

MIT License
