![CRAFT Logo](images/craft.png) ![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

# CRAFT: Chemical Reaction Analysis and Feature-based Training 

CRAFT is a comprehensive machine learning framework designed for chemical reaction prediction and optimization. It combines traditional ML algorithms, neural networks, and graph neural networks with Bayesian optimization to predict reaction outcomes and find optimal reaction conditions.

[English](README.md) | [简体中文](README_zh-CN.md)

## 📁 Project Structure

```
craft/
├── core/                    # Core framework components
│   ├── run_manager.py      # Experiment management
│   ├── config_loader.py    # Configuration loading
│   └── trainer_setup.py    # Model training setup
├── models/                  # Model implementations
│   ├── sklearn_models.py   # Traditional ML models
│   ├── ann.py              # Neural networks
│   └── gnn_models.py       # Graph neural networks
├── optimization/            # Bayesian optimization
│   ├── optimizer.py        # Main optimization engine
│   └── space_loader.py     # Search space management
├── utils/                   # Utility functions
├── examples/                # Example configurations and scripts
│   └── configs/            # Configuration files
├── data/                    # Data directory
└── output/                  # Results and trained models
```

## 📋 Quick Start

### 🔧 Environment Setup

CRAFT provides multiple ways to set up your environment. Choose the method that works best for you:

#### 🐍 Option 1: Setup with pip (Recommended)

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/craft.git
cd craft
```

2. **Create and activate virtual environment**:
```bash
python3 -m venv craft
source craft/bin/activate  # On Windows: craft\Scripts\activate
```

3. **Install PyTorch (required for neural networks)**:
```bash
# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8 (adjust based on your CUDA version):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric:
pip install torch_geometric torch_cluster torch_scatter torch_sparse
```

4. **Install other dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 🐍 Option 2: Conda Environment

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/craft.git
cd craft
```

2. **Install PyTorch first**:
```bash
# For CUDA (adjust version as needed):
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install PyTorch Geometric:
conda install pyg -c pyg
```

3. **Create conda environment**:
```bash
conda env create -f environment.yml
conda activate craft
```

#### 🔍 Verify Installation

Test your installation:
```bash
python -c "
import numpy, pandas, sklearn, rdkit, optuna, rich
print('✅ Core packages imported successfully')
try:
    import torch, torch_geometric
    print('✅ PyTorch and PyTorch Geometric available')
except ImportError:
    print('⚠️  PyTorch/PyTorch Geometric not found - please install manually')
"
```

### 📋 System Requirements

- **Python**: 3.8+ (3.9 recommended)
- **Operating System**: Linux, macOS, Windows (WSL recommended)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **GPU**: Optional (CUDA-compatible GPU for faster neural network training)

### 🧪 Prepare Your Data

Place your reaction data in CSV format in the `data/` directory:

```bash
mkdir -p data
# Copy your reaction data CSV file to data/
```

**Note**: Ensure your virtual environment is activated before running any CRAFT commands:
```bash
# For pip installation:
source craft/bin/activate

# For conda installation:
conda activate craft
```

### Basic Usage

#### 1. Quick Training (Recommended for beginners)
```bash
python run_training_only.py --config examples/configs/quick_start.yaml
```

#### 2. Full Model Training
```bash
# Simple regression training
python run_training_only.py --config examples/configs/regression_training_simple.yaml

# Classification training
python run_training_only.py --config examples/configs/classification_training_simple.yaml

# Training with 5-fold cross-validation
python run_training_only.py --config examples/configs/regression_training_kfold.yaml
```

#### 3. Bayesian Optimization (using pre-trained model)
```bash
python run_optimization.py --config examples/configs/bayesian_optimization_only.yaml
```

#### 4. End-to-End Workflow (Training + Optimization)
```bash
python run_full_workflow.py --config examples/configs/end_to_end_workflow.yaml
```

## ⚙️ Environment Configuration Files

CRAFT includes several configuration files to help you set up your environment:

### Environment Files

| File | Purpose | Use Case |
|------|---------|----------|
| `requirements.txt` | Python package dependencies | Standard pip installation |
| `environment.yml` | Conda environment specification | Conda users, reproducible environments |

### Environment Management Tips

For smooth environment management:

- **Virtual Environment**: Always use a dedicated virtual environment named `craft`
- **PyTorch Installation**: Install PyTorch separately based on your system configuration
- **Dependency Isolation**: Keep CRAFT dependencies separate from your system Python
- **Version Compatibility**: Use the recommended package versions for best results

Quick setup checklist:
```bash
# 1. Create environment
python3 -m venv craft  # or: conda env create -f environment.yml

# 2. Activate environment  
source craft/bin/activate  # or: conda activate craft

# 3. Install PyTorch (see installation options above)

# 4. Install other dependencies
pip install -r requirements.txt
```

## 🔧 Configuration Files

CRAFT provides various pre-configured YAML files for different scenarios:

### Training Configurations

| Configuration | Description | Use Case |
|--------------|-------------|----------|
| `quick_start.yaml` | Minimal setup for testing | First-time users, quick experiments |
| `regression_training_simple.yaml` | Basic regression training | Standard regression tasks |
| `regression_training_kfold.yaml` | 5-fold cross-validation | Robust model evaluation |
| `regression_training_split.yaml` | Train/validation/test split | Model development |
| `classification_training_simple.yaml` | Basic classification | Classification tasks |
| `classification_training_kfold.yaml` | Classification with CV | Robust classification |
| `training_with_features.yaml` | Rich feature engineering | Complex molecular datasets |
| `training_without_features.yaml` | Minimal features | Simple datasets |
| `gnn_training.yaml` | Graph neural networks | Advanced molecular modeling |

### Optimization Configurations

| Configuration | Description | Use Case |
|--------------|-------------|----------|
| `bayesian_optimization_only.yaml` | Standalone optimization | Using pre-trained models |
| `end_to_end_workflow.yaml` | Complete pipeline | Full automation |

## 📊 Supported Algorithms

### Traditional Machine Learning
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost, Histogram Gradient Boosting
- **Tree Ensembles**: Random Forest, Extra Trees, AdaBoost
- **Linear Models**: Ridge, LASSO, ElasticNet, Bayesian Ridge
- **Kernel Methods**: Gaussian Process Regression, Kernel Ridge Regression, SVR
- **Instance-based**: k-Nearest Neighbors
- **Linear**: Stochastic Gradient Descent

### Neural Networks
- **Traditional ANN**: PyTorch-based Artificial Neural Networks
- **Graph Neural Networks**: GCN, GAT, MPNN, Graph Transformer, Ensemble GNN

## 🧬 Feature Engineering

CRAFT automatically generates molecular features from SMILES strings:

- **Morgan Fingerprints**: Circular fingerprints with customizable radius and bits
- **MACCS Keys**: 166-bit structural keys
- **RDKit Descriptors**: 200+ molecular descriptors
- **Custom Features**: Support for precomputed features

## 📈 Data Splitting Strategies

1. **Train/Test Split**: Simple 80/20 split
2. **Train/Validation/Test Split**: 70/15/15 split for model development
3. **K-Fold Cross-Validation**: Robust evaluation with stratified sampling

## 🎯 Bayesian Optimization

Find optimal reaction conditions using trained models:

- **Acquisition Functions**: Expected Improvement (EI), Upper Confidence Bound (UCB), Probability of Improvement (POI)
- **Search Spaces**: Discrete (catalyst libraries) and continuous (temperature, time) variables
- **Multi-objective**: Support for multiple optimization targets
- **Constraints**: Chemical and practical constraints

## 📝 Example Data Format

Your CSV file should contain SMILES strings and target values:

```csv
Catalyst,Reactant1,Reactant2,Temperature,Solvent,yield
CC(C)P(c1ccccc1)c1ccccc1,CC(=O)c1ccccc1,NCc1ccccc1,80,toluene,95.2
CCc1ccc(P(CCc2ccccc2)CCc2ccccc2)cc1,CC(=O)c1ccccc1,NCc1ccccc1,60,THF,87.5
...
```

## 🛠️ Advanced Usage

### Custom Configuration

Create your own YAML configuration file based on the examples:

```yaml
experiment_name: "My_Experiment"
task_type: "regression"

data:
  source_mode: "single_file"
  single_file_config:
    main_file_path: "data/my_reactions.csv"
    smiles_col: ["Catalyst", "Reactant1", "Reactant2"]
    target_col: "yield"

training:
  models_to_run:
    - "xgb"
    - "lgbm"
    - "rf"
  n_trials: 20

# ... additional configuration
```

### Programmatic Usage

```python
from core.run_manager import start_experiment_run
from core.config_loader import load_config

# Load configuration
config = load_config("my_config.yaml")

# Run experiment
results = start_experiment_run(config)

# Access results
best_model = max(results['results'], key=lambda x: x['test_r2'])
print(f"Best model: {best_model['model_name']} (R² = {best_model['test_r2']:.4f})")
```

## 📊 Output and Results

CRAFT generates comprehensive outputs:

- **Trained Models**: Serialized models in multiple formats
- **Predictions**: CSV files with predictions and uncertainties
- **Metrics**: Detailed performance metrics and cross-validation results
- **Feature Importance**: Analysis of important molecular features
- **Visualizations**: Learning curves, feature importance plots
- **Optimization Results**: Top-ranked reaction conditions

## 🔧 Troubleshooting

### Common Installation Issues

#### RDKit Installation Problems
```bash
# If RDKit installation fails with pip, try conda:
conda install -c conda-forge rdkit

# Or use the conda environment setup:
./setup.sh conda
```

#### PyTorch Geometric Issues
```bash
# For CUDA compatibility issues:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only installation:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Memory Issues
- **Large datasets**: Consider using data sampling or batch processing
- **GPU memory**: Reduce batch size in neural network training
- **System memory**: Close other applications or use a machine with more RAM

#### Permission Issues
```bash
# Fix virtual environment permissions:
sudo chown -R $USER:$USER craft/
```

### Environment Verification

If you encounter import errors, verify your environment:

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(torch|rdkit|optuna|sklearn|pandas)"

# Test core imports
python -c "
try:
    import torch, rdkit, optuna, sklearn, pandas, numpy
    print('✅ All core packages available')
except ImportError as e:
    print(f'❌ Missing package: {e}')
"
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

