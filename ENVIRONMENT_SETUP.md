# CHEMIA Environment Setup Guide

## 🚀 Quick Setup Options

### Option 1: Automated Setup (Recommended)
```bash
# Run the automated setup script
./setup_environment.sh
```

### Option 2: Using Improved Environment Files
```bash
# For new environment (recommended)
conda env create -f environment.yml
conda activate chemia

# Or for pip users
pip install -r requirements.txt
```

## 📦 Complete Dependency List

### Core Scientific Computing

- ✅ numpy>=1.21.0,<2.0
- ✅ pandas>=1.3.0,<3.0
- ✅ scipy>=1.7.0,<2.0

### Machine Learning
- ✅ scikit-learn>=1.0.2,<2.0
- ✅ xgboost>=1.6.0,<4.0
- ✅ lightgbm>=3.3.0,<5.0
- ✅ catboost>=1.0.4,<2.0
- ✅ optuna>=3.0.0,<5.0

### Visualization (Standard Installation)
- ✅ matplotlib>=3.5.0,<4.0
- ✅ seaborn>=0.11.0,<1.0 
- ✅ plotly>=5.0.0,<6.0

### Chemical Informatics
- ✅ rdkit>=2022.03.1

### Data Processing
- ✅ pyyaml>=6.0,<7.0
- ✅ joblib>=1.1.0,<2.0
- ✅ tqdm>=4.62.0,<5.0
- ✅ rich>=12.0.0,<14.0

### System Dependencies
- ✅ typing-extensions>=4.0.0,<5.0
- ✅ packaging>=21.0,<25.0

## 🔍 Environment Verification

### Quick Verification
```bash
# Activate your environment
conda activate craft  # or chemia

# Run verification
python -c "
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
print('✅ All visualization packages working!')
print(f'Seaborn version: {sns.__version__}')
"
```

### Comprehensive Check
```bash
# Run full package check
python << 'EOF'
packages = [
    'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib',
    'seaborn', 'plotly', 'xgboost', 'lightgbm', 'catboost',
    'optuna', 'rdkit', 'yaml', 'joblib', 'rich', 'tqdm'
]

print("=== Package Verification ===")
all_good = True
for pkg in packages:
    try:
        if pkg == 'sklearn':
            import sklearn as module
        elif pkg == 'yaml':
            import yaml as module
        elif pkg == 'rdkit':
            import rdkit as module
        else:
            module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {pkg}: {version}")
    except ImportError:
        print(f"❌ {pkg}: MISSING")
        all_good = False

if all_good:
    print("\n🎉 All packages are correctly installed!")
else:
    print("\n⚠️ Some packages are missing")
EOF
```

## Environment Status Summary

| Component | Status | Version | Notes |
|-----------|--------|---------|-------|
| Python | ✅ | 3.12.11 | Compatible |
| NumPy | ✅ | 1.26.4 | Core dependency |
| Pandas | ✅ | 1.5.3 | Data processing |
| Scikit-learn | ✅ | 1.7.0 | ML algorithms |
| Seaborn | ✅ | 0.13.2 | Visualization |
| Matplotlib | ✅ | 3.10.3 | Plotting backend |
| Plotly | ✅ | 6.1.2 | Interactive plots |
| XGBoost | ✅ | 3.0.2 | Gradient boosting |
| LightGBM | ✅ | 4.6.0 | Fast GBM |
| CatBoost | ✅ | 1.2.8 | Categorical GBM |
| Optuna | ✅ | 4.4.0 | Hyperparameter optimization |
| RDKit | ✅ | 2025.03.3 | Chemical informatics |

