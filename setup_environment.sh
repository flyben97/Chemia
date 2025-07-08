#!/bin/bash

# CHEMIA Environment Setup Script
# This script ensures all dependencies are properly installed

set -e  # Exit on any error

echo "=========================================="
echo "CHEMIA Environment Setup"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    print_error "Please install Miniconda or Anaconda first"
    exit 1
fi

print_status "Conda found: $(conda --version)"

# Function to check if environment exists
env_exists() {
    conda env list | grep -q "^$1 "
}

# Check current environment setup
print_header "Environment Detection"

CRAFT_EXISTS=false
CHEMIA_EXISTS=false

if env_exists "craft"; then
    CRAFT_EXISTS=true
    print_status "Found existing 'craft' environment"
fi

if env_exists "chemia"; then
    CHEMIA_EXISTS=true
    print_status "Found existing 'chemia' environment"
fi

# Determine which environment to use/create
TARGET_ENV=""
if [[ "$CRAFT_EXISTS" == true ]]; then
    TARGET_ENV="craft"
    print_status "Using existing 'craft' environment"
elif [[ "$CHEMIA_EXISTS" == true ]]; then
    TARGET_ENV="chemia"
    print_status "Using existing 'chemia' environment"
else
    TARGET_ENV="chemia"
    print_status "Will create new 'chemia' environment"
fi

# Function to install missing packages
install_missing_packages() {
    local env_name=$1
    print_header "Checking and Installing Missing Packages in $env_name"
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $env_name
    
    # Check and install packages
    python << EOF
import sys
import subprocess

packages_to_check = [
    ('numpy', 'numpy>=1.21.0'),
    ('pandas', 'pandas>=1.3.0'),
    ('scipy', 'scipy>=1.7.0'),
    ('sklearn', 'scikit-learn>=1.0.2'),
    ('matplotlib', 'matplotlib>=3.5.0'),
    ('seaborn', 'seaborn>=0.11.0'),
    ('plotly', 'plotly>=5.0.0'),
    ('xgboost', 'xgboost>=1.6.0'),
    ('lightgbm', 'lightgbm>=3.3.0'),
    ('catboost', 'catboost>=1.0.4'),
    ('optuna', 'optuna>=3.0.0'),
    ('rdkit', 'rdkit>=2022.03.1'),
    ('yaml', 'pyyaml>=6.0'),
    ('joblib', 'joblib>=1.1.0'),
    ('rich', 'rich>=12.0.0'),
    ('tqdm', 'tqdm>=4.62.0'),
    ('typing_extensions', 'typing-extensions>=4.0.0'),
    ('packaging', 'packaging>=21.0')
]

missing_conda = []
missing_pip = []

print("Checking package status...")
for import_name, install_name in packages_to_check:
    try:
        if import_name == 'sklearn':
            import sklearn
        elif import_name == 'yaml':
            import yaml
        elif import_name == 'rdkit':
            import rdkit
        else:
            __import__(import_name)
        print(f"✅ {install_name}: Already installed")
    except ImportError:
        print(f"❌ {install_name}: Missing")
        if install_name.startswith(('rdkit', 'matplotlib', 'seaborn')):
            missing_conda.append(install_name.split('>=')[0])
        else:
            missing_pip.append(install_name)

# Install missing conda packages
if missing_conda:
    print(f"\\nInstalling via conda: {', '.join(missing_conda)}")
    conda_cmd = ['conda', 'install', '-c', 'conda-forge', '-y'] + missing_conda
    subprocess.run(conda_cmd, check=True)

# Install missing pip packages  
if missing_pip:
    print(f"\\nInstalling via pip: {', '.join(missing_pip)}")
    pip_cmd = ['pip', 'install'] + missing_pip
    subprocess.run(pip_cmd, check=True)

if not missing_conda and not missing_pip:
    print("\\n🎉 All packages are already installed!")

EOF
}

# Main installation logic
if [[ "$TARGET_ENV" == "craft" ]] || [[ "$TARGET_ENV" == "chemia" ]]; then
    if [[ "$CRAFT_EXISTS" == false ]] && [[ "$CHEMIA_EXISTS" == false ]]; then
        print_header "Creating New Environment"
        
        # Choose which environment file to use
        if [[ -f "environment_improved.yml" ]]; then
            print_status "Using improved environment configuration"
            conda env create -f environment_improved.yml
            TARGET_ENV="chemia"
        elif [[ -f "environment.yml" ]]; then
            print_status "Using standard environment configuration"
            conda env create -f environment.yml
            TARGET_ENV="craft"
        else
            print_error "No environment.yml file found!"
            exit 1
        fi
    else
        # Environment exists, just install missing packages
        install_missing_packages $TARGET_ENV
    fi
else
    print_error "No suitable environment found or created"
    exit 1
fi

# Final verification
print_header "Final Verification"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $TARGET_ENV

python << 'EOF'
print("=== Final Package Verification ===")

essential_packages = [
    'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 
    'seaborn', 'plotly', 'xgboost', 'lightgbm', 'catboost',
    'optuna', 'rdkit', 'yaml', 'joblib', 'rich', 'tqdm'
]

all_good = True
for pkg in essential_packages:
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
    print("\n🎉 环境配置完成！所有必需包都已正确安装!")
    print("🚀 现在可以运行 CHEMIA 项目了!")
else:
    print("\n⚠️  某些包仍然缺失，请检查上述输出")
EOF

print_header "Setup Complete"
print_status "Environment: $TARGET_ENV"
print_status "To activate: conda activate $TARGET_ENV"
print_status "To verify: python -c 'import seaborn; print(\"Seaborn version:\", seaborn.__version__)'"

echo "=========================================="
echo "CHEMIA Environment Setup Complete!"
echo "==========================================" 