# CRAFT Configuration Files Guide

This directory contains various pre-configured YAML files for different use cases. Choose the appropriate configuration based on your needs.

## 📋 Quick Selection Guide

### For Beginners
- **`quick_start.yaml`** - Start here! Minimal setup for first-time users

### By Task Type
- **Regression Tasks**: `regression_training_*.yaml`
- **Classification Tasks**: `classification_training_*.yaml`
- **Graph Neural Networks**: `gnn_training.yaml`

### By Data Splitting Strategy
- **Simple Split**: `*_simple.yaml` (80/20 train/test)
- **K-Fold Cross-Validation**: `*_kfold.yaml` (5-fold CV)
- **Train/Valid/Test**: `*_split.yaml` (70/15/15 split)

### By Feature Complexity
- **Rich Features**: `training_with_features.yaml`
- **Basic Features**: `training_without_features.yaml`

### By Workflow Type
- **Training Only**: `*_training_*.yaml`
- **Optimization Only**: `bayesian_optimization_only.yaml`
- **End-to-End**: `end_to_end_workflow.yaml`

## 📁 Complete File List

### Training Configurations

| File | Task | Split Strategy | Features | Models |
|------|------|---------------|----------|---------|
| `quick_start.yaml` | Regression | Train/Test | Basic | 3 models |
| `regression_training_simple.yaml` | Regression | Train/Test | Standard | 5 models |
| `regression_training_kfold.yaml` | Regression | 5-Fold CV | Standard | 8 models |
| `regression_training_split.yaml` | Regression | Train/Valid/Test | Rich | 6 models |
| `classification_training_simple.yaml` | Classification | Train/Test | Standard | 5 models |
| `classification_training_kfold.yaml` | Classification | 5-Fold CV | Standard | 7 models |
| `training_with_features.yaml` | Regression | 5-Fold CV | Rich | 6 models |
| `training_without_features.yaml` | Regression | Train/Test | Minimal | 5 models |
| `gnn_training.yaml` | Regression | 5-Fold CV | GNN + Standard | 7 models |
| `gnn_pure_training.yaml` | Regression | 5-Fold CV | Pure GNN | 5 GNN models |

### Optimization Configurations

| File | Description | Input Required |
|------|-------------|----------------|
| `bayesian_optimization_only.yaml` | Standalone optimization | Pre-trained model |
| `end_to_end_workflow.yaml` | Complete pipeline | Raw data only |

## 🚀 Usage Examples

### Training a Model
```bash
# Quick start (recommended for beginners)
python run_training_only.py --config examples/configs/quick_start.yaml

# Comprehensive regression with cross-validation
python run_training_only.py --config examples/configs/regression_training_kfold.yaml

# Classification with rich features
python run_training_only.py --config examples/configs/classification_training_kfold.yaml
```

### Running Optimization
```bash
# Using a pre-trained model
python run_optimization.py --config examples/configs/bayesian_optimization_only.yaml

# End-to-end workflow (train + optimize)
python run_full_workflow.py --config examples/configs/end_to_end_workflow.yaml
```

## 🔧 Customization Guide

### Modifying Configurations

1. **Copy a base configuration**:
   ```bash
   cp examples/configs/regression_training_simple.yaml my_config.yaml
   ```

2. **Edit key parameters**:
   - `experiment_name`: Your experiment name
   - `data.single_file_config.main_file_path`: Path to your data
   - `training.models_to_run`: Select desired algorithms
   - `training.n_trials`: Adjust hyperparameter optimization trials

3. **Run with your configuration**:
   ```bash
   python run_training_only.py --config my_config.yaml
   ```

### Common Modifications

#### Change Data Path
```yaml
data:
  single_file_config:
    main_file_path: "data/my_reactions.csv"  # Update this
    smiles_col: ["Catalyst", "Reactant1", "Reactant2"]
    target_col: "yield"
```

#### Select Different Models
```yaml
training:
  models_to_run:
    - "xgb"           # XGBoost
    - "lgbm"          # LightGBM
    - "rf"            # Random Forest
    - "gpr"           # Gaussian Process
    # Add or remove as needed
```

#### Adjust Feature Engineering
```yaml
features:
  per_smiles_col_generators:
    Catalyst: 
      - type: "morgan"
        radius: 2
        nBits: 1024
      - type: "rdkit_descriptors"  # Add more features
```

## 📊 Configuration Comparison

| Feature | Simple | K-Fold | Split | Rich Features |
|---------|--------|---------|-------|---------------|
| Models | 3-5 | 7-8 | 6 | 6 |
| Hyperparameter Trials | 5-10 | 15 | 20 | 25 |
| Feature Types | 1-2 | 2-3 | 2-3 | 3-4 |
| Cross-Validation | ❌ | ✅ | ❌ | ✅ |
| Validation Set | ❌ | ❌ | ✅ | ❌ |
| Execution Time | Fast | Medium | Medium | Slow |
| Robustness | Low | High | Medium | High |

## 🎯 Choosing the Right Configuration

### For Different Scenarios

**Quick Testing/Prototyping**:
- Use: `quick_start.yaml`
- Why: Fast execution, basic but reliable

**Research/Publication**:
- Use: `*_kfold.yaml` configurations
- Why: Robust evaluation with cross-validation

**Model Development**:
- Use: `*_split.yaml` configurations
- Why: Separate validation set for hyperparameter tuning

**Production Pipeline**:
- Use: `end_to_end_workflow.yaml`
- Why: Complete automation from training to optimization

**Small Datasets (<500 samples)**:
- Use: K-fold configurations with GPR models
- Why: Better generalization with limited data

**Large Datasets (>5000 samples)**:
- Use: Train/test split with gradient boosting
- Why: Faster training, less overfitting risk

## 🧬 Special Note on Graph Neural Networks (GNNs)

**Important**: GNNs work differently from traditional ML models!

### Why GNNs Don't Need Molecular Fingerprints
- **Direct Graph Processing**: GNNs convert SMILES directly to molecular graphs (atoms→nodes, bonds→edges)
- **Learned Representations**: They learn molecular features automatically through graph convolutions
- **No Fixed Features**: Unlike Morgan/MACCS fingerprints, GNN features are learned end-to-end

### GNN Configuration Options
- **`gnn_pure_training.yaml`**: ✅ **Recommended** - Pure GNN training without traditional fingerprints
- **`gnn_training.yaml`**: Mixed GNN + traditional models (fingerprints only used by traditional models)
- **`config_gnn_demo.yaml`**: ❌ **Fixed** - Previously had incorrect fingerprint configuration

### When to Use GNNs
- **Small to medium datasets** (GNNs can learn from graph structure)
- **Novel molecular scaffolds** (better generalization than fixed fingerprints)
- **When you need interpretability** (attention weights show important atoms/bonds)

## 💡 Tips and Best Practices

1. **Start Simple**: Begin with `quick_start.yaml` to ensure everything works
2. **Gradual Complexity**: Move to more complex configurations as needed
3. **Data Size Matters**: Use cross-validation for small datasets, simple splits for large ones
4. **Feature Engineering**: Start with basic features, add complexity if needed
5. **Model Selection**: Include both fast (RF) and powerful (XGB, LGBM) models
6. **Hyperparameter Trials**: Start with fewer trials for testing, increase for final runs
7. **GNN Usage**: Use `gnn_pure_training.yaml` for pure graph-based learning

## 🔍 Troubleshooting

**Common Issues**:
- **File not found**: Check data paths in configuration
- **Memory errors**: Reduce feature complexity or model count
- **Slow execution**: Reduce `n_trials` or number of models
- **Poor performance**: Try richer features or more models

**Getting Help**:
- Check the main README for detailed documentation
- Look at existing configurations for examples
- Modify step by step from working configurations

---

**Happy modeling with CRAFT! 🧪✨**

# Advanced Graph Construction Configurations

The CRAFT framework now supports advanced graph construction modes for handling multiple SMILES inputs. These modes enable sophisticated modeling of multi-component chemical systems and fusion with experimental features.

## 🌟 Advanced Graph Construction Modes

### 1. Feature-Level Concatenation (`feature_concat`)

**Configuration**: [`feature_concatenation_mode.yaml`](feature_concatenation_mode.yaml)

**What it does**:
- Extracts molecular-level descriptors from each SMILES
- Concatenates all features into a single vector
- Trains on the concatenated representation

**Best for**:
- When molecular interactions are less important
- Faster training needs
- Traditional ML-style behavior with GNN-derived features

**Example**:
```yaml
graph_construction:
  mode: "feature_concat"
  molecular_feature_size: 50  # Each molecule → 50-dim vector
```

### 2. Reaction Graph Construction (`reaction_graph`)

**Configuration**: [`reaction_graph_mode.yaml`](reaction_graph_mode.yaml)

**What it does**:
- Creates a unified reaction graph with all molecules
- Adds a virtual "reaction center" node
- Connects molecules to center with typed edges based on roles

**Best for**:
- Chemical reaction modeling
- Capturing inter-molecular interactions
- Systems where molecular roles matter (catalyst, reactant, product)

**Example**:
```yaml
graph_construction:
  mode: "reaction_graph"
  molecule_roles:
    catalyst_smiles: "catalyst"
    substrate_smiles: "reactant"
    product_smiles: "product"
```

### 3. Custom Feature Fusion (`custom_fusion`)

**Configuration**: [`custom_fusion_mode.yaml`](custom_fusion_mode.yaml)

**What it does**:
- Builds molecular graphs normally
- Extracts graph embeddings via GNN
- Fuses embeddings with user-provided experimental features
- Supports multiple fusion methods (concatenation, attention, gating, transformer)

**Best for**:
- Combining molecular structure with experimental conditions
- Reaction yield prediction with temperature, pressure, time
- Any scenario with heterogeneous data (graphs + tabular)

**Example**:
```yaml
graph_construction:
  mode: "custom_fusion"
  fusion_method: "attention"
  custom_feature_columns:
    - "temperature"
    - "pressure"
    - "reaction_time"
```

### 4. Traditional Batch Mode (`batch`)

**Default mode** - batches individual molecular graphs together
- Maintains compatibility with existing configurations
- Each molecule processed separately, then batched

## 🔧 Fusion Methods for Custom Features

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Concatenate** | Simple concatenation | Fast, stable | No learned interaction |
| **Attention** | Multi-head attention fusion | Interpretable, adaptive | More parameters |
| **Gated** | Gating mechanism | Can ignore irrelevant features | Complex |
| **Transformer** | Full transformer layers | Most sophisticated | May overfit |

## 📊 Configuration Comparison

| Feature | Batch | Feature Concat | Reaction Graph | Custom Fusion |
|---------|-------|----------------|----------------|---------------|
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Memory** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Inter-molecular** | ❌ | ❌ | ✅ | Partial |
| **Custom Features** | ❌ | ❌ | ❌ | ✅ |
| **Interpretability** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Flexibility** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🚀 Usage Examples

### Basic Usage

```bash
# Feature concatenation mode
python run_training_only.py --config examples/configs/feature_concatenation_mode.yaml

# Reaction graph mode  
python run_training_only.py --config examples/configs/reaction_graph_mode.yaml

# Custom fusion mode
python run_training_only.py --config examples/configs/custom_fusion_mode.yaml

# Advanced features demo (all modes)
python run_training_only.py --config examples/configs/advanced_graph_features.yaml
```

### Programmatic Usage

```python
from optimizers import GNNOptimizer
from utils.advanced_graph_builder import GraphConstructionMode

# Initialize with advanced graph construction
optimizer = GNNOptimizer(
    model_name='gcn',
    smiles_columns=['catalyst', 'reactant_1', 'reactant_2'],
    graph_construction_mode='reaction_graph',
    molecule_roles={
        'catalyst': 'catalyst',
        'reactant_1': 'reactant', 
        'reactant_2': 'reactant'
    }
)

# For custom fusion
fusion_config = {
    'fusion_method': 'attention',
    'custom_feature_dim': 5,
    'graph_embed_dim': 128
}

optimizer = GNNOptimizer(
    model_name='gat',
    smiles_columns=['compound'],
    graph_construction_mode='custom_fusion',
    custom_fusion_config=fusion_config
)
```

## 📈 Performance Considerations

### When to Use Each Mode

**Feature Concatenation**:
- ✅ Small datasets
- ✅ Fast prototyping needed
- ✅ Traditional ML pipeline integration
- ❌ Complex molecular interactions

**Reaction Graph**:
- ✅ Chemical reaction datasets
- ✅ Multi-component systems
- ✅ Catalyst design
- ❌ Very large molecules (memory)

**Custom Fusion**:
- ✅ Experimental data available
- ✅ Reaction conditions matter
- ✅ Heterogeneous datasets
- ❌ Limited additional features

## 🔍 Advanced Analysis Features

### Ablation Studies
All advanced modes support ablation studies to validate the benefit of advanced features:

```yaml
evaluation:
  ablation_study:
    enabled: true
    test_graph_only: true      # GNN without custom features
    test_custom_only: true     # Traditional ML on custom features only
    test_traditional_vs_gnn: true  # Compare with molecular descriptors
```

### Visualization and Interpretation

- **Reaction graphs**: Visualize unified reaction structures
- **Attention weights**: See which features the model focuses on
- **Fusion analysis**: Understand how custom features influence predictions
- **Feature importance**: Rank custom features by importance

## 💡 Tips and Best Practices

1. **Start Simple**: Begin with batch mode, then try advanced modes
2. **Data Quality**: Ensure molecule roles are correctly assigned for reaction graphs
3. **Feature Engineering**: Normalize custom features before fusion
4. **Ablation Testing**: Always validate that advanced features improve performance
5. **Memory Management**: Use smaller batch sizes for reaction graph mode
6. **Fusion Method**: Start with concatenation, then try attention for custom fusion

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Out of memory" with reaction graph mode
- **Solution**: Reduce batch size, use smaller molecules, enable gradient accumulation

**Issue**: Custom fusion not improving performance  
- **Solution**: Check feature correlation, try different fusion methods, ensure proper normalization

**Issue**: Reaction graph training very slow
- **Solution**: Reduce number of molecules per reaction, use simpler GNN models first

**Issue**: Invalid molecule roles
- **Solution**: Check column names in `molecule_roles` match `smiles_columns`

### Debug Mode

Enable detailed logging for debugging:

```yaml
compute:
  debug_mode: true
  log_graph_construction: true
  save_intermediate_results: true
``` 