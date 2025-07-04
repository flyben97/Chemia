# 原始数据分割保存功能 (Original Data Splits Feature)

## 功能概述

这个新功能会在训练过程中自动保存原始数据的分割版本，让您可以轻松检查哪些数据被用于训练、验证和测试。

## 功能特点

- 🔍 **易于检查**: 将原始数据按照训练时的相同分割方式保存为单独的CSV文件
- 📊 **完整信息**: 保留原始数据的所有列，包括SMILES、目标值和其他特征
- 📈 **分割统计**: 自动生成分割摘要，显示每个数据集的大小和百分比
- 🔄 **一致性**: 确保与模型训练使用的是完全相同的数据分割

## 输出文件结构

运行训练后，您会在输出目录中找到一个新的文件夹：

```
output/your_experiment_name/
├── models/                     # 训练的模型
├── data_splits/               # 处理后的特征数据
├── original_data_splits/      # 新增！原始数据分割
│   ├── train_original_data.csv    # 训练集原始数据
│   ├── test_original_data.csv     # 测试集原始数据
│   ├── val_original_data.csv      # 验证集原始数据 (如果使用 train_valid_test 模式)
│   └── data_split_summary.csv     # 分割摘要统计
└── run_config.json           # 运行配置
```

## 文件说明

### 1. 数据分割文件
- **train_original_data.csv**: 包含用于训练的原始数据行
- **test_original_data.csv**: 包含用于测试的原始数据行  
- **val_original_data.csv**: 包含用于验证的原始数据行（仅在使用 `train_valid_test` 模式时）

### 2. 分割摘要文件 (data_split_summary.csv)
包含以下列：
- `split`: 数据集名称 (train/val/test)
- `count`: 样本数量
- `percentage`: 占总数据的百分比

## 使用示例

### 方法1: 使用现有配置

只需正常运行您的训练配置，功能会自动激活：

```bash
# 使用现有配置
python run_training_only.py --config your_config.yaml

# 或使用完整工作流
python run_full_workflow.py --config config_full_workflow.yaml
```

### 方法2: 使用示例配置

我们提供了一个专门的示例配置：

```bash
python run_training_only.py --config examples/configs/config_with_original_splits.yaml
```

### 方法3: 运行演示

```bash
python demo_original_splits.py
```

## 配置选项

功能支持所有现有的数据分割模式：

### 交叉验证模式 (推荐用于小数据集)
```yaml
split_mode: "cross_validation"
split_config:
  cross_validation:
    n_splits: 5
    test_size_for_cv: 0.2
    random_state: 42
```

### 传统分割模式
```yaml
split_mode: "train_valid_test"
split_config:
  train_valid_test:
    test_size: 0.2
    valid_size: 0.1
    random_state: 42
```

## 实际应用场景

### 1. 数据质量检查
检查训练集和测试集中的数据分布是否合理：
```python
import pandas as pd

# 加载分割后的数据
train_data = pd.read_csv('output/your_run/original_data_splits/train_original_data.csv')
test_data = pd.read_csv('output/your_run/original_data_splits/test_original_data.csv')

# 检查目标值分布
print("训练集ee值分布:")
print(train_data['ee'].describe())
print("\n测试集ee值分布:")
print(test_data['ee'].describe())
```

### 2. 特定样本分析
查找特定的反应条件在哪个数据集中：
```python
# 查找特定配体的分布
ligand_of_interest = "your_ligand_smiles"
train_count = len(train_data[train_data['Ligand'] == ligand_of_interest])
test_count = len(test_data[test_data['Ligand'] == ligand_of_interest])
print(f"目标配体在训练集: {train_count} 个, 测试集: {test_count} 个")
```

### 3. 重现性验证
确保模型评估的可重现性：
```python
# 使用完全相同的测试集重新评估模型
test_smiles = test_data[['Reactant1', 'Reactant2', 'Ligand']].values
true_values = test_data['ee'].values
# 然后使用这些数据进行预测...
```

## 技术细节

- 数据分割在特征生成**之前**进行，确保索引一致性
- 使用与模型训练相同的随机种子，保证分割一致性
- 自动处理缺失值清理后的索引映射
- 支持所有现有的数据源模式和分割配置

## 注意事项

1. **存储空间**: 这会增加输出目录的大小，因为保存了原始数据的副本
2. **大数据集**: 对于非常大的数据集，考虑是否需要这个功能
3. **隐私数据**: 确保原始数据的保存符合您的数据安全要求

## 故障排除

如果出现问题，请检查：

1. **权限**: 确保有写入输出目录的权限
2. **磁盘空间**: 确保有足够的磁盘空间保存额外的CSV文件
3. **数据格式**: 确保原始数据可以正常读取和保存为CSV格式

## 向后兼容性

此功能完全向后兼容，不会影响现有的工作流程或配置文件。如果您不需要这个功能，可以忽略生成的 `original_data_splits/` 文件夹。 