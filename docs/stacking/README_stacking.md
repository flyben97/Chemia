# CRAFT 模型堆叠（Stacking）使用指南

## 概述

模型堆叠（Stacking）是一种强大的集成学习技术，通过组合多个基础模型的预测结果来获得更好的预测性能。CRAFT框架提供了完整的模型堆叠功能，支持多种堆叠策略和使用场景。

## 主要特性

### 🎯 多种堆叠策略
- **简单平均**：直接平均所有模型的预测结果
- **加权平均**：根据模型性能分配不同权重
- **元学习器**：使用机器学习模型学习如何组合基础模型

### 🔧 支持的元学习器
- **Ridge回归**：适合回归任务，具有正则化效果
- **随机森林**：适合复杂非线性关系
- **逻辑回归**：适合分类任务

### 📊 任务支持
- **回归任务**：连续值预测
- **分类任务**：类别预测，支持概率输出

### 🆕 自动数据加载功能
- **智能数据读取**：自动从CRAFT实验目录读取train/valid/test数据
- **格式自动转换**：原始数据自动转换为预测所需格式
- **灵活数据源**：支持自动加载和用户自定义数据两种方式
- **一键集成**：完全自动化的模型选择、数据加载和权重优化

## 安装和导入

```python
from model_stacking import ModelStacker, create_ensemble, auto_ensemble, smart_ensemble_with_meta_learner
from data_loader import create_validation_dataset, load_custom_validation_data
```

## 快速开始

### 1. 完全自动化集成（推荐）

```python
# 🚀 一行代码搞定：自动加载数据、自动选择模型、自动优化权重
from model_stacking import auto_ensemble

stacker = auto_ensemble("output/my_experiment")
prediction = stacker.predict_single({
    'SMILES': 'CCO',
    'Solvent_1_SMILES': 'CC(=O)O',
    'Solvent_2_SMILES': 'CCN'
})
```

### 2. 智能元学习器集成

```python
# 🧠 自动选择模型 + 训练元学习器
from model_stacking import smart_ensemble_with_meta_learner

stacker = smart_ensemble_with_meta_learner(
    experiment_dir="output/my_experiment",
    meta_method="ridge"  # 或 "rf", "logistic"
)
prediction = stacker.predict_single(test_sample)
```

### 3. 传统手动配置

```python
# 手动配置方式（保持向后兼容）
stacker = ModelStacker("output/my_experiment")
stacker.add_model("xgb", weight=0.4)
stacker.add_model("lgbm", weight=0.3) 
stacker.add_model("catboost", weight=0.3)
stacker.set_stacking_method("weighted_average")
```

## 数据加载方式

### 方式1：自动从实验目录加载（推荐）

CRAFT训练时会自动保存数据拆分到 `experiment_dir/original_data_splits/` 目录：

```
output/my_experiment/
├── original_data_splits/
│   ├── train_original_data.csv     # 训练集原始数据
│   ├── val_original_data.csv       # 验证集原始数据  
│   ├── test_original_data.csv      # 测试集原始数据
│   └── data_split_summary.csv      # 拆分摘要统计
```

使用自动加载：

```python
# 完全自动：自动加载验证数据进行模型选择和权重优化
stacker = auto_ensemble("output/my_experiment")

# 自动训练元学习器
stacker.fit_meta_model(auto_load=True, validation_size=100)

# 自动评估（自动加载测试数据）
evaluation = stacker.evaluate(auto_load=True, use_test_set=True)
```

### 方式2：自定义验证数据

```python
# 方法1：手动提供数据
custom_validation = [
    {'SMILES': 'CCO', 'Solvent_1_SMILES': 'CC(=O)O', 'Solvent_2_SMILES': 'CCN'},
    # ... 更多样本
]
custom_labels = [12.5, 8.3, 15.2]

stacker = auto_ensemble(
    experiment_dir="output/my_experiment",
    validation_data=custom_validation,
    true_labels=custom_labels,
    auto_load_validation=False
)

# 方法2：从CSV文件加载
from data_loader import load_custom_validation_data

val_data, val_labels = load_custom_validation_data(
    validation_file="my_validation_data.csv",
    target_column="target_value"
)

stacker = auto_ensemble(
    experiment_dir,
    validation_data=val_data,
    true_labels=val_labels,
    auto_load_validation=False
)
```

## 基础使用

### 1. 创建基础堆叠器

```python
# 方法1：从实验目录创建
stacker = ModelStacker(experiment_dir="output/my_experiment")

# 添加基础模型
stacker.add_model("xgb", weight=0.4)
stacker.add_model("lgbm", weight=0.3)
stacker.add_model("catboost", weight=0.3)

# 设置堆叠方法
stacker.set_stacking_method("weighted_average")
```

### 2. 快速创建集成

```python
# 一行代码创建加权集成
stacker = create_ensemble(
    experiment_dir="output/my_experiment",
    model_names=["xgb", "lgbm", "catboost"],
    weights=[0.5, 0.3, 0.2],
    method="weighted_average"
)
```

### 3. 自动优化集成

```python
# 根据验证性能自动选择模型和权重
stacker = auto_ensemble(
    experiment_dir="output/my_experiment",
    auto_load_validation=True,  # 🆕 自动加载验证数据
    validation_size=100,
    available_models=['xgb', 'lgbm', 'catboost', 'rf']
)
```

## 堆叠方法详解

### 1. 简单平均（Simple Average）
```python
stacker.set_stacking_method("simple_average")
```
- **优点**：简单、稳定、不容易过拟合
- **缺点**：不考虑模型性能差异
- **适用场景**：模型性能相近时

### 2. 加权平均（Weighted Average）
```python
stacker.set_stacking_method("weighted_average")
```
- **优点**：考虑模型性能差异，手动控制权重
- **缺点**：需要预先知道模型性能
- **适用场景**：已知各模型相对性能时

### 3. 元学习器（Meta-Learner）

#### Ridge回归元学习器
```python
stacker.set_stacking_method("ridge")
stacker.fit_meta_model(auto_load=True)  # 🆕 自动加载验证数据
```
- **优点**：自动学习组合方式，有正则化
- **缺点**：需要额外的验证数据
- **适用场景**：回归任务，追求最佳性能

#### 随机森林元学习器
```python
stacker.set_stacking_method("rf")
stacker.fit_meta_model(auto_load=True)
```
- **优点**：处理非线性关系，特征重要性
- **缺点**：可能过拟合，计算复杂
- **适用场景**：复杂的组合关系

#### 逻辑回归元学习器（分类任务）
```python
stacker.set_stacking_method("logistic")
stacker.fit_meta_model(auto_load=True)
```
- **优点**：输出概率，可解释性强
- **缺点**：假设线性关系
- **适用场景**：分类任务

## 预测和评估

### 单样本预测
```python
sample = {
    'SMILES': 'CCO',
    'Solvent_1_SMILES': 'CC(=O)O',
    'Solvent_2_SMILES': 'CCN'
}

prediction = stacker.predict_single(sample)
print(f"预测结果: {prediction}")
```

### 批量预测
```python
results = stacker.predict(test_data)
print(f"预测结果: {results['predictions']}")
print(f"使用模型: {results['model_names']}")
print(f"堆叠方法: {results['stacking_method']}")
```

### 性能评估
```python
# 🆕 自动评估（自动加载测试数据）
evaluation = stacker.evaluate(auto_load=True, use_test_set=True)

# 回归任务指标
if evaluation.get('r2') is not None:
    print(f"R² Score: {evaluation['r2']:.4f}")
    print(f"RMSE: {evaluation['rmse']:.4f}")
    print(f"MAE: {evaluation['mae']:.4f}")

# 分类任务指标
if evaluation.get('accuracy') is not None:
    print(f"Accuracy: {evaluation['accuracy']:.4f}")

# 基础模型性能对比
print("基础模型性能:")
for model_name, perf in evaluation['base_model_performance'].items():
    print(f"  {model_name}: {perf}")
```

## 保存和加载

### 保存堆叠模型
```python
stacker.save("models/my_stacked_model.pkl")
```

### 加载堆叠模型
```python
loaded_stacker = ModelStacker.load("models/my_stacked_model.pkl")
prediction = loaded_stacker.predict_single(test_sample)
```

## 高级技巧

### 1. 多层堆叠
```python
# 第一层：创建基础集成
ensemble1 = create_ensemble(experiment_dir, ["xgb", "lgbm"], [0.6, 0.4])
ensemble2 = create_ensemble(experiment_dir, ["catboost", "rf"], [0.7, 0.3])

# 第二层：手动组合
pred1 = ensemble1.predict_single(sample)
pred2 = ensemble2.predict_single(sample)
final_pred = 0.6 * pred1 + 0.4 * pred2
```

### 2. 智能权重优化
```python
# 🆕 完全自动化的权重优化
stacker = auto_ensemble(
    experiment_dir="output/my_experiment",
    auto_load_validation=True
)

# 权重会根据验证集性能自动计算
print("自动优化的权重:")
for model_name, weight in stacker.model_weights.items():
    print(f"  {model_name}: {weight:.3f}")
```

### 3. 置信度估算
```python
result = stacker.predict([sample])
base_predictions = result['base_predictions'][0]

# 计算预测不确定性
std_dev = np.std(base_predictions)
confidence = 1 / (1 + std_dev)
prediction_interval = [
    np.mean(base_predictions) - 2 * std_dev,
    np.mean(base_predictions) + 2 * std_dev
]

print(f"置信度: {confidence:.4f}")
print(f"预测区间: {prediction_interval}")
```

## 最佳实践

### 1. 选择基础模型
- **多样性**：选择不同类型的算法（树模型、线性模型、神经网络等）
- **性能**：所有基础模型都应该有合理的性能
- **互补性**：模型在不同数据子集上有不同的强弱项

### 2. 数据加载策略
- **优先自动加载**：使用CRAFT实验目录的数据拆分，确保数据一致性
- **合理验证集大小**：平衡计算效率和评估准确性（建议50-200样本）
- **测试集保护**：用验证集进行模型选择，测试集仅用于最终评估

### 3. 权重设置策略
- **自动优化**：优先使用 `auto_ensemble()` 的自动权重计算
- **基于性能**：根据验证集R²或准确率设置权重
- **基于稳定性**：性能稳定的模型给予更高权重

### 4. 验证策略
- **数据拆分一致性**：使用CRAFT训练时相同的数据拆分
- **交叉验证**：对于小数据集，考虑使用交叉验证评估
- **时间分割**：时间序列数据使用时间分割验证

### 5. 避免过拟合
- **简单方法优先**：先尝试加权平均，再考虑元学习器
- **正则化**：元学习器使用正则化防止过拟合
- **验证数据分离**：确保验证数据与训练数据独立

## 性能提升建议

### 1. 模型选择
```python
# 好的组合：不同类型算法
models = ["xgb", "lgbm", "catboost", "rf", "ann"]

# 避免：相似算法
models = ["xgb", "lgbm"]  # 都是梯度提升
```

### 2. 智能集成
```python
# 🆕 智能元学习器：自动选择 + 元学习器训练
stacker = smart_ensemble_with_meta_learner(
    experiment_dir="output/my_experiment",
    meta_method="ridge",
    validation_size=100
)
```

### 3. 元学习器选择
- **数据量大**：随机森林或神经网络
- **数据量小**：Ridge回归或简单平均
- **需要概率**：逻辑回归（分类）

## 常见问题解决

### Q1：如何确保数据格式正确？
**解决方案：**
- 使用自动数据加载功能，确保格式一致性
- 检查SMILES列名和特征列是否匹配训练时的配置
- 查看 `experiment_dir/run_config.json` 确认原始配置

### Q2：验证数据加载失败？
**解决方案：**
- 检查 `original_data_splits` 目录是否存在
- 确认实验是否使用了正确的数据拆分模式
- 使用自定义验证数据作为备选方案

### Q3：堆叠后性能没有提升？
**解决方案：**
- 检查基础模型是否足够多样化
- 尝试不同的权重分配策略
- 使用更大的验证集评估性能

### Q4：元学习器过拟合？
**解决方案：**
- 增加正则化强度
- 使用更简单的元学习器（如Ridge）
- 增加验证数据量

### Q5：预测时间太长？
**解决方案：**
- 减少基础模型数量
- 使用加权平均代替元学习器
- 限制验证数据集大小

## 完整示例

查看 `stacking_example.py` 文件获取完整的使用示例，包括：
- 基础堆叠使用
- 自动数据加载
- 元学习器训练
- 性能评估对比
- 高级堆叠技巧
- 保存和加载模型
- 自定义验证数据

## 文件说明

- `model_stacking.py`：主要功能模块
- `data_loader.py`：数据加载工具模块（新增）
- `stacking_example.py`：详细使用示例
- `quick_stacking.py`：快速堆叠工具
- `README_stacking.md`：本使用指南

## 数据格式要求

### 输入数据格式
```python
# 字典格式（单样本）
sample = {
    'SMILES': 'CCO',
    'Solvent_1_SMILES': 'CC(=O)O',
    'Solvent_2_SMILES': 'CCN',
    # 其他特征列（如果有）
}

# 列表格式（多样本）
samples = [
    {'SMILES': 'CCO', 'Solvent_1_SMILES': 'CC(=O)O', ...},
    {'SMILES': 'c1ccccc1', 'Solvent_1_SMILES': 'CNC(=O)N', ...},
    # ...
]

# DataFrame格式也支持
df = pd.DataFrame(samples)
```

### 标签格式
```python
# 回归任务
labels = [12.5, 8.3, 15.2, ...]  # 连续值

# 分类任务
labels = ['class_A', 'class_B', 'class_A', ...]  # 类别标签
# 或
labels = [0, 1, 0, ...]  # 编码后的类别
```

通过合理使用模型堆叠和自动数据加载功能，您可以充分发挥CRAFT框架中多个训练模型的优势，获得更加稳定和准确的预测结果！

## 🚀 快速上手指令

```bash
# 1. 最简单的使用
python -c "
from model_stacking import auto_ensemble
stacker = auto_ensemble('output/your_experiment')
print('✓ 自动集成创建完成')
"

# 2. 运行完整示例
python stacking_example.py

# 3. 快速堆叠测试
python quick_stacking.py output/your_experiment
``` 