# INTERNCHEMIA 模型堆叠完整教程

> 📚 深入学习INTERNCHEMIA模型堆叠技术，从基础概念到高级应用

## 目录

1. [概述](#概述)
2. [快速开始](#快速开始)
3. [API接口详解](#api接口详解)
4. [YAML配置指南](#yaml配置指南)
5. [实际案例分析](#实际案例分析)
6. [最佳实践](#最佳实践)
7. [故障排除](#故障排除)

## 概述

### 什么是模型堆叠？

模型堆叠（Stacking）是一种高级集成学习技术，通过组合多个基础模型的预测结果来获得更准确、更稳定的预测性能。

### 为什么使用模型堆叠？

- **性能提升**: 通常比最佳单模型提升5-15%
- **减少过拟合**: 多模型组合降低单模型的偏差
- **提高稳定性**: 减少预测方差，提供更可靠的结果

### CRAFT堆叠系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   CRAFT 堆叠系统                         │
├─────────────────────────────────────────────────────────┤
│  核心层: ModelStacker, StackingPredictor               │
│  工具层: stacking_ensemble, stacking_config            │
│  应用层: YAML配置, 命令行工具, Python API               │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 方式1: 自动集成（推荐新手）

```python
from utils.stacking_ensemble import auto_ensemble

# 完全自动化：选择模型、优化权重、评估性能
stacker = auto_ensemble("output/my_experiment")

# 单样本预测
sample = {"SMILES": "CCO", "temperature": 80}
prediction = stacker.predict_single(sample)
print(f"预测结果: {prediction}")
```

### 方式2: 手动配置（精确控制）

```python
from model_stacking import ModelStacker

# 创建堆叠器
stacker = ModelStacker("output/my_experiment")
stacker.add_model("xgb", weight=0.4)
stacker.add_model("lgbm", weight=0.6)
stacker.set_stacking_method("weighted_average")

# 进行预测
prediction = stacker.predict_single(sample)
```

### 方式3: YAML配置

```yaml
# config.yaml
stacking:
  experiment_dir: "output/my_experiment"
  method: "weighted_average"
  models:
    - name: "xgb"
      weight: 0.4
    - name: "lgbm"
      weight: 0.6
```

```python
from model_stacking import ModelStacker
stacker = ModelStacker.from_yaml_config("config.yaml")
```

## API接口详解

### 核心类：ModelStacker

```python
class ModelStacker:
    def __init__(self, experiment_dir=None, models=None)
    def add_model(self, model_name: str, weight: float = 1.0)
    def set_stacking_method(self, method: str)
    def predict(self, data) -> Dict[str, Any]
    def predict_single(self, sample) -> Union[float, str, int]
    def evaluate(self, auto_load=True) -> Dict[str, Any]
    def save(self, filepath: str)
    
    @classmethod
    def from_yaml_config(cls, config_path: str)
```

### 工具函数

```python
from utils.stacking_ensemble import (
    create_ensemble,           # 快速创建集成
    auto_ensemble,            # 自动优化集成
    smart_ensemble_with_meta_learner,  # 智能元学习器
)

from utils.stacking_config import (
    create_sample_stacking_config,  # 创建示例配置
    get_config_template,       # 获取配置模板
)
```

## YAML配置指南

### 基本配置结构

```yaml
stacking:              # 必需：堆叠配置
  experiment_dir: ""   # 必需：实验目录
  method: ""           # 必需：堆叠方法
  models: []           # 必需：模型列表
  meta_model: {}       # 可选：元模型配置

evaluation: {}         # 可选：评估配置
save: {}              # 可选：保存配置
```

### 详细配置示例

```yaml
stacking:
  experiment_dir: "output/reaction_prediction"
  method: "weighted_average"
  
  models:
    - name: "xgb"
      weight: 0.4
      enabled: true
    - name: "lgbm"
      weight: 0.3
      enabled: true
    - name: "catboost"
      weight: 0.3
      enabled: true
      
  meta_model:
    auto_train: true
    validation:
      auto_load: true
      size: 200

evaluation:
  auto_evaluate: true
  use_test_set: true

save:
  save_stacker: true
  save_path: "output/ensemble_model.pkl"
```

### 命令行工具

```bash
# 创建示例配置
python stacking_yaml_demo.py --create-sample-config

# 使用配置运行
python stacking_yaml_demo.py --config my_config.yaml
```

## 实际案例分析

### 案例1: 分子性质预测

```python
from utils.stacking_ensemble import auto_ensemble

# 自动创建最优集成
stacker = auto_ensemble(
    experiment_dir="output/solubility_prediction",
    validation_size=200,
    available_models=['xgb', 'lgbm', 'catboost']
)

# 评估性能
evaluation = stacker.evaluate(auto_load=True)
print(f"集成模型 R²: {evaluation['r2']:.4f}")
```

### 案例2: 反应收率预测（元学习器）

```yaml
# reaction_stacking.yaml
stacking:
  experiment_dir: "output/reaction_yield"
  method: "ridge"
  models:
    - name: "xgb"
      enabled: true
    - name: "lgbm"  
      enabled: true
    - name: "catboost"
      enabled: true
  meta_model:
    auto_train: true
    validation:
      auto_load: true
      size: 150
```

运行：
```bash
python stacking_yaml_demo.py --config reaction_stacking.yaml
```

## 最佳实践

### 1. 模型选择策略

#### ✅ 推荐组合
```python
# 算法多样性组合
models = ["xgb", "lgbm", "catboost", "rf", "ann"]

# 基于集成类型的组合
tree_models = ["xgb", "lgbm", "catboost", "rf"]
```

#### ❌ 避免组合
```python
# 相似算法组合
similar_models = ["xgb", "lgbm"]  # 都是梯度提升
```

### 2. 权重设置原则

#### 自动权重优化（推荐）
```python
# 让系统自动计算最优权重
stacker = auto_ensemble("output/my_experiment")
print("系统计算的权重:", stacker.model_weights)
```

#### 基于性能的权重
```python
# 假设验证集性能
model_r2 = {"xgb": 0.86, "lgbm": 0.84, "catboost": 0.85}
weights = {"xgb": 0.5, "catboost": 0.3, "lgbm": 0.2}
```

### 3. 堆叠方法选择

- **简单任务**: 使用加权平均，快速有效
- **复杂任务**: 使用元学习器，可能获得更好性能
- **生产环境**: 加权平均更稳定

## 故障排除

### 常见错误及解决方案

#### 1. 模型加载失败
**错误**: `❌ 添加模型失败 xgb: [Errno 2] No such file or directory`

**解决**:
```python
# 检查路径和模型文件
import os
experiment_dir = "output/my_experiment"
print(f"目录存在: {os.path.exists(experiment_dir)}")
model_files = [f for f in os.listdir(experiment_dir) if f.endswith('.pkl')]
print(f"模型文件: {model_files}")
```

#### 2. 验证数据加载失败
**错误**: `自动加载验证数据失败`

**解决**:
```python
# 手动提供验证数据
validation_data = [{"SMILES": "CCO"}, {"SMILES": "CC(=O)O"}]
validation_labels = [12.5, 8.3]

stacker = auto_ensemble(
    experiment_dir=experiment_dir,
    validation_data=validation_data,
    true_labels=validation_labels,
    auto_load_validation=False
)
```

### 调试工具

```python
# 检查堆叠器状态
def check_stacker_status(stacker):
    print("📊 堆叠器状态检查:")
    print(f"  实验目录: {stacker.experiment_dir}")
    print(f"  基础模型数量: {len(stacker.base_models)}")
    print(f"  模型列表: {list(stacker.base_models.keys())}")
    print(f"  权重分配: {stacker.model_weights}")
    print(f"  堆叠方法: {stacker.stacking_method}")

check_stacker_status(stacker)
```

## 高级技巧

### 1. 多层堆叠

```python
# 第一层：创建专业化集成
tree_ensemble = create_ensemble(
    experiment_dir, 
    ["xgb", "lgbm", "catboost"], 
    method="weighted_average"
)

# 第二层：组合不同类型的集成
def two_layer_prediction(sample):
    tree_pred = tree_ensemble.predict_single(sample)
    # 其他集成预测...
    final_pred = 0.7 * tree_pred + 0.3 * other_pred
    return final_pred
```

### 2. 不确定性量化

```python
def predict_with_uncertainty(stacker, sample):
    """估计预测不确定性"""
    result = stacker.predict([sample])
    base_predictions = result['base_predictions'][0]
    
    return {
        'mean': np.mean(base_predictions),
        'std': np.std(base_predictions),
        'uncertainty': np.std(base_predictions) / np.mean(base_predictions)
    }

uncertainty_result = predict_with_uncertainty(stacker, test_sample)
print(f"预测: {uncertainty_result['mean']:.4f} ± {uncertainty_result['std']:.4f}")
```

## 性能优化

### 计算效率
- 选择最佳的3-5个模型，避免过多模型
- 使用批量预测而非循环单个预测
- 合理控制验证数据大小

### 方法选择（按速度排序）
1. `simple_average` - 最快
2. `weighted_average` - 很快  
3. `ridge` - 中等
4. `rf` - 较慢

## 总结

CRAFT模型堆叠系统提供了从简单到高级的完整解决方案：

1. **入门级**: 使用`auto_ensemble()`一键创建
2. **进阶级**: YAML配置文件精确控制
3. **专家级**: 自定义元学习器和高级技巧

通过合理使用这些功能，您可以充分发挥多个CRAFT模型的优势，获得更准确、更稳定的预测结果。

## 参考资源

- [简洁README](CRAFT_STACKING_README.md) - 快速上手指南
- [示例代码](examples/stacking/) - 实用示例和模板
- [API文档](docs/stacking/) - 详细API参考
