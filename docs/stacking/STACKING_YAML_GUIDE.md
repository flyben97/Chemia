# CRAFT 模型堆叠 YAML 配置指南

这个指南展示如何使用YAML配置文件来对训练好的CRAFT模型进行堆叠，获得更强大的预测性能。

## 🚀 快速开始

### 1. 准备工作

确保你已经有训练好的CRAFT模型：

```bash
# 假设你的模型在这个目录下
ls output/my_experiment/
# 应该能看到: best_xgb.pkl, best_lgbm.pkl, best_catboost.pkl 等文件
```

### 2. 创建配置文件

创建一个简单的配置文件 `my_stacking.yaml`：

```yaml
stacking:
  experiment_dir: "output/my_experiment"
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

evaluation:
  auto_evaluate: true
  use_test_set: true

save:
  save_stacker: true
  save_path: "output/ensemble_model.pkl"
```

### 3. 运行模型堆叠

```bash
python stacking_yaml_demo.py --config my_stacking.yaml
```

## 📋 配置文件详解

### 基本配置结构

```yaml
stacking:              # 堆叠配置部分
  experiment_dir: ""   # 实验目录路径
  method: ""           # 堆叠方法
  models: []           # 模型列表
  meta_model: {}       # 元模型配置（可选）

evaluation: {}         # 评估配置
save: {}              # 保存配置
advanced: {}          # 高级选项（可选）
```

### 堆叠方法说明

#### 1. 简单平均 (Simple Average)
```yaml
stacking:
  method: "simple_average"
  models:
    - name: "xgb"
      enabled: true
    - name: "lgbm"
      enabled: true
```

所有模型的预测结果取平均值，权重相等。

#### 2. 加权平均 (Weighted Average)
```yaml
stacking:
  method: "weighted_average"
  models:
    - name: "xgb"
      weight: 0.4      # 权重越大，影响越大
      enabled: true
    - name: "lgbm"
      weight: 0.6
      enabled: true
```

根据指定权重对模型预测结果进行加权平均。

#### 3. 元学习器 (Meta-Learner)
```yaml
stacking:
  method: "ridge"      # 可选: ridge, rf, logistic
  models:
    - name: "xgb"
      enabled: true
    - name: "lgbm"
      enabled: true
  meta_model:
    auto_train: true
    validation:
      auto_load: true
      size: 200
```

使用机器学习算法自动学习如何组合基础模型的预测结果。

## 📁 完整使用案例

### 案例1：基础加权集成

**场景**: 对XGBoost、LightGBM、CatBoost进行加权集成

**配置文件** (`config_basic_ensemble.yaml`):
```yaml
stacking:
  experiment_dir: "output/reaction_prediction"
  method: "weighted_average"
  models:
    - name: "xgb"
      weight: 0.4
      enabled: true
    - name: "lgbm"
      weight: 0.35
      enabled: true
    - name: "catboost"
      weight: 0.25
      enabled: true

evaluation:
  auto_evaluate: true
  use_test_set: true
  compare_with_base: true

save:
  save_stacker: true
  save_path: "output/basic_ensemble.pkl"
  results_dir: "output/ensemble_results"
```

**运行**:
```bash
python stacking_yaml_demo.py --config config_basic_ensemble.yaml
```

**预期输出**:
```
🚀 开始YAML配置模型堆叠
============================================================
📋 加载配置文件: config_basic_ensemble.yaml

🔧 创建模型堆叠器...
✓ 添加模型: xgb (权重: 0.4)
✓ 添加模型: lgbm (权重: 0.35)
✓ 添加模型: catboost (权重: 0.25)
✓ 设置堆叠方法: weighted_average
✓ 从YAML配置创建堆叠器: config_basic_ensemble.yaml

📊 开始自动评估...
📈 评估结果:
----------------------------------------
  R² Score: 0.8756
  RMSE: 0.2134
  MAE: 0.1642

🔍 基础模型性能比较:
----------------------------------------
  xgb: R²=0.8621, RMSE=0.2245
  lgbm: R²=0.8598, RMSE=0.2267
  catboost: R²=0.8632, RMSE=0.2238

💾 保存堆叠器到: output/basic_ensemble.pkl
✓ 堆叠模型已保存到: output/basic_ensemble.pkl

✅ 模型堆叠完成!
🎉 模型堆叠成功完成!
   堆叠方法: weighted_average
   模型数量: 3
```

### 案例2：智能元学习器

**场景**: 使用Ridge回归自动学习最优组合权重

**配置文件** (`config_meta_ensemble.yaml`):
```yaml
stacking:
  experiment_dir: "output/reaction_prediction"
  method: "ridge"
  models:
    - name: "xgb"
      enabled: true
    - name: "lgbm"
      enabled: true
    - name: "catboost"
      enabled: true
    - name: "rf"
      enabled: true
    - name: "ann"
      enabled: true
  meta_model:
    auto_train: true
    validation:
      auto_load: true
      size: 150

evaluation:
  auto_evaluate: true
  use_test_set: true
  compare_with_base: true

save:
  save_stacker: true
  save_path: "output/meta_ensemble.pkl"
  results_dir: "output/meta_results"
  save_evaluation: true
```

**运行**:
```bash
python stacking_yaml_demo.py --config config_meta_ensemble.yaml
```

### 案例3：在Python代码中使用

你也可以在Python代码中直接使用YAML配置：

```python
from model_stacking import load_stacking_config_from_yaml

# 从YAML文件创建堆叠器
stacker = load_stacking_config_from_yaml("my_config.yaml")

# 进行预测
test_sample = {
    "SMILES": "CCO",
    "temperature": 80,
    "pressure": 1.0
}

prediction = stacker.predict_single(test_sample)
print(f"预测结果: {prediction}")

# 批量预测
test_data = [
    {"SMILES": "CCO", "temperature": 80},
    {"SMILES": "CC(=O)O", "temperature": 100}
]

results = stacker.predict(test_data)
print(f"批量预测: {results['predictions']}")
```

### 案例4：加载已保存的模型

```python
from model_stacking import ModelStacker

# 加载已保存的堆叠器
stacker = ModelStacker.load("output/ensemble_model.pkl")

# 直接使用
prediction = stacker.predict_single({"SMILES": "CCO"})
```

## 🔧 高级功能

### 1. 动态模型启用/禁用

```yaml
stacking:
  models:
    - name: "xgb"
      weight: 0.4
      enabled: true
    - name: "lgbm"
      weight: 0.3
      enabled: true
    - name: "catboost"
      weight: 0.3
      enabled: false    # 临时禁用
```

### 2. 自动数据加载控制

```yaml
stacking:
  meta_model:
    validation:
      auto_load: true   # 自动从实验目录加载验证数据
      size: 200         # 限制验证数据大小
```

### 3. 详细的保存选项

```yaml
save:
  save_stacker: true
  save_path: "models/my_ensemble.pkl"
  results_dir: "results/ensemble_analysis"
  save_evaluation: true
  save_config_copy: true
```

## 🛠️ 命令行工具

### 创建示例配置文件

```bash
# 创建默认示例配置
python stacking_yaml_demo.py --create-sample-config

# 指定输出路径
python stacking_yaml_demo.py --create-sample-config --output my_config.yaml
```

### 使用配置文件运行

```bash
# 基本使用
python stacking_yaml_demo.py --config config_stacking.yaml

# 查看帮助
python stacking_yaml_demo.py --help
```

## 🎯 最佳实践

### 1. 权重设置建议

- **性能优先**: 根据各模型在验证集上的表现设置权重
- **多样性优先**: 给不同类型的模型（如树模型vs神经网络）相似权重
- **经验法则**: XGBoost和LightGBM通常表现相近，可以给较高权重

### 2. 模型选择建议

- **至少使用3个模型**: 确保集成的稳定性
- **算法多样性**: 选择不同类型的算法（梯度提升、随机森林、神经网络等）
- **性能过滤**: 只选择在验证集上表现良好的模型

### 3. 堆叠方法选择

- **简单任务**: 使用加权平均，快速有效
- **复杂任务**: 使用元学习器，可能获得更好性能
- **生产环境**: 加权平均更稳定，元学习器需要更多验证

## ❗ 常见问题

### Q1: 配置文件验证失败

**错误**: `ValueError: 配置文件必须包含 'stacking' 部分`

**解决**: 确保YAML文件包含正确的stacking部分：
```yaml
stacking:
  experiment_dir: "your/path"
  # ... 其他配置
```

### Q2: 模型加载失败

**错误**: `❌ 添加模型失败 xgb: ...`

**解决**: 
1. 检查experiment_dir路径是否正确
2. 确保模型文件存在（如`best_xgb.pkl`）
3. 检查模型是否训练完成

### Q3: 验证数据加载失败

**错误**: `自动加载验证数据失败`

**解决**:
1. 确保实验目录包含原始数据
2. 手动提供验证数据：
```python
stacker.fit_meta_model(validation_data=your_data, true_labels=your_labels)
```

### Q4: 权重不生效

**问题**: 设置了权重但结果没有变化

**解决**: 确保使用的是`weighted_average`方法：
```yaml
stacking:
  method: "weighted_average"  # 不是 "simple_average"
```

## 📚 更多资源

- 查看 `model_stacking.py` 了解详细的API文档
- 运行 `python model_stacking.py` 查看内置示例
- 参考现有的配置文件模板：
  - `config_stacking.yaml`: 完整配置
  - `config_stacking_simple.yaml`: 简化配置  
  - `config_stacking_meta.yaml`: 元学习器配置

## 🔄 更新日志

- **v1.0**: 基础YAML配置支持
- **v1.1**: 添加元学习器支持
- **v1.2**: 增强验证和错误处理
- **v1.3**: 添加命令行工具和示例生成 