# CRAFT 堆叠模型API 快速开始

这是CRAFT模型堆叠功能的简洁API，提供了比`model_stacking.py`更易用的接口。

## 🚀 快速使用

### 1️⃣ 方法一：YAML配置方式

```python
from stacking_api import load_stacker_from_config, stack_predict

# 加载堆叠器
stacker = load_stacker_from_config("my_stacking_config.yaml")

# 进行预测
result = stack_predict(stacker, {"SMILES": "CCO", "temperature": 80})
print(f"预测结果: {result['predictions']}")
```

### 2️⃣ 方法二：程序化创建

```python
from stacking_api import create_stacker, stack_predict_single

# 创建堆叠器
stacker = create_stacker(
    experiment_dir="output/my_experiment",
    model_names=["xgb", "lgbm", "catboost"],
    weights=[0.4, 0.3, 0.3],
    method="weighted_average"
)

# 单个样本预测
prediction = stack_predict_single(stacker, {"SMILES": "CCO"})
print(f"预测值: {prediction}")
```

### 3️⃣ 方法三：一步预测

```python
from stacking_api import quick_stack_predict

# 一步完成加载和预测
result = quick_stack_predict("config.yaml", test_data)
```

## 📋 创建配置文件

### 使用预定义模板

```python
from utils.stacking_config import get_config_template

# 获取基础加权模板
config = get_config_template("basic_weighted", "output/my_experiment")

# 保存到文件
from utils.stacking_config import save_yaml_config
save_yaml_config(config, "my_stacking.yaml")
```

### 可用模板

- `basic_weighted`: 基础加权平均
- `simple_average`: 简单平均
- `meta_learner`: 智能元学习器

### 手动创建配置

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

## 🔧 API 参考

### 核心类

```python
class StackingPredictor:
    def predict(self, data) -> Dict[str, Any]           # 批量预测
    def predict_single(self, sample) -> float|str|int   # 单样本预测
    def evaluate(self, auto_load=True) -> Dict[str, Any] # 评估性能
    def save(self, filepath: str) -> None               # 保存模型
    def get_info(self) -> Dict[str, Any]                # 获取信息
```

### 主要函数

```python
# 加载和创建
load_stacker_from_config(config_path: str) -> StackingPredictor
create_stacker(experiment_dir, model_names, weights=None, method="weighted_average") -> StackingPredictor

# 预测
stack_predict(predictor, data) -> Dict[str, Any]
stack_predict_single(predictor, sample) -> float|str|int
quick_stack_predict(config_path, data) -> Dict[str, Any]
```

## 💡 最佳实践

### 堆叠方法选择

- **简单任务**: `weighted_average` - 快速有效
- **复杂任务**: `ridge` - 元学习器自动优化权重
- **快速原型**: `simple_average` - 无需设置权重

### 权重设置

```python
# 基于模型性能设置权重
weights = [0.4, 0.3, 0.3]  # XGBoost表现最好，给最高权重

# 或让元学习器自动学习
method = "ridge"  # 自动学习最优组合
```

### 模型选择

```python
# 推荐组合：多样性 + 性能
model_names = ["xgb", "lgbm", "catboost", "rf"]  # 不同算法类型
```

## 🏃‍♂️ 完整示例

```python
#!/usr/bin/env python3
from stacking_api import create_stacker, stack_predict
from utils.stacking_config import create_sample_stacking_config

# 1. 创建配置文件
config = create_sample_stacking_config(
    experiment_dir="output/reaction_prediction",
    model_names=["xgb", "lgbm", "catboost"],
    weights=[0.4, 0.35, 0.25],
    output_path="reaction_stacking.yaml"
)

# 2. 加载堆叠器
from stacking_api import load_stacker_from_config
stacker = load_stacker_from_config("reaction_stacking.yaml")

# 3. 进行预测
test_data = [
    {"SMILES": "CCO", "temperature": 80, "pressure": 1.0},
    {"SMILES": "CC(=O)O", "temperature": 100, "pressure": 1.5}
]

results = stack_predict(stacker, test_data)
print(f"堆叠预测结果: {results['predictions']}")
print(f"使用方法: {results['stacking_method']}")
print(f"模型数量: {len(results['model_names'])}")

# 4. 评估性能
evaluation = stacker.evaluate()
print(f"R² Score: {evaluation.get('r2', 'N/A')}")

# 5. 保存堆叠器
stacker.save("my_reaction_ensemble.pkl")
```

## 📚 相关文件

- `stacking_api.py` - 主要API接口
- `utils/stacking_config.py` - 配置工具
- `model_stacking.py` - 核心实现
- `stacking_yaml_demo.py` - 命令行工具
- `STACKING_YAML_GUIDE.md` - 详细指南

## ⚡ 快速命令

```bash
# 创建示例配置
python -c "from utils.stacking_config import get_config_template, save_yaml_config; save_yaml_config(get_config_template('basic_weighted', 'output/my_exp'), 'my_config.yaml')"

# 运行演示
python stacking_api_demo.py

# 命令行工具
python stacking_yaml_demo.py --config my_config.yaml
``` 