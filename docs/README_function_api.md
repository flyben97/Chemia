# CRAFT 函数式预测API

这是一个简单的函数式接口，让您可以在Python代码中直接调用CRAFT模型进行预测，而无需使用命令行或配置文件。

## 🚀 快速开始

### 基本用法

```python
from prediction_api import load_model, predict_single

# 加载模型
predictor = load_model("output/my_experiment", "xgb")

# 预测单个样本
result = predict_single(predictor, {
    "SMILES": "CCO", 
    "Solvent_1_SMILES": "CC(=O)O",
    "Solvent_2_SMILES": "CCN"
})

print(f"预测值: {result}")
```

### 一步预测

```python
from prediction_api import quick_predict

# 一步完成加载和预测
result = quick_predict("output/my_experiment", "xgb", {
    "SMILES": "CCO",
    "Solvent_1_SMILES": "CC(=O)O"
})

print(f"预测结果: {result['predictions'][0]}")
```

## 📚 主要功能

### 1. 模型加载
```python
from prediction_api import load_model

# 从实验目录加载
predictor = load_model("output/experiment_dir", "model_name")

# 从直接文件路径加载
from prediction_api import load_model_from_files
predictor = load_model_from_files(
    model_path="path/to/model.json",
    config_path="path/to/config.json",
    scaler_path="path/to/scaler.joblib"  # 可选
)
```

### 2. 单样本预测
```python
# 返回单个预测值
prediction = predict_single(predictor, sample_dict)

# 返回详细结果（包含概率、任务类型等）
result = predict(predictor, sample_dict)
```

### 3. 批量预测
```python
# 使用字典列表
samples = [
    {"SMILES": "CCO", "Solvent_1_SMILES": "CC(=O)O"},
    {"SMILES": "c1ccccc1", "Solvent_1_SMILES": "CNC(=O)N"}
]
results = predict(predictor, samples)

# 使用DataFrame
import pandas as pd
df = pd.DataFrame(samples)
results = predict(predictor, df)
```

## 🔧 支持的输入格式

### 字典格式（单个样本）
```python
sample = {
    "SMILES": "CCO",
    "Solvent_1_SMILES": "CC(=O)O",
    "Solvent_2_SMILES": "CCN",
    "feature_1": 1.2,  # 预计算特征（如果有）
    "feature_2": 3.4
}
```

### 列表格式（多个样本）
```python
samples = [
    {"SMILES": "CCO", "Solvent_1_SMILES": "CC(=O)O"},
    {"SMILES": "c1ccccc1", "Solvent_1_SMILES": "CNC(=O)N"}
]
```

### DataFrame格式
```python
df = pd.DataFrame({
    "SMILES": ["CCO", "c1ccccc1"],
    "Solvent_1_SMILES": ["CC(=O)O", "CNC(=O)N"]
})
```

## 📊 返回结果格式

### 单样本预测（predict_single）
```python
result = predict_single(predictor, sample)
# 返回: 12.3456 (单个数值)
```

### 详细预测结果（predict）
```python
result = predict(predictor, sample)
# 返回字典:
{
    'predictions': array([12.3456]),     # 预测值
    'probabilities': None,               # 分类概率（分类任务才有）
    'task_type': 'regression',           # 任务类型
    'n_samples': 1                       # 样本数量
}
```

### 分类任务结果
```python
# 分类任务会包含标签和概率
{
    'predictions': array(['High']),           # 解码后的标签
    'predictions_encoded': array([1]),        # 编码的预测值
    'probabilities': array([[0.23, 0.77]]), # 各类别概率
    'task_type': 'classification',
    'n_samples': 1
}
```

## 💡 实际应用示例

### 集成到计算函数中
```python
def calculate_reaction_yield(reactant, solvent1, solvent2):
    """计算反应收率"""
    sample = {
        'SMILES': reactant,
        'Solvent_1_SMILES': solvent1,
        'Solvent_2_SMILES': solvent2
    }
    
    result = quick_predict("output/my_experiment", "xgb", sample)
    predicted_yield = result['predictions'][0]
    
    return {
        'yield': predicted_yield,
        'confidence': 'high' if predicted_yield > 0.8 else 'low'
    }

# 使用
yield_info = calculate_reaction_yield('CCO', 'CC(=O)O', 'CCN')
print(f"预测收率: {yield_info['yield']:.2%}")
```

### 批量筛选化合物
```python
def screen_compounds(compound_list, solvent):
    """批量筛选化合物"""
    samples = [
        {'SMILES': smiles, 'Solvent_1_SMILES': solvent}
        for smiles in compound_list
    ]
    
    results = quick_predict("output/my_experiment", "xgb", samples)
    predictions = results['predictions']
    
    # 筛选高活性化合物
    good_compounds = [
        compound_list[i] for i, pred in enumerate(predictions)
        if pred > 0.7
    ]
    
    return good_compounds

# 使用
compounds = ['CCO', 'c1ccccc1', 'CCCC', 'CCN']
good_ones = screen_compounds(compounds, 'CC(=O)O')
print(f"高活性化合物: {good_ones}")
```

### 模型比较
```python
def compare_models(sample, models=['xgb', 'lgbm', 'catboost']):
    """比较不同模型的预测结果"""
    experiment_dir = "output/my_experiment"
    results = {}
    
    for model_name in models:
        try:
            result = quick_predict(experiment_dir, model_name, sample)
            results[model_name] = result['predictions'][0]
        except Exception as e:
            results[model_name] = f"Error: {e}"
    
    return results

# 使用
sample = {'SMILES': 'CCO', 'Solvent_1_SMILES': 'CC(=O)O'}
comparison = compare_models(sample)
for model, prediction in comparison.items():
    print(f"{model}: {prediction}")
```

## ⚠️ 注意事项

1. **输入数据格式**：确保输入数据包含训练时使用的所有必需列
2. **模型路径**：确保实验目录路径正确，包含训练好的模型文件
3. **特征一致性**：预测时会自动使用训练时相同的特征生成方法
4. **错误处理**：建议使用try-except处理可能的错误

## 🔗 相关文件

- **`prediction_api.py`** - 主要API模块
- **`prediction_api_example.py`** - 详细使用示例
- **`run_prediction_standalone.py`** - 命令行预测工具
- **`quick_predict.py`** - 简化的命令行工具

## 🆘 常见问题

**Q: 如何知道需要哪些输入列？**
A: 查看训练时的配置文件，或者从训练数据中查看SMILES列和特征列。

**Q: 预测很慢怎么办？**
A: 对于大批量预测，考虑分批处理，或者预先加载模型避免重复加载。

**Q: 如何处理预测错误？**
A: 使用try-except捕获异常，检查输入数据格式和模型路径是否正确。

---

这个函数式API让您可以轻松地将CRAFT模型集成到更大的计算流程中，无需处理复杂的配置文件或命令行参数。 