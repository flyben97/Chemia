# CRAFT 模型堆叠示例

这个目录包含了CRAFT模型堆叠功能的示例代码、配置文件和演示脚本。

## 📁 目录结构

```
examples/stacking/
├── README.md                    # 本文件，使用说明
├── configs/                     # 配置文件模板
│   ├── config_stacking.yaml         # 完整配置示例
│   ├── config_stacking_simple.yaml  # 简化配置示例  
│   ├── config_stacking_meta.yaml    # 元学习器配置示例
│   ├── template_basic_weighted.yaml # 基础加权模板
│   ├── template_simple_average.yaml # 简单平均模板
│   └── template_meta_learner.yaml   # 元学习器模板
└── demos/                       # 演示脚本
    ├── stacking_yaml_demo.py        # YAML配置演示
    ├── stacking_api_demo.py         # API接口演示
    ├── yaml_stacking_example.py     # YAML使用示例
    ├── stacking_example.py          # 基础堆叠示例
    └── quick_stacking.py            # 快速堆叠工具
```

## 🚀 快速开始

### 1. 使用YAML配置方式

```bash
# 运行YAML配置演示
cd examples/stacking/demos
python stacking_yaml_demo.py --config ../configs/config_stacking_simple.yaml

# 创建示例配置文件
python stacking_yaml_demo.py --create-sample-config
```

### 2. 使用API接口

```bash
# 运行API演示
python stacking_api_demo.py
```

### 3. 程序化使用

```python
from model_stacking import ModelStacker
from utils.stacking_ensemble import create_ensemble, auto_ensemble

# 方式1: 手动创建
stacker = ModelStacker("output/my_experiment")
stacker.add_model("xgb", weight=0.4)
stacker.add_model("lgbm", weight=0.6)

# 方式2: 使用工具函数
stacker = create_ensemble(
    "output/my_experiment", 
    ["xgb", "lgbm", "catboost"],
    weights=[0.4, 0.3, 0.3]
)

# 方式3: 自动优化
stacker = auto_ensemble("output/my_experiment")
```

## 📋 配置文件说明

### config_stacking_simple.yaml
最基础的配置，适合日常使用：
- 使用加权平均方法
- 包含3个模型
- 自动评估和保存

### config_stacking_meta.yaml  
使用元学习器的高级配置：
- 使用Ridge回归作为元模型
- 自动训练元学习器
- 包含完整的验证配置

### 模板文件
- `template_*.yaml`: 预定义的配置模板
- 可以通过`get_config_template()`函数获取

## 🔧 工具模块说明

### 核心模块
- `model_stacking.py`: 核心ModelStacker类
- `stacking_api.py`: 简化的API接口

### 工具模块 (utils/)
- `utils.stacking_ensemble`: 集成创建工具
- `utils.stacking_config`: 配置处理工具  
- `utils.stacking_evaluation`: 评估分析工具

## 📊 使用场景

### 1. 快速原型开发
使用`config_stacking_simple.yaml`进行快速测试

### 2. 生产环境部署
使用`config_stacking.yaml`的完整配置

### 3. 研究和实验
使用`config_stacking_meta.yaml`探索元学习器

### 4. 自动化工作流
使用`auto_ensemble()`自动选择最佳模型组合

## ⚠️ 注意事项

1. **实验目录**: 确保指定的experiment_dir包含训练好的模型
2. **模型兼容性**: 所有基础模型必须针对同一任务和数据集训练
3. **内存使用**: 堆叠会同时加载多个模型，注意内存消耗
4. **验证数据**: 元学习器需要验证数据进行训练

## 🆘 故障排除

### 常见问题
1. **模型加载失败**: 检查experiment_dir路径和模型文件
2. **配置验证错误**: 参考示例配置文件格式
3. **内存不足**: 减少同时使用的模型数量
4. **预测不一致**: 确保所有模型使用相同的特征和预处理

### 调试技巧
```bash
# 检查可用模型
python -c "from utils.stacking_ensemble import find_available_models; print(find_available_models('output/my_experiment'))"

# 获取推荐配置
python -c "from utils.stacking_ensemble import get_ensemble_recommendations; print(get_ensemble_recommendations('output/my_experiment'))"
```

## 📚 更多文档

详细文档位于 `docs/stacking/` 目录：
- 完整的API参考
- 最佳实践指南
- 高级配置说明
- 性能优化建议 