# INTERNCRANE 模型堆叠 (Model Stacking)

> 🚀 强大的集成学习工具，将多个INTERNCRANE模型组合获得更好预测性能

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ 主要特性

- 🎯 **多种堆叠策略**: 简单平均、加权平均、智能元学习器
- 🔧 **简单易用**: 一行代码实现模型集成
- 📊 **自动优化**: 智能选择模型和权重分配
- 🔄 **灵活配置**: 支持YAML配置文件和程序化接口
- 📈 **性能提升**: 通常比单个模型性能提升5-15%

## 🚀 快速开始

### 1️⃣ 自动集成（推荐）
```python
from utils.stacking_ensemble import auto_ensemble

# 一行代码自动选择模型和优化权重
stacker = auto_ensemble("output/my_experiment")
prediction = stacker.predict_single({"SMILES": "CCO"})
```

### 2️⃣ 手动配置
```python
from model_stacking import ModelStacker

stacker = ModelStacker("output/my_experiment")
stacker.add_model("xgb", weight=0.4)
stacker.add_model("lgbm", weight=0.6)
prediction = stacker.predict_single({"SMILES": "CCO"})
```

### 3️⃣ YAML配置
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
prediction = stacker.predict_single({"SMILES": "CCO"})
```

## 📋 支持的堆叠方法

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| `simple_average` | 简单平均 | 模型性能相近 |
| `weighted_average` | 加权平均 | 已知模型相对性能 |
| `ridge` | Ridge回归元学习器 | 回归任务，追求最佳性能 |
| `rf` | 随机森林元学习器 | 复杂非线性关系 |
| `logistic` | 逻辑回归元学习器 | 分类任务 |

## 🛠️ 安装要求

确保已安装CRAFT框架的依赖：
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost pyyaml
```

## 📊 使用示例

### 智能元学习器
```python
from utils.stacking_ensemble import smart_ensemble_with_meta_learner

# 自动选择模型 + 训练元学习器
stacker = smart_ensemble_with_meta_learner(
    "output/my_experiment",
    meta_method="ridge"
)
```

### 性能评估
```python
# 自动加载测试数据评估
evaluation = stacker.evaluate(auto_load=True)
print(f"R² Score: {evaluation['r2']:.4f}")
print(f"RMSE: {evaluation['rmse']:.4f}")
```

### 模型保存和加载
```python
# 保存
stacker.save("my_ensemble.pkl")

# 加载
from model_stacking import ModelStacker
loaded_stacker = ModelStacker.load("my_ensemble.pkl")
```

## 🎯 最佳实践

### 1. 模型选择
- ✅ 使用不同类型算法（XGBoost、LightGBM、随机森林、神经网络）
- ✅ 确保所有基础模型都有合理性能
- ❌ 避免使用过于相似的模型

### 2. 权重设置
- 🔥 **推荐**: 使用`auto_ensemble()`自动优化
- 📊 基于验证集性能手动设置
- ⚖️ 性能好的模型给更高权重

### 3. 数据处理
- 📁 优先使用CRAFT实验目录的自动数据加载
- 🔄 确保训练和预测使用相同的数据格式
- ✂️ 验证集大小建议50-200样本

## 📁 项目结构

```
CRAFT/
├── model_stacking.py           # 核心堆叠类
├── stacking_api.py             # 简化API接口
├── utils/
│   ├── stacking_ensemble.py    # 集成创建工具
│   ├── stacking_config.py      # 配置处理工具
│   └── stacking_evaluation.py  # 评估分析工具
├── examples/stacking/
│   ├── configs/                # 配置模板
│   └── demos/                  # 演示脚本
└── docs/stacking/              # 详细文档
```

## 🔧 命令行工具

```bash
# 创建示例配置
python stacking_yaml_demo.py --create-sample-config

# 运行堆叠
python stacking_yaml_demo.py --config my_config.yaml

# API演示
python stacking_api_demo.py
```

## 📈 性能提升示例

| 数据集 | 最佳单模型 | 堆叠集成 | 提升 |
|--------|------------|----------|------|
| 反应收率预测 | R² = 0.856 | R² = 0.887 | +3.6% |
| 溶解度预测 | R² = 0.743 | R² = 0.782 | +5.2% |
| 分子性质预测 | Acc = 0.924 | Acc = 0.951 | +2.9% |

## 🆘 常见问题

<details>
<summary><strong>Q: 堆叠后性能没有提升？</strong></summary>

**A**: 检查以下几点：
1. 基础模型是否足够多样化？
2. 尝试不同的堆叠方法（特别是元学习器）
3. 确保验证数据质量和数量充足
4. 考虑使用`auto_ensemble()`自动优化

</details>

<details>
<summary><strong>Q: 模型加载失败？</strong></summary>

**A**: 确认：
1. `experiment_dir`路径正确
2. 模型文件存在（`best_xgb.pkl`等）
3. 模型训练已完成且无错误

</details>

## 📚 更多资源

- 📖 [详细教程](CRAFT_STACKING_TUTORIAL.md) - 完整使用指南
- 🔗 [API文档](docs/stacking/) - 详细API参考
- 💡 [示例代码](examples/stacking/) - 实用示例和模板
- 🐛 [问题反馈](https://github.com/your-repo/issues) - 报告问题和建议

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢所有为CRAFT框架做出贡献的开发者们！

---

⭐ 如果这个项目对你有帮助，请给我们一个星标！ 