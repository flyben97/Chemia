# CRAFT 预测和模型堆叠文件总览

本文档提供CRAFT预测系统所有文件的完整总览，包括基础预测功能和高级模型堆叠功能。

## 📁 文件结构

```
CRAFT预测系统/
├── 🎯 基础预测功能
│   ├── config_prediction.yaml          # 通用预测配置模板
│   ├── config_prediction_example.yaml  # 配置使用示例（中文注释）
│   ├── run_prediction_standalone.py    # 功能完整的独立预测脚本
│   ├── quick_predict.py                # 简化快速预测工具
│   ├── prediction_api.py               # 函数式预测API
│   └── prediction_api_example.py       # 函数式API使用示例
│
├── 🔗 模型堆叠功能
│   ├── model_stacking.py               # 模型堆叠核心模块
│   ├── stacking_example.py             # 堆叠详细使用示例
│   ├── quick_stacking.py               # 快速堆叠工具
│   └── README_stacking.md              # 堆叠使用指南
│
├── 🆕 自动数据加载功能
│   └── data_loader.py                  # 数据加载工具模块
│
└── 📚 文档资料
    ├── README_prediction.md            # 基础预测使用指南
    ├── README_function_api.md          # 函数式API专门指南
    └── PREDICTION_FILES_SUMMARY.md     # 本文档
```

## 🎯 基础预测功能

### 1. 配置文件
- **`config_prediction.yaml`**：通用配置模板，包含所有可能选项
- **`config_prediction_example.yaml`**：实用配置示例，预填写常用配置和中文注释

### 2. 预测工具
- **`run_prediction_standalone.py`**：功能最完整的预测脚本，支持四种使用模式
- **`quick_predict.py`**：最简化的一键预测工具
- **`prediction_api.py`**：可在代码中调用的函数式API

### 3. 示例代码
- **`prediction_api_example.py`**：详细的API使用示例，覆盖各种使用场景

## 🔗 模型堆叠功能

### 4. 堆叠核心
- **`model_stacking.py`**：模型堆叠主模块，支持多种堆叠策略和元学习器

### 5. 堆叠工具
- **`quick_stacking.py`**：简化的一键堆叠工具
- **`stacking_example.py`**：详细的堆叠使用示例

### 6. 堆叠文档
- **`README_stacking.md`**：完整的堆叠使用指南和最佳实践

## 🆕 自动数据加载功能

### 7. 数据加载工具
- **`data_loader.py`**：智能数据加载模块，自动从CRAFT实验目录读取验证数据

## 📊 功能对比表

| 功能特性 | quick_predict | run_prediction | prediction_api | model_stacking | quick_stacking |
|---------|---------------|----------------|----------------|----------------|----------------|
| 使用难度 | ⭐ 最简单 | ⭐⭐ 简单 | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 复杂 | ⭐⭐ 简单 |
| 功能完整性 | ⭐⭐ 基础 | ⭐⭐⭐⭐⭐ 完整 | ⭐⭐⭐⭐ 完整 | ⭐⭐⭐⭐⭐ 高级 | ⭐⭐⭐ 中等 |
| 自动数据加载 | ❌ | ❌ | ❌ | ✅ 完整支持 | ✅ 支持 |
| 配置灵活性 | ❌ | ✅ 完整 | ❌ | ✅ 完整 | ❌ |
| 交互模式 | ❌ | ✅ | ❌ | ❌ | ❌ |
| 批量处理 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 代码集成 | ❌ | ❌ | ✅ 专门设计 | ✅ | ✅ |
| 模型集成 | ❌ | ❌ | ❌ | ✅ 核心功能 | ✅ 简化版 |
| 错误处理 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🚀 使用场景推荐

### 快速开始（5分钟）
```bash
# 1. 最简单预测
python quick_predict.py output/my_experiment "CCO"

# 2. 最简单堆叠（🆕 自动数据加载）
python -c "
from model_stacking import auto_ensemble
stacker = auto_ensemble('output/my_experiment')
print('✓ 自动集成创建完成')
"
```

### 初学者用户
1. **single预测**: `quick_predict.py` - 一行命令搞定
2. **简单集成**: `quick_stacking.py` - 自动堆叠多个模型
3. **配置学习**: `config_prediction_example.yaml` - 学习配置选项

### 有经验的用户
1. **灵活预测**: `run_prediction_standalone.py` - 完整的配置控制
2. **自动集成**: `model_stacking.py` 中的 `auto_ensemble()` - 自动优化
3. **自定义集成**: `model_stacking.py` - 完全控制堆叠策略

### 开发者集成
1. **函数调用**: `prediction_api.py` - 直接在代码中调用
2. **智能集成**: `smart_ensemble_with_meta_learner()` - 元学习器自动训练
3. **自动数据**: `data_loader.py` - 自动读取训练时的数据拆分

### 研究和实验
1. **高级堆叠**: `model_stacking.py` - 多种元学习器和策略
2. **性能对比**: `stacking_example.py` - 详细的实验示例
3. **自定义评估**: 自动从实验目录加载测试数据进行评估

## 🆕 新增亮点功能

### 1. 完全自动化集成
```python
# 一行代码完成：数据加载 + 模型选择 + 权重优化
stacker = auto_ensemble("output/my_experiment")
```

### 2. 智能元学习器
```python
# 自动选择模型 + 训练元学习器
stacker = smart_ensemble_with_meta_learner(
    "output/my_experiment",
    meta_method="ridge"
)
```

### 3. 自动数据管理
- 自动读取CRAFT训练时的原始数据拆分
- 智能格式转换和验证数据准备
- 支持从`original_data_splits/`目录自动加载
- 备用自定义数据加载选项

### 4. 一键性能评估
```python
# 自动加载测试数据进行评估
evaluation = stacker.evaluate(auto_load=True)
```

## 📚 学习路径

### 第一步：基础预测（30分钟）
1. 运行 `python quick_predict.py output/my_experiment "CCO"`
2. 阅读 `README_prediction.md` 的快速开始部分
3. 试试 `config_prediction_example.yaml` 的配置

### 第二步：模型堆叠（1小时）
1. 🆕 尝试自动集成：`auto_ensemble("output/my_experiment")`
2. 运行 `stacking_example.py` 查看详细示例
3. 阅读 `README_stacking.md` 了解原理

### 第三步：高级应用（2小时）
1. 🆕 测试智能元学习器：`smart_ensemble_with_meta_learner()`
2. 学习 `prediction_api.py` 进行代码集成
3. 🆕 探索自动数据加载功能

### 第四步：实际项目（按需）
1. 根据具体需求选择合适的工具
2. 🆕 使用自动数据加载确保数据一致性
3. 参考最佳实践进行优化

## 💡 最佳实践

### 数据一致性（🆕）
- 优先使用自动数据加载，确保与训练时数据拆分一致
- 验证SMILES列名和特征列配置是否匹配
- 检查 `experiment_dir/run_config.json` 确认原始配置

### 模型选择
- 从单模型预测开始，确保基础功能正常
- 🆕 使用 `auto_ensemble()` 进行智能模型选择和权重优化
- 对于复杂任务，尝试元学习器集成

### 性能优化
- 🆕 使用自动优化功能避免手动调参
- 对于大数据集，限制验证数据大小提高效率
- 选择合适的堆叠方法平衡性能和复杂度

### 部署建议
- 生产环境使用 `prediction_api.py` 的函数接口
- 🆕 保存完整的堆叠模型以便重复使用
- 定期使用测试集评估模型性能

## 🔧 故障排除

### 常见问题

1. **数据加载失败**（🆕）
   - 检查 `original_data_splits/` 目录是否存在
   - 确认实验使用了正确的数据拆分模式
   - 使用自定义验证数据作为备选

2. **模型加载错误**
   - 验证实验目录路径正确性
   - 检查模型文件是否完整
   - 确认模型训练成功完成

3. **预测格式错误**
   - 🆕 使用自动数据加载确保格式一致
   - 检查SMILES列名是否匹配
   - 验证输入数据类型正确

4. **堆叠性能不佳**
   - 🆕 尝试自动优化权重计算
   - 确保基础模型足够多样化
   - 使用更大的验证集进行评估

### 获取帮助
- 查看相关README文件的详细说明
- 运行示例代码了解正确用法
- 🆕 检查自动数据加载的状态信息
- 使用详细的错误输出进行调试

## 🎉 总结

CRAFT预测系统现在提供了从最简单的一键预测到最高级的智能集成的完整解决方案。新增的自动数据加载功能大大简化了使用流程，确保了数据一致性，让用户能够更专注于模型的预测性能而不是数据管理细节。

选择适合您需求的工具，从简单开始，逐步探索更高级的功能！ 