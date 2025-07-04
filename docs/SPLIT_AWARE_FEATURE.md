# CRAFT 模型堆叠：Split-Aware 功能详解

## 🎯 功能概述

**Split-Aware** 是CRAFT模型堆叠系统的新功能，它能够智能地根据原实验的数据拆分策略（`split_mode`）来选择最合适的验证数据集进行堆叠模型训练。这确保了堆叠验证与原实验的数据拆分策略保持一致，避免数据泄露并维持实验的严谨性。

## 🔍 问题背景

在模型堆叠过程中，我们需要验证数据来训练元学习器或评估堆叠效果。传统做法是：
1. 优先使用 `validation set`
2. 如果没有 `validation set`，则使用 `test set`

但这种做法没有考虑到原实验的数据拆分策略，可能导致：
- **数据使用不一致**：原实验是cross_validation但堆叠却用validation set
- **潜在的数据泄露**：不合理的数据集选择可能破坏实验设计

## ✨ Split-Aware 工作原理

### 当 `split_aware: true` 时

系统会自动检测原实验的 `split_mode` 并智能选择验证数据：

#### 📊 Train-Valid-Test 模式
```yaml
# 原实验配置
split_mode: "train_valid_test"
split_config:
  train_valid_test:
    valid_size: 0.05  # 5% validation
    test_size: 0.05   # 5% test
```

**选择策略**：优先使用 **validation set** 进行堆叠验证
- ✅ **正确做法**：使用专门预留的validation set训练元学习器
- ✅ **保护test set**：确保test set仅用于最终评估，避免污染

#### 🔄 Cross-Validation 模式
```yaml
# 原实验配置  
split_mode: "cross_validation"
split_config:
  cross_validation:
    n_folds: 5
    test_size_for_cv: 0.2  # 20% test set
```

**选择策略**：使用 **test set** 进行堆叠验证
- ✅ **符合CV逻辑**：Cross-validation模式通常没有专门的validation set
- ✅ **一致性**：与原实验的数据使用策略保持一致

### 当 `split_aware: false` 时

使用传统逻辑：
1. 优先使用 validation set
2. 没有validation set时使用 test set

## 🚀 使用方法

### 1. YAML配置

```yaml
stacking:
  experiment_dir: output/your_experiment
  method: ridge
  models:
    - name: xgb
      enabled: true
    - name: catboost  
      enabled: true
  
  meta_model:
    auto_train: true
    validation:
      auto_load: true
      size: 200
      split_aware: true  # 🔑 启用智能数据选择
```

### 2. Python API

```python
from model_stacking import ModelStacker

# 创建堆叠器
stacker = ModelStacker(experiment_dir="output/your_experiment")
stacker.add_model("xgb")
stacker.add_model("catboost")
stacker.set_stacking_method("ridge")

# 训练元模型（启用split_aware）
stacker.fit_meta_model(
    auto_load=True,
    validation_size=200,
    split_aware=True  # 🔑 启用智能数据选择
)
```

### 3. 便捷函数

```python
from utils.stacking_ensemble import auto_ensemble, smart_ensemble_with_meta_learner

# 自动集成（智能数据选择）
stacker = auto_ensemble(
    experiment_dir="output/your_experiment",
    split_aware=True
)

# 智能元学习器集成
stacker = smart_ensemble_with_meta_learner(
    experiment_dir="output/your_experiment",
    meta_learner="ridge",
    split_aware=True
)
```

## 📊 实际效果展示

### 运行输出示例

```bash
🔄 自动从实验目录加载验证数据...
📊 数据拆分摘要:
  train: 1469 样本 (90.0%)
  val: 82 样本 (5.0%)
  test: 82 样本 (5.0%)

🎯 [split-aware] train_valid_test模式：使用validation set进行堆叠验证
🎯 使用全部 82 个样本进行验证（从validation集）
✓ 自动加载验证数据: 82 样本
```

### 性能对比

| 模型类型 | R² Score | RMSE | 备注 |
|---------|----------|------|------|
| Ridge元学习器 | 0.7069 | 5.4668 | 使用validation set训练 |
| 加权平均堆叠 | 0.7401 | 5.1479 | 智能权重分配 |
| XGBoost单模型 | 0.7434 | 5.1152 | 性能最佳 |
| CatBoost单模型 | 0.6740 | 5.7661 | 性能较差 |

## 🛡️ 最佳实践

### 1. 推荐配置

对于大多数场景，建议启用 `split_aware`:

```yaml
validation:
  auto_load: true
  size: 200
  split_aware: true  # ✅ 推荐启用
```

### 2. 配置选择指南

| 原实验模式 | split_aware=true | split_aware=false |
|-----------|------------------|-------------------|
| train_valid_test | 使用validation set | 使用validation set |
| cross_validation | 使用test set | 使用validation set (可能不存在) |

### 3. 注意事项

- ✅ **数据一致性**：split_aware确保堆叠验证策略与原实验一致
- ✅ **避免泄露**：合理的数据集选择避免train/valid/test界限模糊
- ⚠️ **样本数量**：确保选择的验证集有足够样本训练元学习器

## 🔄 向后兼容性

- **默认值**：`split_aware: false`，保持原有行为不变
- **渐进迁移**：可以逐步在新配置中启用此功能
- **完全兼容**：现有配置文件无需修改即可继续使用

## 📁 配置文件示例

项目提供了两个示例配置：

1. **config_stacking_meta.yaml**：使用元学习器 + split_aware
2. **config_stacking_split_aware.yaml**：使用加权平均 + split_aware

## 🎉 总结

Split-Aware功能提供了：
- 🎯 **智能数据选择**：根据原实验策略自动选择合适的验证数据
- 🛡️ **实验严谨性**：保持与原实验的数据拆分策略一致
- 🔄 **灵活配置**：可选择启用或禁用，完全向后兼容
- 📊 **透明过程**：详细的日志输出，清晰显示数据选择逻辑

这个功能让CRAFT的模型堆叠更加智能和严谨，确保了实验的科学性和结果的可靠性。 