# CRAFT 仅优化模式使用说明

## 概述

这个工具允许您使用已经训练好的模型直接进行贝叶斯优化，而无需重新训练模型。非常适合以下场景：
- 您已经有了满意的训练模型
- 想要测试不同的反应条件搜索空间
- 需要快速进行多次优化实验

## 文件说明

- `config_optimization_only.yaml` - 配置文件
- `run_optimization_only.py` - 主执行脚本
- `view_parameter_space.py` - 辅助脚本：查看参数空间文件内容
- `README_optimization_only.md` - 本说明文档

## 使用步骤

### 1. 修改配置文件

编辑 `config_optimization_only.yaml`，主要需要修改以下部分：

#### 1.1 指定已训练模型的位置

**方式一：使用训练运行目录（推荐）**
```yaml
model_source:
  source_run_dir: "output/E2E_Reaction_Opt_regression_20250630_153026"  # 修改为您的模型目录
  model_to_use: "xgb"  # 修改为您想使用的模型（xgb, lgbm, rf等）
```

**方式二：直接指定文件路径**
```yaml
model_source:
  direct_paths:
    model_file: "path/to/your/model.json"
    scaler_file: "path/to/your/scaler.joblib"
    encoder_file: "path/to/your/encoder.joblib"  # 仅分类任务需要
    config_file: "path/to/your/run_config.json"
```

#### 1.2 设置反应条件

修改固定组分：
```yaml
optimization_config:
  fixed_components:
    Reactant1: 'O=C(OC)CCC#C'  # 修改为您的反应物1
    Reactant2: 'O=P(C1=CC=CC=C1)(C2=CC=CC=C2)C(Br)CCC'  # 修改为您的反应物2
```

修改搜索空间（确保文件路径正确）：
```yaml
  reaction_components:
    Ligand:
      mode: "search"  # 搜索模式：优化过程中会变化
      file: 'data/ligand_space.csv'  # 确保文件存在
    catalyst:
      mode: "fixed"   # 固定模式：使用指定行的数据
      file: 'data/catalysts_space.csv'
      row_index: 5    # 使用第6行数据 (0-based索引)
    base:
      mode: "fixed"   # 固定模式
      file: 'data/base_space.csv'
      row_index: 2    # 使用第3行数据
    # ... 其他组分
```

**组件模式说明：**
- `mode: "search"` - 搜索模式：优化过程中会在整个空间中搜索
- `mode: "fixed"` - 固定模式：使用指定的固定值，不参与优化

#### 1.3 调整优化参数

```yaml
  bayesian_optimization:
    init_points: 10      # 初始随机探索点数
    n_iter: 100          # 优化迭代次数
    random_state: 42     # 随机种子
    top_k_results: 5     # 保存前k个最佳结果
```

### 2. 运行优化

```bash
python run_optimization_only.py --config config_optimization_only.yaml
```

或者使用默认配置文件：
```bash
python run_optimization_only.py
```

## 输出说明

### 终端输出
- 简洁的进度信息
- 模型加载状态
- 优化完成提示

### 输出文件
程序会在 `output/optimization_only_YYYYMMDD_HHMMSS/` 目录下生成：

1. **`top_5_optimized_conditions.csv`** - 最佳反应条件（文件名中的数字根据配置中的 `top_k_results` 决定）
2. **`optimization_run.log`** - 详细的优化日志
3. **其他优化过程文件** - 包括详细的迭代记录等

### 结果文件格式
CSV文件包含以下列：
- 各反应组分的值（如 Ligand, Base, Catalyst, Solvent, Temperature）
- 预测的目标值（如 ee 值）
- 其他相关信息

## 常见问题

### Q1: 如何找到我的模型目录？
A: 模型目录通常在 `output/` 文件夹下，格式类似 `实验名_任务类型_日期时间`，例如：
- `output/E2E_Reaction_Opt_regression_20250630_153026`
- `output/MyExperiment_regression_20250701_120000`

### Q2: 如何知道有哪些可用模型？
A: 在模型目录的 `models/` 子文件夹中，您可以看到所有训练过的模型，例如：
- `xgboost_regressor/`
- `lightgbm_regressor/`
- `random_forest_regressor/`

模型名称对应关系：
- `xgb` → `xgboost_regressor`
- `lgbm` → `lightgbm_regressor` 
- `rf` → `random_forest_regressor`
- 等等

### Q3: 搜索空间文件格式要求？
A: 搜索空间文件应该是CSV格式，包含：
- 显示列（display_col 指定）
- 特征列（如果 is_feature_source 为 True）
- 正确的分隔符（sep 参数）
- Index列（用于标识每一行）

### Q4: 如何确定要固定使用哪一行的数据？
A: 有两种方法查看参数空间文件：

**方法一：使用辅助脚本（推荐）**
```bash
python view_parameter_space.py data/catalysts_space.csv
python view_parameter_space.py data/base_space.csv  
python view_parameter_space.py data/solvent_space.csv
python view_parameter_space.py data/ligand_space.csv \t  # 如果是制表符分隔
```

**方法二：手动查看文件**
1. 打开文件（如 `data/catalysts_space.csv`）
2. 找到您想使用的催化剂、碱或溶剂
3. 记住它在文件中的行号（注意：第一行数据是row_index=0）
4. 在配置中设置对应的 `row_index`

例如，如果催化剂文件内容如下：
```
Index,Compound,Feature1,Feature2,...
0,Catalyst_A,0.1,0.2,...
1,Catalyst_B,0.3,0.4,...
2,Catalyst_C,0.5,0.6,...
```
要使用 Catalyst_B，应设置 `row_index: 1`

### Q5: 优化结果不理想怎么办？
A: 可以尝试：
- 增加 `n_iter` 参数（更多优化迭代）
- 增加 `init_points` 参数（更多初始探索）
- 检查搜索空间是否合理
- 尝试不同的预训练模型

## 高级使用

### 完整的固定+搜索配置示例
```yaml
optimization_config:
  reaction_components:
    # 搜索配体空间
    Ligand:
      mode: "search"
      file: 'data/ligand_space.csv'
      display_col: 'SMILES'
      sep: '\t'
    
    # 固定使用第3个催化剂 (row_index=2)
    catalyst:
      mode: "fixed"
      file: 'data/catalysts_space.csv'
      row_index: 2
      display_col: 'Compound'
      is_feature_source: True
      feature_slice: "2:"
    
    # 固定使用第5个碱 (row_index=4)  
    base:
      mode: "fixed"
      file: 'data/base_space.csv'
      row_index: 4
      display_col: 'Base'
      is_feature_source: True
      feature_slice: "2:"
    
    # 固定使用第1个溶剂 (row_index=0)
    solvent:
      mode: "fixed"
      file: 'data/solvent_space.csv'
      row_index: 0
      display_col: 'Name'
      is_feature_source: True
      feature_slice: "3:"
    
    # 搜索温度范围
    temperature:
      mode: "search"
      file: 'data/temperature_space.csv'
      display_col: 'Temp'
      is_feature_source: true
      feature_slice: "1:"
```

### 自定义输出目录前缀
```yaml
output:
  output_dir_prefix: "my_optimization"  # 输出目录将变为 output/my_optimization_YYYYMMDD_HHMMSS
```

### 使用不同的随机种子
```yaml
bayesian_optimization:
  random_state: 123  # 使用不同的随机种子获得不同的优化路径
```

### 保存更多或更少的结果
```yaml
bayesian_optimization:
  top_k_results: 10  # 保存前10个最佳结果，输出文件名会相应变为 top_10_optimized_conditions.csv
``` 