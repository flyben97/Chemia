# CRAFT 模型预测使用指南

本指南将介绍如何使用训练好的CRAFT模型进行预测，包括两种主要方式：YAML配置文件和独立Python脚本。

## 📁 文件说明

### 1. 配置文件
- **`config_prediction.yaml`** - 预测配置文件，支持详细的预测参数配置

### 2. 执行脚本
- **`run_prediction_standalone.py`** - 独立预测脚本，支持多种使用模式

## 🚀 使用方式

### 方式一：使用YAML配置文件（推荐）

#### 1. 编辑配置文件
打开 `config_prediction.yaml` 并根据您的需求修改以下关键配置：

```yaml
# 选择预测模式
prediction_mode: "experiment_directory"  # 或 "direct_files"

# 实验目录模式（推荐）
experiment_directory_mode:
  run_directory: "output/your_experiment_run_directory_here"  # 修改为您的实验目录
  model_name: "xgb"  # 修改为您要使用的模型名称

# 输入输出配置
data:
  input_file: "data/new_data_for_prediction.csv"  # 修改为您的输入文件
  output_file: "predictions/prediction_results.csv"  # 修改为输出路径
```

#### 2. 运行预测
```bash
python run_prediction_standalone.py --config config_prediction.yaml
```

### 方式二：直接命令行使用（实验目录模式）

如果您有完整的实验运行目录：

```bash
python run_prediction_standalone.py \
  --run_dir output/S04_agent_5_a_regression_20240101_120000 \
  --model_name xgb \
  --input data/new_data.csv \
  --output predictions/results.csv
```

### 方式三：直接命令行使用（文件模式）

如果您有单独的模型文件：

```bash
python run_prediction_standalone.py \
  --model_path path/to/your/model.json \
  --config_path path/to/run_config.json \
  --input data/new_data.csv \
  --output predictions/results.csv \
  --scaler_path path/to/scaler.joblib \
  --encoder_path path/to/label_encoder.joblib
```

### 方式四：交互模式（初学者推荐）

运行交互模式，程序会引导您逐步设置：

```bash
python run_prediction_standalone.py --interactive
```

## 📊 输入数据格式

您的输入CSV文件应包含与训练时相同的列，例如：

```csv
SMILES,Solvent_1_SMILES,Solvent_2_SMILES,Temp,feat_1,feat_2,feat_3
CCO,CC(=O)O,CCN,25.0,1.2,3.4,5.6
c1ccccc1,CNC(=O)N,CC,30.0,2.1,4.3,6.5
```

**重要说明：**
- SMILES列：必须与训练时使用的SMILES列名称一致
- 预计算特征：如果训练时使用了预计算特征，预测时也需要提供相同的特征
- 不需要包含目标变量（target column）

## 🎯 输出结果

### 回归任务输出
```csv
SMILES,Solvent_1_SMILES,prediction,prediction_timestamp,model_type,task_type
CCO,CC(=O)O,12.3456,2024-01-01T12:00:00,XGBRegressor,regression
```

### 分类任务输出
```csv
SMILES,prediction_label,prediction_encoded,proba_class_0,proba_class_1,prediction_timestamp,model_type,task_type
CCO,High,1,0.2345,0.7655,2024-01-01T12:00:00,XGBClassifier,classification
```

## ⚙️ 配置选项详解

### 预测模式
- **`experiment_directory`**: 使用完整的实验目录（推荐）
- **`direct_files`**: 直接指定模型文件路径

### 重要配置项

```yaml
prediction:
  batch_size: 1000  # 批处理大小（大数据集时有用）
  save_probabilities: true  # 是否保存分类概率
  output_format:
    include_input_data: true  # 是否在输出中包含输入数据
    add_prediction_metadata: true  # 是否添加预测元数据
    precision: 4  # 预测结果的小数位数

logging:
  verbose: false  # 是否显示详细日志
  save_log: true  # 是否保存预测日志

advanced:
  memory_efficient: true  # 内存优化模式
  skip_invalid_rows: true  # 跳过无效行而不是报错
```

## 🔧 常见问题解决

### 问题1：找不到模型文件
**错误信息**: `Model directory not found`

**解决方案**:
1. 检查实验目录路径是否正确
2. 确认模型名称是否与训练时一致
3. 验证模型目录下是否存在模型文件

### 问题2：特征生成失败
**错误信息**: `Error processing features`

**解决方案**:
1. 确保输入数据包含所有必需的SMILES列
2. 检查SMILES格式是否有效
3. 使用 `--verbose` 参数查看详细错误信息

### 问题3：内存不足
**解决方案**:
1. 在配置文件中设置 `batch_size` 为较小值（如100-500）
2. 启用 `memory_efficient: true`
3. 考虑将大文件分批处理

## 📚 使用示例

### 示例1：快速回归预测
```bash
# 使用训练好的XGBoost模型预测新化合物的性质
python run_prediction_standalone.py \
  --run_dir output/S04_agent_5_a_regression_20240101_120000 \
  --model_name xgb \
  --input data/new_compounds.csv \
  --output predictions/compound_properties.csv
```

### 示例2：分类任务预测
```bash
# 使用LightGBM模型进行化合物分类
python run_prediction_standalone.py \
  --run_dir output/classification_experiment_20240101 \
  --model_name lgbm \
  --input data/compounds_to_classify.csv \
  --output predictions/compound_classes.csv \
  --verbose  # 显示详细处理过程
```

### 示例3：批量预测大数据集
编辑 `config_prediction.yaml`:
```yaml
prediction_mode: "experiment_directory"
experiment_directory_mode:
  run_directory: "output/your_experiment"
  model_name: "catboost"
data:
  input_file: "data/large_dataset.csv"
  output_file: "predictions/large_results.csv"
prediction:
  batch_size: 500  # 小批量处理
advanced:
  memory_efficient: true
  chunk_size: 2000
```

然后运行：
```bash
python run_prediction_standalone.py --config config_prediction.yaml
```

## 💡 最佳实践

1. **使用YAML配置文件**：对于复杂的预测任务，使用配置文件可以确保可重现性
2. **备份重要预测**：将重要的预测结果和配置文件一起保存
3. **验证输入数据**：预测前检查输入数据的格式和完整性
4. **选择合适的模型**：根据训练结果选择性能最好的模型进行预测
5. **处理大数据集**：对于大型数据集，使用批处理和内存优化选项

## 🆘 获取帮助

查看所有命令行选项：
```bash
python run_prediction_standalone.py --help
```

使用交互模式获得引导：
```bash
python run_prediction_standalone.py --interactive
```

---

**提示**: 如果您是第一次使用，建议先用交互模式熟悉流程，然后再使用YAML配置文件进行批量预测。 