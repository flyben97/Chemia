![CRAFT Logo](images/craft.png) ![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

# CRAFT: 化学反应分析与基于特征的训练框架

CRAFT 是一个综合性的机器学习框架，专为化学反应预测和优化而设计。它结合了传统机器学习算法、神经网络和图神经网络，并配合贝叶斯优化来预测反应结果并找到最优反应条件。

[English](README.md) | 简体中文

## 📁 项目结构

```
craft/
├── core/                    # 核心框架组件
│   ├── run_manager.py      # 实验管理
│   ├── config_loader.py    # 配置加载
│   └── trainer_setup.py    # 模型训练设置
├── models/                  # 模型实现
│   ├── sklearn_models.py   # 传统机器学习模型
│   ├── ann.py              # 神经网络
│   └── gnn_models.py       # 图神经网络
├── optimization/            # 贝叶斯优化
│   ├── optimizer.py        # 主优化引擎
│   └── space_loader.py     # 搜索空间管理
├── utils/                   # 工具函数
├── examples/                # 示例配置和脚本
│   └── configs/            # 配置文件
├── data/                    # 数据目录
└── output/                  # 结果和训练模型
```

## 📋 快速开始

### 安装

1. **克隆仓库**：
```bash
git clone https://github.com/your-username/craft.git
cd craft
```

2. **安装依赖**：
```bash
pip install -r requirements.txt
```

3. **准备数据**：将您的反应数据（CSV格式）放置在 `data/` 目录中。

### 基本使用

#### 1. 快速训练（推荐初学者使用）
```bash
python run_training_only.py --config examples/configs/quick_start.yaml
```

#### 2. 完整模型训练
```bash
# 简单回归训练
python run_training_only.py --config examples/configs/regression_training_simple.yaml

# 分类训练
python run_training_only.py --config examples/configs/classification_training_simple.yaml

# 5折交叉验证训练
python run_training_only.py --config examples/configs/regression_training_kfold.yaml
```

#### 3. 贝叶斯优化（使用预训练模型）
```bash
python run_optimization.py --config examples/configs/bayesian_optimization_only.yaml
```

#### 4. 端到端工作流程（训练+优化）
```bash
python run_full_workflow.py --config examples/configs/end_to_end_workflow.yaml
```

## 🔧 配置文件

CRAFT 为不同场景提供了各种预配置的YAML文件：

### 训练配置

| 配置文件 | 描述 | 使用场景 |
|---------|------|----------|
| `quick_start.yaml` | 最小化测试设置 | 初次使用者，快速实验 |
| `regression_training_simple.yaml` | 基础回归训练 | 标准回归任务 |
| `regression_training_kfold.yaml` | 5折交叉验证 | 稳健的模型评估 |
| `regression_training_split.yaml` | 训练/验证/测试分割 | 模型开发 |
| `classification_training_simple.yaml` | 基础分类训练 | 分类任务 |
| `classification_training_kfold.yaml` | 分类交叉验证 | 稳健的分类 |
| `training_with_features.yaml` | 丰富的特征工程 | 复杂分子数据集 |
| `training_without_features.yaml` | 最小特征 | 简单数据集 |
| `gnn_training.yaml` | 图神经网络 | 高级分子建模 |

### 优化配置

| 配置文件 | 描述 | 使用场景 |
|---------|------|----------|
| `bayesian_optimization_only.yaml` | 独立优化 | 使用预训练模型 |
| `end_to_end_workflow.yaml` | 完整流水线 | 全自动化 |

## 📊 支持的算法

### 传统机器学习
- **梯度提升方法**：XGBoost、LightGBM、CatBoost、直方图梯度提升
- **树集成方法**：随机森林、极端随机树、AdaBoost
- **线性模型**：岭回归、LASSO、ElasticNet、贝叶斯岭回归
- **核方法**：高斯过程回归、核岭回归、支持向量回归
- **基于实例的方法**：k近邻
- **线性方法**：随机梯度下降

### 神经网络
- **传统神经网络**：基于PyTorch的人工神经网络
- **图神经网络**：GCN、GAT、MPNN、图Transformer、集成GNN

## 🧬 特征工程

CRAFT 自动从SMILES字符串生成分子特征：

- **Morgan指纹**：可定制半径和位数的圆形指纹
- **MACCS密钥**：166位结构密钥
- **RDKit描述符**：200+种分子描述符
- **自定义特征**：支持预计算特征

## 📈 数据分割策略

1. **训练/测试分割**：简单的80/20分割
2. **训练/验证/测试分割**：70/15/15分割，用于模型开发
3. **K折交叉验证**：使用分层采样的稳健评估

## 🎯 贝叶斯优化

使用训练好的模型寻找最优反应条件：

- **获取函数**：期望改进(EI)、置信上界(UCB)、改进概率(POI)
- **搜索空间**：离散（催化剂库）和连续（温度、时间）变量
- **多目标**：支持多个优化目标
- **约束条件**：化学和实际约束

## 📝 数据格式示例

您的CSV文件应包含SMILES字符串和目标值：

```csv
Catalyst,Reactant1,Reactant2,Temperature,Solvent,yield
CC(C)P(c1ccccc1)c1ccccc1,CC(=O)c1ccccc1,NCc1ccccc1,80,toluene,95.2
CCc1ccc(P(CCc2ccccc2)CCc2ccccc2)cc1,CC(=O)c1ccccc1,NCc1ccccc1,60,THF,87.5
...
```

## 🛠️ 高级用法

### 自定义配置

基于示例创建您自己的YAML配置文件：

```yaml
experiment_name: "我的实验"
task_type: "regression"

data:
  source_mode: "single_file"
  single_file_config:
    main_file_path: "data/my_reactions.csv"
    smiles_col: ["Catalyst", "Reactant1", "Reactant2"]
    target_col: "yield"

training:
  models_to_run:
    - "xgb"
    - "lgbm"
    - "rf"
  n_trials: 20

# ... 额外配置
```

### 程序化使用

```python
from core.run_manager import start_experiment_run
from core.config_loader import load_config

# 加载配置
config = load_config("my_config.yaml")

# 运行实验
results = start_experiment_run(config)

# 访问结果
best_model = max(results['results'], key=lambda x: x['test_r2'])
print(f"最佳模型: {best_model['model_name']} (R² = {best_model['test_r2']:.4f})")
```

## 📊 输出和结果

CRAFT 生成综合性输出：

- **训练模型**：多种格式的序列化模型
- **预测结果**：包含预测值和不确定性的CSV文件
- **评估指标**：详细的性能指标和交叉验证结果
- **特征重要性**：重要分子特征的分析
- **可视化**：学习曲线、特征重要性图表
- **优化结果**：排名前列的反应条件

## 📚 运行脚本说明

### 主要运行脚本

1. **`run_training_only.py`** - 专门用于模型训练
   - 支持多种算法的并行训练
   - 详细的结果分析和可视化
   - 适合模型开发和比较

2. **`run_optimization.py`** - 独立的贝叶斯优化
   - 使用预训练模型进行优化
   - 支持自定义搜索空间
   - 适合条件优化任务

3. **`run_full_workflow.py`** - 端到端工作流程
   - 自动训练→选择最佳模型→优化
   - 完全自动化的流水线
   - 适合生产环境使用

### 使用建议

- **初学者**：从 `quick_start.yaml` 开始
- **模型开发**：使用 `run_training_only.py` 配合不同的训练配置
- **生产使用**：使用 `run_full_workflow.py` 进行端到端处理
- **条件优化**：使用 `run_optimization.py` 基于已有模型

## 🤝 贡献

我们欢迎贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详情。

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。
