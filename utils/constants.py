# utils/constants.py
"""
公共常量定义模块
集中管理项目中的默认值、阈值和 Magic Number，避免硬编码散落各处。
"""

# UI / Console 默认配置
DEFAULT_CONSOLE_WIDTH = 120

# 数据验证默认参数
DEFAULT_SMILES_SAMPLE_SIZE = 200
DEFAULT_SMILES_MIN_VALID_RATIO = 0.8
DEFAULT_SMILES_DETECT_SAMPLE_SIZE = 100
DEFAULT_SMILES_DETECT_CONFIDENCE = 0.7

# 特征工程默认参数
DEFAULT_MORGAN_RADIUS = 2
DEFAULT_MORGAN_NBITS = 2048
DEFAULT_BATCH_SIZE = 32

# 预测时跳过特征生成的数值列启发式阈值
PREDICTION_SKIP_FEATUREGEN_NUMERIC_THRESHOLD = 10

# 输出目录默认根路径
DEFAULT_OUTPUT_ROOT = "output"
DEFAULT_BATCH_OUTPUT_PREFIX = "batch_output"

# 模型名称映射（补充 io_handler 中缺失的别名）
MODEL_NAME_ALIASES = {
    'xgb': 'xgboost',
    'lgb': 'lgbm',
    'cb': 'catboost',
    'rf': 'randomforest',
    'et': 'extratrees',
    'gb': 'gbdt',
    'gbr': 'gbdt',
    'gradboost': 'gbdt',
    'gradientboosting': 'gbdt',
    'svr': 'svr',
    'svc': 'svc',
    'knn': 'kneighbors',
    'gp': 'gpr',
    'gpc': 'gpr',
    'mlp': 'ann',
    'dnn': 'ann',
    'enet': 'elasticnet',
    'br': 'bayesianridge',
    'sgd': 'sgd',
    'sgdregressor': 'sgd',
    'sgdclassifier': 'sgd',
    'lr': 'logisticregression',
    'logreg': 'logisticregression',
}
