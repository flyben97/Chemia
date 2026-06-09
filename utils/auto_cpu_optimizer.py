#!/usr/bin/env python3
"""
自动CPU优化器
在后台自动应用系统级CPU优化，不影响模型参数搜索
"""

import os
import sys
import psutil
import multiprocessing as mp
import threading
import logging
from typing import Dict, Any, Optional, Union, List
import warnings
import numpy as np

# 设置系统级CPU优化环境变量
def _setup_cpu_environment():
    """设置CPU优化环境变量"""
    cpu_count = os.cpu_count() or 1
    optimal_threads = min(8, max(1, cpu_count - 1)) if cpu_count > 1 else 1

    # 设置各种数值计算库的线程数
    env_vars = {
        'OMP_NUM_THREADS': str(optimal_threads),
        'MKL_NUM_THREADS': str(optimal_threads),
        'NUMBA_NUM_THREADS': str(optimal_threads),
        'OPENBLAS_NUM_THREADS': str(optimal_threads),
        'VECLIB_MAXIMUM_THREADS': str(optimal_threads),
        'BLIS_NUM_THREADS': str(optimal_threads),
        'KMP_DUPLICATE_LIB_OK': 'TRUE',  # 避免库冲突
        'KMP_AFFINITY': 'granularity=fine,compact,1,0',  # CPU亲和性
        'KMP_BLOCKTIME': '1',  # 减少线程等待时间
    }

    for key, value in env_vars.items():
        os.environ[key] = value

# 环境变量设置已改为显式调用，不再在导入时自动执行。
# 如需自动设置，请在脚本开头调用: from utils.auto_cpu_optimizer import setup_cpu_environment; setup_cpu_environment()

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor, CatBoostClassifier
    BOOSTING_AVAILABLE = True
except ImportError:
    BOOSTING_AVAILABLE = False

logger = logging.getLogger(__name__)


class AutoCPUOptimizer:
    """自动CPU优化器 - 系统级优化，不影响模型参数"""

    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(AutoCPUOptimizer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化自动CPU优化器"""
        if self._initialized:
            return

        self.system_info = self._get_system_info()
        self.optimal_threads = self._calculate_optimal_threads()
        self.memory_limit_gb = self._calculate_memory_limit()

        # 应用系统级优化
        self._apply_system_optimizations()

        # 标记为已初始化
        AutoCPUOptimizer._initialized = True

        logger.info(f"AutoCPUOptimizer initialized: {self.optimal_threads} threads, {self.memory_limit_gb:.1f}GB memory limit")

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'cpu_count': os.cpu_count(),
            'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'platform': sys.platform
        }

    def _calculate_optimal_threads(self) -> int:
        """计算最优线程数"""
        cpu_count = self.system_info['cpu_count']

        if cpu_count <= 2:
            return cpu_count
        elif cpu_count <= 4:
            return cpu_count - 1
        elif cpu_count <= 8:
            return min(6, cpu_count - 1)
        elif cpu_count <= 16:
            return min(8, cpu_count - 2)
        else:
            return min(12, cpu_count // 2)

    def _calculate_memory_limit(self) -> float:
        """计算内存限制"""
        available_memory = self.system_info['memory_available_gb']
        return available_memory * 0.75  # 保留25%内存给系统

    def _apply_system_optimizations(self):
        """应用系统级优化"""
        # PyTorch优化
        if TORCH_AVAILABLE:
            torch.set_num_threads(self.optimal_threads)
            torch.set_num_interop_threads(self.optimal_threads)

            # 启用CPU优化
            if hasattr(torch.backends, 'mkldnn'):
                torch.backends.mkldnn.enabled = True
            if hasattr(torch.backends, 'mkl'):
                torch.backends.mkl.enabled = True

            # 设置内存分配策略
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = False  # CPU环境下关闭

        # NumPy优化
        try:
            import numpy as np
            # 设置NumPy错误处理
            np.seterr(all='ignore')
        except ImportError:
            pass

        # 抑制警告
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)

    def optimize_sklearn_model(self, model, model_name: str = None) -> Any:
        """自动优化Scikit-learn模型的系统级参数"""
        if not SKLEARN_AVAILABLE:
            return model

        model_name = model_name or model.__class__.__name__.lower()

        # 设置并行参数
        if hasattr(model, 'n_jobs'):
            model.n_jobs = self.optimal_threads

        # 随机森林和Extra Trees优化
        if any(name in model_name for name in ['randomforest', 'extratrees']):
            if hasattr(model, 'n_jobs'):
                model.n_jobs = self.optimal_threads

        # SVM优化
        elif any(name in model_name for name in ['svm', 'svc', 'svr']):
            if hasattr(model, 'cache_size'):
                # 根据可用内存设置缓存大小
                cache_size_mb = min(512, int(self.memory_limit_gb * 1024 * 0.1))
                model.cache_size = cache_size_mb

        # KNN优化
        elif 'kneighbors' in model_name:
            if hasattr(model, 'n_jobs'):
                model.n_jobs = self.optimal_threads
            if hasattr(model, 'algorithm') and model.algorithm == 'auto':
                model.algorithm = 'ball_tree'  # 通常在CPU上更快

        return model

    def optimize_boosting_model(self, model, model_name: str = None) -> Any:
        """自动优化梯度提升模型的系统级参数"""
        if not BOOSTING_AVAILABLE:
            return model

        model_name = model_name or model.__class__.__name__.lower()

        # XGBoost优化
        if 'xgb' in model_name or 'xgboost' in model_name:
            if hasattr(model, 'n_jobs'):
                model.n_jobs = self.optimal_threads
            if hasattr(model, 'tree_method'):
                model.tree_method = 'hist'  # CPU上最快的方法
            if hasattr(model, 'verbosity'):
                model.verbosity = 0

        # LightGBM优化
        elif 'lgb' in model_name or 'lightgbm' in model_name:
            if hasattr(model, 'n_jobs'):
                model.n_jobs = self.optimal_threads
            if hasattr(model, 'num_threads'):
                model.num_threads = self.optimal_threads
            if hasattr(model, 'verbose'):
                model.verbose = -1
            if hasattr(model, 'device_type'):
                model.device_type = 'cpu'

        # CatBoost优化
        elif 'catboost' in model_name:
            if hasattr(model, 'thread_count'):
                model.thread_count = self.optimal_threads
            if hasattr(model, 'task_type'):
                model.task_type = 'CPU'
            if hasattr(model, 'verbose'):
                model.verbose = False

        return model

    def optimize_pytorch_model(self, model, model_name: str = None) -> Any:
        """自动优化PyTorch模型的系统级参数"""
        if not TORCH_AVAILABLE:
            return model

        # 确保模型在CPU上
        model = model.cpu()

        # 设置为评估模式进行优化
        if hasattr(model, 'eval'):
            model.eval()

        return model

    def get_optimal_batch_size(self,
                              model_type: str,
                              n_samples: int,
                              n_features: int,
                              base_batch_size: int = None) -> int:
        """获取最优批次大小"""
        if base_batch_size is None:
            # 默认批次大小
            base_batch_sizes = {
                'sklearn': min(1000, n_samples),
                'xgboost': min(1000, n_samples),
                'lightgbm': min(1000, n_samples),
                'catboost': min(500, n_samples),
                'pytorch': min(256, n_samples),
                'tabnet': min(512, n_samples)
            }
            base_batch_size = base_batch_sizes.get(model_type.lower(), 256)

        # 根据内存调整批次大小
        estimated_memory_per_sample = n_features * 4 / (1024**2)  # MB per sample
        max_batch_by_memory = int(self.memory_limit_gb * 1024 * 0.3 / estimated_memory_per_sample)

        optimal_batch_size = min(base_batch_size, max_batch_by_memory, n_samples)

        # 确保批次大小合理
        return max(16, optimal_batch_size)

    def optimize_data_types(self, X, y=None):
        """优化数据类型以节省内存"""
        # 转换为float32以节省内存
        if hasattr(X, 'dtype') and X.dtype != np.float32:
            X = X.astype(np.float32)

        if y is not None and hasattr(y, 'dtype') and y.dtype != np.float32:
            y = y.astype(np.float32)

        # 确保数据连续存储
        if hasattr(X, 'flags') and not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)

        if y is not None and hasattr(y, 'flags') and not y.flags['C_CONTIGUOUS']:
            y = np.ascontiguousarray(y)

        return (X, y)

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()

        return {
            'cpu_usage_percent': cpu_percent,
            'memory_usage_percent': memory_info.percent,
            'memory_available_gb': memory_info.available / (1024**3),
            'optimal_threads': self.optimal_threads,
            'memory_limit_gb': self.memory_limit_gb,
            'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else None
        }


# 全局实例
_auto_cpu_optimizer = None

def get_auto_cpu_optimizer() -> AutoCPUOptimizer:
    """获取全局AutoCPUOptimizer实例"""
    global _auto_cpu_optimizer
    if _auto_cpu_optimizer is None:
        _auto_cpu_optimizer = AutoCPUOptimizer()
    return _auto_cpu_optimizer


def auto_optimize_model(model, model_name: str = None) -> Any:
    """自动优化模型的系统级参数"""
    optimizer = get_auto_cpu_optimizer()

    # 根据模型类型选择优化方法
    model_name = model_name or model.__class__.__name__.lower()

    if SKLEARN_AVAILABLE and isinstance(model, BaseEstimator):
        return optimizer.optimize_sklearn_model(model, model_name)
    elif BOOSTING_AVAILABLE and any(lib in model_name for lib in ['xgb', 'lgb', 'catboost']):
        return optimizer.optimize_boosting_model(model, model_name)
    elif TORCH_AVAILABLE and hasattr(model, 'parameters'):
        return optimizer.optimize_pytorch_model(model, model_name)
    else:
        return model


def auto_optimize_data(X, y=None):
    """自动优化数据类型和内存布局"""
    optimizer = get_auto_cpu_optimizer()
    return optimizer.optimize_data_types(X, y)


def get_optimal_batch_size(model_type: str, n_samples: int, n_features: int, base_batch_size: int = None) -> int:
    """获取最优批次大小"""
    optimizer = get_auto_cpu_optimizer()
    return optimizer.get_optimal_batch_size(model_type, n_samples, n_features, base_batch_size)


def get_system_status() -> Dict[str, Any]:
    """获取系统状态"""
    optimizer = get_auto_cpu_optimizer()
    return optimizer.get_system_status()


# 显式环境变量设置函数（替代导入时自动执行）
def setup_cpu_environment():
    """显式设置CPU优化环境变量"""
    _setup_cpu_environment()


# 自动应用优化的装饰器
def auto_cpu_optimize(func):
    """装饰器：自动应用CPU优化"""
    def wrapper(*args, **kwargs):
        # 确保AutoCPUOptimizer已初始化
        get_auto_cpu_optimizer()
        return func(*args, **kwargs)
    return wrapper


if __name__ == "__main__":
    # 测试自动CPU优化器
    optimizer = get_auto_cpu_optimizer()

    print("=== 自动CPU优化器测试 ===")
    print(f"系统信息: {optimizer.system_info}")
    print(f"最优线程数: {optimizer.optimal_threads}")
    print(f"内存限制: {optimizer.memory_limit_gb:.1f} GB")

    # 测试系统状态
    status = get_system_status()
    print(f"当前系统状态: {status}")

    # 测试数据优化
    import numpy as np
    X = np.random.randn(1000, 50).astype(np.float64)
    y = np.random.randn(1000).astype(np.float64)

    print(f"原始数据类型: X={X.dtype}, y={y.dtype}")
    X_opt, y_opt = auto_optimize_data(X, y)
    print(f"优化后数据类型: X={X_opt.dtype}, y={y_opt.dtype}")

    # 测试批次大小优化
    batch_size = get_optimal_batch_size('xgboost', 10000, 100)
    print(f"推荐批次大小: {batch_size}")
