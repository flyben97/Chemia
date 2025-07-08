"""
SVR 性能优化工具
提供SVR模型的自动优化建议和数据预处理功能
"""

import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import time
from typing import Dict, Any, Tuple, Optional

class SVROptimizer:
    """SVR模型性能优化器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.performance_stats = {}
    
    def analyze_dataset(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        分析数据集特征，给出SVR优化建议
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            数据集分析结果和优化建议
        """
        n_samples, n_features = X.shape
        
        analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'feature_mean': np.mean(X, axis=0).mean(),
            'feature_std': np.std(X, axis=0).mean(),
            'target_range': np.max(y) - np.min(y),
            'target_std': np.std(y),
            'memory_usage_mb': X.nbytes / (1024 * 1024),
            'recommendations': {}
        }
        
        # 根据数据集大小给出建议
        if n_samples < 1000:
            analysis['recommendations']['dataset_size'] = 'small'
            analysis['recommendations']['kernel'] = ['linear', 'rbf', 'poly']
            analysis['recommendations']['c_range'] = (0.1, 100)
            analysis['recommendations']['max_iter'] = 10000
        elif n_samples < 5000:
            analysis['recommendations']['dataset_size'] = 'medium'
            analysis['recommendations']['kernel'] = ['linear', 'rbf']
            analysis['recommendations']['c_range'] = (0.1, 50)
            analysis['recommendations']['max_iter'] = 5000
        else:
            analysis['recommendations']['dataset_size'] = 'large'
            analysis['recommendations']['kernel'] = ['linear']  # 大数据集建议只用线性核
            analysis['recommendations']['c_range'] = (0.1, 10)
            analysis['recommendations']['max_iter'] = 3000
            
        # 检查特征缩放需求
        feature_scales = np.std(X, axis=0)
        if np.max(feature_scales) / np.min(feature_scales) > 10:
            analysis['recommendations']['scaling'] = 'required'
        else:
            analysis['recommendations']['scaling'] = 'optional'
            
        # 内存使用建议
        if analysis['memory_usage_mb'] > 500:
            analysis['recommendations']['cache_size'] = 200
        elif analysis['memory_usage_mb'] > 100:
            analysis['recommendations']['cache_size'] = 500
        else:
            analysis['recommendations']['cache_size'] = 1000
            
        return analysis
    
    def get_optimized_params(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        基于数据集分析获取优化的SVR参数
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            优化的SVR参数字典
        """
        analysis = self.analyze_dataset(X, y)
        recommendations = analysis['recommendations']
        
        params = {
            'kernel': recommendations['kernel'][0],  # 选择推荐的第一个核函数
            'C': recommendations['c_range'][1] / 2,  # 选择C范围的中间值
            'cache_size': recommendations['cache_size'],
            'max_iter': recommendations['max_iter'],
            'tol': 1e-3,
            'shrinking': True
        }
        
        # 为非线性核添加gamma参数
        if params['kernel'] != 'linear':
            params['gamma'] = 'scale'  # 使用sklearn的自动缩放
            
        # 根据目标变量范围调整epsilon
        target_std = analysis['target_std']
        params['epsilon'] = max(0.01, min(0.1, target_std * 0.1))
        
        return params
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray, 
                       fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理数据以优化SVR性能
        
        Args:
            X: 特征矩阵
            y: 目标变量
            fit_scaler: 是否拟合缩放器
            
        Returns:
            预处理后的特征矩阵和目标变量
        """
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        # 检查并处理异常值
        X_scaled = np.clip(X_scaled, -5, 5)  # 限制在合理范围内
        
        return X_scaled, y
    
    def train_with_monitoring(self, X: np.ndarray, y: np.ndarray, 
                            params: Optional[Dict[str, Any]] = None,
                            test_size: float = 0.2) -> Dict[str, Any]:
        """
        训练SVR模型并监控性能
        
        Args:
            X: 特征矩阵
            y: 目标变量
            params: SVR参数，如果为None则自动优化
            test_size: 测试集比例
            
        Returns:
            训练结果和性能统计
        """
        # 分析数据集
        analysis = self.analyze_dataset(X, y)
        
        # 获取优化参数
        if params is None:
            params = self.get_optimized_params(X, y)
            
        # 预处理数据
        X_processed, y_processed = self.preprocess_data(X, y)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size, random_state=42
        )
        
        # 训练模型并监控时间
        start_time = time.time()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            svr = SVR(**params)
            svr.fit(X_train, y_train)
            
        training_time = time.time() - start_time
        
        # 预测和评估
        start_time = time.time()
        y_pred = svr.predict(X_test)
        prediction_time = time.time() - start_time
        
        # 计算评估指标
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        results = {
            'model': svr,
            'dataset_analysis': analysis,
            'used_params': params,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'n_support': svr.n_support_,
            'metrics': {
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            },
            'performance_warnings': []
        }
        
        # 性能警告
        if training_time > 60:  # 训练时间超过1分钟
            results['performance_warnings'].append(
                f"训练时间较长 ({training_time:.1f}秒)，建议使用线性核或减少数据量"
            )
            
        if analysis['n_samples'] > 10000 and params['kernel'] != 'linear':
            results['performance_warnings'].append(
                "大数据集建议使用线性核以提高训练速度"
            )
            
        return results
    
    def suggest_alternatives(self, X: np.ndarray, y: np.ndarray) -> Dict[str, str]:
        """
        当SVR性能不佳时，建议替代算法
        
        Args:
            X: 特征矩阵 
            y: 目标变量
            
        Returns:
            替代算法建议
        """
        analysis = self.analyze_dataset(X, y)
        suggestions = {}
        
        if analysis['n_samples'] > 10000:
            suggestions['large_dataset'] = [
                'Linear Regression (Ridge/Lasso)',
                'Random Forest',
                'Gradient Boosting (XGBoost/LightGBM)',
                'Neural Networks'
            ]
            
        if analysis['n_features'] > analysis['n_samples']:
            suggestions['high_dimensional'] = [
                'Ridge Regression',
                'Lasso Regression', 
                'ElasticNet',
                'Random Forest'
            ]
            
        if analysis['memory_usage_mb'] > 1000:
            suggestions['memory_efficient'] = [
                'SGD Regressor',
                'Online Learning algorithms',
                'Mini-batch algorithms'
            ]
            
        return suggestions

def optimize_svr_config(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    便捷函数：为给定数据集优化SVR配置
    
    Args:
        X: 特征矩阵
        y: 目标变量
        
    Returns:
        优化的配置建议
    """
    optimizer = SVROptimizer()
    analysis = optimizer.analyze_dataset(X, y)
    optimized_params = optimizer.get_optimized_params(X, y)
    alternatives = optimizer.suggest_alternatives(X, y)
    
    return {
        'analysis': analysis,
        'optimized_params': optimized_params,
        'alternatives': alternatives,
        'preprocessing_required': analysis['recommendations']['scaling'] == 'required'
    } 