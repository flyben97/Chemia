#!/usr/bin/env python3
"""
CRAFT 堆叠评估工具模块

这个模块提供堆叠模型的评估、分析和可视化功能，包括：
- 模型性能评估
- 基础模型对比分析
- 堆叠效果分析
- 评估结果导出
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loader import prepare_stacking_data

def evaluate_stacking_performance(stacker, 
                                test_data: Optional[Union[Dict, List[Dict], pd.DataFrame]] = None,
                                true_labels: Optional[Union[List, np.ndarray]] = None,
                                auto_load: bool = True,
                                use_test_set: bool = True,
                                evaluate_both_sets: bool = True) -> Dict[str, Any]:
    """
    评估堆叠模型性能
    
    Args:
        stacker: 堆叠模型实例
        test_data: 测试数据（可选）
        true_labels: 真实标签（可选）
        auto_load: 是否自动加载数据
        use_test_set: 是否使用测试集（当evaluate_both_sets=False时生效）
        evaluate_both_sets: 是否同时评估validation和test数据集
    
    Returns:
        dict: 评估结果，包含单个或两个数据集的结果
    """
    
    if not auto_load:
        # 如果不自动加载，使用提供的数据进行单一评估
        return _evaluate_single_dataset(stacker, test_data, true_labels, "provided_data")
    
    if stacker.experiment_dir is None:
        raise ValueError("需要设置experiment_dir或提供test_data和true_labels")
    
    data_info = prepare_stacking_data(stacker.experiment_dir)
    
    if not evaluate_both_sets:
        # 原有逻辑：只评估一个数据集
        if use_test_set and 'test_data' in data_info:
            test_data = data_info['test_data']
            true_labels = data_info['test_labels']
            if test_data is not None:
                print(f"✓ 自动加载测试数据: {len(test_data)} 样本")
            return _evaluate_single_dataset(stacker, test_data, true_labels, "test")
        elif 'validation_data' in data_info:
            test_data = data_info['validation_data']
            true_labels = data_info['validation_labels']
            if test_data is not None:
                print(f"✓ 自动加载验证数据用于评估: {len(test_data)} 样本")
            return _evaluate_single_dataset(stacker, test_data, true_labels, "validation")
        else:
            raise ValueError("无法加载测试或验证数据")
    
    # 新功能：同时评估validation和test数据集
    evaluation_results = {
        'stacking_method': stacker.stacking_method,
        'n_models': len(stacker.base_models),
        'model_names': list(stacker.base_models.keys()),
        'evaluation_mode': 'both_sets'
    }
    
    datasets_evaluated = []
    
    # 评估validation数据集
    if 'validation_data' in data_info and data_info['validation_data'] is not None:
        val_data = data_info['validation_data']
        val_labels = data_info['validation_labels']
        val_size = len(val_data) if val_data is not None else 0
        print(f"✓ 自动加载验证数据: {val_size} 样本")
        
        val_evaluation = _evaluate_single_dataset(stacker, val_data, val_labels, "validation")
        evaluation_results['validation_set'] = val_evaluation
        datasets_evaluated.append('validation')
    
    # 评估test数据集
    if 'test_data' in data_info and data_info['test_data'] is not None:
        test_data = data_info['test_data']
        test_labels = data_info['test_labels']
        test_size = len(test_data) if test_data is not None else 0
        print(f"✓ 自动加载测试数据: {test_size} 样本")
        
        test_evaluation = _evaluate_single_dataset(stacker, test_data, test_labels, "test")
        evaluation_results['test_set'] = test_evaluation
        datasets_evaluated.append('test')
    
    if not datasets_evaluated:
        raise ValueError("无法加载任何评估数据集")
    
    evaluation_results['datasets_evaluated'] = datasets_evaluated
    
    return evaluation_results


def _evaluate_single_dataset(stacker, data, true_labels, dataset_name: str) -> Dict[str, Any]:
    """评估单个数据集的性能"""
    if data is None or true_labels is None:
        raise ValueError(f"数据集 {dataset_name} 的数据或标签为空")
    
    result = stacker.predict(data)
    predictions = result['predictions']
    y_true = np.array(true_labels)
    
    evaluation = {
        'dataset_name': dataset_name,
        'n_samples': len(y_true),
        'stacking_method': stacker.stacking_method,
        'model_names': list(stacker.base_models.keys())
    }
    
    # 计算堆叠模型性能
    if stacker.task_type == 'regression':
        mse = mean_squared_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - predictions))
        
        evaluation.update({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
    else:
        accuracy = accuracy_score(y_true, predictions)
        evaluation['accuracy'] = accuracy
        
        if result['probabilities'] is not None:
            try:
                logloss = log_loss(y_true, result['probabilities'])
                evaluation['log_loss'] = logloss
            except Exception:
                pass
    
    # 评估各个基础模型的性能
    base_predictions = result['base_predictions']
    model_names = result['model_names']
    
    base_performance = {}
    for i, model_name in enumerate(model_names):
        base_pred = base_predictions[:, i]
        if stacker.task_type == 'regression':
            base_r2 = r2_score(y_true, base_pred)
            base_rmse = np.sqrt(mean_squared_error(y_true, base_pred))
            base_mae = np.mean(np.abs(y_true - base_pred))
            base_performance[model_name] = {'r2': base_r2, 'rmse': base_rmse, 'mae': base_mae}
        else:
            base_acc = accuracy_score(y_true, base_pred)
            base_performance[model_name] = {'accuracy': base_acc}
    
    evaluation['base_model_performance'] = base_performance
    
    return evaluation

def generate_evaluation_report(evaluation: Dict[str, Any]) -> str:
    """生成评估报告"""
    report = []
    report.append("=" * 60)
    report.append("CRAFT 模型堆叠评估报告")
    report.append("=" * 60)
    
    # 基本信息
    report.append(f"堆叠方法: {evaluation['stacking_method']}")
    report.append(f"模型数量: {evaluation['n_models']}")
    report.append(f"模型列表: {', '.join(evaluation['model_names'])}")
    report.append("")
    
    # 堆叠模型性能
    report.append("📊 堆叠模型性能:")
    report.append("-" * 30)
    
    if 'r2' in evaluation:
        report.append(f"R² Score: {evaluation['r2']:.4f}")
        report.append(f"RMSE: {evaluation['rmse']:.4f}")
        report.append(f"MAE: {evaluation['mae']:.4f}")
        report.append(f"MSE: {evaluation['mse']:.4f}")
    elif 'accuracy' in evaluation:
        report.append(f"Accuracy: {evaluation['accuracy']:.4f}")
        if 'log_loss' in evaluation:
            report.append(f"Log Loss: {evaluation['log_loss']:.4f}")
    
    report.append("")
    
    # 基础模型性能
    report.append("🔍 基础模型性能对比:")
    report.append("-" * 30)
    
    base_performance = evaluation.get('base_model_performance', {})
    for model_name, metrics in base_performance.items():
        if 'r2' in metrics:
            report.append(f"{model_name:>10}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        elif 'accuracy' in metrics:
            report.append(f"{model_name:>10}: Accuracy={metrics['accuracy']:.4f}")
    
    return "\n".join(report)
