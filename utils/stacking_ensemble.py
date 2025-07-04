#!/usr/bin/env python3
"""
CRAFT 堆叠集成工具模块

这个模块提供各种便捷的堆叠集成创建函数，包括：
- 快速创建集成模型
- 自动选择最优模型组合  
- 智能元学习器集成
- 模型性能评估和比较
"""

import os
import sys
import contextlib
import numpy as np
from typing import Dict, List, Any, Union, Optional
from sklearn.metrics import r2_score, accuracy_score

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prediction_api import load_model
from data_loader import create_validation_dataset

def create_ensemble(experiment_dir: str, model_names: List[str], 
                   weights: Optional[List[float]] = None,
                   method: str = "weighted_average"):
    """
    快速创建模型集成
    
    Args:
        experiment_dir: 实验目录路径
        model_names: 模型名称列表
        weights: 模型权重列表（可选）
        method: 堆叠方法
    
    Returns:
        ModelStacker: 配置好的堆叠器
    """
    from model_stacking import ModelStacker
    
    if weights is None:
        weights = [1.0] * len(model_names)
    
    if len(weights) != len(model_names):
        raise ValueError("权重数量必须与模型数量一致")
    
    # 创建堆叠器
    stacker = ModelStacker(experiment_dir=experiment_dir)
    stacker.set_stacking_method(method)
    
    # 添加模型
    for model_name, weight in zip(model_names, weights):
        stacker.add_model(model_name, weight)
    
    return stacker

def auto_ensemble(experiment_dir: str, 
                 validation_data: Optional[Union[Dict, List[Dict]]] = None,
                 true_labels: Optional[Union[List, np.ndarray]] = None,
                 available_models: Optional[List[str]] = None,
                 auto_load_validation: bool = True,
                 validation_size: int = 100,
                 split_aware: bool = False):
    """
    自动创建最优集成模型
    
    Args:
        experiment_dir: 实验目录路径
        validation_data: 验证数据（可选）
        true_labels: 真实标签（可选）
        available_models: 可用模型列表
        auto_load_validation: 是否自动加载验证数据
        validation_size: 验证数据大小限制
        split_aware: 是否根据原实验的split_mode智能选择数据集
    
    Returns:
        ModelStacker: 最优堆叠器
    """
    from model_stacking import ModelStacker
    
    if available_models is None:
        available_models = ['xgb', 'lgbm', 'catboost', 'rf', 'ann']
    
    print("🔍 自动寻找最优集成模型...")
    
    # 自动加载验证数据
    if validation_data is None and true_labels is None and auto_load_validation:
        try:
            print("🔄 自动从实验目录加载验证数据...")
            validation_data, true_labels = create_validation_dataset(
                experiment_dir, 
                validation_size=validation_size,
                split_aware=split_aware
            )
            print(f"✓ 自动加载验证数据: {len(validation_data)} 样本")
        except Exception as e:
            raise ValueError(f"自动加载验证数据失败: {e}. 请手动提供validation_data和true_labels")
    
    if validation_data is None or true_labels is None:
        raise ValueError("必须提供validation_data和true_labels，或设置auto_load_validation=True")
    
    # 加载并评估各个模型
    valid_models = []
    model_scores = {}
    
    for model_name in available_models:
        try:
            with contextlib.suppress(Exception):
                predictor = load_model(experiment_dir, model_name)
                result = predictor.predict(validation_data)
                predictions = result['predictions']
                
                if predictor.task_type == 'regression':
                    score = r2_score(true_labels, predictions)
                else:
                    score = accuracy_score(true_labels, predictions)
                
                valid_models.append(model_name)
                model_scores[model_name] = score
                print(f"  ✓ {model_name}: {score:.4f}")
        except Exception:
            print(f"  ❌ {model_name}: 无法加载")
    
    if len(valid_models) < 2:
        raise ValueError("至少需要2个有效模型才能创建集成")
    
    # 根据性能计算权重
    scores = np.array([model_scores[name] for name in valid_models])
    scores = np.maximum(scores, 0)  # 确保非负
    weights = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones(len(scores)) / len(scores)
    
    # 创建加权集成
    stacker = create_ensemble(experiment_dir, valid_models, weights.tolist(), "weighted_average")
    
    print(f"✓ 创建集成模型，包含 {len(valid_models)} 个模型")
    for name, weight in zip(valid_models, weights):
        print(f"  - {name}: {weight:.3f}")
    
    return stacker

def smart_ensemble_with_meta_learner(experiment_dir: str,
                                   validation_data: Optional[Union[Dict, List[Dict]]] = None,
                                   true_labels: Optional[Union[List, np.ndarray]] = None,
                                   available_models: Optional[List[str]] = None,
                                   auto_load_validation: bool = True,
                                   validation_size: int = 100,
                                   meta_learner: str = "ridge",
                                   split_aware: bool = False):
    """
    创建带有元学习器的智能集成模型
    
    Args:
        experiment_dir: 实验目录路径
        validation_data: 验证数据（可选）
        true_labels: 真实标签（可选）
        available_models: 可用模型列表
        auto_load_validation: 是否自动加载验证数据
        validation_size: 验证数据大小限制
        meta_learner: 元学习器类型 ("ridge", "rf", "logistic")
        split_aware: 是否根据原实验的split_mode智能选择数据集
    
    Returns:
        ModelStacker: 智能堆叠器
    """
    from model_stacking import ModelStacker
    
    if available_models is None:
        available_models = ['xgb', 'lgbm', 'catboost', 'rf', 'ann']
    
    print("🧠 创建智能元学习器集成...")
    
    # 自动加载验证数据
    if validation_data is None and true_labels is None and auto_load_validation:
        try:
            print("🔄 自动从实验目录加载验证数据...")
            validation_data, true_labels = create_validation_dataset(
                experiment_dir, 
                validation_size=validation_size,
                split_aware=split_aware
            )
            print(f"✓ 自动加载验证数据: {len(validation_data)} 样本")
        except Exception as e:
            raise ValueError(f"自动加载验证数据失败: {e}. 请手动提供validation_data和true_labels")
    
    if validation_data is None or true_labels is None:
        raise ValueError("必须提供validation_data和true_labels，或设置auto_load_validation=True")
    
    # 加载所有可用模型
    valid_models = []
    for model_name in available_models:
        try:
            with contextlib.suppress(Exception):
                predictor = load_model(experiment_dir, model_name)
                valid_models.append(model_name)
                print(f"  ✓ {model_name}: 加载成功")
        except Exception:
            print(f"  ❌ {model_name}: 无法加载")
    
    if len(valid_models) < 2:
        raise ValueError("至少需要2个有效模型才能创建集成")
    
    # 创建堆叠器并设置元学习器
    stacker = ModelStacker(experiment_dir=experiment_dir)
    stacker.set_stacking_method(meta_learner)
    
    # 添加所有有效模型
    for model_name in valid_models:
        stacker.add_model(model_name, weight=1.0)
    
    # 训练元学习器
    stacker.fit_meta_model(
        validation_data=validation_data,
        true_labels=true_labels,
        auto_load=False  # 已经手动提供了数据
    )
    
    print(f"✓ 智能集成模型创建完成，包含 {len(valid_models)} 个模型")
    print(f"✓ 元学习器: {meta_learner}")
    
    return stacker

def compare_ensemble_methods(experiment_dir: str,
                           model_names: Optional[List[str]] = None,
                           methods: Optional[List[str]] = None,
                           validation_size: int = 100) -> Dict[str, Dict[str, Any]]:
    """
    比较不同堆叠方法的性能
    
    Args:
        experiment_dir: 实验目录路径
        model_names: 模型名称列表
        methods: 堆叠方法列表
        validation_size: 验证数据大小
    
    Returns:
        dict: 各方法的评估结果
    """
    if model_names is None:
        model_names = ['xgb', 'lgbm', 'catboost']
    
    if methods is None:
        methods = ['simple_average', 'weighted_average', 'ridge']
    
    results = {}
    
    print("🔬 比较不同堆叠方法...")
    
    for method in methods:
        try:
            print(f"🔄 测试方法: {method}")
            
            if method in ['simple_average', 'weighted_average']:
                stacker = create_ensemble(experiment_dir, model_names, method=method)
            else:
                stacker = smart_ensemble_with_meta_learner(
                    experiment_dir=experiment_dir,
                    available_models=model_names,
                    validation_size=validation_size,
                    meta_learner=method
                )
            
            evaluation = stacker.evaluate(auto_load=True)
            results[method] = evaluation
            
            # 显示主要指标
            if stacker.task_type == 'regression':
                r2 = evaluation.get('r2', 'N/A')
                rmse = evaluation.get('rmse', 'N/A')
                print(f"  ✓ R²: {r2:.4f}, RMSE: {rmse:.4f}")
            else:
                acc = evaluation.get('accuracy', 'N/A')
                print(f"  ✓ Accuracy: {acc:.4f}")
                
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results[method] = {'error': str(e)}
    
    return results

def find_available_models(experiment_dir: str) -> List[str]:
    """
    查找实验目录中可用的模型
    
    Args:
        experiment_dir: 实验目录路径
    
    Returns:
        list: 可用模型名称列表
    """
    available_models = []
    candidate_models = ['xgb', 'lgbm', 'catboost', 'rf', 'ann', 'gbdt', 'extratrees']
    
    for model_name in candidate_models:
        try:
            # 尝试加载模型来检查是否存在
            load_model(experiment_dir, model_name)
            available_models.append(model_name)
        except Exception:
            pass
    
    return available_models

def get_ensemble_recommendations(experiment_dir: str) -> Dict[str, Any]:
    """
    获取集成方法推荐
    
    Args:
        experiment_dir: 实验目录路径
    
    Returns:
        dict: 推荐信息
    """
    available_models = find_available_models(experiment_dir)
    n_models = len(available_models)
    
    recommendations = {
        'available_models': available_models,
        'n_models': n_models,
        'recommendations': []
    }
    
    if n_models < 2:
        recommendations['recommendations'].append({
            'type': 'warning',
            'message': f"只找到 {n_models} 个模型，至少需要2个模型才能进行集成"
        })
    elif n_models == 2:
        recommendations['recommendations'].extend([
            {
                'type': 'method',
                'name': 'simple_average',
                'reason': '模型数量较少，简单平均即可'
            },
            {
                'type': 'method', 
                'name': 'weighted_average',
                'reason': '可以根据性能手动设置权重'
            }
        ])
    else:
        recommendations['recommendations'].extend([
            {
                'type': 'method',
                'name': 'auto_ensemble',
                'reason': '自动选择最佳权重组合'
            },
            {
                'type': 'method',
                'name': 'smart_ensemble_with_meta_learner',
                'reason': '使用元学习器自动学习最优组合策略'
            }
        ])
    
    return recommendations 