#!/usr/bin/env python3
"""
CRAFT 模型堆叠配置工具

这个模块提供YAML配置文件的加载、验证和处理功能。
包含堆叠配置的验证逻辑和工具函数。
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: YAML配置文件路径
    
    Returns:
        dict: 配置字典
    
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML格式错误
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML文件解析错误: {e}")

def validate_stacking_config(config: Dict[str, Any]) -> None:
    """
    验证模型堆叠YAML配置文件
    
    Args:
        config: 从YAML文件加载的配置字典
    
    Raises:
        ValueError: 配置验证失败时抛出
    """
    if 'stacking' not in config:
        raise ValueError("配置文件必须包含 'stacking' 部分")
    
    stacking_config = config['stacking']
    
    # 验证基本配置
    if 'experiment_dir' not in stacking_config:
        raise ValueError("配置文件必须指定 'stacking.experiment_dir'")
    
    # 验证堆叠方法
    method = stacking_config.get('method', 'weighted_average')
    valid_methods = ["simple_average", "weighted_average", "ridge", "rf", "logistic"]
    if method not in valid_methods:
        raise ValueError(f"不支持的堆叠方法: {method}. 支持的方法: {valid_methods}")
    
    # 验证模型配置
    if 'models' not in stacking_config:
        raise ValueError("配置文件必须包含 'stacking.models' 列表")
    
    models = stacking_config['models']
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("'stacking.models' 必须是非空列表")
    
    for i, model_config in enumerate(models):
        if not isinstance(model_config, dict):
            raise ValueError(f"模型配置 {i} 必须是字典")
        
        if 'name' not in model_config:
            raise ValueError(f"模型配置 {i} 必须包含 'name' 字段")
        
        weight = model_config.get('weight', 1.0)
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(f"模型 {model_config['name']} 的权重必须是正数")
        
        enabled = model_config.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError(f"模型 {model_config['name']} 的 'enabled' 字段必须是布尔值")
    
    # 验证元模型配置（如果存在）
    if 'meta_model' in stacking_config:
        meta_config = stacking_config['meta_model']
        if not isinstance(meta_config, dict):
            raise ValueError("'stacking.meta_model' 必须是字典")
        
        if 'validation' in meta_config:
            validation_config = meta_config['validation']
            if not isinstance(validation_config, dict):
                raise ValueError("'stacking.meta_model.validation' 必须是字典")
            
            size = validation_config.get('size', 100)
            if not isinstance(size, int) or size <= 0:
                raise ValueError("验证数据大小必须是正整数")
            
            split_aware = validation_config.get('split_aware', False)
            if not isinstance(split_aware, bool):
                raise ValueError("'split_aware' 必须是布尔值")

def extract_stacking_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从完整配置中提取堆叠相关配置
    
    Args:
        config: 完整的配置字典
    
    Returns:
        dict: 堆叠配置字典
    """
    validate_stacking_config(config)
    return config['stacking']

def extract_model_configs(stacking_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    提取启用的模型配置列表
    
    Args:
        stacking_config: 堆叠配置字典
    
    Returns:
        list: 启用的模型配置列表
    """
    models = stacking_config.get('models', [])
    return [model for model in models if model.get('enabled', True)]

def get_model_weights(model_configs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    从模型配置中提取权重字典
    
    Args:
        model_configs: 模型配置列表
    
    Returns:
        dict: 模型名称到权重的映射
    """
    weights = {}
    for model_config in model_configs:
        name = model_config['name']
        weight = model_config.get('weight', 1.0)
        weights[name] = weight
    return weights

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    归一化权重，使权重和为1
    
    Args:
        weights: 原始权重字典
    
    Returns:
        dict: 归一化的权重字典
    """
    total_weight = sum(weights.values())
    if total_weight == 0:
        # 如果所有权重都是0，使用均等权重
        equal_weight = 1.0 / len(weights)
        return {name: equal_weight for name in weights.keys()}
    
    return {name: weight / total_weight for name, weight in weights.items()}

def save_yaml_config(config: Dict[str, Any], output_path: str) -> None:
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        output_path: 输出文件路径
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def create_sample_stacking_config(
    experiment_dir: str,
    model_names: List[str],
    weights: Optional[List[float]] = None,
    method: str = "weighted_average",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建示例堆叠配置
    
    Args:
        experiment_dir: 实验目录路径
        model_names: 模型名称列表
        weights: 权重列表（可选）
        method: 堆叠方法
        output_path: 输出路径（可选，如果提供则保存到文件）
    
    Returns:
        dict: 配置字典
    """
    if weights is None:
        weights = [1.0] * len(model_names)
    
    if len(weights) != len(model_names):
        raise ValueError("权重数量必须与模型数量一致")
    
    # 创建模型配置
    models = []
    for name, weight in zip(model_names, weights):
        models.append({
            'name': name,
            'weight': weight,
            'enabled': True
        })
    
    # 创建完整配置
    config = {
        'stacking': {
            'experiment_dir': experiment_dir,
            'method': method,
            'models': models
        },
        'evaluation': {
            'auto_evaluate': True,
            'use_test_set': True
        },
        'save': {
            'save_stacker': True,
            'save_path': f"output/ensemble_{method}.pkl"
        }
    }
    
    # 如果是元学习器方法，添加相应配置
    if method in ['ridge', 'rf', 'logistic']:
        config['stacking']['meta_model'] = {
            'auto_train': True,
            'validation': {
                'auto_load': True,
                'size': 100
            }
        }
    
    # 保存到文件（如果指定）
    if output_path:
        save_yaml_config(config, output_path)
    
    return config

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典
    
    Args:
        configs: 要合并的配置字典
    
    Returns:
        dict: 合并后的配置字典
    """
    merged = {}
    for config in configs:
        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged

# 预定义的配置模板
STACKING_CONFIG_TEMPLATES = {
    'basic_weighted': {
        'stacking': {
            'method': 'weighted_average',
            'models': [
                {'name': 'xgb', 'weight': 0.4, 'enabled': True},
                {'name': 'lgbm', 'weight': 0.3, 'enabled': True},
                {'name': 'catboost', 'weight': 0.3, 'enabled': True}
            ]
        },
        'evaluation': {'auto_evaluate': True, 'use_test_set': True},
        'save': {'save_stacker': True}
    },
    
    'simple_average': {
        'stacking': {
            'method': 'simple_average',
            'models': [
                {'name': 'xgb', 'enabled': True},
                {'name': 'lgbm', 'enabled': True},
                {'name': 'catboost', 'enabled': True}
            ]
        },
        'evaluation': {'auto_evaluate': True, 'use_test_set': True},
        'save': {'save_stacker': True}
    },
    
    'meta_learner': {
        'stacking': {
            'method': 'ridge',
            'models': [
                {'name': 'xgb', 'enabled': True},
                {'name': 'lgbm', 'enabled': True},
                {'name': 'catboost', 'enabled': True},
                {'name': 'rf', 'enabled': True}
            ],
            'meta_model': {
                'auto_train': True,
                'validation': {'auto_load': True, 'size': 200, 'split_aware': False}
            }
        },
        'evaluation': {'auto_evaluate': True, 'use_test_set': True, 'compare_with_base': True},
        'save': {'save_stacker': True, 'save_evaluation': True}
    }
}

def get_config_template(template_name: str, experiment_dir: str) -> Dict[str, Any]:
    """
    获取预定义的配置模板
    
    Args:
        template_name: 模板名称 ('basic_weighted', 'simple_average', 'meta_learner')
        experiment_dir: 实验目录路径
    
    Returns:
        dict: 配置字典
    
    Raises:
        ValueError: 模板不存在
    """
    if template_name not in STACKING_CONFIG_TEMPLATES:
        available = list(STACKING_CONFIG_TEMPLATES.keys())
        raise ValueError(f"未知模板: {template_name}. 可用模板: {available}")
    
    template = STACKING_CONFIG_TEMPLATES[template_name].copy()
    template['stacking']['experiment_dir'] = experiment_dir
    
    return template 