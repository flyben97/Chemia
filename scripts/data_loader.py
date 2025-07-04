#!/usr/bin/env python3
"""
INTERNCRANE 数据加载模块

自动读取INTERNCRANE训练时生成的数据拆分，支持从实验目录加载
train/valid/test数据，同时保持用户自定义数据的灵活性。
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
import warnings

def load_experiment_config(experiment_dir: str) -> Dict[str, Any]:
    """
    从实验目录加载配置文件
    
    Args:
        experiment_dir: 实验目录路径
    
    Returns:
        dict: 实验配置
    """
    config_path = os.path.join(experiment_dir, "run_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def load_original_data_splits(experiment_dir: str) -> Dict[str, pd.DataFrame]:
    """
    加载原始数据拆分
    
    Args:
        experiment_dir: 实验目录路径
    
    Returns:
        dict: 包含train, val, test数据的字典
    """
    splits_dir = os.path.join(experiment_dir, "original_data_splits")
    
    if not os.path.exists(splits_dir):
        raise FileNotFoundError(f"原始数据拆分目录不存在: {splits_dir}")
    
    data_splits = {}
    
    # 加载各个数据集
    split_files = {
        'train': 'train_original_data.csv',
        'val': 'val_original_data.csv', 
        'test': 'test_original_data.csv'
    }
    
    for split_name, filename in split_files.items():
        filepath = os.path.join(splits_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                data_splits[split_name] = df
                print(f"✓ 加载 {split_name} 数据: {len(df)} 样本")
            except Exception as e:
                warnings.warn(f"无法加载 {split_name} 数据: {e}")
        else:
            if split_name != 'val':  # val可能不存在（交叉验证模式）
                warnings.warn(f"{split_name} 数据文件不存在: {filepath}")
    
    # 加载拆分摘要
    summary_path = os.path.join(splits_dir, "data_split_summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        print("\n📊 数据拆分摘要:")
        for _, row in summary_df.iterrows():
            print(f"  {row['split']}: {row['count']} 样本 ({row['percentage']})")
    
    return data_splits

def load_processed_data_splits(experiment_dir: str) -> Dict[str, np.ndarray]:
    """
    加载处理后的特征数据拆分
    
    Args:
        experiment_dir: 实验目录路径
    
    Returns:
        dict: 包含X和y数据的字典
    """
    data_splits_dir = os.path.join(experiment_dir, "data_splits")
    
    if not os.path.exists(data_splits_dir):
        raise FileNotFoundError(f"数据拆分目录不存在: {data_splits_dir}")
    
    # 查找数据集前缀（通常是实验名称）
    files = os.listdir(data_splits_dir)
    prefixes = set()
    for file in files:
        if file.endswith('.csv'):
            parts = file.split('_')
            if len(parts) >= 2:
                prefixes.add(parts[0])
    
    if not prefixes:
        raise FileNotFoundError(f"在 {data_splits_dir} 中找不到数据文件")
    
    # 使用第一个找到的前缀
    prefix = sorted(prefixes)[0]
    print(f"🔍 使用数据集前缀: {prefix}")
    
    processed_data = {}
    
    # 加载各种数据文件
    data_files = {
        'X_train': f'{prefix}_X_train.csv',
        'y_train': f'{prefix}_y_train.csv',
        'X_val': f'{prefix}_X_val.csv',
        'y_val': f'{prefix}_y_val.csv',
        'X_test': f'{prefix}_X_test.csv',
        'y_test': f'{prefix}_y_test.csv'
    }
    
    for data_name, filename in data_files.items():
        filepath = os.path.join(data_splits_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # 转换为numpy数组
                if data_name.startswith('y_'):
                    processed_data[data_name] = df.iloc[:, 0].values  # 目标值通常是第一列
                else:
                    processed_data[data_name] = df.values
                print(f"✓ 加载 {data_name}: {df.shape}")
            except Exception as e:
                if data_name not in ['X_val', 'y_val']:  # val可能不存在
                    warnings.warn(f"无法加载 {data_name}: {e}")
    
    return processed_data

def prepare_stacking_data(experiment_dir: str, 
                         use_original: bool = True) -> Dict[str, Any]:
    """
    为模型堆叠准备数据
    
    Args:
        experiment_dir: 实验目录路径
        use_original: 是否使用原始数据（True）还是处理后的特征数据（False）
    
    Returns:
        dict: 包含验证和测试数据的字典
    """
    config = load_experiment_config(experiment_dir)
    
    if use_original:
        # 使用原始数据（推荐用于验证模型堆叠）
        data_splits = load_original_data_splits(experiment_dir)
        
        # 提取目标列
        target_col = config['data']['single_file_config']['target_col']
        
        result = {
            'config': config,
            'target_column': target_col,
            'split_mode': config.get('split_mode', 'unknown')
        }
        
        # 准备验证数据和标签
        if 'val' in data_splits and len(data_splits['val']) > 0:
            val_df = data_splits['val']
            result['validation_data'] = val_df.drop(columns=[target_col]).to_dict('records')
            result['validation_labels'] = val_df[target_col].tolist()
            print(f"✓ 准备验证数据: {len(result['validation_data'])} 样本")
        
        # 准备测试数据和标签
        if 'test' in data_splits and len(data_splits['test']) > 0:
            test_df = data_splits['test']
            result['test_data'] = test_df.drop(columns=[target_col]).to_dict('records')
            result['test_labels'] = test_df[target_col].tolist()
            print(f"✓ 准备测试数据: {len(result['test_data'])} 样本")
        
        # 如果没有验证集，使用部分训练数据作为验证
        if 'validation_data' not in result and 'train' in data_splits:
            train_df = data_splits['train']
            # 使用后20%的训练数据作为验证
            val_size = max(10, int(len(train_df) * 0.2))
            val_df = train_df.tail(val_size)
            
            result['validation_data'] = val_df.drop(columns=[target_col]).to_dict('records')
            result['validation_labels'] = val_df[target_col].tolist()
            print(f"⚠️  使用训练数据尾部作为验证: {len(result['validation_data'])} 样本")
        
        return result
    
    else:
        # 使用处理后的特征数据
        processed_data = load_processed_data_splits(experiment_dir)
        
        result = {
            'config': config,
            'split_mode': config.get('split_mode', 'unknown'),
            'processed_data': processed_data
        }
        
        # 准备验证数据
        if 'X_val' in processed_data and 'y_val' in processed_data:
            result['validation_features'] = processed_data['X_val']
            result['validation_labels'] = processed_data['y_val']
            print(f"✓ 准备验证特征: {processed_data['X_val'].shape}")
        
        # 准备测试数据
        if 'X_test' in processed_data and 'y_test' in processed_data:
            result['test_features'] = processed_data['X_test']
            result['test_labels'] = processed_data['y_test']
            print(f"✓ 准备测试特征: {processed_data['X_test'].shape}")
        
        return result

def format_sample_for_prediction(sample_dict: Dict[str, Any], 
                                config: Dict[str, Any]) -> Dict[str, Any]:
    """
    将原始数据样本格式化为预测所需的格式
    
    Args:
        sample_dict: 原始样本字典
        config: 实验配置
    
    Returns:
        dict: 格式化后的样本字典
    """
    # 提取SMILES列和其他特征列
    smiles_cols = config['data']['single_file_config']['smiles_col']
    target_col = config['data']['single_file_config']['target_col']
    
    formatted_sample = {}
    
    # 复制SMILES列
    for col in smiles_cols:
        if col in sample_dict:
            formatted_sample[col] = sample_dict[col]
    
    # 复制其他非目标列
    precomputed_features = config['data']['single_file_config'].get('precomputed_features', {})
    feature_columns = precomputed_features.get('feature_columns')
    
    if feature_columns:
        # 处理预计算特征列
        for key, value in sample_dict.items():
            if key not in smiles_cols and key != target_col:
                formatted_sample[key] = value
    
    return formatted_sample

def create_validation_dataset(experiment_dir: str,
                            validation_size: int = 50,
                            use_test_if_no_val: bool = True,
                            split_aware: bool = False) -> Tuple[List[Dict], List]:
    """
    创建用于模型堆叠验证的数据集
    
    Args:
        experiment_dir: 实验目录路径
        validation_size: 验证数据大小限制
        use_test_if_no_val: 如果没有验证集，是否使用测试集（当split_aware=False时生效）
        split_aware: 是否根据原实验的split_mode智能选择数据集
                    True: train_valid_test模式使用validation，cross_validation模式使用test
                    False: 按照use_test_if_no_val参数的传统逻辑
    
    Returns:
        tuple: (验证数据列表, 真实标签列表)
    """
    data_info = prepare_stacking_data(experiment_dir, use_original=True)
    config = data_info['config']
    
    # 获取原实验的数据拆分模式
    split_mode = config.get('split_mode', 'unknown')
    
    val_data = None
    val_labels = None
    source = None
    
    if split_aware:
        # 根据原实验的split_mode智能选择数据集
        if split_mode == 'train_valid_test':
            # train_valid_test模式：优先使用validation set进行堆叠验证
            if 'validation_data' in data_info:
                val_data = data_info['validation_data']
                val_labels = data_info['validation_labels']
                source = "validation"
                print(f"🎯 [split-aware] train_valid_test模式：使用validation set进行堆叠验证")
            elif 'test_data' in data_info:
                val_data = data_info['test_data']
                val_labels = data_info['test_labels']
                source = "test"
                print(f"⚠️  [split-aware] 没有validation set，使用test set进行堆叠验证")
            else:
                raise ValueError("train_valid_test模式下没有可用的validation或test数据")
        
        elif split_mode == 'cross_validation':
            # cross_validation模式：使用test set进行堆叠验证（因为没有专门的validation set）
            if 'test_data' in data_info:
                val_data = data_info['test_data']
                val_labels = data_info['test_labels']
                source = "test"
                print(f"🎯 [split-aware] cross_validation模式：使用test set进行堆叠验证")
            else:
                # 如果没有test set，使用部分train数据
                if 'validation_data' in data_info:
                    val_data = data_info['validation_data']
                    val_labels = data_info['validation_labels']
                    source = "validation(from_train)"
                    print(f"⚠️  [split-aware] cross_validation模式下没有test set，使用训练数据的一部分")
                else:
                    raise ValueError("cross_validation模式下没有可用的test或validation数据")
        
        else:
            print(f"⚠️  [split-aware] 未知的split_mode: {split_mode}，回退到传统逻辑")
            split_aware = False  # 回退到传统逻辑
    
    if not split_aware:
        # 传统逻辑：优先使用验证数据
        if 'validation_data' in data_info:
            val_data = data_info['validation_data']
            val_labels = data_info['validation_labels']
            source = "validation"
        elif use_test_if_no_val and 'test_data' in data_info:
            val_data = data_info['test_data']
            val_labels = data_info['test_labels']
            source = "test"
            print("⚠️  没有验证集，使用测试集进行堆叠验证")
        else:
            raise ValueError("没有可用的验证数据")
    
    if val_data is None or val_labels is None:
        raise ValueError("无法获取验证数据")
    
    # 限制数据大小以提高效率
    if len(val_data) > validation_size:
        indices = np.random.choice(len(val_data), validation_size, replace=False)
        val_data = [val_data[i] for i in indices]
        val_labels = [val_labels[i] for i in indices]
        print(f"🎯 随机选择 {validation_size} 个样本进行验证（从{source}集）")
    else:
        print(f"🎯 使用全部 {len(val_data)} 个样本进行验证（从{source}集）")
    
    # 格式化样本
    formatted_data = []
    for sample in val_data:
        formatted_sample = format_sample_for_prediction(sample, config)
        formatted_data.append(formatted_sample)
    
    return formatted_data, val_labels

def load_custom_validation_data(validation_file: str,
                               labels_file: Optional[str] = None,
                               target_column: Optional[str] = None) -> Tuple[List[Dict], List]:
    """
    加载用户自定义的验证数据
    
    Args:
        validation_file: 验证数据文件路径（CSV）
        labels_file: 标签文件路径（可选，如果None则从validation_file中读取）
        target_column: 目标列名（如果labels_file为None时需要）
    
    Returns:
        tuple: (验证数据列表, 真实标签列表)
    """
    # 加载验证数据
    val_df = pd.read_csv(validation_file)
    
    if labels_file is not None:
        # 从单独的标签文件加载
        labels_df = pd.read_csv(labels_file)
        if len(labels_df.columns) == 1:
            labels = labels_df.iloc[:, 0].tolist()
        else:
            raise ValueError("标签文件应该只包含一列")
        
        # 验证数据不包含目标列
        validation_data = val_df.to_dict('records')
    
    else:
        # 从验证文件中分离特征和标签
        if target_column is None:
            raise ValueError("如果不提供单独的标签文件，必须指定target_column")
        
        if target_column not in val_df.columns:
            raise ValueError(f"目标列 '{target_column}' 不在数据中")
        
        labels = val_df[target_column].tolist()
        feature_df = val_df.drop(columns=[target_column])
        validation_data = feature_df.to_dict('records')
    
    print(f"✓ 加载自定义验证数据: {len(validation_data)} 样本")
    return validation_data, labels

# 示例用法
if __name__ == "__main__":
    print("INTERNCRANE 数据加载模块示例")
    print("=" * 50)
    
    # 示例实验目录
    example_dir = "output/S04_agent_5_a_regression_20250101_120000"
    
    print("1. 加载实验配置:")
    try:
        config = load_experiment_config(example_dir)
        print(f"  实验名称: {config['experiment_name']}")
        print(f"  任务类型: {config['task_type']}")
        print(f"  数据拆分模式: {config['split_mode']}")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
    
    print("\n2. 加载原始数据拆分:")
    try:
        data_splits = load_original_data_splits(example_dir)
        print(f"  可用数据集: {list(data_splits.keys())}")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
    
    print("\n3. 准备堆叠验证数据:")
    try:
        val_data, val_labels = create_validation_dataset(example_dir, validation_size=20)
        print(f"  验证数据: {len(val_data)} 样本")
        print(f"  示例样本键: {list(val_data[0].keys()) if val_data else []}")
        print(f"  标签范围: {min(val_labels):.3f} - {max(val_labels):.3f}")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
    
    print("\n4. 自定义数据示例:")
    print("""
# 使用自定义验证数据
val_data, val_labels = load_custom_validation_data(
    validation_file="my_validation_data.csv",
    target_column="target_value"
)

# 或者使用分离的文件
val_data, val_labels = load_custom_validation_data(
    validation_file="my_features.csv",
    labels_file="my_labels.csv"
)
    """) 