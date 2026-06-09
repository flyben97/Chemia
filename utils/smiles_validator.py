# utils/smiles_validator.py
"""
SMILES 验证器 - 验证和检测 SMILES 列
SMILES Validator - Validate and detect SMILES columns
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from rdkit import Chem
from rich.console import Console

console = Console()


def validate_smiles(smiles_str: str) -> bool:
    """
    验证单个 SMILES 字符串是否有效

    Args:
        smiles_str: SMILES 字符串

    Returns:
        是否有效
    """
    if not isinstance(smiles_str, str):
        return False

    try:
        mol = Chem.MolFromSmiles(smiles_str)
        return mol is not None
    except Exception:
        return False


def validate_smiles_columns(df: pd.DataFrame,
                           smiles_cols: List[str],
                           sample_size: int = 200,
                           min_valid_ratio: float = 0.8,
                           show_details: bool = True,
                           random_state: int = None) -> Tuple[bool, Dict[str, Any]]:
    """
    验证 SMILES 列的有效性

    Args:
        df: 数据框
        smiles_cols: SMILES 列名列表
        sample_size: 验证的样本数
        min_valid_ratio: 最小有效比例
        show_details: 是否显示详细信息

    Returns:
        (所有列都有效, 验证结果字典)
    """
    validation_results = {}
    all_valid = True

    for col in smiles_cols:
        if col not in df.columns:
            validation_results[col] = {
                'is_valid_column': False,
                'error_message': f'列 "{col}" 不存在于数据框中',
                'valid_count': 0,
                'sample_size_checked': 0,
                'valid_ratio': 0.0
            }
            all_valid = False
            continue

        # 获取样本
        sample_size_actual = min(sample_size, len(df))
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(len(df), size=sample_size_actual, replace=False)
        sample_data = df.iloc[sample_indices][col]

        # 验证 SMILES
        valid_count = 0
        for smiles in sample_data:
            if validate_smiles(smiles):
                valid_count += 1

        valid_ratio = valid_count / sample_size_actual if sample_size_actual > 0 else 0
        is_valid = valid_ratio >= min_valid_ratio

        validation_results[col] = {
            'is_valid_column': is_valid,
            'error_message': None if is_valid else f'有效 SMILES 比例过低: {valid_ratio:.1%}',
            'valid_count': valid_count,
            'sample_size_checked': sample_size_actual,
            'valid_ratio': valid_ratio
        }

        if not is_valid:
            all_valid = False

        if show_details:
            status = "✓" if is_valid else "✗"
            console.print(f"  {status} {col}: {valid_count}/{sample_size_actual} 有效 ({valid_ratio:.1%})")

    return all_valid, validation_results


def suggest_potential_smiles_columns(df: pd.DataFrame, random_state: int = None) -> List[str]:
    """
    建议可能的 SMILES 列

    Args:
        df: 数据框

    Returns:
        可能的 SMILES 列名列表
    """
    potential_cols = []

    # 基于列名的启发式方法
    smiles_keywords = ['smiles', 'smi', 'mol', 'structure', 'canonical', 'smilesstring']

    for col in df.columns:
        col_lower = col.lower()

        # 检查列名
        if any(keyword in col_lower for keyword in smiles_keywords):
            potential_cols.append(col)
            continue

        # 检查列内容 (采样)
        if df[col].dtype == 'object':
            sample_size = min(100, len(df))
            rng = np.random.default_rng(random_state)
            sample_indices = rng.choice(len(df), size=sample_size, replace=False)
            sample_data = df.iloc[sample_indices][col]

            # 检查是否看起来像 SMILES
            valid_count = 0
            for value in sample_data:
                if isinstance(value, str) and validate_smiles(value):
                    valid_count += 1

            # 如果超过 50% 的样本是有效 SMILES，则认为是 SMILES 列
            if valid_count / sample_size > 0.5:
                potential_cols.append(col)

    return potential_cols


def detect_smiles_columns(df: pd.DataFrame,
                         confidence_threshold: float = 0.7,
                         random_state: int = None) -> List[str]:
    """
    自动检测 SMILES 列

    Args:
        df: 数据框
        confidence_threshold: 置信度阈值

    Returns:
        检测到的 SMILES 列名列表
    """
    detected_cols = []

    for col in df.columns:
        if df[col].dtype != 'object':
            continue

        # 采样检查
        sample_size = min(100, len(df))
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(len(df), size=sample_size, replace=False)
        sample_data = df.iloc[sample_indices][col]

        # 计算有效 SMILES 的比例
        valid_count = 0
        for value in sample_data:
            if isinstance(value, str) and validate_smiles(value):
                valid_count += 1

        confidence = valid_count / sample_size if sample_size > 0 else 0

        if confidence >= confidence_threshold:
            detected_cols.append(col)

    return detected_cols


def standardize_smiles(smiles_str: str) -> str:
    """
    标准化 SMILES 字符串

    Args:
        smiles_str: SMILES 字符串

    Returns:
        标准化的 SMILES 字符串，如果无效则返回原字符串
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass

    return smiles_str


def standardize_smiles_column(df: pd.DataFrame, col: str) -> pd.Series:
    """
    标准化 SMILES 列

    Args:
        df: 数据框
        col: 列名

    Returns:
        标准化后的 Series
    """
    return df[col].apply(standardize_smiles)


def get_smiles_statistics(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """
    获取 SMILES 列的统计信息

    Args:
        df: 数据框
        col: 列名

    Returns:
        统计信息字典
    """
    stats = {
        'total_count': len(df),
        'non_null_count': df[col].notna().sum(),
        'null_count': df[col].isna().sum(),
        'valid_smiles_count': 0,
        'invalid_smiles_count': 0,
        'unique_smiles_count': 0,
        'average_length': 0.0
    }

    # 计算有效/无效 SMILES
    valid_smiles = []
    for smiles in df[col].dropna():
        if isinstance(smiles, str):
            if validate_smiles(smiles):
                stats['valid_smiles_count'] += 1
                valid_smiles.append(smiles)
            else:
                stats['invalid_smiles_count'] += 1

    stats['unique_smiles_count'] = len(set(valid_smiles))

    # 计算平均长度
    if valid_smiles:
        stats['average_length'] = np.mean([len(s) for s in valid_smiles])

    return stats


def validate_and_clean_smiles(df: pd.DataFrame,
                             smiles_cols: List[str],
                             remove_invalid: bool = False) -> pd.DataFrame:
    """
    验证并清理 SMILES 列

    Args:
        df: 数据框
        smiles_cols: SMILES 列名列表
        remove_invalid: 是否删除包含无效 SMILES 的行

    Returns:
        清理后的数据框
    """
    df_clean = df.copy()

    if remove_invalid:
        # 删除包含无效 SMILES 的行
        for col in smiles_cols:
            if col in df_clean.columns:
                valid_mask = df_clean[col].apply(
                    lambda x: isinstance(x, str) and validate_smiles(x)
                )
                df_clean = df_clean[valid_mask]
    else:
        # 标准化 SMILES
        for col in smiles_cols:
            if col in df_clean.columns:
                df_clean[col] = standardize_smiles_column(df_clean, col)

    return df_clean
