"""
Configuration Parser with Column Range Support

This module provides utilities for parsing configuration files with support
for column range specifications in custom feature columns.
"""

import re
import pandas as pd
from typing import List, Union, Dict, Any, Optional


class ColumnRangeParser:
    """Parser for column range specifications"""
    
    def __init__(self):
        # 支持的范围语法模式
        self.range_patterns = {
            'name_range': r'^([a-zA-Z_]\w*)_(\d+):([a-zA-Z_]\w*)_(\d+)$',  # feature_1:feature_10
            'name_simple': r'^([a-zA-Z_]\w*):([a-zA-Z_]\w*)$',             # col_start:col_end
            'index_range': r'^(\d+):(\d+)$',                               # 5:15
            'name_prefix': r'^([a-zA-Z_]\w*)_(\d+):(\d+)$'                 # temperature_1:5
        }
    
    def parse_column_specification(self, 
                                 column_spec: Union[str, List[str]], 
                                 available_columns: Optional[List[str]] = None) -> List[str]:
        """
        解析列规范，支持范围语法
        
        Args:
            column_spec: 列规范，可以是字符串、列表或混合
            available_columns: 可用的列名列表（用于验证）
            
        Returns:
            解析后的列名列表
            
        Examples:
            # 单个列
            "temperature" -> ["temperature"]
            
            # 列名范围
            "temperature_1:temperature_5" -> ["temperature_1", "temperature_2", ..., "temperature_5"]
            
            # 前缀+数字范围  
            "feature_1:10" -> ["feature_1", "feature_2", ..., "feature_10"]
            
            # 索引范围（需要available_columns）
            "5:8" -> [available_columns[5], available_columns[6], available_columns[7], available_columns[8]]
            
            # 混合列表
            ["temperature", "pressure_1:3", "time"] -> ["temperature", "pressure_1", "pressure_2", "pressure_3", "time"]
        """
        if isinstance(column_spec, str):
            return self._parse_single_spec(column_spec, available_columns)
        elif isinstance(column_spec, list):
            result = []
            for spec in column_spec:
                result.extend(self._parse_single_spec(spec, available_columns))
            return result
        else:
            raise ValueError(f"Unsupported column specification type: {type(column_spec)}")
    
    def _parse_single_spec(self, spec: str, available_columns: Optional[List[str]] = None) -> List[str]:
        """解析单个列规范"""
        
        # 检查是否是范围语法
        for pattern_name, pattern in self.range_patterns.items():
            match = re.match(pattern, spec)
            if match:
                return self._expand_range(pattern_name, match, available_columns)
        
        # 如果不是范围语法，返回原始规范
        return [spec]
    
    def _expand_range(self, pattern_name: str, match, available_columns: Optional[List[str]] = None) -> List[str]:
        """展开范围规范"""
        
        if pattern_name == 'name_range':
            # feature_1:feature_10 格式
            prefix1, start_num, prefix2, end_num = match.groups()
            if prefix1 != prefix2:
                raise ValueError(f"Range prefixes must match: {prefix1} vs {prefix2}")
            
            start_idx = int(start_num)
            end_idx = int(end_num)
            return [f"{prefix1}_{i}" for i in range(start_idx, end_idx + 1)]
        
        elif pattern_name == 'name_simple':
            # col_start:col_end 格式
            start_col, end_col = match.groups()
            if available_columns is None:
                raise ValueError("available_columns required for name_simple range")
            
            try:
                start_idx = available_columns.index(start_col)
                end_idx = available_columns.index(end_col)
                return available_columns[start_idx:end_idx + 1]
            except ValueError as e:
                raise ValueError(f"Column not found in available_columns: {e}")
        
        elif pattern_name == 'index_range':
            # 5:15 格式
            start_idx, end_idx = map(int, match.groups())
            if available_columns is None:
                raise ValueError("available_columns required for index_range")
            
            if end_idx >= len(available_columns):
                raise ValueError(f"End index {end_idx} exceeds available columns length {len(available_columns)}")
            
            return available_columns[start_idx:end_idx + 1]
        
        elif pattern_name == 'name_prefix':
            # temperature_1:5 格式
            prefix, start_num, end_num = match.groups()
            start_idx = int(start_num)
            end_idx = int(end_num)
            return [f"{prefix}_{i}" for i in range(start_idx, end_idx + 1)]
        
        else:
            raise ValueError(f"Unknown pattern: {pattern_name}")
    
    def validate_columns(self, parsed_columns: List[str], available_columns: List[str]) -> List[str]:
        """验证解析后的列是否都存在"""
        missing = [col for col in parsed_columns if col not in available_columns]
        if missing:
            raise ValueError(f"Columns not found in dataset: {missing}")
        return parsed_columns


def parse_custom_feature_columns(config: Dict[str, Any], 
                                available_columns: Optional[List[str]] = None) -> List[str]:
    """
    从配置中解析自定义特征列
    
    Args:
        config: 配置字典
        available_columns: 数据集中可用的列名
        
    Returns:
        解析后的特征列名列表
    """
    parser = ColumnRangeParser()
    
    # 获取自定义特征列配置
    custom_feature_columns = config.get('data', {}).get('custom_feature_columns', [])
    
    if not custom_feature_columns:
        return []
    
    # 解析列规范
    parsed_columns = parser.parse_column_specification(custom_feature_columns, available_columns)
    
    # 如果有可用列信息，进行验证
    if available_columns:
        parsed_columns = parser.validate_columns(parsed_columns, available_columns)
    
    return parsed_columns


# 使用示例和测试
if __name__ == "__main__":
    parser = ColumnRangeParser()
    
    # 测试数据
    test_columns = [f"feature_{i}" for i in range(20)] + ["temperature", "pressure", "time"]
    
    print("🧪 Column Range Parser Tests")
    print("=" * 50)
    
    test_cases = [
        # 基本用法
        "temperature",
        ["temperature", "pressure"],
        
        # 前缀+数字范围
        "feature_1:5",
        "feature_10:15",
        
        # 名称范围
        "feature_1:feature_5",
        
        # 索引范围
        "0:3",
        "15:17",
        
        # 混合用法
        ["temperature", "feature_1:3", "pressure", "5:7"]
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = parser.parse_column_specification(test_case, test_columns)
            print(f"Test {i}: {test_case}")
            print(f"  Result: {result}")
            print(f"  Count: {len(result)} columns")
            print()
        except Exception as e:
            print(f"Test {i}: {test_case}")
            print(f"  Error: {e}")
            print()
    
    print("✅ All tests completed!") 