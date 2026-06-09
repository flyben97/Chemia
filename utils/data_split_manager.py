# utils/data_split_manager.py
"""
数据拆分管理工具
用于查看、验证和管理训练过程中保存的数据拆分
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DataSplitManager:
    """数据拆分管理器"""

    def __init__(self, experiment_dir: str):
        """
        初始化数据拆分管理器

        Args:
            experiment_dir: 实验输出目录路径
        """
        self.experiment_dir = experiment_dir
        self.original_splits_dir = os.path.join(experiment_dir, 'original_data_splits')
        self.raw_original_splits_dir = os.path.join(experiment_dir, 'raw_original_data_splits')
        self.processed_splits_dir = os.path.join(experiment_dir, 'data_splits')

    def list_available_splits(self) -> Dict[str, List[str]]:
        """列出所有可用的数据拆分"""
        splits = {
            'raw_original': [],
            'cleaned_original': [],
            'processed': []
        }

        # 检查原始数据拆分
        if os.path.exists(self.raw_original_splits_dir):
            for file in os.listdir(self.raw_original_splits_dir):
                if file.endswith('.csv') and not file.startswith('complete_') and not file.endswith('_summary.csv'):
                    splits['raw_original'].append(file)

        # 检查清理后的原始数据拆分
        if os.path.exists(self.original_splits_dir):
            for file in os.listdir(self.original_splits_dir):
                if file.endswith('.csv') and not file.endswith('_summary.csv'):
                    splits['cleaned_original'].append(file)

        # 检查处理后的数据拆分
        if os.path.exists(self.processed_splits_dir):
            for file in os.listdir(self.processed_splits_dir):
                if file.endswith('.csv'):
                    splits['processed'].append(file)

        return splits

    def get_split_info(self) -> pd.DataFrame:
        """获取数据拆分的详细信息"""
        info_data = []

        # 原始数据信息
        raw_summary_file = os.path.join(self.raw_original_splits_dir, 'raw_data_split_summary.csv')
        if os.path.exists(raw_summary_file):
            raw_summary = pd.read_csv(raw_summary_file)
            for _, row in raw_summary.iterrows():
                info_data.append({
                    'type': 'raw_original',
                    'dataset': row['dataset'],
                    'count': row['count'],
                    'percentage': row['percentage'],
                    'description': row['description']
                })

        # 清理后数据信息
        cleaned_summary_file = os.path.join(self.original_splits_dir, 'data_split_summary.csv')
        if os.path.exists(cleaned_summary_file):
            cleaned_summary = pd.read_csv(cleaned_summary_file)
            for _, row in cleaned_summary.iterrows():
                info_data.append({
                    'type': 'cleaned_original',
                    'dataset': row['split'],
                    'count': row['count'],
                    'percentage': row['percentage'],
                    'description': f"Cleaned {row['split']} split"
                })

        return pd.DataFrame(info_data)

    def load_split_data(self, split_type: str, split_name: str) -> Optional[pd.DataFrame]:
        """
        加载指定的数据拆分

        Args:
            split_type: 数据类型 ('raw_original', 'cleaned_original', 'processed')
            split_name: 拆分名称 ('train', 'val', 'test')

        Returns:
            DataFrame或None
        """
        if split_type == 'raw_original':
            file_path = os.path.join(self.raw_original_splits_dir, f'{split_name}_raw_original_data.csv')
        elif split_type == 'cleaned_original':
            file_path = os.path.join(self.original_splits_dir, f'{split_name}_original_data.csv')
        elif split_type == 'processed':
            # 处理后的数据可能有多个文件，选择主要的
            possible_files = [
                f'processed_dataset_{split_name}.csv',
                f'raw_dataset_{split_name}.csv'
            ]
            file_path = None
            for filename in possible_files:
                potential_path = os.path.join(self.processed_splits_dir, filename)
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break
        else:
            raise ValueError(f"Invalid split_type: {split_type}")

        if file_path and os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            return None

    def compare_splits(self, split_name: str = 'train') -> pd.DataFrame:
        """
        比较同一拆分的不同版本

        Args:
            split_name: 要比较的拆分名称

        Returns:
            比较结果DataFrame
        """
        comparison_data = []

        # 加载不同版本的数据
        raw_data = self.load_split_data('raw_original', split_name)
        cleaned_data = self.load_split_data('cleaned_original', split_name)
        processed_data = self.load_split_data('processed', split_name)

        # 比较基本信息
        for data_type, data in [('raw_original', raw_data),
                               ('cleaned_original', cleaned_data),
                               ('processed', processed_data)]:
            if data is not None:
                comparison_data.append({
                    'data_type': data_type,
                    'rows': len(data),
                    'columns': len(data.columns),
                    'column_names': ', '.join(data.columns[:5]) + ('...' if len(data.columns) > 5 else ''),
                    'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                    'has_features': any('_' in col for col in data.columns if col not in ['smiles', 'target', 'ee'])
                })

        return pd.DataFrame(comparison_data)

    def validate_splits(self) -> Dict[str, bool]:
        """验证数据拆分的完整性"""
        validation_results = {}

        # 检查必要的目录是否存在
        validation_results['raw_original_dir_exists'] = os.path.exists(self.raw_original_splits_dir)
        validation_results['cleaned_original_dir_exists'] = os.path.exists(self.original_splits_dir)
        validation_results['processed_dir_exists'] = os.path.exists(self.processed_splits_dir)

        # 检查训练集是否存在
        validation_results['raw_train_exists'] = self.load_split_data('raw_original', 'train') is not None
        validation_results['cleaned_train_exists'] = self.load_split_data('cleaned_original', 'train') is not None
        validation_results['processed_train_exists'] = self.load_split_data('processed', 'train') is not None

        # 检查索引映射文件
        index_mapping_file = os.path.join(self.raw_original_splits_dir, 'index_mapping.csv')
        validation_results['index_mapping_exists'] = os.path.exists(index_mapping_file)

        return validation_results

    def get_index_mapping(self) -> Optional[pd.DataFrame]:
        """获取原始索引到清理索引的映射"""
        mapping_file = os.path.join(self.raw_original_splits_dir, 'index_mapping.csv')
        if os.path.exists(mapping_file):
            return pd.read_csv(mapping_file)
        return None

    def export_split_for_reuse(self, split_type: str, split_name: str, output_file: str) -> bool:
        """
        导出数据拆分供其他项目使用

        Args:
            split_type: 数据类型
            split_name: 拆分名称
            output_file: 输出文件路径

        Returns:
            是否成功
        """
        data = self.load_split_data(split_type, split_name)
        if data is not None:
            data.to_csv(output_file, index=False)
            return True
        return False

    def create_custom_split(self, raw_data_file: str, train_ratio: float = 0.7,
                           val_ratio: float = 0.15, test_ratio: float = 0.15,
                           output_dir: str = None, random_state: int = 42) -> Dict[str, str]:
        """
        从原始数据创建自定义拆分

        Args:
            raw_data_file: 原始数据文件路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            output_dir: 输出目录
            random_state: 随机种子

        Returns:
            输出文件路径字典
        """
        from sklearn.model_selection import train_test_split

        if output_dir is None:
            output_dir = os.path.join(self.experiment_dir, 'custom_splits')

        os.makedirs(output_dir, exist_ok=True)

        # 读取原始数据
        df = pd.read_csv(raw_data_file)

        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("训练、验证、测试集比例之和必须等于1")

        # 第一次拆分：分离测试集
        if test_ratio > 0:
            df_train_val, df_test = train_test_split(
                df, test_size=test_ratio, random_state=random_state
            )
        else:
            df_train_val = df
            df_test = pd.DataFrame()

        # 第二次拆分：分离训练集和验证集
        if val_ratio > 0 and len(df_train_val) > 0:
            val_size_adjusted = val_ratio / (train_ratio + val_ratio)
            df_train, df_val = train_test_split(
                df_train_val, test_size=val_size_adjusted, random_state=random_state
            )
        else:
            df_train = df_train_val
            df_val = pd.DataFrame()

        # 保存拆分
        output_files = {}

        if len(df_train) > 0:
            train_file = os.path.join(output_dir, 'train_custom.csv')
            df_train.to_csv(train_file, index=False)
            output_files['train'] = train_file

        if len(df_val) > 0:
            val_file = os.path.join(output_dir, 'val_custom.csv')
            df_val.to_csv(val_file, index=False)
            output_files['val'] = val_file

        if len(df_test) > 0:
            test_file = os.path.join(output_dir, 'test_custom.csv')
            df_test.to_csv(test_file, index=False)
            output_files['test'] = test_file

        # 保存拆分信息
        split_info = pd.DataFrame([
            {'split': 'train', 'count': len(df_train), 'ratio': len(df_train)/len(df)},
            {'split': 'val', 'count': len(df_val), 'ratio': len(df_val)/len(df)},
            {'split': 'test', 'count': len(df_test), 'ratio': len(df_test)/len(df)}
        ])

        info_file = os.path.join(output_dir, 'split_info.csv')
        split_info.to_csv(info_file, index=False)
        output_files['info'] = info_file

        return output_files


def main():
    """命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="数据拆分管理工具")
    parser.add_argument('experiment_dir', help="实验输出目录路径")
    parser.add_argument('--list', action='store_true', help="列出所有可用的数据拆分")
    parser.add_argument('--info', action='store_true', help="显示数据拆分详细信息")
    parser.add_argument('--validate', action='store_true', help="验证数据拆分完整性")
    parser.add_argument('--compare', help="比较指定拆分的不同版本 (train/val/test)")
    parser.add_argument('--export', nargs=3, metavar=('TYPE', 'SPLIT', 'OUTPUT'),
                       help="导出数据拆分 (type split_name output_file)")
    parser.add_argument('--create-split', nargs=2, metavar=('INPUT', 'OUTPUT'),
                       help="从原始数据创建自定义拆分")
    parser.add_argument('--train-ratio', type=float, default=0.7, help="训练集比例")
    parser.add_argument('--val-ratio', type=float, default=0.15, help="验证集比例")
    parser.add_argument('--test-ratio', type=float, default=0.15, help="测试集比例")

    args = parser.parse_args()

    if not os.path.exists(args.experiment_dir):
        print(f"错误: 实验目录不存在: {args.experiment_dir}")
        return

    manager = DataSplitManager(args.experiment_dir)

    if args.list:
        splits = manager.list_available_splits()
        print("可用的数据拆分:")
        for split_type, files in splits.items():
            print(f"\n{split_type}:")
            for file in files:
                print(f"  - {file}")

    elif args.info:
        info_df = manager.get_split_info()
        print("数据拆分详细信息:")
        print(info_df.to_string(index=False))

    elif args.validate:
        results = manager.validate_splits()
        print("数据拆分验证结果:")
        for check, result in results.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}: {result}")

    elif args.compare:
        comparison_df = manager.compare_splits(args.compare)
        print(f"{args.compare} 拆分比较:")
        print(comparison_df.to_string(index=False))

    elif args.export:
        split_type, split_name, output_file = args.export
        success = manager.export_split_for_reuse(split_type, split_name, output_file)
        if success:
            print(f"✓ 成功导出到: {output_file}")
        else:
            print(f"✗ 导出失败")

    elif args.create_split:
        input_file, output_dir = args.create_split
        try:
            output_files = manager.create_custom_split(
                input_file, args.train_ratio, args.val_ratio, args.test_ratio, output_dir
            )
            print("✓ 自定义拆分创建成功:")
            for split_name, file_path in output_files.items():
                print(f"  {split_name}: {file_path}")
        except Exception as e:
            print(f"✗ 创建失败: {e}")

    else:
        print("请指定操作选项。使用 --help 查看帮助。")


if __name__ == "__main__":
    main()
