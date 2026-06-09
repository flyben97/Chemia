# utils/simple_features.py
"""
简化的分子特征计算接口
支持通过简单命令为SMILES计算各种特征
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Union, Optional, Dict, Any
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.feature_generator import calculate_features_from_smiles


class SimpleFeatureCalculator:
    """
    简化的分子特征计算器
    支持单个SMILES和批量计算
    """

    # 支持的特征类型映射
    FEATURE_TYPES = {
        # 指纹特征
        'maccs': {
            'type': 'maccs',
            'description': 'MACCS Keys (166 bits)',
            'category': 'fingerprint'
        },
        'morgan': {
            'type': 'morgan',
            'description': 'Morgan Fingerprints (default: 2048 bits, radius=2)',
            'category': 'fingerprint',
            'default_params': {'nBits': 2048, 'radius': 2}
        },
        'rdkit': {
            'type': 'rdkit',
            'description': 'RDKit Fingerprints (default: 2048 bits)',
            'category': 'fingerprint',
            'default_params': {'nBits': 2048}
        },
        'atompair': {
            'type': 'atompair',
            'description': 'Atom Pair Fingerprints (default: 2048 bits)',
            'category': 'fingerprint',
            'default_params': {'nBits': 2048}
        },
        'torsion': {
            'type': 'torsion',
            'description': 'Topological Torsion Fingerprints (default: 2048 bits)',
            'category': 'fingerprint',
            'default_params': {'nBits': 2048}
        },

        # 描述符
        'rdkit_descriptors': {
            'type': 'rdkit_descriptors',
            'description': 'RDKit Molecular Descriptors (~200+ features)',
            'category': 'descriptor'
        },

        # 嵌入特征
        'chemberta': {
            'type': 'chemberta',
            'description': 'ChemBERTa Embeddings (768 dimensions)',
            'category': 'embedding'
        },
        'molt5': {
            'type': 'molt5',
            'description': 'MolT5 Embeddings (768 dimensions)',
            'category': 'embedding'
        },
        'chemroberta': {
            'type': 'chemroberta',
            'description': 'ChemRoBERTa Embeddings (768 dimensions)',
            'category': 'embedding'
        },
        'unimol': {
            'type': 'unimol',
            'description': 'Uni-Mol V1 Embeddings (512 dimensions)',
            'category': 'embedding',
            'default_params': {'model_version': 'v1'}
        },
        'unimolv2': {
            'type': 'unimol',
            'description': 'Uni-Mol V2 Embeddings (512 dimensions, default: 84m)',
            'category': 'embedding',
            'default_params': {'model_version': 'v2', 'model_size': '84m'}
        },
        'unimolv2_164m': {
            'type': 'unimol',
            'description': 'Uni-Mol V2 164M Embeddings (512 dimensions)',
            'category': 'embedding',
            'default_params': {'model_version': 'v2', 'model_size': '164m'}
        },
        'unimolv2_310m': {
            'type': 'unimol',
            'description': 'Uni-Mol V2 310M Embeddings (512 dimensions)',
            'category': 'embedding',
            'default_params': {'model_version': 'v2', 'model_size': '310m'}
        },
        'unimolv2_570m': {
            'type': 'unimol',
            'description': 'Uni-Mol V2 570M Embeddings (512 dimensions)',
            'category': 'embedding',
            'default_params': {'model_version': 'v2', 'model_size': '570m'}
        },
        'unimolv2_1.1b': {
            'type': 'unimol',
            'description': 'Uni-Mol V2 1.1B Embeddings (512 dimensions)',
            'category': 'embedding',
            'default_params': {'model_version': 'v2', 'model_size': '1.1B'}
        }
    }

    def __init__(self, output_dir: str = "./feature_output", parallel: bool = True):
        """
        初始化特征计算器

        Args:
            output_dir: 输出目录
            parallel: 是否使用并行计算（仅对指纹特征有效）
        """
        self.output_dir = output_dir
        self.parallel = parallel
        os.makedirs(output_dir, exist_ok=True)

    def list_available_features(self) -> None:
        """列出所有可用的特征类型"""
        print("可用的特征类型:")
        print("=" * 60)

        categories = {}
        for name, info in self.FEATURE_TYPES.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((name, info['description']))

        for category, features in categories.items():
            print(f"\n{category.upper()} 特征:")
            print("-" * 30)
            for name, desc in features:
                print(f"  {name:<15} - {desc}")

        print(f"\n使用示例:")
        print(f"  calculator.calculate('CC(C)C', 'morgan')")
        print(f"  calculator.calculate(['CC(C)C', 'CCO'], 'maccs')")

    def calculate(self,
                  smiles: Union[str, List[str]],
                  feature_type: str,
                  output_file: Optional[str] = None,
                  **kwargs) -> Optional[pd.DataFrame]:
        """
        计算分子特征

        Args:
            smiles: 单个SMILES字符串或SMILES列表
            feature_type: 特征类型 (maccs, morgan, unimolv2, molt5等)
            output_file: 输出CSV文件路径 (可选)
            **kwargs: 特征计算的额外参数

        Returns:
            包含特征的DataFrame，失败时返回None
        """
        # 验证特征类型
        if feature_type not in self.FEATURE_TYPES:
            print(f"错误: 不支持的特征类型 '{feature_type}'")
            print("可用类型:", list(self.FEATURE_TYPES.keys()))
            return None

        # 处理输入
        if isinstance(smiles, str):
            smiles_list = [smiles]
            is_single = True
        else:
            smiles_list = list(smiles)
            is_single = False

        if not smiles_list:
            print("错误: 没有提供SMILES")
            return None

        # 获取特征配置
        feature_config = self.FEATURE_TYPES[feature_type]
        actual_type = feature_config['type']

        # 合并默认参数和用户参数
        params = feature_config.get('default_params', {}).copy()
        params.update(kwargs)

        print(f"正在计算 {feature_config['description']} 特征...")
        print(f"输入: {len(smiles_list)} 个SMILES")

        # 计算特征
        try:
            features_df = calculate_features_from_smiles(
                smiles_list=smiles_list,
                feature_type=actual_type,
                output_dir_for_logs=self.output_dir,
                parallel=self.parallel,
                **params
            )

            if features_df is None or features_df.empty:
                print("特征计算失败")
                return None

            print(f"✓ 特征计算完成: {features_df.shape}")

            # 保存到文件
            if output_file:
                features_df.to_csv(output_file, index=True)
                print(f"✓ 结果已保存到: {output_file}")

            return features_df

        except Exception as e:
            print(f"特征计算出错: {e}")
            return None

    def calculate_multiple_types(self,
                                smiles: Union[str, List[str]],
                                feature_types: List[str],
                                output_file: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        计算多种类型的特征并合并

        Args:
            smiles: SMILES字符串或列表
            feature_types: 特征类型列表
            output_file: 输出文件路径

        Returns:
            合并后的特征DataFrame
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            smiles_list = list(smiles)

        all_features = []

        for feature_type in feature_types:
            print(f"\n计算 {feature_type} 特征...")
            features_df = self.calculate(smiles_list, feature_type)

            if features_df is not None:
                all_features.append(features_df)
            else:
                print(f"跳过 {feature_type} 特征（计算失败）")

        if not all_features:
            print("所有特征计算都失败了")
            return None

        # 合并所有特征
        print("\n合并特征...")
        combined_df = pd.concat(all_features, axis=1)

        print(f"✓ 合并完成: {combined_df.shape}")

        if output_file:
            combined_df.to_csv(output_file, index=True)
            print(f"✓ 合并结果已保存到: {output_file}")

        return combined_df

    def batch_calculate_from_file(self,
                                 input_file: str,
                                 smiles_column: str,
                                 feature_type: str,
                                 output_file: Optional[str] = None,
                                 **kwargs) -> Optional[pd.DataFrame]:
        """
        从CSV文件批量计算特征

        Args:
            input_file: 输入CSV文件路径
            smiles_column: SMILES列名
            feature_type: 特征类型
            output_file: 输出文件路径
            **kwargs: 特征计算参数

        Returns:
            包含原始数据和特征的DataFrame
        """
        try:
            # 读取输入文件
            df = pd.read_csv(input_file)
            print(f"读取输入文件: {input_file}")
            print(f"数据形状: {df.shape}")

            if smiles_column not in df.columns:
                print(f"错误: 找不到SMILES列 '{smiles_column}'")
                print(f"可用列: {list(df.columns)}")
                return None

            # 提取SMILES
            smiles_list = df[smiles_column].dropna().tolist()
            print(f"有效SMILES数量: {len(smiles_list)}")

            # 计算特征
            features_df = self.calculate(smiles_list, feature_type, **kwargs)

            if features_df is None:
                return None

            # 合并原始数据和特征
            # 重新索引以匹配原始数据
            features_df.index = range(len(features_df))
            df_with_features = pd.concat([df.head(len(features_df)), features_df], axis=1)

            if output_file:
                df_with_features.to_csv(output_file, index=False)
                print(f"✓ 结果已保存到: {output_file}")

            return df_with_features

        except Exception as e:
            print(f"批量计算出错: {e}")
            return None


def main():
    """命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="简化的分子特征计算器")
    parser.add_argument('--smiles', '-s', help="单个SMILES字符串")
    parser.add_argument('--smiles-list', nargs='+', help="多个SMILES字符串")
    parser.add_argument('--input-file', '-i', help="输入CSV文件路径")
    parser.add_argument('--smiles-column', default='smiles', help="SMILES列名 (默认: smiles)")
    parser.add_argument('--feature-type', '-f', required=True,
                       choices=list(SimpleFeatureCalculator.FEATURE_TYPES.keys()),
                       help="特征类型")
    parser.add_argument('--output', '-o', help="输出CSV文件路径")
    parser.add_argument('--output-dir', default='./feature_output', help="输出目录")
    parser.add_argument('--no-parallel', action='store_true', help="禁用并行计算")
    parser.add_argument('--list-features', action='store_true', help="列出所有可用特征类型")

    # 特征参数
    parser.add_argument('--radius', type=int, default=2, help="Morgan指纹半径")
    parser.add_argument('--nbits', type=int, default=2048, help="指纹位数")

    # UniMol参数
    parser.add_argument('--model-version', choices=['v1', 'v2'], help="UniMol模型版本")
    parser.add_argument('--model-size', choices=['84m', '164m', '310m', '570m', '1.1B'],
                       help="UniMol V2模型大小")
    parser.add_argument('--remove-hs', action='store_true', help="UniMol: 移除氢原子")

    args = parser.parse_args()

    # 创建计算器
    calculator = SimpleFeatureCalculator(
        output_dir=args.output_dir,
        parallel=not args.no_parallel
    )

    # 列出特征类型
    if args.list_features:
        calculator.list_available_features()
        return

    # 准备特征参数
    feature_params = {}
    if args.feature_type in ['morgan']:
        feature_params['radius'] = args.radius
    if args.feature_type in ['morgan', 'rdkit', 'atompair', 'torsion']:
        feature_params['nBits'] = args.nbits

    # UniMol参数
    if 'unimol' in args.feature_type.lower():
        if args.model_version:
            feature_params['model_version'] = args.model_version
        if args.model_size:
            feature_params['model_size'] = args.model_size
        if args.remove_hs:
            feature_params['remove_hs'] = True

    # 执行计算
    if args.input_file:
        # 从文件批量计算
        result = calculator.batch_calculate_from_file(
            input_file=args.input_file,
            smiles_column=args.smiles_column,
            feature_type=args.feature_type,
            output_file=args.output,
            **feature_params
        )
    elif args.smiles:
        # 单个SMILES
        result = calculator.calculate(
            smiles=args.smiles,
            feature_type=args.feature_type,
            output_file=args.output,
            **feature_params
        )
        if result is not None:
            print(f"\n特征预览:")
            print(result.head())
    elif args.smiles_list:
        # 多个SMILES
        result = calculator.calculate(
            smiles=args.smiles_list,
            feature_type=args.feature_type,
            output_file=args.output,
            **feature_params
        )
        if result is not None:
            print(f"\n特征预览:")
            print(result.head())
    else:
        print("错误: 请提供SMILES字符串、SMILES列表或输入文件")
        parser.print_help()


if __name__ == "__main__":
    main()
