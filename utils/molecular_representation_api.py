#!/usr/bin/env python3
"""
分子表征 API

提供统一的接口来计算分子的指纹和 Embedding。
支持所有预训练模型和分子指纹类型。

使用示例:
    from utils.molecular_representation_api import MolecularRepresentationAPI

    api = MolecularRepresentationAPI()

    # 计算 Morgan 指纹
    fp = api.get_fingerprint("CCO", fingerprint_type="morgan", radius=2, nBits=2048)

    # 计算 MACCS 指纹
    maccs = api.get_fingerprint("CCO", fingerprint_type="maccs")

    # 计算 UniMol Embedding
    embedding = api.get_embedding("CCO", model_type="unimol")

    # 计算 ChemBERTa Embedding
    chemberta = api.get_embedding("CCO", model_type="chemberta")

    # 批量计算
    smiles_list = ["CCO", "c1ccccc1", "CC(C)C"]
    fps = api.get_fingerprints(smiles_list, fingerprint_type="morgan")
    embeddings = api.get_embeddings(smiles_list, model_type="unimol")
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.mol_fp_features import calculate_molecular_features
from utils.transformer_embeddings import (
    get_chemberta_embedding,
    get_molt5_embedding,
    get_chemroberta_embedding,
    get_roberta_embedding
)
from utils.unimol_embedding import get_unimol_embedding
from utils.transformer_embeddings import get_matbert_embedding
from utils.smiles_validator import standardize_smiles as _standardize_smiles


def _standardize_smiles_list(smiles_list: List[str], enabled: bool = True) -> List[str]:
    """
    对SMILES列表进行标准化（内部辅助函数）。
    
    Args:
        smiles_list: SMILES字符串列表
        enabled: 是否启用标准化（默认True）
        
    Returns:
        标准化后的SMILES列表（如果启用），否则返回原列表
    """
    if not enabled:
        return smiles_list
    return [_standardize_smiles(smi) for smi in smiles_list]


class MolecularRepresentationAPI:
    """
    分子表征 API

    提供统一的接口来计算分子的指纹和 Embedding。
    """

    # 支持的指纹类型
    FINGERPRINT_TYPES = {
        'morgan': '分子 Morgan 指纹',
        'maccs': 'MACCS 键指纹',
        'rdkit': 'RDKit 拓扑指纹',
        'atompair': '原子对指纹',
        'torsion': '扭转指纹'
    }

    # 支持的预训练模型
    EMBEDDING_MODELS = {
        'chemberta': 'ChemBERTa (768维)',
        'molt5': 'MolT5 (768维)',
        'chemroberta': 'ChemRoBERTa (768维)',
        'roberta': 'RoBERTa (768维)',
        'unimol': 'UniMol (512维)',
        'unimolv2_84m': 'UniMolv2 84M (768维)',
        'unimolv2_164m': 'UniMolv2 164M (768维)',
        'unimolv2_310m': 'UniMolv2 310M (1024维)',
        'unimolv2_570m': 'UniMolv2 570M (1536维)',
        'unimolv2_1b': 'UniMolv2 1.1B (1536维)',
        'matbert': 'MatBERT (768维) - 材料公式'
    }

    def __init__(self):
        """初始化 API"""
        self.fingerprint_types = list(self.FINGERPRINT_TYPES.keys())
        self.embedding_models = list(self.EMBEDDING_MODELS.keys())

    # ==================== 指纹计算 ====================

    def get_fingerprint(
        self,
        smiles: str,
        fingerprint_type: str = 'morgan',
        radius: int = 2,
        nBits: int = 2048,
        descriptors: bool = False,
        standardize_smiles: bool = True
    ) -> Optional[np.ndarray]:
        """
        计算单个分子的指纹

        Args:
            smiles: SMILES 字符串
            fingerprint_type: 指纹类型 ('morgan', 'maccs', 'rdkit', 'atompair', 'torsion')
            radius: Morgan 指纹的半径（仅对 morgan 有效）
            nBits: 指纹位数（maccs 固定 167）
            descriptors: 是否同时计算分子描述符
            standardize_smiles: 是否对SMILES进行标准化（默认True，使用RDKit canonical SMILES）

        Returns:
            np.ndarray: 指纹向量，如果 SMILES 无效则返回 None

        示例:
            >>> api = MolecularRepresentationAPI()
            >>> fp = api.get_fingerprint("CCO", fingerprint_type="morgan")
            >>> print(fp.shape)  # (2048,)
        """
        if fingerprint_type not in self.FINGERPRINT_TYPES:
            raise ValueError(f"不支持的指纹类型: {fingerprint_type}。支持的类型: {self.fingerprint_types}")

        # SMILES标准化
        processed_smiles = _standardize_smiles(smiles) if standardize_smiles else smiles

        df = calculate_molecular_features(
            processed_smiles,
            fp_type=fingerprint_type,
            descriptors=descriptors,
            radius=radius,
            nBits=nBits
        )

        if df is None or df.empty:
            return None

        return df.iloc[0].values

    def get_fingerprints(
        self,
        smiles_list: List[str],
        fingerprint_type: str = 'morgan',
        radius: int = 2,
        nBits: int = 2048,
        descriptors: bool = False,
        batch_size: int = 100,
        standardize_smiles: bool = True
    ) -> Optional[np.ndarray]:
        """
        批量计算分子指纹

        Args:
            smiles_list: SMILES 字符串列表
            fingerprint_type: 指纹类型
            radius: Morgan 指纹的半径
            nBits: 指纹位数
            descriptors: 是否同时计算分子描述符
            batch_size: 批处理大小
            standardize_smiles: 是否对SMILES进行标准化（默认True，使用RDKit canonical SMILES）

        Returns:
            np.ndarray: 指纹矩阵 (n_molecules, n_features)

        示例:
            >>> api = MolecularRepresentationAPI()
            >>> smiles = ["CCO", "c1ccccc1", "CC(C)C"]
            >>> fps = api.get_fingerprints(smiles, fingerprint_type="morgan")
            >>> print(fps.shape)  # (3, 2048)
        """
        if fingerprint_type not in self.FINGERPRINT_TYPES:
            raise ValueError(f"不支持的指纹类型: {fingerprint_type}。支持的类型: {self.fingerprint_types}")

        # SMILES标准化
        processed_smiles_list = _standardize_smiles_list(smiles_list, enabled=standardize_smiles)

        all_fps = []

        for i in range(0, len(processed_smiles_list), batch_size):
            batch = processed_smiles_list[i:i+batch_size]

            for smiles in batch:
                fp = self.get_fingerprint(
                    smiles,
                    fingerprint_type=fingerprint_type,
                    radius=radius,
                    nBits=nBits,
                    descriptors=descriptors
                )
                if fp is not None:
                    all_fps.append(fp)

        if not all_fps:
            return None

        return np.array(all_fps)

    def get_descriptors(self, smiles: str, standardize_smiles: bool = True) -> Optional[np.ndarray]:
        """
        计算单个分子的 RDKit 描述符

        Args:
            smiles: SMILES 字符串
            standardize_smiles: 是否对SMILES进行标准化（默认True，使用RDKit canonical SMILES）

        Returns:
            np.ndarray: 描述符向量

        示例:
            >>> api = MolecularRepresentationAPI()
            >>> desc = api.get_descriptors("CCO")
            >>> print(desc.shape)  # (~200,)
        """
        # SMILES标准化
        processed_smiles = _standardize_smiles(smiles) if standardize_smiles else smiles

        df = calculate_molecular_features(
            processed_smiles,
            fp_type=None,
            descriptors=True
        )

        if df is None or df.empty:
            return None

        return df.iloc[0].values

    def get_all_descriptors(self, smiles_list: List[str], standardize_smiles: bool = True) -> Optional[np.ndarray]:
        """
        批量计算分子描述符

        Args:
            smiles_list: SMILES 字符串列表
            standardize_smiles: 是否对SMILES进行标准化（默认True，使用RDKit canonical SMILES）

        Returns:
            np.ndarray: 描述符矩阵
        """
        # SMILES标准化
        processed_smiles_list = _standardize_smiles_list(smiles_list, enabled=standardize_smiles)

        all_descs = []

        for smiles in processed_smiles_list:
            desc = self.get_descriptors(smiles, standardize_smiles=False)  # 已经标准化过了
            if desc is not None:
                all_descs.append(desc)

        if not all_descs:
            return None

        return np.array(all_descs)

    # ==================== Embedding 计算 ====================

    def get_embedding(
        self,
        smiles: str,
        model_type: str = 'unimol',
        batch_size: int = 32,
        standardize_smiles: bool = True
    ) -> Optional[np.ndarray]:
        """
        计算单个分子的 Embedding

        Args:
            smiles: SMILES 字符串
            model_type: 模型类型 ('chemberta', 'molt5', 'chemroberta', 'roberta', 'unimol', 等)
            batch_size: 批处理大小
            standardize_smiles: 是否对SMILES进行标准化（默认True，使用RDKit canonical SMILES）

        Returns:
            np.ndarray: Embedding 向量

        示例:
            >>> api = MolecularRepresentationAPI()
            >>> emb = api.get_embedding("CCO", model_type="unimol")
            >>> print(emb.shape)  # (512,)
        """
        if model_type not in self.EMBEDDING_MODELS:
            raise ValueError(f"不支持的模型: {model_type}。支持的模型: {self.embedding_models}")

        embeddings = self.get_embeddings([smiles], model_type=model_type, batch_size=batch_size, standardize_smiles=standardize_smiles)

        if embeddings is None or len(embeddings) == 0:
            return None

        return embeddings[0]

    def get_embeddings(
        self,
        smiles_list: List[str],
        model_type: str = 'unimol',
        batch_size: int = 32,
        standardize_smiles: bool = True
    ) -> Optional[np.ndarray]:
        """
        批量计算分子 Embedding

        Args:
            smiles_list: SMILES 字符串列表
            model_type: 模型类型
            batch_size: 批处理大小
            standardize_smiles: 是否对SMILES进行标准化（默认True，使用RDKit canonical SMILES）

        Returns:
            np.ndarray: Embedding 矩阵 (n_molecules, embedding_dim)

        示例:
            >>> api = MolecularRepresentationAPI()
            >>> smiles = ["CCO", "c1ccccc1", "CC(C)C"]
            >>> embs = api.get_embeddings(smiles, model_type="unimol")
            >>> print(embs.shape)  # (3, 512)
        """
        if model_type not in self.EMBEDDING_MODELS:
            raise ValueError(f"不支持的模型: {model_type}。支持的模型: {self.embedding_models}")

        # SMILES标准化
        processed_smiles_list = _standardize_smiles_list(smiles_list, enabled=standardize_smiles)

        # 调用相应的 embedding 函数
        if model_type == 'chemberta':
            return get_chemberta_embedding(processed_smiles_list, batch_size=batch_size)
        elif model_type == 'molt5':
            return get_molt5_embedding(processed_smiles_list, batch_size=batch_size)
        elif model_type == 'chemroberta':
            return get_chemroberta_embedding(processed_smiles_list, batch_size=batch_size)
        elif model_type == 'roberta':
            return get_roberta_embedding(processed_smiles_list, batch_size=batch_size)
        elif model_type == 'unimol':
            return get_unimol_embedding(processed_smiles_list, model_version='v1', standardize_smiles=False)  # 已标准化
        elif model_type.startswith('unimolv2'):
            # 解析 UniMolv2 版本
            size_map = {
                'unimolv2_84m': '84m',
                'unimolv2_164m': '164m',
                'unimolv2_310m': '310m',
                'unimolv2_570m': '570m',
                'unimolv2_1b': '1.1B'
            }
            model_size = size_map.get(model_type, '310m')
            return get_unimol_embedding(processed_smiles_list, model_version='v2', model_size=model_size, standardize_smiles=False)  # 已标准化
        elif model_type == 'matbert':
            return get_matbert_embedding(processed_smiles_list, batch_size=batch_size)
        else:
            raise ValueError(f"不支持的模型: {model_type}")

    # ==================== 工具方法 ====================

    def list_fingerprint_types(self) -> Dict[str, str]:
        """列出所有支持的指纹类型"""
        return self.FINGERPRINT_TYPES

    def list_embedding_models(self) -> Dict[str, str]:
        """列出所有支持的 Embedding 模型"""
        return self.EMBEDDING_MODELS

    def get_fingerprint_info(self, fingerprint_type: str) -> Dict[str, Any]:
        """获取指纹类型的详细信息"""
        if fingerprint_type not in self.FINGERPRINT_TYPES:
            raise ValueError(f"不支持的指纹类型: {fingerprint_type}")

        info = {
            'name': self.FINGERPRINT_TYPES[fingerprint_type],
            'type': fingerprint_type
        }

        if fingerprint_type == 'maccs':
            info['fixed_bits'] = 167
        else:
            info['configurable_bits'] = True
            info['default_bits'] = 2048

        if fingerprint_type == 'morgan':
            info['configurable_radius'] = True
            info['default_radius'] = 2

        return info

    def get_embedding_info(self, model_type: str) -> Dict[str, Any]:
        """获取 Embedding 模型的详细信息"""
        if model_type not in self.EMBEDDING_MODELS:
            raise ValueError(f"不支持的模型: {model_type}")

        # 提取维度信息
        name = self.EMBEDDING_MODELS[model_type]

        # 从名称中提取维度
        if '768维' in name:
            dim = 768
        elif '512维' in name:
            dim = 512
        elif '1024维' in name:
            dim = 1024
        elif '1536维' in name:
            dim = 1536
        else:
            dim = None

        return {
            'name': name,
            'model_type': model_type,
            'embedding_dim': dim
        }


# ==================== 便捷函数 ====================

def get_fingerprint(
    smiles: str,
    fingerprint_type: str = 'morgan',
    standardize_smiles: bool = True,
    **kwargs
) -> Optional[np.ndarray]:
    """
    快速计算单个分子的指纹

    Args:
        smiles: SMILES 字符串
        fingerprint_type: 指纹类型
        standardize_smiles: 是否对SMILES进行标准化（默认True）
        **kwargs: 其他参数（radius, nBits 等）

    Returns:
        np.ndarray: 指纹向量
    """
    api = MolecularRepresentationAPI()
    return api.get_fingerprint(smiles, fingerprint_type=fingerprint_type, standardize_smiles=standardize_smiles, **kwargs)


def get_fingerprints(
    smiles_list: List[str],
    fingerprint_type: str = 'morgan',
    standardize_smiles: bool = True,
    **kwargs
) -> Optional[np.ndarray]:
    """
    快速批量计算分子指纹

    Args:
        smiles_list: SMILES 字符串列表
        fingerprint_type: 指纹类型
        standardize_smiles: 是否对SMILES进行标准化（默认True）
        **kwargs: 其他参数

    Returns:
        np.ndarray: 指纹矩阵
    """
    api = MolecularRepresentationAPI()
    return api.get_fingerprints(smiles_list, fingerprint_type=fingerprint_type, standardize_smiles=standardize_smiles, **kwargs)


def get_embedding(
    smiles: str,
    model_type: str = 'unimol',
    standardize_smiles: bool = True,
    **kwargs
) -> Optional[np.ndarray]:
    """
    快速计算单个分子的 Embedding

    Args:
        smiles: SMILES 字符串
        model_type: 模型类型
        standardize_smiles: 是否对SMILES进行标准化（默认True）
        **kwargs: 其他参数

    Returns:
        np.ndarray: Embedding 向量
    """
    api = MolecularRepresentationAPI()
    return api.get_embedding(smiles, model_type=model_type, standardize_smiles=standardize_smiles, **kwargs)


def get_embeddings(
    smiles_list: List[str],
    model_type: str = 'unimol',
    standardize_smiles: bool = True,
    **kwargs
) -> Optional[np.ndarray]:
    """
    快速批量计算分子 Embedding

    Args:
        smiles_list: SMILES 字符串列表
        model_type: 模型类型
        standardize_smiles: 是否对SMILES进行标准化（默认True）
        **kwargs: 其他参数

    Returns:
        np.ndarray: Embedding 矩阵
    """
    api = MolecularRepresentationAPI()
    return api.get_embeddings(smiles_list, model_type=model_type, standardize_smiles=standardize_smiles, **kwargs)


if __name__ == '__main__':
    # 使用示例
    api = MolecularRepresentationAPI()

    print("=" * 60)
    print("分子表征 API 使用示例")
    print("=" * 60)

    # 示例 SMILES
    smiles_list = ["CCO", "c1ccccc1", "CC(C)C"]

    # 1. 计算 Morgan 指纹
    print("\n1. 计算 Morgan 指纹")
    fps = api.get_fingerprints(smiles_list, fingerprint_type="morgan")
    print(f"   形状: {fps.shape}")

    # 2. 计算 MACCS 指纹
    print("\n2. 计算 MACCS 指纹")
    maccs = api.get_fingerprints(smiles_list, fingerprint_type="maccs")
    print(f"   形状: {maccs.shape}")

    # 3. 计算 UniMol Embedding
    print("\n3. 计算 UniMol Embedding")
    embs = api.get_embeddings(smiles_list, model_type="unimol")
    print(f"   形状: {embs.shape}")

    # 4. 列出支持的类型
    print("\n4. 支持的指纹类型:")
    for fp_type, desc in api.list_fingerprint_types().items():
        print(f"   - {fp_type}: {desc}")

    print("\n5. 支持的 Embedding 模型:")
    for model, desc in api.list_embedding_models().items():
        print(f"   - {model}: {desc}")
