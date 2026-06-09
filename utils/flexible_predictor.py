# utils/flexible_predictor.py
"""
灵活的模型预测器 - 支持从任意路径加载模型进行预测
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.io_handler import (
    load_model_from_path,
    load_scaler_from_path,
    load_label_encoder_from_path,
    load_config_from_path
)
from utils.predictor_api import Predictor


class FlexiblePredictor:
    """
    灵活的模型预测器，支持从任意路径加载模型和相关文件
    """

    def __init__(self, model_path: str, config_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 label_encoder_path: Optional[str] = None,
                 task_type: str = 'regression'):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径 (.json, .cbm, .joblib, .zip等)
            config_path: 配置文件路径 (可选，如果不提供会使用默认配置)
            scaler_path: 标准化器路径 (可选)
            label_encoder_path: 标签编码器路径 (可选，分类任务需要)
            task_type: 任务类型 ('regression' 或 'classification')
        """
        self.model_path = model_path
        self.config_path = config_path
        self.scaler_path = scaler_path
        self.label_encoder_path = label_encoder_path
        self.task_type = task_type

        # 验证文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型
        print(f"正在加载模型: {model_path}")
        self.model = load_model_from_path(model_path, task_type)

        # 加载配置
        if config_path and os.path.exists(config_path):
            print(f"正在加载配置: {config_path}")
            self.config = load_config_from_path(config_path)
        else:
            print("使用默认配置")
            self.config = self._get_default_config()

        # 加载预处理器
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            print(f"正在加载标准化器: {scaler_path}")
            self.scaler = load_scaler_from_path(scaler_path)

        self.label_encoder = None
        if label_encoder_path and os.path.exists(label_encoder_path):
            print(f"正在加载标签编码器: {label_encoder_path}")
            self.label_encoder = load_label_encoder_from_path(label_encoder_path)

        # 创建预测器实例
        self.predictor = Predictor(
            model=self.model,
            scaler=self.scaler,
            label_encoder=self.label_encoder,
            run_config=self.config,
            output_dir="./predictions"
        )

        print("✓ 模型加载完成，可以开始预测")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'task_type': self.task_type,
            'data': {
                'single_file_config': {
                    'smiles_col': ['substrate', 'ligand', 'base', 'solvent'],
                    'target_col': 'ee'
                }
            },
            'features': {
                'smiles_features': {
                    'enabled': True,
                    'feature_types': ['morgan_fingerprints', 'rdkit_descriptors']
                }
            }
        }

    def predict(self, input_data: Union[str, pd.DataFrame],
                output_path: Optional[str] = None) -> pd.DataFrame:
        """
        进行预测

        Args:
            input_data: 输入数据，可以是CSV文件路径或DataFrame
            output_path: 输出文件路径 (可选)

        Returns:
            包含预测结果的DataFrame
        """
        # 处理输入数据
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"输入文件不存在: {input_data}")
            print(f"正在读取输入文件: {input_data}")
            df = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("input_data必须是文件路径或DataFrame")

        print(f"输入数据形状: {df.shape}")

        # 进行预测
        results = self.predictor.predict_from_df(df)

        if results is None or results.empty:
            print("预测失败或无结果")
            return pd.DataFrame()

        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results.to_csv(output_path, index=False)
            print(f"预测结果已保存到: {output_path}")

        print(f"预测完成，结果形状: {results.shape}")
        return results

    def predict_single(self, **kwargs) -> float:
        """
        预测单个样本

        Args:
            **kwargs: 反应组分，例如 substrate='CC', ligand='CCO', base='NaOH', solvent='DMSO'

        Returns:
            预测值
        """
        # 创建单行DataFrame
        df = pd.DataFrame([kwargs])

        # 进行预测
        results = self.predict(df)

        if results.empty:
            raise ValueError("预测失败")

        return results['prediction'].iloc[0]

    @classmethod
    def from_training_output(cls, training_output_dir: str, model_name: str = None) -> 'FlexiblePredictor':
        """
        从训练输出目录创建预测器

        Args:
            training_output_dir: 训练输出目录
            model_name: 模型名称 (如果不指定，会自动查找)

        Returns:
            FlexiblePredictor实例
        """
        from utils.io_handler import get_full_model_name, find_model_file

        # 查找配置文件
        config_path = os.path.join(training_output_dir, 'run_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 加载配置获取任务类型
        config = load_config_from_path(config_path)
        task_type = config.get('task_type', 'regression')

        # 如果没有指定模型名称，查找最佳模型
        if model_name is None:
            # 查找results.json或类似文件来确定最佳模型
            results_files = ['results.json', 'training_results.json']
            for results_file in results_files:
                results_path = os.path.join(training_output_dir, results_file)
                if os.path.exists(results_path):
                    import json
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    if 'results' in results and results['results']:
                        # 选择第一个模型作为默认
                        model_name = results['results'][0]['model_name']
                        break

            if model_name is None:
                # 如果找不到结果文件，列出可用模型
                models_dir = os.path.join(training_output_dir, 'models')
                if os.path.exists(models_dir):
                    available_models = os.listdir(models_dir)
                    if available_models:
                        model_name = available_models[0].replace('_model', '')
                        print(f"自动选择模型: {model_name}")

        if model_name is None:
            raise ValueError("无法确定模型名称，请手动指定")

        # 构建文件路径
        full_model_name = get_full_model_name(model_name)
        model_dir = os.path.join(training_output_dir, 'models', full_model_name)
        data_splits_dir = os.path.join(training_output_dir, 'data_splits')

        model_path = find_model_file(model_dir, full_model_name)
        scaler_path = os.path.join(data_splits_dir, 'processed_dataset_scaler.joblib')
        encoder_path = os.path.join(data_splits_dir, 'processed_dataset_label_encoder.joblib')

        # 检查文件存在性
        scaler_path = scaler_path if os.path.exists(scaler_path) else None
        encoder_path = encoder_path if os.path.exists(encoder_path) and task_type != 'regression' else None

        return cls(
            model_path=model_path,
            config_path=config_path,
            scaler_path=scaler_path,
            label_encoder_path=encoder_path,
            task_type=task_type
        )


def main():
    """命令行接口示例"""
    import argparse

    parser = argparse.ArgumentParser(description="灵活的模型预测器")
    parser.add_argument('--model', required=True, help="模型文件路径")
    parser.add_argument('--config', help="配置文件路径")
    parser.add_argument('--scaler', help="标准化器文件路径")
    parser.add_argument('--encoder', help="标签编码器文件路径")
    parser.add_argument('--task-type', default='regression', choices=['regression', 'classification'], help="任务类型")
    parser.add_argument('--input', required=True, help="输入CSV文件路径")
    parser.add_argument('--output', help="输出CSV文件路径")

    args = parser.parse_args()

    # 创建预测器
    predictor = FlexiblePredictor(
        model_path=args.model,
        config_path=args.config,
        scaler_path=args.scaler,
        label_encoder_path=args.encoder,
        task_type=args.task_type
    )

    # 进行预测
    results = predictor.predict(args.input, args.output)
    print(f"预测完成，处理了 {len(results)} 个样本")


if __name__ == "__main__":
    main()
