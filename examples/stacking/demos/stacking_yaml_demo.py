#!/usr/bin/env python3
"""
CRAFT 模型堆叠 YAML 配置演示

这个脚本演示如何使用YAML配置文件来进行模型堆叠。
支持从配置文件自动加载模型、设置堆叠方法、训练元模型等功能。

使用方法:
    python stacking_yaml_demo.py --config config_stacking.yaml

需要先有训练好的模型在指定的实验目录中。
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_stacking import ModelStacker, load_stacking_config_from_yaml

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件未找到: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ YAML文件解析错误: {e}")
        sys.exit(1)

def save_results(results: Dict[str, Any], config: Dict[str, Any]):
    """保存结果到文件"""
    save_config = config.get('save', {})
    results_dir = save_config.get('results_dir', 'output/stacking_results')
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存评估结果
    if save_config.get('save_evaluation', True) and 'evaluation' in results:
        eval_file = os.path.join(results_dir, 'evaluation_results.yaml')
        with open(eval_file, 'w', encoding='utf-8') as f:
            yaml.dump(results['evaluation'], f, default_flow_style=False, allow_unicode=True)
        print(f"✓ 评估结果已保存到: {eval_file}")
    
    # 保存预测结果
    if save_config.get('save_predictions', True) and 'predictions' in results:
        pred_file = os.path.join(results_dir, 'predictions.csv')
        predictions_df = pd.DataFrame(results['predictions'])
        predictions_df.to_csv(pred_file, index=False)
        print(f"✓ 预测结果已保存到: {pred_file}")
    
    # 保存配置文件副本
    if save_config.get('save_config_copy', True):
        config_copy = os.path.join(results_dir, 'stacking_config.yaml')
        with open(config_copy, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"✓ 配置文件副本已保存到: {config_copy}")

def run_stacking_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    使用YAML配置文件运行模型堆叠
    
    Args:
        config_path: YAML配置文件路径
    
    Returns:
        dict: 运行结果
    """
    print("🚀 开始YAML配置模型堆叠")
    print("=" * 60)
    
    # 加载配置文件
    print(f"📋 加载配置文件: {config_path}")
    config = load_yaml_config(config_path)
    
    # 从YAML配置创建堆叠器
    print("\n🔧 创建模型堆叠器...")
    try:
        stacker = load_stacking_config_from_yaml(config_path)
    except Exception as e:
        print(f"❌ 创建堆叠器失败: {e}")
        return {'success': False, 'error': str(e)}
    
    results = {'success': True, 'config_path': config_path}
    
    # 自动评估（如果配置）
    eval_config = config.get('evaluation', {})
    if eval_config.get('auto_evaluate', True):
        print("\n📊 开始自动评估...")
        try:
            use_test_set = eval_config.get('use_test_set', True)
            evaluation = stacker.evaluate(auto_load=True, use_test_set=use_test_set)
            results['evaluation'] = evaluation
            
            print("📈 评估结果:")
            print("-" * 40)
            
            # 显示主要指标
            if stacker.task_type == 'regression':
                if 'r2' in evaluation:
                    print(f"  R² Score: {evaluation['r2']:.4f}")
                if 'rmse' in evaluation:
                    print(f"  RMSE: {evaluation['rmse']:.4f}")
                if 'mae' in evaluation:
                    print(f"  MAE: {evaluation['mae']:.4f}")
            else:
                if 'accuracy' in evaluation:
                    print(f"  Accuracy: {evaluation['accuracy']:.4f}")
                if 'log_loss' in evaluation:
                    print(f"  Log Loss: {evaluation['log_loss']:.4f}")
            
            # 显示基础模型性能比较
            if eval_config.get('compare_with_base', True) and 'base_model_performance' in evaluation:
                print("\n🔍 基础模型性能比较:")
                print("-" * 40)
                base_perf = evaluation['base_model_performance']
                for model_name, metrics in base_perf.items():
                    if stacker.task_type == 'regression':
                        r2 = metrics.get('r2', 'N/A')
                        rmse = metrics.get('rmse', 'N/A')
                        print(f"  {model_name}: R²={r2:.4f}, RMSE={rmse:.4f}")
                    else:
                        acc = metrics.get('accuracy', 'N/A')
                        print(f"  {model_name}: Accuracy={acc:.4f}")
            
        except Exception as e:
            print(f"⚠️  评估失败: {e}")
            results['evaluation_error'] = str(e)
    
    # 保存堆叠器（如果配置）
    save_config = config.get('save', {})
    if save_config.get('save_stacker', True):
        save_path = save_config.get('save_path', 'output/stacked_models/ensemble_model.pkl')
        print(f"\n💾 保存堆叠器到: {save_path}")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            stacker.save(save_path)
            results['saved_model_path'] = save_path
        except Exception as e:
            print(f"⚠️  保存失败: {e}")
            results['save_error'] = str(e)
    
    # 保存结果
    try:
        save_results(results, config)
    except Exception as e:
        print(f"⚠️  保存结果失败: {e}")
    
    print("\n✅ 模型堆叠完成!")
    return results

def create_sample_config(output_path: str = "config_stacking_example.yaml"):
    """创建示例配置文件"""
    sample_config = {
        'stacking': {
            'experiment_dir': 'output/my_experiment',
            'method': 'weighted_average',
            'models': [
                {'name': 'xgb', 'weight': 0.4, 'enabled': True},
                {'name': 'lgbm', 'weight': 0.3, 'enabled': True},
                {'name': 'catboost', 'weight': 0.3, 'enabled': True}
            ],
            'meta_model': {
                'auto_train': False,
                'validation': {
                    'auto_load': True,
                    'size': 100
                }
            }
        },
        'evaluation': {
            'auto_evaluate': True,
            'use_test_set': True,
            'compare_with_base': True
        },
        'save': {
            'save_stacker': True,
            'save_path': 'output/stacked_models/example_ensemble.pkl',
            'results_dir': 'output/stacking_results',
            'save_evaluation': True,
            'save_config_copy': True
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ 示例配置文件已创建: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="CRAFT 模型堆叠 YAML 配置演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用
  python stacking_yaml_demo.py --config config_stacking.yaml
  
  # 创建示例配置文件
  python stacking_yaml_demo.py --create-sample-config
  
  # 指定输出路径创建示例配置
  python stacking_yaml_demo.py --create-sample-config --output my_config.yaml

注意事项:
  1. 确保实验目录中有训练好的模型
  2. 配置文件中的模型名称必须与实际模型文件匹配
  3. 使用元学习器时会自动训练，需要验证数据
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='YAML配置文件路径'
    )
    
    parser.add_argument(
        '--create-sample-config',
        action='store_true',
        help='创建示例配置文件'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='config_stacking_example.yaml',
        help='示例配置文件输出路径 (默认: config_stacking_example.yaml)'
    )
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_sample_config(args.output)
        return
    
    if not args.config:
        print("❌ 请指定配置文件路径或使用 --create-sample-config 创建示例配置")
        parser.print_help()
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 运行模型堆叠
    results = run_stacking_from_yaml(args.config)
    
    if results['success']:
        print(f"\n🎉 模型堆叠成功完成!")
        if 'evaluation' in results:
            eval_result = results['evaluation']
            print(f"   堆叠方法: {eval_result.get('stacking_method', 'unknown')}")
            print(f"   模型数量: {eval_result.get('n_models', 'unknown')}")
    else:
        print(f"\n❌ 模型堆叠失败: {results.get('error', 'unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main() 