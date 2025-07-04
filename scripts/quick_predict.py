#!/usr/bin/env python3
"""
INTERNCRANE 快速预测脚本

这是一个简化的预测脚本，提供最基本的模型预测功能。
适合快速预测任务，无需复杂的配置。

使用示例：
    python quick_predict.py /path/to/experiment_dir xgb input.csv output.csv
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from run_prediction_standalone import PredictionRunner, console

def quick_predict(experiment_dir: str, model_name: str, input_file: str, output_file: str, verbose: bool = False):
    """
    快速预测函数
    
    Args:
        experiment_dir: 训练实验目录路径
        model_name: 模型名称 (如 xgb, lgbm, catboost)
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        verbose: 是否显示详细信息
    """
    
    console.print(f"[bold blue]🚀 INTERNCRANE 快速预测[/bold blue]")
    console.print(f"[cyan]实验目录:[/cyan] {experiment_dir}")
    console.print(f"[cyan]模型名称:[/cyan] {model_name}")
    console.print(f"[cyan]输入文件:[/cyan] {input_file}")
    console.print(f"[cyan]输出文件:[/cyan] {output_file}")
    console.print("-" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        console.print(f"[bold red]❌ 输入文件不存在:[/bold red] {input_file}")
        return False
    
    # 检查实验目录是否存在
    if not os.path.exists(experiment_dir):
        console.print(f"[bold red]❌ 实验目录不存在:[/bold red] {experiment_dir}")
        return False
    
    # 创建配置
    config = {
        'prediction_mode': 'experiment_directory',
        'experiment_directory_mode': {
            'run_directory': experiment_dir,
            'model_name': model_name
        },
        'data': {
            'input_file': input_file,
            'output_file': output_file
        },
        'logging': {
            'verbose': verbose
        },
        'prediction': {
            'save_probabilities': True,
            'output_format': {
                'include_input_data': True,
                'add_prediction_metadata': True,
                'precision': 4
            }
        },
        'advanced': {
            'memory_efficient': True,
            'skip_invalid_rows': True
        }
    }
    
    # 创建预测运行器并执行
    runner = PredictionRunner(config)
    success = runner.run_prediction_pipeline()
    
    if success:
        console.print(f"\n[bold green]✅ 预测完成![/bold green]")
        console.print(f"[green]结果已保存到:[/green] {output_file}")
        
        # 显示简单的结果统计
        try:
            import pandas as pd
            result_df = pd.read_csv(output_file)
            console.print(f"[cyan]预测样本数:[/cyan] {len(result_df)}")
            
            # 如果是回归任务，显示预测值范围
            if 'prediction' in result_df.columns:
                pred_min = result_df['prediction'].min()
                pred_max = result_df['prediction'].max()
                pred_mean = result_df['prediction'].mean()
                console.print(f"[cyan]预测值范围:[/cyan] {pred_min:.4f} ~ {pred_max:.4f} (平均: {pred_mean:.4f})")
            
            # 如果是分类任务，显示类别分布
            elif 'prediction_label' in result_df.columns:
                class_counts = result_df['prediction_label'].value_counts()
                console.print(f"[cyan]类别分布:[/cyan]")
                for class_name, count in class_counts.items():
                    console.print(f"  - {class_name}: {count} 个样本")
                    
        except Exception as e:
            console.print(f"[yellow]注意: 无法读取结果统计: {e}[/yellow]")
    else:
        console.print(f"\n[bold red]❌ 预测失败![/bold red]")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="INTERNCRANE 快速预测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python quick_predict.py output/my_experiment xgb input.csv output.csv
  
  # 显示详细信息
  python quick_predict.py output/my_experiment xgb input.csv output.csv --verbose
  
  # 使用其他模型
  python quick_predict.py output/my_experiment lgbm input.csv output.csv
  python quick_predict.py output/my_experiment catboost input.csv output.csv

支持的模型名称:
  - xgb (XGBoost)
  - lgbm (LightGBM) 
  - catboost (CatBoost)
  - rf (Random Forest)
  - ann (人工神经网络)
  - 以及其他在训练时使用的模型
        """
    )
    
    parser.add_argument('experiment_dir', type=str, 
                       help='训练实验目录路径')
    parser.add_argument('model_name', type=str,
                       help='模型名称 (如: xgb, lgbm, catboost)')
    parser.add_argument('input_file', type=str,
                       help='输入CSV文件路径')
    parser.add_argument('output_file', type=str,
                       help='输出CSV文件路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    # 执行预测
    success = quick_predict(
        experiment_dir=args.experiment_dir,
        model_name=args.model_name,
        input_file=args.input_file,
        output_file=args.output_file,
        verbose=args.verbose
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 