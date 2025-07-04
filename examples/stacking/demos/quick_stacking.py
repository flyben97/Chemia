#!/usr/bin/env python3
"""
CRAFT 快速模型堆叠工具

最简单的一键模型堆叠，只需要提供实验目录即可快速获得集成预测。
适合快速测试和部署场景。
"""

import sys
import os
from typing import Dict, List, Any, Union, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_stacking import ModelStacker, create_ensemble, auto_ensemble

def quick_stack(experiment_dir: str, 
               sample: Union[Dict[str, Any], List[Dict[str, Any]]],
               models: Optional[List[str]] = None,
               method: str = "weighted_average",
               auto_optimize: bool = True) -> Dict[str, Any]:
    """
    一键模型堆叠和预测
    
    Args:
        experiment_dir: 实验目录路径
        sample: 要预测的样本（单个字典或字典列表）
        models: 模型名称列表（可选，默认尝试常见模型）
        method: 堆叠方法（默认加权平均）
        auto_optimize: 是否使用自动优化（根据验证数据计算权重）
    
    Returns:
        dict: 预测结果
    """
    if models is None:
        models = ['xgb', 'lgbm', 'catboost', 'rf', 'ann']
    
    try:
        if auto_optimize:
            # 🆕 使用自动优化：自动加载验证数据，自动计算权重
            print("🔄 使用自动优化模式...")
            stacker = auto_ensemble(
                experiment_dir=experiment_dir,
                auto_load_validation=True,
                validation_size=50,  # 快速验证，使用较小数据集
                available_models=models
            )
        else:
            # 传统模式：等权重堆叠
            print("🔄 使用等权重堆叠模式...")
            stacker = ModelStacker(experiment_dir=experiment_dir)
            loaded_models = []
            
            for model_name in models:
                try:
                    stacker.add_model(model_name, weight=1.0)
                    loaded_models.append(model_name)
                except Exception:
                    continue
            
            if len(loaded_models) == 0:
                raise ValueError("无法加载任何模型")
            
            if len(loaded_models) == 1:
                print(f"⚠️  只有一个可用模型: {loaded_models[0]}，无法进行堆叠")
                stacker.set_stacking_method("simple_average")
            else:
                # 设置等权重
                equal_weight = 1.0 / len(loaded_models)
                for model_name in loaded_models:
                    stacker.model_weights[model_name] = equal_weight
                
                stacker.set_stacking_method(method)
                print(f"✅ 成功加载 {len(loaded_models)} 个模型: {loaded_models}")
        
        # 进行预测
        results = stacker.predict(sample)
        
        # 添加额外信息
        results['loaded_models'] = list(stacker.base_models.keys())
        results['experiment_dir'] = experiment_dir
        results['auto_optimized'] = auto_optimize
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"快速堆叠失败: {e}")

def ensemble_predict(experiment_dir: str, 
                    sample: Union[Dict[str, Any], List[Dict[str, Any]]],
                    auto_optimize: bool = True) -> float:
    """
    最简单的集成预测，直接返回预测值
    
    Args:
        experiment_dir: 实验目录路径
        sample: 单个样本字典或样本列表
        auto_optimize: 是否使用自动优化
    
    Returns:
        预测值
    """
    if isinstance(sample, dict):
        # 单样本预测
        results = quick_stack(experiment_dir, sample, auto_optimize=auto_optimize)
        return results['predictions'][0]
    else:
        # 多样本预测
        results = quick_stack(experiment_dir, sample, auto_optimize=auto_optimize)
        return results['predictions']

def best_ensemble(experiment_dir: str,
                 validation_data: Optional[Union[Dict, List[Dict]]] = None,
                 true_labels: Optional[Union[List, Any]] = None,
                 sample: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """
    自动寻找最佳集成并预测
    
    Args:
        experiment_dir: 实验目录路径
        validation_data: 验证数据（可选，如果为None则自动加载）
        true_labels: 真实标签（可选，如果为None则自动加载）
        sample: 要预测的样本（可选）
    
    Returns:
        dict: 预测结果和集成信息
    """
    try:
        # 🆕 使用自动集成
        print("🔍 自动寻找最佳集成...")
        if validation_data is None and true_labels is None:
            stacker = auto_ensemble(
                experiment_dir=experiment_dir,
                auto_load_validation=True,
                validation_size=100
            )
        else:
            # 确保有效的验证数据和标签
            if validation_data is not None and true_labels is not None:
                stacker = auto_ensemble(
                    experiment_dir=experiment_dir,
                    validation_data=validation_data,
                    true_labels=true_labels,
                    auto_load_validation=False
                )
            else:
                # 如果只有一个为None，回退到自动加载
                print("⚠️  验证数据不完整，使用自动加载模式")
                stacker = auto_ensemble(
                    experiment_dir=experiment_dir,
                    auto_load_validation=True,
                    validation_size=100
                )
        
        result = {
            'stacker': stacker,
            'optimization': 'auto',
            'model_weights': stacker.model_weights,
            'models_used': list(stacker.base_models.keys())
        }
        
        # 如果提供了样本，进行预测
        if sample is not None:
            predictions = stacker.predict(sample)
            result.update(predictions)
        
        return result
        
    except Exception as e:
        print(f"⚠️  自动优化失败，使用默认方法: {e}")
        if sample is not None:
            fallback_result = quick_stack(experiment_dir, sample, auto_optimize=False)
            fallback_result['optimization'] = 'fallback'
            return fallback_result
        else:
            # 如果没有样本，只返回堆叠器信息
            try:
                stacker = ModelStacker(experiment_dir=experiment_dir)
                models = ['xgb', 'lgbm', 'catboost', 'rf', 'ann']
                for model_name in models:
                    try:
                        stacker.add_model(model_name, weight=1.0)
                    except Exception:
                        continue
                
                if not stacker.base_models:
                    raise ValueError("无法加载任何模型")
                
                return {
                    'stacker': stacker,
                    'optimization': 'fallback',
                    'model_weights': stacker.model_weights,
                    'models_used': list(stacker.base_models.keys())
                }
            except Exception as inner_e:
                raise RuntimeError(f"自动集成和备用方法都失败: {e}, {inner_e}")

def evaluate_ensemble(experiment_dir: str, auto_optimize: bool = True) -> Dict[str, Any]:
    """
    评估集成模型性能（新增功能）
    
    Args:
        experiment_dir: 实验目录路径
        auto_optimize: 是否使用自动优化
        
    Returns:
        dict: 评估结果
    """
    try:
        if auto_optimize:
            stacker = auto_ensemble(experiment_dir=experiment_dir)
        else:
            # 创建基础堆叠器用于评估
            stacker = ModelStacker(experiment_dir=experiment_dir)
            models = ['xgb', 'lgbm', 'catboost', 'rf', 'ann']
            loaded_count = 0
            for model_name in models:
                try:
                    stacker.add_model(model_name, weight=1.0)
                    loaded_count += 1
                except Exception:
                    continue
            
            if loaded_count == 0:
                raise ValueError("无法加载任何模型")
            
            # 设置等权重
            equal_weight = 1.0 / loaded_count
            for model_name in stacker.base_models.keys():
                stacker.model_weights[model_name] = equal_weight
            
            stacker.set_stacking_method("weighted_average")
        
        # 🆕 自动评估（自动加载测试数据）
        evaluation = stacker.evaluate(auto_load=True, use_test_set=True)
        
        print("📊 集成模型评估结果:")
        print(f"  堆叠方法: {evaluation['stacking_method']}")
        print(f"  模型数量: {evaluation['n_models']}")
        print(f"  使用模型: {evaluation['model_names']}")
        
        if evaluation.get('r2') is not None:
            print(f"  R²: {evaluation['r2']:.4f}")
            print(f"  RMSE: {evaluation['rmse']:.4f}")
            print(f"  MAE: {evaluation['mae']:.4f}")
        
        if evaluation.get('accuracy') is not None:
            print(f"  准确率: {evaluation['accuracy']:.4f}")
        
        return evaluation
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return {'error': str(e)}

# 命令行接口
def main():
    """命令行主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python quick_stacking.py <experiment_dir>")
        print()
        print("示例:")
        print("  python quick_stacking.py output/my_experiment")
        return
    
    experiment_dir = sys.argv[1]
    
    print("🚀 CRAFT 快速模型堆叠")
    print(f"📁 实验目录: {experiment_dir}")
    print()
    
    # 示例数据（用户需要根据实际情况修改）
    sample_data = {
        'SMILES': 'CCO',
        'Solvent_1_SMILES': 'CC(=O)O',
        'Solvent_2_SMILES': 'CCN'
    }
    
    try:
        # 快速堆叠预测
        print("正在进行快速堆叠...")
        results = quick_stack(experiment_dir, sample_data)
        
        print("=" * 50)
        print("🎯 预测结果")
        print("=" * 50)
        print(f"预测值: {results['predictions'][0]}")
        print(f"使用模型: {results['loaded_models']}")
        print(f"堆叠方法: {results['stacking_method']}")
        
        if len(results['loaded_models']) > 1:
            print("\n📊 各模型预测:")
            base_predictions = results['base_predictions'][0]
            for i, model_name in enumerate(results['model_names']):
                print(f"  {model_name}: {base_predictions[i]:.4f}")
        
        print(f"\n✅ 堆叠完成！")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n💡 请检查:")
        print("  1. 实验目录路径是否正确")
        print("  2. 目录中是否包含训练好的模型")
        print("  3. 样本数据格式是否匹配训练时的特征")

if __name__ == "__main__":
    # 如果作为脚本运行，执行命令行接口
    main()
    
    print("\n" + "=" * 60)
    print("📖 使用示例")
    print("=" * 60)
    print("""
# 1. 最简单的使用
from quick_stacking import ensemble_predict

prediction = ensemble_predict("output/my_experiment", {
    'SMILES': 'CCO',
    'Solvent_1_SMILES': 'CC(=O)O', 
    'Solvent_2_SMILES': 'CCN'
})
print(f"预测结果: {prediction}")

# 2. 获取详细结果
from quick_stacking import quick_stack

results = quick_stack("output/my_experiment", sample_data)
print(f"集成预测: {results['predictions']}")
print(f"使用模型: {results['loaded_models']}")

# 3. 自动优化
from quick_stacking import best_ensemble

results = best_ensemble(
    "output/my_experiment",
    validation_data,
    true_labels,
    test_sample
)
    """)
    print("\n💡 提示: 修改上述示例中的数据路径和特征字段以匹配您的实际情况") 