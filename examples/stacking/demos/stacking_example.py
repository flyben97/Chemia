#!/usr/bin/env python3
"""
CRAFT 模型堆叠使用示例

这个文件展示了如何使用 model_stacking.py 模块进行模型堆叠，
包含各种堆叠策略和使用场景的示例代码。
"""

import numpy as np
import pandas as pd
from model_stacking import ModelStacker, create_ensemble, auto_ensemble, smart_ensemble_with_meta_learner

def example_1_basic_stacking():
    """示例1：基础模型堆叠"""
    print("=" * 60)
    print("示例1：基础模型堆叠")
    print("=" * 60)
    
    # 替换为您的实验目录路径
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    
    try:
        # 创建堆叠器
        stacker = ModelStacker(experiment_dir=experiment_dir)
        
        # 添加基础模型（根据性能设置权重）
        stacker.add_model("xgb", weight=0.4)      # XGBoost权重最高
        stacker.add_model("lgbm", weight=0.3)     # LightGBM次之
        stacker.add_model("catboost", weight=0.3) # CatBoost
        
        # 设置堆叠方法为加权平均
        stacker.set_stacking_method("weighted_average")
        
        # 准备测试数据
        test_data = [
            {
                'SMILES': 'CCO',
                'Solvent_1_SMILES': 'CC(=O)O',
                'Solvent_2_SMILES': 'CCN',
            },
            {
                'SMILES': 'c1ccccc1',
                'Solvent_1_SMILES': 'CNC(=O)N',
                'Solvent_2_SMILES': 'CC',
            }
        ]
        
        # 进行预测
        results = stacker.predict(test_data)
        
        print(f"堆叠方法: {results['stacking_method']}")
        print(f"使用模型: {results['model_names']}")
        print(f"样本数量: {results['n_samples']}")
        print(f"集成预测: {results['predictions']}")
        
        # 显示各个基础模型的预测结果
        print("\n各模型预测对比:")
        base_predictions = results['base_predictions']
        for i, model_name in enumerate(results['model_names']):
            print(f"  {model_name}: {base_predictions[:, i]}")
        
        print(f"  集成结果: {results['predictions']}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请确保实验目录路径正确，并且包含训练好的模型")

def example_2_meta_learning():
    """示例2：使用元学习器进行堆叠（自动数据加载）"""
    print("\n" + "=" * 60)
    print("示例2：使用元学习器进行堆叠（自动数据加载）")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    
    try:
        # 创建堆叠器
        stacker = ModelStacker(experiment_dir=experiment_dir)
        
        # 添加基础模型
        stacker.add_model("xgb")
        stacker.add_model("lgbm")
        stacker.add_model("catboost")
        
        # 设置元学习器方法
        stacker.set_stacking_method("ridge")  # 使用Ridge回归作为元学习器
        
        # 🆕 自动训练元学习器（自动从实验目录读取验证数据）
        stacker.fit_meta_model(auto_load=True, validation_size=50)
        
        # 使用元学习器进行预测
        test_data = [
            {'SMILES': 'CCC', 'Solvent_1_SMILES': 'CCO', 'Solvent_2_SMILES': 'CC(=O)O'}
        ]
        
        results = stacker.predict(test_data)
        print(f"元学习器预测: {results['predictions']}")
        print(f"基础模型预测: {results['base_predictions'][0]}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

def example_3_quick_ensemble():
    """示例3：快速创建集成模型"""
    print("\n" + "=" * 60)
    print("示例3：快速创建集成模型")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    
    try:
        # 快速创建加权集成
        stacker = create_ensemble(
            experiment_dir=experiment_dir,
            model_names=["xgb", "lgbm", "catboost"],
            weights=[0.5, 0.3, 0.2],  # 基于性能设置权重
            method="weighted_average"
        )
        
        # 测试数据
        sample = {
            'SMILES': 'CCO',
            'Solvent_1_SMILES': 'CC(=O)O',
            'Solvent_2_SMILES': 'CCN'
        }
        
        # 单样本预测
        prediction = stacker.predict_single(sample)
        print(f"快速集成预测: {prediction}")
        
        # 批量预测
        samples = [sample] * 3
        results = stacker.predict(samples)
        print(f"批量预测结果: {results['predictions']}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

def example_4_auto_ensemble():
    """示例4：完全自动集成（自动数据加载）"""
    print("\n" + "=" * 60)
    print("示例4：完全自动集成（自动数据加载）")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    
    try:
        # 🆕 完全自动化：自动加载验证数据，自动选择模型，自动计算权重
        stacker = auto_ensemble(
            experiment_dir=experiment_dir,
            auto_load_validation=True,  # 自动从实验目录读取验证数据
            validation_size=100,        # 验证数据大小限制
            available_models=['xgb', 'lgbm', 'catboost', 'rf']  # 可用模型列表
        )
        
        # 测试预测
        test_sample = {'SMILES': 'CCC', 'Solvent_1_SMILES': 'CCO', 'Solvent_2_SMILES': 'CC(=O)O'}
        prediction = stacker.predict_single(test_sample)
        print(f"自动集成预测: {prediction}")
        
        # 🆕 自动评估（自动加载测试数据）
        evaluation = stacker.evaluate(auto_load=True, use_test_set=True)
        print(f"自动评估结果:")
        print(f"  R²: {evaluation.get('r2', 'N/A'):.4f}")
        print(f"  RMSE: {evaluation.get('rmse', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

def example_5_model_evaluation():
    """示例5：模型性能评估和比较"""
    print("\n" + "=" * 60)
    print("示例5：模型性能评估和比较")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    
    try:
        # 创建不同的堆叠器进行比较
        stackers = {
            "简单平均": create_ensemble(experiment_dir, ["xgb", "lgbm", "catboost"], method="simple_average"),
            "加权平均": create_ensemble(experiment_dir, ["xgb", "lgbm", "catboost"], [0.5, 0.3, 0.2], "weighted_average"),
        }
        
        # 准备测试数据和真实标签
        test_data = [
            {'SMILES': 'CCO', 'Solvent_1_SMILES': 'CC(=O)O', 'Solvent_2_SMILES': 'CCN'},
            {'SMILES': 'c1ccccc1', 'Solvent_1_SMILES': 'CNC(=O)N', 'Solvent_2_SMILES': 'CC'},
            {'SMILES': 'CCCC', 'Solvent_1_SMILES': 'O', 'Solvent_2_SMILES': 'CCO'},
        ]
        true_labels = [12.5, 8.3, 15.2]  # 示例真实标签
        
        print("堆叠方法性能比较:")
        print("-" * 40)
        
        for method_name, stacker in stackers.items():
            try:
                evaluation = stacker.evaluate(test_data, true_labels)
                print(f"\n{method_name}:")
                if evaluation.get('r2') is not None:
                    print(f"  R² Score: {evaluation['r2']:.4f}")
                    print(f"  RMSE: {evaluation['rmse']:.4f}")
                    print(f"  MAE: {evaluation['mae']:.4f}")
                
                # 显示基础模型性能
                print("  基础模型性能:")
                for model_name, perf in evaluation['base_model_performance'].items():
                    if 'r2' in perf:
                        print(f"    {model_name}: R²={perf['r2']:.4f}, RMSE={perf['rmse']:.4f}")
                    else:
                        print(f"    {model_name}: Accuracy={perf['accuracy']:.4f}")
                        
            except Exception as e:
                print(f"  {method_name}: 评估失败 - {e}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

def example_6_save_and_load():
    """示例6：保存和加载堆叠模型"""
    print("\n" + "=" * 60)
    print("示例6：保存和加载堆叠模型")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    
    try:
        # 创建堆叠器
        stacker = create_ensemble(
            experiment_dir=experiment_dir,
            model_names=["xgb", "lgbm"],
            weights=[0.6, 0.4],
            method="weighted_average"
        )
        
        # 保存堆叠模型
        save_path = "models/my_stacked_model.pkl"
        stacker.save(save_path)
        
        # 加载堆叠模型
        loaded_stacker = ModelStacker.load(save_path)
        
        # 使用加载的模型进行预测
        test_sample = {'SMILES': 'CCO', 'Solvent_1_SMILES': 'CC(=O)O', 'Solvent_2_SMILES': 'CCN'}
        
        original_pred = stacker.predict_single(test_sample)
        loaded_pred = loaded_stacker.predict_single(test_sample)
        
        print(f"原始模型预测: {original_pred}")
        print(f"加载模型预测: {loaded_pred}")
        
        # 类型安全的比较
        try:
            if isinstance(original_pred, (int, float)) and isinstance(loaded_pred, (int, float)):
                consistency = "✓" if abs(float(original_pred) - float(loaded_pred)) < 1e-6 else "✗"
            else:
                consistency = "✓" if original_pred == loaded_pred else "✗"
            print(f"预测一致性: {consistency}")
        except Exception:
            print(f"预测一致性: {'✓' if original_pred == loaded_pred else '✗'}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

def example_7_advanced_stacking():
    """示例7：高级堆叠技巧"""
    print("\n" + "=" * 60)
    print("示例7：高级堆叠技巧")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    
    try:
        # 1. 多层堆叠：先创建基础集成，再进行二次堆叠
        print("🔧 多层堆叠示例:")
        
        # 第一层：创建两个不同的基础集成
        ensemble1 = create_ensemble(experiment_dir, ["xgb", "lgbm"], [0.6, 0.4], "weighted_average")
        ensemble2 = create_ensemble(experiment_dir, ["catboost"], [1.0], "weighted_average")  # 单模型集成
        
        # 第二层：组合两个集成（这里简化为权重平均）
        test_sample = {'SMILES': 'CCO', 'Solvent_1_SMILES': 'CC(=O)O', 'Solvent_2_SMILES': 'CCN'}
        
        pred1 = ensemble1.predict_single(test_sample)
        pred2 = ensemble2.predict_single(test_sample)
        
        # 类型安全的计算
        if isinstance(pred1, (int, float)) and isinstance(pred2, (int, float)):
            final_pred = 0.6 * float(pred1) + 0.4 * float(pred2)
            print(f"  集成1预测: {pred1:.4f}")
            print(f"  集成2预测: {pred2:.4f}")
            print(f"  最终预测: {final_pred:.4f}")
        else:
            print(f"  集成1预测: {pred1}")
            print(f"  集成2预测: {pred2}")
            print("  注意：非数值预测，无法进行算术组合")
        
        # 2. 动态权重调整
        print("\n🔧 动态权重调整:")
        
        def calculate_dynamic_weights(models, validation_data, true_labels):
            """根据验证性能动态计算权重"""
            performances = []
            for model_name in models:
                try:
                    # 这里简化为随机性能，实际中应该用验证集评估
                    perf = np.random.uniform(0.7, 0.95)  # 模拟R²分数
                    performances.append(max(0, perf))
                except:
                    performances.append(0)
            
            # 归一化权重
            total = sum(performances)
            if total > 0:
                weights = [p / total for p in performances]
            else:
                weights = [1.0 / len(models)] * len(models)
            
            return weights
        
        # 动态计算权重
        models = ["xgb", "lgbm", "catboost"]
        dynamic_weights = calculate_dynamic_weights(models, None, None)
        
        dynamic_ensemble = create_ensemble(experiment_dir, models, dynamic_weights, "weighted_average")
        dynamic_pred = dynamic_ensemble.predict_single(test_sample)
        
        print(f"  动态权重: {[f'{w:.3f}' for w in dynamic_weights]}")
        print(f"  动态集成预测: {dynamic_pred}")
        
        # 3. 置信度估算
        print("\n🔧 预测置信度估算:")
        
        result = dynamic_ensemble.predict([test_sample])
        base_preds = result['base_predictions'][0]
        std_dev = np.std(base_preds)
        mean_pred = np.mean(base_preds)
        confidence = 1 / (1 + std_dev)  # 简单的置信度计算
        
        print(f"  基础预测: {base_preds}")
        print(f"  标准差: {std_dev:.4f}")
        print(f"  置信度: {confidence:.4f}")
        print(f"  预测区间: [{mean_pred - 2*std_dev:.4f}, {mean_pred + 2*std_dev:.4f}]")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

def example_8_classification_stacking():
    """示例8：分类任务的模型堆叠"""
    print("\n" + "=" * 60)
    print("示例8：分类任务的模型堆叠")
    print("=" * 60)
    
    # 注意：这个示例需要分类任务的实验目录
    classification_experiment_dir = "output/classification_experiment"
    
    print("📝 分类任务堆叠特点:")
    print("  - 支持概率预测的聚合")
    print("  - 可以使用投票机制")
    print("  - 元学习器可以学习类别概率分布")
    
    print("\n示例代码:")
    print("""
# 分类任务堆叠
stacker = ModelStacker(experiment_dir=classification_experiment_dir)
stacker.add_model("xgb", weight=0.4)
stacker.add_model("lgbm", weight=0.3)
stacker.add_model("rf", weight=0.3)

# 使用逻辑回归作为元学习器（分类任务）
stacker.set_stacking_method("logistic")

# 训练元学习器
stacker.fit_meta_model(validation_data, true_class_labels)

# 预测（返回类别和概率）
results = stacker.predict(test_data)
predicted_classes = results['predictions']
class_probabilities = results['probabilities']
    """)

def example_9_smart_meta_learner():
    """示例9：智能元学习器集成（新增）"""
    print("\n" + "=" * 60)
    print("示例9：智能元学习器集成（新增）")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    
    try:
        # 🆕 智能创建元学习器集成：
        # 1. 自动加载验证数据
        # 2. 自动选择最佳模型组合
        # 3. 自动训练元学习器
        stacker = smart_ensemble_with_meta_learner(
            experiment_dir=experiment_dir,
            validation_size=100,
            meta_method="ridge",  # 可选: "ridge", "rf", "logistic"
            available_models=['xgb', 'lgbm', 'catboost', 'rf', 'ann']
        )
        
        # 测试预测
        test_sample = {'SMILES': 'CCO', 'Solvent_1_SMILES': 'CC(=O)O', 'Solvent_2_SMILES': 'CCN'}
        prediction = stacker.predict_single(test_sample)
        print(f"智能集成预测: {prediction}")
        
        # 自动评估
        evaluation = stacker.evaluate(auto_load=True)
        print(f"集成性能:")
        print(f"  R²: {evaluation.get('r2', 'N/A'):.4f}")
        print(f"  使用模型: {evaluation['model_names']}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

def example_10_custom_validation_data():
    """示例10：使用自定义验证数据"""
    print("\n" + "=" * 60)
    print("示例10：使用自定义验证数据")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    
    try:
        # 使用自定义验证数据进行堆叠
        print("🔧 使用自定义验证数据:")
        
        # 准备自定义验证数据
        custom_validation = [
            {'SMILES': 'CCO', 'Solvent_1_SMILES': 'CC(=O)O', 'Solvent_2_SMILES': 'CCN'},
            {'SMILES': 'c1ccccc1', 'Solvent_1_SMILES': 'CNC(=O)N', 'Solvent_2_SMILES': 'CC'},
            {'SMILES': 'CCCC', 'Solvent_1_SMILES': 'O', 'Solvent_2_SMILES': 'CCO'},
        ]
        custom_labels = [12.5, 8.3, 15.2]  # 自定义真实标签
        
        # 方法1：手动提供验证数据
        stacker = auto_ensemble(
            experiment_dir=experiment_dir,
            validation_data=custom_validation,
            true_labels=custom_labels,
            auto_load_validation=False  # 关闭自动加载
        )
        
        # 方法2：从CSV文件加载自定义验证数据
        print("\n📂 从文件加载自定义验证数据的示例:")
        print("""
from data_loader import load_custom_validation_data

# 从单个文件加载（包含特征和标签）
val_data, val_labels = load_custom_validation_data(
    validation_file="my_validation_data.csv",
    target_column="target_value"
)

# 从分离的文件加载
val_data, val_labels = load_custom_validation_data(
    validation_file="my_features.csv",
    labels_file="my_labels.csv"
)

# 使用加载的数据
stacker = auto_ensemble(
    experiment_dir,
    validation_data=val_data,
    true_labels=val_labels,
    auto_load_validation=False
)
        """)
        
        print(f"✓ 自定义验证数据集成完成")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    print("CRAFT 模型堆叠使用示例")
    print("请确保修改实验目录路径为您实际的路径")
    print()
    
    # 运行所有示例
    example_1_basic_stacking()
    example_2_meta_learning()
    example_3_quick_ensemble()
    example_4_auto_ensemble()
    example_5_model_evaluation()
    example_6_save_and_load()
    example_7_advanced_stacking()
    example_8_classification_stacking()
    example_9_smart_meta_learner()
    example_10_custom_validation_data()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
    print("\n💡 提示:")
    print("1. 模型堆叠通常能提升预测性能，特别是当基础模型具有互补性时")
    print("2. 加权平均适合快速部署，元学习器适合追求最佳性能")
    print("3. 建议在独立的验证集上评估堆叠模型的性能")
    print("4. 保存训练好的堆叠模型以便后续使用") 