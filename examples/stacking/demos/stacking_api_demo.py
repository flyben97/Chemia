#!/usr/bin/env python3
"""
CRAFT 堆叠模型API 使用演示

这个脚本展示如何使用新的stacking_api.py来进行模型堆叠，
包括YAML配置方式和程序化方式。
"""

import os
import sys
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入新的API
from stacking_api import (
    load_stacker_from_config, create_stacker, stack_predict, 
    stack_predict_single, quick_stack_predict, StackingPredictor
)
from utils.stacking_config import (
    create_sample_stacking_config, get_config_template, save_yaml_config
)

def demo_yaml_config_approach():
    """演示YAML配置方式"""
    print("🔧 方法1：使用YAML配置文件")
    print("=" * 50)
    
    # 创建示例配置
    experiment_dir = "output/my_experiment"  # 假设的实验目录
    
    print("📋 1.1 创建配置文件...")
    config = create_sample_stacking_config(
        experiment_dir=experiment_dir,
        model_names=['xgb', 'lgbm', 'catboost'],
        weights=[0.4, 0.3, 0.3],
        method='weighted_average',
        output_path='demo_stacking_config.yaml'
    )
    print("✓ 配置文件已创建: demo_stacking_config.yaml")
    
    # 显示配置内容
    print("\n📄 配置文件内容:")
    print("-" * 30)
    import yaml
    print(yaml.dump(config, default_flow_style=False))
    
    try:
        print("🚀 1.2 从配置文件加载堆叠器...")
        # 加载堆叠器（这里会失败，因为没有实际的模型文件）
        stacker = load_stacker_from_config('demo_stacking_config.yaml')
        print(f"✓ 堆叠器加载成功: {stacker.get_info()}")
        
        print("\n🔮 1.3 进行预测...")
        # 示例预测
        test_sample = {
            "SMILES": "CCO",
            "Solvent_1_SMILES": "CC(=O)O",
            "temperature": 80
        }
        
        prediction = stack_predict_single(stacker, test_sample)
        print(f"✓ 预测结果: {prediction}")
        
    except Exception as e:
        print(f"❌ 演示失败 (正常，因为没有真实模型): {e}")
        print("   在实际使用中，请确保experiment_dir中有训练好的模型")

def demo_programmatic_approach():
    """演示程序化方式"""
    print("\n\n🛠️  方法2：程序化创建堆叠器")
    print("=" * 50)
    
    try:
        print("🔧 2.1 创建堆叠器...")
        stacker = create_stacker(
            experiment_dir="output/my_experiment",
            model_names=['xgb', 'lgbm'],
            weights=[0.6, 0.4],
            method='weighted_average'
        )
        print(f"✓ 堆叠器创建成功: {stacker.get_info()}")
        
        print("\n📊 2.2 模型信息:")
        info = stacker.get_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n🔮 2.3 批量预测示例...")
        test_data = [
            {"SMILES": "CCO", "temperature": 80},
            {"SMILES": "CC(=O)O", "temperature": 100}
        ]
        
        results = stack_predict(stacker, test_data)
        print(f"✓ 预测完成: {results}")
        
    except Exception as e:
        print(f"❌ 演示失败 (正常，因为没有真实模型): {e}")

def demo_config_templates():
    """演示配置模板功能"""
    print("\n\n📋 方法3：使用预定义配置模板")
    print("=" * 50)
    
    # 可用的模板
    templates = ['basic_weighted', 'simple_average', 'meta_learner']
    
    for template_name in templates:
        try:
            print(f"\n📝 3.{templates.index(template_name)+1} 创建 {template_name} 模板...")
            config = get_config_template(template_name, "output/my_experiment")
            
            # 保存模板到文件
            template_file = f"template_{template_name}.yaml"
            save_yaml_config(config, template_file)
            print(f"✓ 模板已保存: {template_file}")
            
            # 显示堆叠配置部分
            stacking_config = config['stacking']
            print(f"   方法: {stacking_config['method']}")
            print(f"   模型: {[m['name'] for m in stacking_config['models']]}")
            
        except Exception as e:
            print(f"❌ 模板创建失败: {e}")

def demo_quick_functions():
    """演示快速功能"""
    print("\n\n⚡ 方法4：快速功能")
    print("=" * 50)
    
    try:
        print("🚀 4.1 一步预测...")
        # 准备测试数据
        test_data = {"SMILES": "CCO", "temperature": 80}
        
        # 一步完成配置加载和预测
        # results = quick_stack_predict('demo_stacking_config.yaml', test_data)
        # print(f"✓ 一步预测结果: {results}")
        print("   (需要有效的配置文件和模型)")
        
        print("\n💾 4.2 保存和加载堆叠器...")
        print("   stacker.save('my_ensemble.pkl')")
        print("   loaded_stacker = StackingPredictor.load('my_ensemble.pkl')")
        
    except Exception as e:
        print(f"❌ 快速功能演示失败: {e}")

def show_api_comparison():
    """显示API对比"""
    print("\n\n📚 API 使用对比")
    print("=" * 50)
    
    print("🔗 旧方式 (model_stacking.py):")
    print("""
from model_stacking import ModelStacker

stacker = ModelStacker("output/my_experiment")
stacker.add_model("xgb", weight=0.4)
stacker.add_model("lgbm", weight=0.3)
stacker.set_stacking_method("weighted_average")
result = stacker.predict(data)
    """)
    
    print("✨ 新方式 (stacking_api.py):")
    print("""
from stacking_api import load_stacker_from_config, stack_predict

# 方式1：YAML配置
stacker = load_stacker_from_config("config.yaml")
result = stack_predict(stacker, data)

# 方式2：程序化创建
stacker = create_stacker("output/my_experiment", 
                        ["xgb", "lgbm"], [0.4, 0.3])
result = stack_predict(stacker, data)

# 方式3：一步预测
result = quick_stack_predict("config.yaml", data)
    """)

def main():
    """主函数"""
    print("🎯 CRAFT 堆叠模型API 使用演示")
    print("=" * 60)
    print("这个演示展示了新的stacking_api.py的各种使用方法")
    print("注意：由于没有真实的训练模型，部分功能会报错，但代码逻辑是正确的")
    
    # 演示各种方法
    demo_yaml_config_approach()
    demo_programmatic_approach()
    demo_config_templates()
    demo_quick_functions()
    show_api_comparison()
    
    print("\n\n🎉 演示完成！")
    print("\n📁 生成的文件:")
    generated_files = [
        'demo_stacking_config.yaml',
        'template_basic_weighted.yaml',
        'template_simple_average.yaml',
        'template_meta_learner.yaml'
    ]
    
    for file in generated_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
    
    print("\n🔧 实际使用步骤:")
    print("1. 训练CRAFT模型：python run_training_only.py --config your_config.yaml")
    print("2. 创建堆叠配置：从模板开始或使用 create_sample_stacking_config()")
    print("3. 运行堆叠：python stacking_yaml_demo.py --config your_stacking_config.yaml")
    print("4. 或在代码中使用：stacker = load_stacker_from_config('config.yaml')")

if __name__ == "__main__":
    main() 