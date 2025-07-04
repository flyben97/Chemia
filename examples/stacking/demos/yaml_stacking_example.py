#!/usr/bin/env python3
"""
CRAFT YAML配置模型堆叠使用示例

这个脚本展示如何在Python代码中使用YAML配置文件进行模型堆叠
"""

from model_stacking import load_stacking_config_from_yaml, ModelStacker

def main():
    print("🎯 CRAFT YAML配置模型堆叠示例")
    print("=" * 50)
    
    # 方法1：从YAML配置文件创建堆叠器
    print("\n📋 方法1：从YAML配置文件创建堆叠器")
    try:
        # 使用简化配置
        stacker = load_stacking_config_from_yaml("config_stacking_simple.yaml")
        print(f"✅ 成功加载堆叠器，包含 {len(stacker.base_models)} 个模型")
        print(f"   堆叠方法: {stacker.stacking_method}")
        print(f"   模型列表: {list(stacker.base_models.keys())}")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("   请确保 experiment_dir 中有训练好的模型")
    
    # 方法2：在代码中使用堆叠器进行预测
    print("\n🔮 方法2：进行预测（示例）")
    print("""
# 单个样本预测
test_sample = {
    "SMILES": "CCO",
    "temperature": 80,
    "pressure": 1.0
}

try:
    prediction = stacker.predict_single(test_sample)
    print(f"预测结果: {prediction}")
except Exception as e:
    print(f"预测失败: {e}")

# 批量预测
test_data = [
    {"SMILES": "CCO", "temperature": 80},
    {"SMILES": "CC(=O)O", "temperature": 100}
]

try:
    results = stacker.predict(test_data)
    print(f"批量预测: {results['predictions']}")
    print(f"预测方法: {results['stacking_method']}")
    print(f"基础模型数量: {len(results['model_names'])}")
except Exception as e:
    print(f"批量预测失败: {e}")
    """)
    
    # 方法3：程序化创建配置
    print("\n🛠️  方法3：程序化创建配置")
    print("""
# 直接在代码中创建堆叠器
stacker = ModelStacker(experiment_dir="output/my_experiment")

# 添加模型
stacker.add_model("xgb", weight=0.4)
stacker.add_model("lgbm", weight=0.3)
stacker.add_model("catboost", weight=0.3)

# 设置堆叠方法
stacker.set_stacking_method("weighted_average")

# 评估模型
evaluation = stacker.evaluate(auto_load=True)
print(f"R² Score: {evaluation['r2']:.4f}")
    """)
    
    print("\n📚 更多使用方法请查看:")
    print("   - STACKING_YAML_GUIDE.md (详细指南)")
    print("   - config_stacking*.yaml (配置模板)")
    print("   - python stacking_yaml_demo.py --help (命令行帮助)")

if __name__ == "__main__":
    main() 