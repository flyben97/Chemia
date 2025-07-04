#!/usr/bin/env python3
"""
INTERNCRANE Quick Stacking Demo - Can be run directly

This script can be run directly, it will automatically use preset configurations for stacking demonstration.

✨ New Features:
- Evaluate performance on validation and test datasets separately
- Automatically compare performance between two datasets
- Smartly identify overfitting issues

If you want to customize configurations, please use stacking_yaml_demo.py script.
"""

import os
import sys

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_quick_stacking_demo():
    """Run stacking demo directly"""
    print("🚀 INTERNCRANE Quick Stacking Demo")
    print("=" * 50)
    
    # Check if there's a config_stacking_meta.yaml file
    config_files = [
        "config_stacking_meta.yaml",
        "config_stacking_example.yaml", 
        "config_stacking_split_aware.yaml",
        "config_stacking_weighted.yaml"
    ]
    
    config_file = None
    for cf in config_files:
        if os.path.exists(cf):
            config_file = cf
            break
    
    if config_file is None:
        print("⚠️  No stacking config file found. Creating basic config...")
        # Create basic configuration
        create_quick_config()
        config_file = "quick_stacking_config.yaml"
    else:
        print(f"✓ Found config file: {config_file}")
    
    try:
        # Run stacking
        from stacking_yaml_demo import run_stacking_from_yaml
        
        print(f"\n🔄 Running stacking with config: {config_file}")
        results = run_stacking_from_yaml(config_file)
        
        if results:
            print("✅ Stacking completed successfully!")
            # Display evaluation mode information
            if 'validation' in results and 'test' in results:
                print("\n📊 Dual Dataset Evaluation Results:")
                print(f"Validation Score: {results['validation'].get('score', 'N/A')}")
                print(f"Test Score: {results['test'].get('score', 'N/A')}")
            else:
                print(f"\n📊 Evaluation Score: {results.get('score', 'N/A')}")
        
        # Display other available configurations
        other_configs = [cf for cf in config_files if cf != config_file and os.path.exists(cf)]
        if other_configs:
            print(f"\n💡 Other available config files: {', '.join(other_configs)}")
            print("💡 You can try running with different configs!")
        
    except Exception as e:
        print(f"❌ Error running stacking: {e}")
        print("💡 Try creating a basic config with create_quick_config()")

def create_quick_config():
    """Create quick configuration file"""
    print("🔧 Creating quick stacking configuration...")
    
    # Find possible experiment directories
    output_dirs = []
    if os.path.exists("output"):
        for item in os.listdir("output"):
            item_path = os.path.join("output", item)
            if os.path.isdir(item_path):
                output_dirs.append(item_path)
    
    if not output_dirs:
        print("❌ No experiment directories found in output/")
        print("💡 Please ensure you have trained models in output/ directory")
        return
    
    # Use the first found directory
    experiment_dir = output_dirs[0]
    print(f"📁 Using experiment directory: {experiment_dir}")
    
    # Create basic YAML config
    config = {
        'stacking': {
            'experiment_dir': experiment_dir,
            'method': 'weighted_average',
            'models': [
                {'name': 'xgb', 'weight': 1.0, 'enabled': True},
                {'name': 'lgbm', 'weight': 1.0, 'enabled': True},
                {'name': 'catboost', 'weight': 1.0, 'enabled': True}
            ]
        },
        'evaluation': {
            'auto_evaluate': True
        }
    }
    
    # Save config
    import yaml
    with open("quick_stacking_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Created quick_stacking_config.yaml")
    print("💡 You can edit this file to customize your stacking configuration")

if __name__ == "__main__":
    run_quick_stacking_demo() 