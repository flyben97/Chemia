#!/usr/bin/env python3
"""
CHEMIA Model Stacking YAML Configuration Demo

This script demonstrates how to use YAML configuration files for model stacking.
Supports automatic model loading, stacking, and evaluation.

Usage:
    python stacking_yaml_demo.py --config config_stacking.yaml

You need to have trained models in the specified experiment directory.
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_stacking import ModelStacker, load_stacking_config_from_yaml

def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config: {config_path}")
        return config
    except Exception as e:
        raise ValueError(f"Failed to load config file {config_path}: {e}")

def save_results_to_file(results: dict, output_dir: str = "output"):
    """Save results to file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        results_dir = os.path.join(output_dir, "stacking_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save evaluation results
        eval_file = os.path.join(results_dir, f"evaluation_{timestamp}.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # Save prediction results
        if 'predictions' in results:
            pred_file = os.path.join(results_dir, f"predictions_{timestamp}.json")
            with open(pred_file, 'w', encoding='utf-8') as f:
                json.dump(results['predictions'], f, ensure_ascii=False, indent=2, default=str)
        
        # Save configuration file copy
        if hasattr(save_results_to_file, 'config_path'):
            config_copy = os.path.join(results_dir, f"config_{timestamp}.yaml")
            import shutil
            shutil.copy2(save_results_to_file.config_path, config_copy)
        
        print(f"✓ Results saved to: {results_dir}")
        return results_dir
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return None

def run_stacking_from_yaml(config_path: str, verbose: bool = True) -> Optional[Dict]:
    """
    Run model stacking workflow from YAML configuration
    
    Args:
        config_path: Path to YAML configuration file
        verbose: Whether to display detailed output
        
    Returns:
        Evaluation results dictionary, or None if failed
    """
    try:
        # Load configuration file
        config = load_yaml_config(config_path)
        save_results_to_file.config_path = config_path
        
        # Create stacker from YAML configuration
        print(f"\n🔄 Creating stacker from YAML config...")
        stacker = ModelStacker.from_yaml_config(config_path)
        
        if verbose:
            print(f"✓ Stacker created with {len(stacker.base_models)} models")
            print(f"✓ Stacking method: {stacker.stacking_method}")
            print(f"✓ Task type: {stacker.task_type}")
        
        # Auto evaluation (if configured)
        evaluation_config = config.get('evaluation', {})
        if evaluation_config.get('auto_evaluate', False):
            print(f"\n📊 Running automatic evaluation...")
            
            try:
                results = stacker.evaluate(auto_load=True, evaluate_both_sets=True)
                
                if verbose:
                    # Check if it's dual dataset evaluation mode
                    if isinstance(results, dict) and 'validation' in results and 'test' in results:
                        # Display validation and test results separately
                        print(f"\n=== Validation Dataset Results ===")
                        val_results = results['validation']
                        if stacker.task_type == 'regression':
                            print(f"Validation RMSE: {val_results.get('rmse', 'N/A'):.4f}")
                            print(f"Validation R²: {val_results.get('r2', 'N/A'):.4f}")
                            print(f"Validation MAE: {val_results.get('mae', 'N/A'):.4f}")
                        else:
                            print(f"Validation Accuracy: {val_results.get('accuracy', 'N/A'):.4f}")
                            print(f"Validation F1: {val_results.get('f1', 'N/A'):.4f}")
                        
                        print(f"\n=== Test Dataset Results ===")
                        test_results = results['test']
                        if stacker.task_type == 'regression':
                            print(f"Test RMSE: {test_results.get('rmse', 'N/A'):.4f}")
                            print(f"Test R²: {test_results.get('r2', 'N/A'):.4f}")
                            print(f"Test MAE: {test_results.get('mae', 'N/A'):.4f}")
                        else:
                            print(f"Test Accuracy: {test_results.get('accuracy', 'N/A'):.4f}")
                            print(f"Test F1: {test_results.get('f1', 'N/A'):.4f}")
                        
                        # Display main metrics
                        print(f"\n=== Summary ===")
                        if stacker.task_type == 'regression':
                            print(f"Validation vs Test RMSE: {val_results.get('rmse', 'N/A'):.4f} vs {test_results.get('rmse', 'N/A'):.4f}")
                        else:
                            print(f"Validation vs Test Accuracy: {val_results.get('accuracy', 'N/A'):.4f} vs {test_results.get('accuracy', 'N/A'):.4f}")
                        
                        # Display base model performance comparison
                        if 'base_model_performance' in val_results:
                            print(f"\n=== Base Model Performance (Validation) ===")
                            for model_name, perf in val_results['base_model_performance'].items():
                                if stacker.task_type == 'regression':
                                    print(f"{model_name}: RMSE={perf.get('rmse', 'N/A'):.4f}, R²={perf.get('r2', 'N/A'):.4f}")
                                else:
                                    print(f"{model_name}: Accuracy={perf.get('accuracy', 'N/A'):.4f}, F1={perf.get('f1', 'N/A'):.4f}")
                        
                        if 'base_model_performance' in test_results:
                            print(f"\n=== Base Model Performance (Test) ===")
                            for model_name, perf in test_results['base_model_performance'].items():
                                if stacker.task_type == 'regression':
                                    print(f"{model_name}: RMSE={perf.get('rmse', 'N/A'):.4f}, R²={perf.get('r2', 'N/A'):.4f}")
                                else:
                                    print(f"{model_name}: Accuracy={perf.get('accuracy', 'N/A'):.4f}, F1={perf.get('f1', 'N/A'):.4f}")
                    else:
                        # Original single dataset display logic
                        # Display main metrics
                        print(f"\n=== Evaluation Results ===")
                        if stacker.task_type == 'regression':
                            print(f"RMSE: {results.get('rmse', 'N/A'):.4f}")
                            print(f"R²: {results.get('r2', 'N/A'):.4f}")
                            print(f"MAE: {results.get('mae', 'N/A'):.4f}")
                        else:
                            print(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}")
                            print(f"F1: {results.get('f1', 'N/A'):.4f}")
                            print(f"Precision: {results.get('precision', 'N/A'):.4f}")
                            print(f"Recall: {results.get('recall', 'N/A'):.4f}")
                        
                        # Display base model performance comparison
                        if 'base_model_performance' in results:
                            print(f"\n=== Base Model Performance ===")
                            for model_name, perf in results['base_model_performance'].items():
                                if stacker.task_type == 'regression':
                                    print(f"{model_name}: RMSE={perf.get('rmse', 'N/A'):.4f}, R²={perf.get('r2', 'N/A'):.4f}")
                                else:
                                    print(f"{model_name}: Accuracy={perf.get('accuracy', 'N/A'):.4f}, F1={perf.get('f1', 'N/A'):.4f}")
                
            except Exception as e:
                print(f"⚠️ Evaluation failed: {e}")
                results = {'error': str(e)}
        else:
            print("⚠️ Auto evaluation disabled in configuration")
            results = {'message': 'Stacker created successfully, but evaluation not performed'}
        
        # Save stacker (if configured)
        save_config = config.get('save', {})
        if save_config.get('save_stacker', False):
            save_path = save_config.get('stacker_path', 'output/stacking_models/stacker.pkl')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            stacker.save(save_path)
            print(f"✓ Stacker saved to: {save_path}")
        
        # Save results
        if evaluation_config.get('save_results', True):
            save_results_to_file(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Error running stacking workflow: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_example_config(output_path: str = "config_stacking_example.yaml"):
    """Create example configuration file"""
    
    example_config = {
        'stacking': {
            'experiment_dir': 'output/my_experiment_20240101_120000',
            'method': 'weighted_average',  # simple_average, weighted_average, ridge, rf, logistic
            'models': [
                {
                    'name': 'xgb',
                    'weight': 1.5,
                    'enabled': True
                },
                {
                    'name': 'lgbm', 
                    'weight': 1.2,
                    'enabled': True
                },
                {
                    'name': 'catboost',
                    'weight': 1.0,
                    'enabled': True
                },
                {
                    'name': 'rf',
                    'weight': 0.8,
                    'enabled': False  # This model will be skipped
                }
            ],
            # Meta-model configuration (optional, for ridge/rf/logistic methods)
            'meta_model': {
                'auto_train': True,
                'validation': {
                    'auto_load': True,
                    'size': 200,
                    'split_aware': True  # Use split-aware validation data loading
                }
            }
        },
        'evaluation': {
            'auto_evaluate': True,
            'save_results': True,
            'evaluate_both_sets': True  # Evaluate both validation and test sets
        },
        'save': {
            'save_stacker': True,
            'stacker_path': 'output/stacking_models/my_stacker.pkl'
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(example_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ Example configuration saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="CHEMIA Model Stacking YAML Configuration Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Basic Usage
  python stacking_yaml_demo.py --config config_stacking.yaml
  
  # Create Example Configuration File
  python stacking_yaml_demo.py --create-sample-config
  
  # Create Example Configuration File with Specific Output Path
  python stacking_yaml_demo.py --create-sample-config --output my_config.yaml

Notes:
  1. Ensure the experiment directory has trained models
  2. Model names in the configuration file must match actual model files
  3. Meta-learners will automatically train if validation data is needed
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='YAML Configuration File Path'
    )
    
    parser.add_argument(
        '--create-sample-config',
        action='store_true',
        help='Create Example Configuration File'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='config_stacking_meta.yaml',
        help='Example Configuration File Output Path (Default: config_stacking_meta.yaml)'
    )
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_example_config(args.output)
        return
    
    if not args.config:
        print("❌ Please specify configuration file path or use --create-sample-config to create example configuration")
        parser.print_help()
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"❌ Configuration file does not exist: {args.config}")
        sys.exit(1)
    
    # Run model stacking
    results = run_stacking_from_yaml(args.config)
    
    if results:
        print("\n✅ Stacking workflow completed!")
    else:
        print("\n❌ Stacking workflow failed!")

if __name__ == "__main__":
    main() 