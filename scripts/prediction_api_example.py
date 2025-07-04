#!/usr/bin/env python3
"""
CRAFT Prediction API Usage Examples

This file demonstrates how to use the prediction_api.py module for functional predictions.
Includes example code for various usage scenarios.
"""

import numpy as np
import pandas as pd
from prediction_api import load_model, predict, predict_single, quick_predict

def example_1_single_prediction():
    """Example 1: Predict single sample"""
    print("=" * 60)
    print("Example 1: Predict Single Sample")
    print("=" * 60)
    
    # Replace with your experiment directory path
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    model_name = "xgb"
    
    try:
        # Load model
        print("Loading model...")
        predictor = load_model(experiment_dir, model_name)
        print(f"✓ Model loaded successfully! Task type: {predictor.task_type}")
        
        # Prepare single sample data
        sample = {
            'SMILES': 'CCO',
            'Solvent_1_SMILES': 'CC(=O)O',
            'Solvent_2_SMILES': 'CCN',
            # If there are pre-computed features, include them too
            # 'feature_1': 1.2,
            # 'feature_2': 3.4,
        }
        
        # Method 1: Use predict_single to get single prediction value
        result = predict_single(predictor, sample)
        print(f"Prediction value: {result}")
        
        # Method 2: Use predict to get detailed results
        detailed_result = predict(predictor, sample)
        print(f"Detailed results: {detailed_result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure the experiment directory path is correct and contains trained models")

def example_2_batch_prediction():
    """Example 2: Batch prediction for multiple samples"""
    print("\n" + "=" * 60)
    print("Example 2: Batch Prediction for Multiple Samples")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    model_name = "xgb"
    
    try:
        # Load model
        predictor = load_model(experiment_dir, model_name)
        
        # Prepare multiple sample data
        samples = [
            {
                'SMILES': 'CCO',
                'Solvent_1_SMILES': 'CC(=O)O',
                'Solvent_2_SMILES': 'CCN',
            },
            {
                'SMILES': 'c1ccccc1',
                'Solvent_1_SMILES': 'CNC(=O)N',
                'Solvent_2_SMILES': 'CC',
            },
            {
                'SMILES': 'CCCC',
                'Solvent_1_SMILES': 'O',
                'Solvent_2_SMILES': 'CCO',
            }
        ]
        
        # Batch prediction
        results = predict(predictor, samples)
        
        print(f"Predicted {results['n_samples']} samples")
        print(f"Prediction results: {results['predictions']}")
        
        # For classification tasks, there will also be probabilities
        if results['probabilities'] is not None:
            print(f"Classification probabilities: {results['probabilities']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def example_3_dataframe_input():
    """Example 3: Use DataFrame as input"""
    print("\n" + "=" * 60)
    print("Example 3: Use DataFrame as Input")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    model_name = "xgb"
    
    try:
        # Load model
        predictor = load_model(experiment_dir, model_name)
        
        # Create DataFrame
        df = pd.DataFrame({
            'SMILES': ['CCO', 'c1ccccc1', 'CCCC'],
            'Solvent_1_SMILES': ['CC(=O)O', 'CNC(=O)N', 'O'],
            'Solvent_2_SMILES': ['CCN', 'CC', 'CCO'],
            # If there are pre-computed features
            # 'feature_1': [1.2, 2.1, 0.8],
            # 'feature_2': [3.4, 4.5, 2.1],
        })
        
        print("Input DataFrame:")
        print(df)
        
        # Prediction
        results = predict(predictor, df)
        
        print(f"\nPrediction results: {results['predictions']}")
        
        # Add prediction results to DataFrame
        df['predictions'] = results['predictions']
        print("\nDataFrame with prediction results:")
        print(df)
    
    except Exception as e:
        print(f"❌ Error: {e}")

def example_4_quick_predict():
    """Example 4: One-step prediction"""
    print("\n" + "=" * 60)
    print("Example 4: One-Step Prediction (No Need to Pre-load Model)")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    model_name = "xgb"
    
    try:
        sample = {
            'SMILES': 'CCO',
            'Solvent_1_SMILES': 'CC(=O)O',
            'Solvent_2_SMILES': 'CCN',
        }
        
        # One-step loading and prediction
        results = quick_predict(experiment_dir, model_name, sample)
        print(f"One-step prediction result: {results}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def example_5_function_integration():
    """Example 5: Integration into other functions"""
    print("\n" + "=" * 60)
    print("Example 5: Integration into Other Functions")
    print("=" * 60)
    
    def calculate_reaction_yield(smiles, solvent1, solvent2, temperature=25.0):
        """
        Function to calculate reaction yield
        This is an example showing how to integrate prediction into larger computational workflows
        """
        experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
        model_name = "xgb"
        
        try:
            # Prepare input data
            sample = {
                'SMILES': smiles,
                'Solvent_1_SMILES': solvent1,
                'Solvent_2_SMILES': solvent2,
                # 'temperature': temperature,  # If model needs temperature feature
            }
            
            # Prediction
            yield_prediction = quick_predict(experiment_dir, model_name, sample)
            
            # Extract prediction value
            predicted_yield = yield_prediction['predictions'][0]
            
            # Can add other computational logic here
            # For example: temperature correction, uncertainty estimation, etc.
            
            return {
                'predicted_yield': predicted_yield,
                'temperature': temperature,
                'confidence': 'high' if predicted_yield > 0.8 else 'medium'
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    # Use this function
    result = calculate_reaction_yield('CCO', 'CC(=O)O', 'CCN', 30.0)
    print(f"Reaction yield calculation result: {result}")

def example_6_model_comparison():
    """Example 6: Compare prediction results from different models"""
    print("\n" + "=" * 60)
    print("Example 6: Compare Prediction Results from Different Models")
    print("=" * 60)
    
    experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
    models_to_compare = ['xgb', 'lgbm', 'catboost']
    
    sample = {
        'SMILES': 'CCO',
        'Solvent_1_SMILES': 'CC(=O)O',
        'Solvent_2_SMILES': 'CCN',
    }
    
    print("Comparing prediction results from different models:")
    print("-" * 40)
    
    for model_name in models_to_compare:
        try:
            result = quick_predict(experiment_dir, model_name, sample)
            prediction = result['predictions'][0]
            print(f"{model_name.upper()}: {prediction:.4f}")
        except Exception as e:
            print(f"{model_name.upper()}: Unable to predict ({e})")

def example_7_error_handling():
    """Example 7: Error handling examples"""
    print("\n" + "=" * 60)
    print("Example 7: Error Handling Examples")
    print("=" * 60)
    
    # Intentionally use wrong path to demonstrate error handling
    wrong_experiment_dir = "output/nonexistent_experiment"
    
    try:
        predictor = load_model(wrong_experiment_dir, "xgb")
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except Exception as e:
        print(f"Other error: {e}")
    
    # Demonstrate data format error
    try:
        experiment_dir = "output/S04_agent_5_a_regression_20250101_120000"
        predictor = load_model(experiment_dir, "xgb")
        
        # Wrong data format
        wrong_data = "this is not valid data"
        result = predict(predictor, wrong_data)  # type: ignore
    except Exception as e:
        print(f"Data format error: {e}")

if __name__ == "__main__":
    print("INTERNCRANE Prediction API Usage Examples")
    print("Please ensure you modify the experiment directory path to your actual path")
    print()
    
    # Run all examples
    example_1_single_prediction()
    example_2_batch_prediction()
    example_3_dataframe_input()
    example_4_quick_predict()
    example_5_function_integration()
    example_6_model_comparison()
    example_7_error_handling()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60) 