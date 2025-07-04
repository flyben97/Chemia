#!/usr/bin/env python3
"""
INTERNCRANE GNN Workflow Demonstration

This script demonstrates how to use Graph Neural Networks for molecular
property prediction using SMILES input in the INTERNCRANE framework.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- SUPPRESS DEBUG LOGS AND WARNINGS ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress graphviz deprecation warnings
warnings.filterwarnings("ignore", message=".*positional args.*")

# Configure logging to suppress DEBUG messages
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('graphviz').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('rdkit').setLevel(logging.WARNING)

# Set root logger to INFO level to avoid DEBUG spam
logging.basicConfig(level=logging.INFO)

def create_demo_data():
    """Create demonstration molecular data with SMILES and properties"""
    
    # Sample SMILES strings and their corresponding properties
    demo_molecules = [
        # Simple alcohols
        ("CCO", 0.5, "ethanol"),
        ("CCCO", 0.6, "propanol"),
        ("CCCCO", 0.7, "butanol"),
        
        # Aromatics
        ("c1ccccc1", 0.3, "benzene"),
        ("Cc1ccccc1", 0.4, "toluene"),
        ("CCc1ccccc1", 0.45, "ethylbenzene"),
        
        # Acids
        ("CC(=O)O", 0.2, "acetic_acid"),
        ("CCC(=O)O", 0.25, "propionic_acid"),
        
        # Amines
        ("CN(C)C", 0.8, "trimethylamine"),
        ("CCN(CC)CC", 0.9, "triethylamine"),
        
        # Heterocycles
        ("c1cccnc1", 0.35, "pyridine"),
        ("c1ccncc1", 0.4, "pyrazine"),
        
        # Functional group combinations
        ("CCN(C)C(=O)c1ccccc1", 0.75, "complex_amide"),
        ("COc1ccc(CC(=O)O)cc1", 0.55, "substituted_acid"),
        ("Nc1ccc(C(=O)O)cc1", 0.65, "amino_acid")
    ]
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            'smiles': smiles,
            'property': prop,
            'name': name,
            'additional_feature': np.random.normal(0, 0.1)  # Random additional feature
        }
        for smiles, prop, name in demo_molecules
    ])
    
    return df


def demo_gnn_training():
    """Demonstrate GNN model training with SMILES data"""
    
    print("🧪 INTERNCRANE GNN Workflow Demonstration")
    print("=" * 50)
    
    # 1. Create demo data
    print("📊 Creating demonstration molecular dataset...")
    df = create_demo_data()
    print(f"   Dataset size: {len(df)} molecules")
    print(f"   Features: {list(df.columns)}")
    print(f"   Sample SMILES: {df['smiles'].iloc[0]}")
    
    # 2. Test SMILES to graph conversion
    print("\n🔬 Testing SMILES to graph conversion...")
    try:
        from utils.smiles_to_graph import SmilesGraphConverter
        
        converter = SmilesGraphConverter()
        feature_dims = converter.get_feature_dimensions()
        
        print(f"   Node features: {feature_dims['node_features']}")
        print(f"   Edge features: {feature_dims['edge_features']}")
        
        # Convert a few examples
        test_smiles = df['smiles'].iloc[:3].tolist()
        for smiles in test_smiles:
            graph = converter.smiles_to_graph(smiles)
            if graph is not None:
                print(f"   ✓ {smiles}: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # 3. Test GNN optimizer
    print("\n🤖 Testing GNN optimizer with simple training...")
    try:
        from optimizers.gnn_optimizer import GNNOptimizer
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        X = df[['smiles', 'additional_feature']]
        y = df['property'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Create GNN optimizer
        gnn_optimizer = GNNOptimizer(
            model_name='gcn',
            smiles_columns=['smiles'],
            n_trials=2,  # Very small for demo
            cv=2,        # Very small for demo
            task_type='regression',
            max_epochs=10,  # Very small for demo
            batch_size=4,   # Very small for demo
            early_stopping_patience=5
        )
        
        print("   ✓ Created GNN optimizer")
        
        # Quick optimization test (very limited)
        print("   Running quick hyperparameter optimization...")
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        best_params, best_score = gnn_optimizer.optimize(
            X_train_split, y_train_split, X_val_split, y_val_split
        )
        
        print(f"   ✓ Best validation score: {best_score:.4f}")
        print(f"   ✓ Best parameters: {list(best_params.keys())}")
        
        # Fit final model
        print("   Fitting final model...")
        gnn_optimizer.fit(X_train, y_train)
        
        # Make predictions
        y_pred = gnn_optimizer.predict(X_test)
        
        # Calculate simple metrics
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"   ✓ Test R²: {r2:.4f}")
        print(f"   ✓ Test RMSE: {rmse:.4f}")
        
        # Show some predictions
        print("\n   Sample predictions:")
        for i in range(min(5, len(X_test))):
            actual = y_test[i]
            predicted = y_pred[i]
            smiles = X_test.iloc[i]['smiles']
            print(f"     {smiles[:20]:<20} | Actual: {actual:.3f} | Predicted: {predicted:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error in GNN training: {e}")
        return False
    
    # 4. Summary
    print("\n🎉 GNN Workflow Demonstration Complete!")
    print("=" * 50)
    print("✅ SMILES to graph conversion: Working")
    print("✅ GNN model creation: Working")
    print("✅ Hyperparameter optimization: Working")
    print("✅ Model training and prediction: Working")
    
    print(f"\n📋 Available GNN Models:")
    available_models = ['gcn', 'gat', 'mpnn', 'afp', 'graph_transformer', 'ensemble_gnn']
    for model in available_models:
        print(f"   • {model.upper()}")
    
    print(f"\n📋 Key Features:")
    print("   • Direct SMILES input (no manual feature engineering)")
    print("   • Automatic graph conversion with rich node/edge features")
    print("   • Support for regression and classification tasks")
    print("   • Integrated hyperparameter optimization")
    print("   • Cross-validation support")
    print("   • Multiple GNN architectures available")
    
    return True


def show_usage_examples():
    """Show examples of how to use GNN models in configuration"""
    
    print("\n📚 Usage Examples")
    print("=" * 30)
    
    print("1. Basic GNN configuration (config.yaml):")
    print("""
task_type: "regression"

features:
  molecular:
    reactant:
      is_feature_source: true
      smiles_column: "reactant_smiles"
    product:
      is_feature_source: true  
      smiles_column: "product_smiles"

training:
  models_to_run: ["gcn", "gat", "mpnn"]
  n_trials: 50
  cv_folds: 5
""")
    
    print("2. Advanced GNN configuration:")
    print("""
training:
  models_to_run: ["afp", "gtn", "ensemble_gnn"]
  n_trials: 100
  cv_folds: 10

gnn_settings:
  max_epochs: 200
  early_stopping_patience: 20
  batch_size: 64
  device: "cuda"  # Use GPU if available
""")
    
    print("3. Running training:")
    print("   python run_training_only.py --config config_gnn.yaml")


if __name__ == "__main__":
    print("Starting CRAFT GNN Workflow Demonstration...")
    
    try:
        success = demo_gnn_training()
        if success:
            show_usage_examples()
        else:
            print("❌ Demo failed. Check error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 