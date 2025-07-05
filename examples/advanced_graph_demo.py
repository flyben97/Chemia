#!/usr/bin/env python3
"""
Advanced Graph Construction Demo Script

This script demonstrates the advanced graph construction features of the CHEMIA framework,
including custom node features, edge features, and graph augmentation techniques.
"""

import sys
import os
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.advanced_graph_builder import AdvancedGraphBuilder, GraphConstructionMode
    from utils.smiles_to_graph import SmilesGraphConverter
    DEMO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced graph features not available: {e}")
    DEMO_AVAILABLE = False

console = Console(width=120)

def create_demo_data():
    """Create synthetic reaction data for demonstration"""
    np.random.seed(42)
    
    # Sample SMILES for demonstration
    catalysts = [
        "CC(C)P(c1ccccc1)c1ccccc1",  # Phosphine catalyst
        "CCN(CC)CC",                  # Amine base
        "Cc1cccc(C)c1P(C(C)(C)C)C(C)(C)C",  # Bulky phosphine
    ]
    
    reactants_1 = [
        "CC(=O)c1ccccc1",            # Acetophenone
        "CC(=O)c1ccc(C)cc1",         # p-Methylacetophenone  
        "CC(=O)c1ccc(F)cc1",         # p-Fluoroacetophenone
    ]
    
    reactants_2 = [
        "NCc1ccccc1",                # Benzylamine
        "NCCc1ccccc1",               # Phenethylamine
        "NC(C)c1ccccc1",             # α-Methylbenzylamine
    ]
    
    products = [
        "CC(NCc1ccccc1)c1ccccc1",    # N-Benzyl-1-phenylethylamine
        "CC(NCCc1ccccc1)c1ccc(C)cc1", # Product 2
        "CC(NC(C)c1ccccc1)c1ccc(F)cc1", # Product 3
    ]
    
    solvents = ["CCO", "CC(C)O", "CCCC"]  # Ethanol, Isopropanol, Butane
    
    # Generate reaction dataset
    n_samples = 50
    data = []
    
    for i in range(n_samples):
        # Select random components
        catalyst = np.random.choice(catalysts)
        reactant1 = np.random.choice(reactants_1)
        reactant2 = np.random.choice(reactants_2) 
        product = np.random.choice(products)
        solvent = np.random.choice(solvents)
        
        # Generate experimental conditions
        temperature = np.random.normal(80, 20)  # °C
        pressure = np.random.normal(2.0, 0.5)   # bar
        time = np.random.exponential(24)        # hours
        catalyst_loading = np.random.normal(5, 2)  # mol%
        solvent_polarity = np.random.random()   # 0-1 scale
        
        # Generate synthetic yield (for demo purposes)
        # Higher temperature and longer time generally increase yield
        base_yield = 50 + temperature * 0.3 + time * 0.5 + catalyst_loading * 2
        noise = np.random.normal(0, 10)
        yield_val = max(0, min(100, base_yield + noise))
        
        data.append({
            'catalyst_smiles': catalyst,
            'reactant_1_smiles': reactant1,
            'reactant_2_smiles': reactant2,
            'product_smiles': product,
            'solvent_smiles': solvent,
            'temperature': temperature,
            'pressure': pressure,
            'reaction_time': time,
            'catalyst_loading': catalyst_loading,
            'solvent_polarity': solvent_polarity,
            'yield': yield_val
        })
    
    return pd.DataFrame(data)

def demo_graph_construction_modes():
    """Demonstrate different graph construction modes"""
    
    if not DEMO_AVAILABLE:
        console.print("[red]Demo not available - missing dependencies[/red]")
        return
    
    console.print(Panel.fit("🧪 Advanced Graph Construction Demo", style="bold blue"))
    
    # Create demo data
    console.print("\n[dim]Creating synthetic reaction dataset...[/dim]")
    df = create_demo_data()
    console.print(f"Generated {len(df)} reaction samples")
    
    # Sample data for demonstration
    sample = df.iloc[0]
    smiles_dict = {
        'catalyst_smiles': sample['catalyst_smiles'],
        'reactant_1_smiles': sample['reactant_1_smiles'], 
        'reactant_2_smiles': sample['reactant_2_smiles'],
        'product_smiles': sample['product_smiles']
    }
    
    custom_features = {
        'temperature': sample['temperature'],
        'pressure': sample['pressure'],
        'reaction_time': sample['reaction_time'],
        'catalyst_loading': sample['catalyst_loading'],
        'solvent_polarity': sample['solvent_polarity']
    }
    
    molecule_roles = {
        'catalyst_smiles': 'catalyst',
        'reactant_1_smiles': 'reactant',
        'reactant_2_smiles': 'reactant', 
        'product_smiles': 'product'
    }
    
    # Demo each construction mode
    modes_to_demo = [
        (GraphConstructionMode.BATCH, "Traditional batch mode"),
        (GraphConstructionMode.FEATURE_CONCAT, "Feature concatenation mode"),
        (GraphConstructionMode.REACTION_GRAPH, "Reaction graph mode"),
        (GraphConstructionMode.CUSTOM_FUSION, "Custom feature fusion mode")
    ]
    
    results_table = Table(title="Graph Construction Results", show_header=True, header_style="bold magenta")
    results_table.add_column("Mode", style="cyan", width=20)
    results_table.add_column("Description", style="white", width=35)
    results_table.add_column("Nodes", style="green", justify="center", width=10)
    results_table.add_column("Edges", style="yellow", justify="center", width=10)
    results_table.add_column("Features", style="blue", justify="center", width=12)
    results_table.add_column("Special", style="red", width=25)
    
    for mode, description in modes_to_demo:
        try:
            console.print(f"\n[bold]Testing {mode.value} mode...[/bold]")
            
            # Configure builder for this mode
            if mode == GraphConstructionMode.CUSTOM_FUSION:
                fusion_config = {
                    'fusion_method': 'attention',
                    'custom_feature_dim': 5,
                    'graph_embed_dim': 128,
                    'output_dim': 256
                }
            else:
                fusion_config = None
            
            builder = AdvancedGraphBuilder(
                construction_mode=mode,
                custom_fusion_config=fusion_config
            )
            
            # Build graph
            if mode == GraphConstructionMode.CUSTOM_FUSION:
                graph = builder.build_graphs(smiles_dict, custom_features, molecule_roles)
            else:
                graph = builder.build_graphs(smiles_dict, molecule_roles=molecule_roles)
            
            # Extract graph info
            if hasattr(graph, 'x') and graph.x is not None:
                num_nodes = graph.x.shape[0]
                num_features = graph.x.shape[1]
            else:
                num_nodes = 0
                num_features = 0
            
            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                num_edges = graph.edge_index.shape[1]
            else:
                num_edges = 0
            
            # Special properties
            special = []
            if hasattr(graph, 'is_feature_concat'):
                special.append("Feature concat")
            if hasattr(graph, 'is_reaction_graph'):
                special.append("Reaction center")
            if hasattr(graph, 'is_custom_fusion'):
                special.append("Custom fusion")
            if hasattr(graph, 'reaction_metadata'):
                special.append(f"RC at {graph.reaction_metadata['reaction_center_idx']}")
            
            special_str = ", ".join(special) if special else "Standard"
            
            results_table.add_row(
                mode.value,
                description,
                str(num_nodes),
                str(num_edges),
                str(num_features),
                special_str
            )
            
            # Mode-specific information
            if mode == GraphConstructionMode.FEATURE_CONCAT:
                if hasattr(graph, 'metadata'):
                    console.print(f"  Feature ranges: {graph.metadata['feature_ranges']}")
            
            elif mode == GraphConstructionMode.REACTION_GRAPH:
                if hasattr(graph, 'reaction_metadata'):
                    console.print(f"  Molecule boundaries: {graph.reaction_metadata['molecule_boundaries']}")
                    console.print(f"  Reaction center at node: {graph.reaction_metadata['reaction_center_idx']}")
            
            elif mode == GraphConstructionMode.CUSTOM_FUSION:
                if hasattr(graph, 'has_custom_features'):
                    console.print(f"  Has custom features: {graph.has_custom_features}")
                if hasattr(graph, 'custom_features'):
                    console.print(f"  Custom feature shape: {graph.custom_features.shape}")
            
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            results_table.add_row(
                mode.value,
                description,
                "Error",
                "Error", 
                "Error",
                str(e)[:20] + "..."
            )
    
    console.print("\n")
    console.print(results_table)

def demo_fusion_methods():
    """Demonstrate different fusion methods for custom features"""
    
    if not DEMO_AVAILABLE:
        return
    
    console.print("\n" + "="*80)
    console.print(Panel.fit("🔗 Custom Feature Fusion Methods Demo", style="bold green"))
    
    fusion_methods = ['concatenate', 'attention', 'gated', 'transformer']
    
    fusion_table = Table(title="Fusion Methods Comparison", show_header=True, header_style="bold magenta")
    fusion_table.add_column("Method", style="cyan", width=15)
    fusion_table.add_column("Parameters", style="yellow", width=15)
    fusion_table.add_column("Complexity", style="blue", width=12)
    fusion_table.add_column("Best For", style="green", width=35)
    
    method_info = {
        'concatenate': {
            'params': 'Linear layer',
            'complexity': 'Low',
            'best_for': 'Simple baseline, fast prototyping'
        },
        'attention': {
            'params': 'Multi-head attn',
            'complexity': 'Medium',
            'best_for': 'Interpretable adaptive fusion'
        },
        'gated': {
            'params': 'Gating network',
            'complexity': 'Medium',
            'best_for': 'Selective feature usage'
        },
        'transformer': {
            'params': 'Full transformer',
            'complexity': 'High',
            'best_for': 'Complex feature interactions'
        }
    }
    
    for method in fusion_methods:
        info = method_info[method]
        fusion_table.add_row(
            method.title(),
            info['params'],
            info['complexity'],
            info['best_for']
        )
    
    console.print(fusion_table)

def demo_use_cases():
    """Show example use cases for each mode"""
    
    console.print("\n" + "="*80)
    console.print(Panel.fit("🎯 Use Case Recommendations", style="bold yellow"))
    
    use_cases = [
        {
            'mode': 'Batch Mode',
            'use_case': 'Single molecule property prediction',
            'example': 'Drug solubility, toxicity prediction',
            'pros': 'Simple, fast, well-tested',
            'cons': 'No multi-molecule interactions'
        },
        {
            'mode': 'Feature Concat',
            'use_case': 'Traditional ML with GNN features',
            'example': 'Catalyst screening with molecular descriptors',
            'pros': 'Fast, interpretable, ML-compatible',
            'cons': 'Loses graph structure'
        },
        {
            'mode': 'Reaction Graph',
            'use_case': 'Chemical reaction modeling',
            'example': 'Yield prediction, reaction optimization',
            'pros': 'Models reaction as system, inter-molecular interactions',
            'cons': 'Complex, requires role annotation'
        },
        {
            'mode': 'Custom Fusion',
            'use_case': 'Multi-modal prediction tasks',
            'example': 'Reaction conditions + molecular structure',
            'pros': 'Combines heterogeneous data, flexible',
            'cons': 'Requires additional features'
        }
    ]
    
    for case in use_cases:
        console.print(f"\n[bold cyan]{case['mode']}[/bold cyan]")
        console.print(f"  Use case: {case['use_case']}")
        console.print(f"  Example: {case['example']}")
        console.print(f"  [green]Pros: {case['pros']}[/green]")
        console.print(f"  [red]Cons: {case['cons']}[/red]")

def main():
    """Run the complete demonstration"""
    
    console.print(Panel.fit("🚀 CHEMIA Advanced Graph Construction Demo", style="bold blue"))
    console.print("\nThis demo showcases the advanced graph construction capabilities of CHEMIA framework.")
    console.print("We'll demonstrate 4 different modes for handling multiple SMILES inputs.\n")
    
    # Check if demo is available
    if not DEMO_AVAILABLE:
        console.print("[red]❌ Demo cannot run - missing required dependencies[/red]")
        console.print("Please install: pip install torch torch-geometric rdkit-pypi")
        return
    
    try:
        # Run demos
        demo_graph_construction_modes()
        demo_fusion_methods() 
        demo_use_cases()
        
        console.print("\n" + "="*80)
        console.print(Panel.fit("✅ Demo Complete!", style="bold green"))
        console.print("\nNext steps:")
        console.print("1. Try the example configurations in examples/configs/")
        console.print("2. Run training with: python run_training_only.py --config examples/configs/advanced_graph_features.yaml")
        console.print("3. Explore the different graph construction modes for your use case")
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    main() 