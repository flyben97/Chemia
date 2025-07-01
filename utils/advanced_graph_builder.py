"""
Advanced Graph Builder for Multiple SMILES Processing

This module provides advanced graph construction methods for handling multiple
SMILES inputs in chemical reaction prediction, including feature concatenation,
reaction graph construction, and custom feature fusion.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any, TYPE_CHECKING
import logging
from enum import Enum

if TYPE_CHECKING:
    # Type checking imports - only for static analysis
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import add_self_loops, to_dense_batch
else:
    # Runtime imports with fallbacks
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.data import Data, Batch
        from torch_geometric.utils import add_self_loops, to_dense_batch
    except ImportError:
        torch = None
        nn = None
        F = None
        Data = None
        Batch = None
        logging.warning("PyTorch Geometric not installed. Advanced graph functionality will not be available.")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    Descriptors = None
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not installed. SMILES processing will not be available.")

from .smiles_to_graph import SmilesGraphConverter


class GraphConstructionMode(Enum):
    """Graph construction modes for multiple SMILES"""
    BATCH = "batch"                    # Original: separate graphs batched together
    FEATURE_CONCAT = "feature_concat"  # Feature-level concatenation
    REACTION_GRAPH = "reaction_graph"  # Unified reaction graph
    CUSTOM_FUSION = "custom_fusion"    # Custom feature + graph embedding fusion


class AdvancedGraphBuilder:
    """Advanced graph builder supporting multiple construction modes"""
    
    def __init__(self, 
                 base_converter: Optional[SmilesGraphConverter] = None,
                 construction_mode: Union[str, GraphConstructionMode] = GraphConstructionMode.BATCH,
                 custom_fusion_config: Optional[Dict[str, Any]] = None):
        """
        Initialize advanced graph builder
        
        Args:
            base_converter: Base SMILES to graph converter
            construction_mode: Graph construction mode
            custom_fusion_config: Configuration for custom fusion mode
        """
        if torch is None or Data is None:
            raise ImportError("PyTorch Geometric is required for advanced graph building")
        
        self.base_converter = base_converter or SmilesGraphConverter()
        
        if isinstance(construction_mode, str):
            self.construction_mode = GraphConstructionMode(construction_mode)
        else:
            self.construction_mode = construction_mode
            
        self.custom_fusion_config = custom_fusion_config or {}
        
        # Initialize mode-specific components
        self._init_mode_components()
    
    def _init_mode_components(self):
        """Initialize components specific to construction mode"""
        if self.construction_mode == GraphConstructionMode.REACTION_GRAPH:
            self._init_reaction_graph_components()
        elif self.construction_mode == GraphConstructionMode.CUSTOM_FUSION:
            self._init_custom_fusion_components()
    
    def _init_reaction_graph_components(self):
        """Initialize reaction graph construction components"""
        # Virtual node types for reaction graph
        self.reaction_node_types = {
            'reactant': 0,
            'product': 1,
            'catalyst': 2,
            'solvent': 3,
            'reaction_center': 4  # Virtual reaction center node
        }
        
        # Virtual edge types for reaction connections
        self.reaction_edge_types = {
            'molecular_bond': 0,      # Normal chemical bonds
            'reaction_participates': 1, # Molecule participates in reaction
            'catalyzes': 2,           # Catalyst relationship
            'solvent_interaction': 3,  # Solvent interaction
            'reaction_flow': 4        # Reaction pathway
        }
    
    def _init_custom_fusion_components(self):
        """Initialize custom fusion components"""
        fusion_config = self.custom_fusion_config
        
        # Fusion method
        self.fusion_method = fusion_config.get('fusion_method', 'concatenate')
        # Options: 'concatenate', 'attention', 'gated', 'transformer'
        
        # Dimension settings
        self.custom_feature_dim = fusion_config.get('custom_feature_dim', 10)
        self.graph_embed_dim = fusion_config.get('graph_embed_dim', 128)
        self.output_dim = fusion_config.get('output_dim', 256)
        
        # Initialize fusion networks if needed
        if self.fusion_method in ['attention', 'gated', 'transformer']:
            self._init_fusion_networks()
    
    def _init_fusion_networks(self):
        """Initialize neural networks for advanced fusion"""
        if self.fusion_method == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=self.graph_embed_dim,
                num_heads=4,
                batch_first=True
            )
        elif self.fusion_method == 'gated':
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.custom_feature_dim + self.graph_embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.graph_embed_dim),
                nn.Sigmoid()
            )
        elif self.fusion_method == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.graph_embed_dim,
                nhead=8,
                batch_first=True
            )
            self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    def build_graphs(self, 
                    smiles_dict: Dict[str, str],
                    custom_features: Optional[Dict[str, Any]] = None,
                    molecule_roles: Optional[Dict[str, str]] = None) -> Any:
        """
        Build graph according to construction mode
        
        Args:
            smiles_dict: Dictionary mapping column names to SMILES
            custom_features: Additional custom features for fusion
            molecule_roles: Role of each molecule (reactant, product, catalyst, etc.)
            
        Returns:
            PyTorch Geometric Data object
        """
        if self.construction_mode == GraphConstructionMode.BATCH:
            return self._build_batch_graph(smiles_dict)
        elif self.construction_mode == GraphConstructionMode.FEATURE_CONCAT:
            return self._build_feature_concat_graph(smiles_dict)
        elif self.construction_mode == GraphConstructionMode.REACTION_GRAPH:
            return self._build_reaction_graph(smiles_dict, molecule_roles)
        elif self.construction_mode == GraphConstructionMode.CUSTOM_FUSION:
            return self._build_custom_fusion_graph(smiles_dict, custom_features)
        else:
            raise ValueError(f"Unknown construction mode: {self.construction_mode}")
    
    def _build_batch_graph(self, smiles_dict: Dict[str, str]) -> Any:
        """Build traditional batched graph (original method)"""
        graphs = []
        for col_name, smiles in smiles_dict.items():
            if smiles and not pd.isna(smiles):
                graph = self.base_converter.smiles_to_graph(smiles)
                if graph is not None:
                    # Add metadata
                    graph.molecule_role = col_name
                    graphs.append(graph)
        
        if len(graphs) == 1:
            return graphs[0]
        elif len(graphs) > 1:
            return Batch.from_data_list(graphs)
        else:
            # Empty graph
            return self._create_empty_graph()
    
    def _build_feature_concat_graph(self, smiles_dict: Dict[str, str]) -> Any:
        """Build graph using feature-level concatenation"""
        all_features = []
        metadata = {'molecule_names': [], 'feature_ranges': []}
        
        start_idx = 0
        for col_name, smiles in smiles_dict.items():
            if smiles and not pd.isna(smiles):
                graph = self.base_converter.smiles_to_graph(smiles)
                if graph is not None:
                    # Extract molecular-level features instead of full graph
                    mol_features = self._extract_molecular_features(smiles)
                    all_features.append(mol_features)
                    
                    metadata['molecule_names'].append(col_name)
                    end_idx = start_idx + len(mol_features)
                    metadata['feature_ranges'].append((start_idx, end_idx))
                    start_idx = end_idx
        
        if all_features:
            # Concatenate all molecular features
            concatenated_features = torch.cat(all_features, dim=0).unsqueeze(0)  # [1, total_features]
            
            # Create a single-node graph with concatenated features
            graph = Data(
                x=concatenated_features,
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=None
            )
            graph.metadata = metadata
            graph.is_feature_concat = True
            return graph
        else:
            return self._create_empty_graph()
    
    def _build_reaction_graph(self, 
                             smiles_dict: Dict[str, str],
                             molecule_roles: Optional[Dict[str, str]] = None) -> Any:
        """Build unified reaction graph with virtual reaction center"""
        if molecule_roles is None:
            # Infer roles from column names
            molecule_roles = self._infer_molecule_roles(smiles_dict)
        
        all_nodes = []
        all_edges = []
        all_edge_types = []
        node_offset = 0
        reaction_metadata = {
            'molecule_boundaries': {},
            'reaction_center_idx': None,
            'molecule_roles': molecule_roles
        }
        
        # Step 1: Add molecular graphs
        for col_name, smiles in smiles_dict.items():
            if smiles and not pd.isna(smiles):
                graph = self.base_converter.smiles_to_graph(smiles)
                if graph is not None:
                    num_nodes = graph.x.size(0)
                    
                    # Add molecular nodes
                    all_nodes.append(graph.x)
                    
                    # Add molecular edges (adjust indices)
                    if graph.edge_index.size(1) > 0:
                        adjusted_edges = graph.edge_index + node_offset
                        all_edges.append(adjusted_edges)
                        # Mark as molecular bonds
                        num_edges = adjusted_edges.size(1)
                        all_edge_types.extend([self.reaction_edge_types['molecular_bond']] * num_edges)
                    
                    # Store molecule boundaries
                    reaction_metadata['molecule_boundaries'][col_name] = (node_offset, node_offset + num_nodes)
                    node_offset += num_nodes
        
        # Step 2: Add virtual reaction center node
        if all_nodes:
            # Create reaction center features (average of all molecular features)
            mol_features = torch.cat(all_nodes, dim=0)
            reaction_center_features = mol_features.mean(dim=0, keepdim=True)
            all_nodes.append(reaction_center_features)
            
            reaction_center_idx = node_offset
            reaction_metadata['reaction_center_idx'] = reaction_center_idx
            
            # Step 3: Connect molecules to reaction center
            for col_name, (start_idx, end_idx) in reaction_metadata['molecule_boundaries'].items():
                role = molecule_roles.get(col_name, 'reactant')
                edge_type = self._get_reaction_edge_type(role)
                
                # Connect all atoms in molecule to reaction center
                for atom_idx in range(start_idx, end_idx):
                    # Bidirectional connection
                    reaction_edges = torch.tensor([[atom_idx, reaction_center_idx],
                                                 [reaction_center_idx, atom_idx]], dtype=torch.long).t()
                    all_edges.append(reaction_edges)
                    all_edge_types.extend([edge_type] * 2)
        
        # Combine all components
        if all_nodes:
            x = torch.cat(all_nodes, dim=0)
            edge_index = torch.cat(all_edges, dim=1) if all_edges else torch.empty(2, 0, dtype=torch.long)
            edge_types = torch.tensor(all_edge_types, dtype=torch.long) if all_edge_types else torch.empty(0, dtype=torch.long)
            
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_types.float().unsqueeze(-1))
            graph.reaction_metadata = reaction_metadata
            graph.is_reaction_graph = True
            return graph
        else:
            return self._create_empty_graph()
    
    def _build_custom_fusion_graph(self, 
                                  smiles_dict: Dict[str, str],
                                  custom_features: Optional[Dict[str, Any]] = None) -> Any:
        """Build graph with custom feature fusion capability"""
        # First build base graph
        base_graph = self._build_batch_graph(smiles_dict)
        
        # Add custom features if provided
        if custom_features:
            # Process custom features
            processed_features = self._process_custom_features(custom_features)
            base_graph.custom_features = processed_features
            base_graph.has_custom_features = True
        else:
            base_graph.has_custom_features = False
        
        base_graph.is_custom_fusion = True
        base_graph.fusion_config = self.custom_fusion_config
        return base_graph
    
    def _extract_molecular_features(self, smiles: str) -> Any:
        """Extract molecular-level features from SMILES"""
        if Chem is None or Descriptors is None or torch is None:
            raise RuntimeError("RDKit and PyTorch are required for molecular feature extraction")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Return zero features for invalid SMILES
            return torch.zeros(50)  # Standard molecular feature size
        
        features = []
        
        # Basic molecular descriptors
        features.extend([
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            mol.GetNumHeavyAtoms(),
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol)
        ])
        
        # Ring descriptors - use basic RDKit functionality
        try:
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            
            # Count aromatic rings
            aromatic_rings = 0
            for ring in ring_info.AtomRings():
                if len(ring) > 0 and mol.GetAtomWithIdx(ring[0]).GetIsAromatic():
                    aromatic_rings += 1
            
            aliphatic_rings = num_rings - aromatic_rings
            
            features.extend([
                num_rings,
                aromatic_rings,
                aliphatic_rings
            ])
        except Exception:
            # Fallback
            features.extend([0.0, 0.0, 0.0])
        
        # Extended descriptors
        try:
            features.extend([
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCsp3(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.BertzCT(mol)
            ])
        except:
            # Fallback if some descriptors fail
            features.extend([0.0] * 6)
        
        # Pad to standard size
        while len(features) < 50:
            features.append(0.0)
        
        return torch.tensor(features[:50], dtype=torch.float)
    
    def _infer_molecule_roles(self, smiles_dict: Dict[str, str]) -> Dict[str, str]:
        """Infer molecule roles from column names"""
        roles = {}
        for col_name in smiles_dict.keys():
            col_lower = col_name.lower()
            if 'catalyst' in col_lower or 'cat' in col_lower:
                roles[col_name] = 'catalyst'
            elif 'product' in col_lower or 'prod' in col_lower:
                roles[col_name] = 'product'
            elif 'solvent' in col_lower or 'solv' in col_lower:
                roles[col_name] = 'solvent'
            elif 'ligand' in col_lower:
                roles[col_name] = 'catalyst'  # Ligands are part of catalyst system
            else:
                roles[col_name] = 'reactant'  # Default
        return roles
    
    def _get_reaction_edge_type(self, role: str) -> int:
        """Get edge type based on molecule role"""
        if role == 'catalyst':
            return self.reaction_edge_types['catalyzes']
        elif role == 'solvent':
            return self.reaction_edge_types['solvent_interaction']
        else:  # reactant, product
            return self.reaction_edge_types['reaction_participates']
    
    def _process_custom_features(self, custom_features: Dict[str, Any]) -> torch.Tensor:
        """Process custom features into tensor format"""
        processed = []
        
        for key, value in custom_features.items():
            if isinstance(value, (int, float)):
                processed.append(float(value))
            elif isinstance(value, (list, tuple)):
                processed.extend([float(v) for v in value])
            elif hasattr(value, 'item'):  # numpy scalar
                processed.append(float(value.item()))
            elif hasattr(value, 'tolist'):  # numpy array
                processed.extend([float(v) for v in value.tolist()])
            else:
                logging.warning(f"Unknown custom feature type for {key}: {type(value)}")
        
        # Pad or truncate to expected dimension
        target_dim = self.custom_feature_dim
        if len(processed) < target_dim:
            processed.extend([0.0] * (target_dim - len(processed)))
        elif len(processed) > target_dim:
            processed = processed[:target_dim]
        
        return torch.tensor(processed, dtype=torch.float)
    
    def _create_empty_graph(self) -> Any:
        """Create empty graph as fallback"""
        if torch is None or Data is None:
            return None
        
        node_features = self.base_converter.get_node_feature_dim()
        return Data(
            x=torch.zeros(1, node_features),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=None
        )
    
    def get_construction_info(self) -> Dict[str, Any]:
        """Get information about the current construction mode"""
        info = {
            'mode': self.construction_mode.value,
            'base_converter_info': self.base_converter.get_feature_dimensions()
        }
        
        if self.construction_mode == GraphConstructionMode.REACTION_GRAPH:
            info.update({
                'reaction_node_types': self.reaction_node_types,
                'reaction_edge_types': self.reaction_edge_types
            })
        elif self.construction_mode == GraphConstructionMode.CUSTOM_FUSION:
            info.update({
                'fusion_config': self.custom_fusion_config,
                'fusion_method': self.fusion_method
            })
        
        return info


class CustomFusionLayer(nn.Module):
    """Neural layer for fusing custom features with graph embeddings"""
    
    def __init__(self, 
                 custom_feature_dim: int,
                 graph_embed_dim: int,
                 output_dim: int,
                 fusion_method: str = 'concatenate'):
        super().__init__()
        
        self.custom_feature_dim = custom_feature_dim
        self.graph_embed_dim = graph_embed_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        
        if fusion_method == 'concatenate':
            self.fusion_layer = nn.Linear(custom_feature_dim + graph_embed_dim, output_dim)
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(graph_embed_dim, num_heads=4, batch_first=True)
            self.feature_proj = nn.Linear(custom_feature_dim, graph_embed_dim)
            self.output_proj = nn.Linear(graph_embed_dim, output_dim)
        elif fusion_method == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(custom_feature_dim + graph_embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, graph_embed_dim),
                nn.Sigmoid()
            )
            self.output_proj = nn.Linear(graph_embed_dim, output_dim)
        elif fusion_method == 'transformer':
            self.feature_proj = nn.Linear(custom_feature_dim, graph_embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=graph_embed_dim,
                nhead=8,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.output_proj = nn.Linear(graph_embed_dim, output_dim)
    
    def forward(self, graph_embedding: torch.Tensor, custom_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse graph embedding with custom features
        
        Args:
            graph_embedding: [batch_size, graph_embed_dim]
            custom_features: [batch_size, custom_feature_dim]
            
        Returns:
            Fused representation: [batch_size, output_dim]
        """
        if self.fusion_method == 'concatenate':
            combined = torch.cat([graph_embedding, custom_features], dim=-1)
            return self.fusion_layer(combined)
        
        elif self.fusion_method == 'attention':
            # Project custom features to same dimension as graph embedding
            custom_projected = self.feature_proj(custom_features).unsqueeze(1)  # [batch, 1, dim]
            graph_embed = graph_embedding.unsqueeze(1)  # [batch, 1, dim]
            
            # Apply attention
            attended, _ = self.attention(graph_embed, custom_projected, custom_projected)
            return self.output_proj(attended.squeeze(1))
        
        elif self.fusion_method == 'gated':
            # Compute gate values
            combined = torch.cat([graph_embedding, custom_features], dim=-1)
            gate_values = self.gate(combined)
            
            # Apply gating
            gated_graph = graph_embedding * gate_values
            return self.output_proj(gated_graph)
        
        elif self.fusion_method == 'transformer':
            # Project custom features
            custom_projected = self.feature_proj(custom_features).unsqueeze(1)
            graph_embed = graph_embedding.unsqueeze(1)
            
            # Combine and apply transformer
            combined = torch.cat([graph_embed, custom_projected], dim=1)  # [batch, 2, dim]
            transformed = self.transformer(combined)
            
            # Pool and project
            pooled = transformed.mean(dim=1)  # [batch, dim]
            return self.output_proj(pooled)


# Example usage and testing
if __name__ == "__main__":
    # Test different construction modes
    test_smiles = {
        'Catalyst': 'CC(C)P(c1ccccc1)c1ccccc1',
        'Reactant1': 'CC(=O)c1ccccc1',
        'Reactant2': 'NCc1ccccc1'
    }
    
    custom_features = {
        'temperature': 80.0,
        'pressure': 1.0,
        'reaction_time': 24.0,
        'additional_vector': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    print("Testing Advanced Graph Builder:")
    
    for mode in GraphConstructionMode:
        try:
            print(f"\n--- Testing {mode.value} mode ---")
            
            builder = AdvancedGraphBuilder(
                construction_mode=mode,
                custom_fusion_config={'custom_feature_dim': 8, 'fusion_method': 'attention'}
            )
            
            if mode == GraphConstructionMode.CUSTOM_FUSION:
                graph = builder.build_graphs(test_smiles, custom_features=custom_features)
            else:
                graph = builder.build_graphs(test_smiles)
            
            print(f"Graph created successfully!")
            if hasattr(graph, 'x') and graph.x is not None:
                print(f"  Nodes: {graph.x.shape[0]}, Features: {graph.x.shape[1]}")
            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                print(f"  Edges: {graph.edge_index.shape[1]}")
            
            # Print mode-specific info
            info = builder.get_construction_info()
            print(f"  Mode info: {info['mode']}")
            
        except Exception as e:
            print(f"Error testing {mode.value}: {e}") 