"""
SMILES to Graph Conversion Utilities

This module provides utilities to convert SMILES strings to graph representations
compatible with PyTorch Geometric for graph neural network training.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
import logging

try:
    import torch
    from torch_geometric.data import Data, Batch  # type: ignore
except ImportError:
    torch = None
    Data = None
    Batch = None
    logging.warning("PyTorch Geometric not installed. GNN functionality will not be available.")

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import rdMolDescriptors, Descriptors  # type: ignore
except ImportError:
    Chem = None
    rdMolDescriptors = None
    Descriptors = None
    logging.warning("RDKit not installed. SMILES processing will not be available.")


class SmilesGraphConverter:
    """Convert SMILES strings to PyTorch Geometric graph objects"""
    
    def __init__(self, 
                 max_nodes: int = 100,
                 add_self_loops: bool = True,
                 use_edge_features: bool = True,
                 use_3d_coords: bool = False):
        """
        Initialize the SMILES to graph converter
        
        Args:
            max_nodes: Maximum number of nodes in a graph
            add_self_loops: Whether to add self-loops to graphs
            use_edge_features: Whether to include edge features
            use_3d_coords: Whether to use 3D coordinates (requires conformer generation)
        """
        if torch is None or Data is None:
            raise ImportError("PyTorch Geometric is required for graph conversion")
        if Chem is None:
            raise ImportError("RDKit is required for SMILES processing")
            
        self.max_nodes = max_nodes
        self.add_self_loops = add_self_loops
        self.use_edge_features = use_edge_features
        self.use_3d_coords = use_3d_coords
        
        # Atom feature vocabulary
        self.atom_features = {
            'atomic_num': list(range(1, 119)),  # Atomic numbers 1-118
            'degree': [0, 1, 2, 3, 4, 5, 6],
            'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.UNSPECIFIED
            ],
            'num_hs': [0, 1, 2, 3, 4],
            'valence': [0, 1, 2, 3, 4, 5, 6]
        }
        
        # Bond feature vocabulary
        self.bond_features = {
            'bond_type': [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
            ],
            'stereo': [
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOANY,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE
            ]
        }
    
    def _one_hot_encode(self, value, vocab: List) -> List[int]:
        """One-hot encode a value given a vocabulary"""
        encoding = [0] * len(vocab)
        if value in vocab:
            encoding[vocab.index(value)] = 1
        else:
            # Unknown value, could add an "unknown" category or leave as all zeros
            pass
        return encoding
    
    def _get_atom_features(self, atom) -> List[float]:
        """Extract features from an RDKit atom"""
        features = []
        
        # Atomic number (one-hot)
        features.extend(self._one_hot_encode(atom.GetAtomicNum(), self.atom_features['atomic_num']))
        
        # Degree (one-hot)
        features.extend(self._one_hot_encode(atom.GetDegree(), self.atom_features['degree']))
        
        # Formal charge (one-hot)
        features.extend(self._one_hot_encode(atom.GetFormalCharge(), self.atom_features['formal_charge']))
        
        # Hybridization (one-hot)
        features.extend(self._one_hot_encode(atom.GetHybridization(), self.atom_features['hybridization']))
        
        # Number of hydrogens (one-hot)
        features.extend(self._one_hot_encode(atom.GetTotalNumHs(), self.atom_features['num_hs']))
        
        # Valence (one-hot)
        features.extend(self._one_hot_encode(atom.GetTotalValence(), self.atom_features['valence']))
        
        # Additional boolean features
        features.extend([
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetMass())
        ])
        
        return features
    
    def _get_bond_features(self, bond) -> List[float]:
        """Extract features from an RDKit bond"""
        features = []
        
        # Bond type (one-hot)
        features.extend(self._one_hot_encode(bond.GetBondType(), self.bond_features['bond_type']))
        
        # Stereo (one-hot)
        features.extend(self._one_hot_encode(bond.GetStereo(), self.bond_features['stereo']))
        
        # Additional boolean features
        features.extend([
            float(bond.GetIsConjugated()),
            float(bond.IsInRing())
        ])
        
        return features
    
    def smiles_to_graph(self, smiles: str):  # type: ignore
        """
        Convert a SMILES string to a PyTorch Geometric Data object
        
        Args:
            smiles: SMILES string
            
        Returns:
            PyTorch Geometric Data object or None if conversion fails
        """
        if Chem is None or torch is None:
            raise RuntimeError("RDKit and PyTorch are required for graph conversion")
            
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)  # type: ignore  # type: ignore
            if mol is None:
                logging.warning(f"Failed to parse SMILES: {smiles}")
                return None
            
            # Add hydrogens if needed
            mol = Chem.AddHs(mol)  # type: ignore  # type: ignore
            
            # Check if molecule is too large
            if mol.GetNumAtoms() > self.max_nodes:
                logging.warning(f"Molecule too large ({mol.GetNumAtoms()} atoms > {self.max_nodes})")
                return None
            
            # Extract node features
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append(self._get_atom_features(atom))
            
            x = torch.tensor(atom_features, dtype=torch.float)  # type: ignore  # type: ignore
            
            # Extract edge indices and features
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Add both directions for undirected graph
                edge_indices.extend([[i, j], [j, i]])
                
                if self.use_edge_features:
                    bond_feat = self._get_bond_features(bond)
                    edge_features.extend([bond_feat, bond_feat])  # Same features for both directions
            
            # Convert to tensors
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()  # type: ignore  # type: ignore
                if self.use_edge_features and edge_features:
                    edge_attr = torch.tensor(edge_features, dtype=torch.float)  # type: ignore  # type: ignore
                else:
                    edge_attr = None
            else:
                # No bonds (single atom)
                edge_index = torch.empty((2, 0), dtype=torch.long)  # type: ignore  # type: ignore
                edge_attr = None
            
            # Add self-loops if requested
            if self.add_self_loops:
                num_nodes = x.size(0)
                self_loop_indices = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)  # type: ignore  # type: ignore
                edge_index = torch.cat([edge_index, self_loop_indices], dim=1)  # type: ignore  # type: ignore
                
                if self.use_edge_features and edge_attr is not None:
                    # Create self-loop features (zeros)
                    self_loop_attr = torch.zeros((num_nodes, edge_attr.size(1)), dtype=torch.float)  # type: ignore  # type: ignore
                    edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)  # type: ignore  # type: ignore
            
            # Create graph data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)  # type: ignore  # type: ignore
            
            # Add molecular descriptors as graph-level features
            data.mol_features = self._get_molecular_features(mol)  # type: ignore
            
            return data
            
        except Exception as e:
            logging.error(f"Error converting SMILES {smiles} to graph: {e}")
            return None
    
    def _get_molecular_features(self, mol):  # type: ignore
        """Extract molecular-level features"""
        if Descriptors is None or rdMolDescriptors is None or torch is None:
            raise RuntimeError("RDKit Descriptors and PyTorch are required")
            
        features = []
        
        # Basic molecular descriptors
        features.extend([
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            mol.GetNumHeavyAtoms(),
            Descriptors.MolWt(mol),  # type: ignore  # type: ignore
            Descriptors.MolLogP(mol),  # type: ignore  # type: ignore
            Descriptors.NumRotatableBonds(mol),  # type: ignore  # type: ignore
            Descriptors.TPSA(mol),  # type: ignore  # type: ignore
            rdMolDescriptors.CalcNumRings(mol),  # type: ignore  # type: ignore  # type: ignore
            rdMolDescriptors.CalcNumAromaticRings(mol),  # type: ignore  # type: ignore  # type: ignore
            # Additional basic descriptors
            float(mol.GetNumAtoms()),
            float(mol.GetNumHeavyAtoms())
        ])
        
        return torch.tensor(features, dtype=torch.float)  # type: ignore
    
    def batch_smiles_to_graphs(self, smiles_list: List[str]):  # type: ignore
        """
        Convert a list of SMILES to a batched graph
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            PyTorch Geometric Batch object
        """
        if Batch is None:
            raise RuntimeError("PyTorch Geometric is required for batch operations")
            
        graphs = []
        for smiles in smiles_list:
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
        
        if not graphs:
            return None
        
        return Batch.from_data_list(graphs)  # type: ignore  # type: ignore
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get the dimensions of node and edge features"""
        # Calculate node feature dimension
        node_dim = (len(self.atom_features['atomic_num']) + 
                   len(self.atom_features['degree']) +
                   len(self.atom_features['formal_charge']) +
                   len(self.atom_features['hybridization']) +
                   len(self.atom_features['num_hs']) +
                   len(self.atom_features['valence']) +
                   3)  # 3 additional features (aromatic, in_ring, mass)
        
        # Calculate edge feature dimension
        edge_dim = (len(self.bond_features['bond_type']) + 
                   len(self.bond_features['stereo']) +
                   2) if self.use_edge_features else 0  # 2 additional features (conjugated, in_ring)
        
        # Molecular feature dimension
        mol_dim = 11  # Number of molecular descriptors
        
        return {
            'node_features': node_dim,
            'edge_features': edge_dim,
            'molecular_features': mol_dim
        }
    
    def get_node_feature_dim(self) -> int:
        """Get the dimension of node features"""
        return self.get_feature_dimensions()['node_features']
    
    def get_edge_feature_dim(self) -> int:
        """Get the dimension of edge features"""
        return self.get_feature_dimensions()['edge_features']


def convert_smiles_dataset(smiles_data: Union[pd.Series, List[str]], 
                          converter: Optional[SmilesGraphConverter] = None):
    """
    Convert a dataset of SMILES to graphs
    
    Args:
        smiles_data: SMILES strings as pandas Series or list
        converter: SmilesGraphConverter instance (creates default if None)
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    if converter is None:
        converter = SmilesGraphConverter()
    
    if isinstance(smiles_data, pd.Series):
        smiles_list = smiles_data.tolist()
    else:
        smiles_list = smiles_data
    
    graphs = []
    for smiles in smiles_list:
        if pd.isna(smiles) or smiles == "":
            continue
        graph = converter.smiles_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
    
    return graphs


# Example usage and testing
if __name__ == "__main__":
    # Test with some example SMILES
    test_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(=O)O",  # Acetic acid
        "CN(C)C"  # Trimethylamine
    ]
    
    converter = SmilesGraphConverter()
    
    print("Feature dimensions:", converter.get_feature_dimensions())
    
    for smiles in test_smiles:
        graph = converter.smiles_to_graph(smiles)
        if graph is not None:
            print(f"SMILES: {smiles}")
            if hasattr(graph, 'x') and graph.x is not None:
                print(f"  Nodes: {graph.x.shape[0]}, Node features: {graph.x.shape[1]}")  # type: ignore
            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                print(f"  Edges: {graph.edge_index.shape[1]}")  # type: ignore
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                print(f"  Edge features: {graph.edge_attr.shape[1]}")  # type: ignore
            if hasattr(graph, 'mol_features') and graph.mol_features is not None:
                print(f"  Molecular features: {graph.mol_features.shape[0]}")  # type: ignore
            print() 