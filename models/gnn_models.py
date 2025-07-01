"""
Graph Neural Network Models for SMILES Processing

This module implements various GNN architectures that can process molecular
graphs converted from SMILES strings for regression and classification tasks.
"""

import math
import numpy as np
from typing import Optional, List, Union, Dict, Any
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from torch_geometric.nn import (  # type: ignore
        GCNConv, GATConv, MessagePassing, global_mean_pool, 
        global_max_pool, global_add_pool, TransformerConv
    )
    from torch_geometric.data import Data, Batch  # type: ignore
    from torch_geometric.utils import add_self_loops, degree  # type: ignore
    from torch_scatter import scatter_add  # type: ignore
except ImportError:
    torch = None
    nn = None
    F = None
    Tensor = None
    MessagePassing = None
    logging.warning("PyTorch Geometric not installed. GNN models will not be available.")


class GCN(nn.Module):  # type: ignore
    """Graph Convolutional Network for molecular property prediction"""
    
    def __init__(self, 
                 node_features: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 output_dim: int = 1,
                 dropout_rate: float = 0.2,
                 task_type: str = 'regression'):
        super(GCN, self).__init__()
        
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for GNN models")
            
        self.task_type = task_type
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))  # type: ignore
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))  # type: ignore
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final prediction layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, data):  # type: ignore
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)  # type: ignore
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)  # type: ignore
        
        # Final prediction
        x = self.classifier(x)
        
        return x


class GAT(nn.Module):  # type: ignore
    """Graph Attention Network for molecular property prediction"""
    
    def __init__(self, 
                 node_features: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 output_dim: int = 1,
                 dropout_rate: float = 0.2,
                 task_type: str = 'regression'):
        super(GAT, self).__init__()
        
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for GNN models")
            
        self.task_type = task_type
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        
        # Graph attention layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(  # type: ignore
            node_features, hidden_dim // num_heads, 
            heads=num_heads, dropout=dropout_rate
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(  # type: ignore
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, dropout=dropout_rate
            ))
        
        # Final GAT layer
        self.convs.append(GATConv(  # type: ignore
            hidden_dim, hidden_dim,
            heads=1, dropout=dropout_rate
        ))
        
        # Final prediction layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, data):  # type: ignore
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph attention layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)  # type: ignore
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)  # type: ignore
        
        # Final prediction
        x = self.classifier(x)
        
        return x


class MPNNLayer(MessagePassing):  # type: ignore
    """Message Passing Neural Network Layer"""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super(MPNNLayer, self).__init__(aggr='add')
        
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for MPNN")
            
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Message function
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(self, x, edge_index, edge_attr=None):  # type: ignore
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):  # type: ignore
        # x_i: target nodes, x_j: source nodes
        if edge_attr is None:
            edge_attr = torch.zeros(x_i.size(0), 1, device=x_i.device)  # type: ignore
        
        # Concatenate node features and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)  # type: ignore
        return self.message_net(msg_input)
    
    def update(self, aggr_out, x):  # type: ignore
        # Combine aggregated messages with node features
        update_input = torch.cat([x, aggr_out], dim=-1)  # type: ignore
        return self.update_net(update_input)


class MPNN(nn.Module):  # type: ignore
    """Message Passing Neural Network for molecular property prediction"""
    
    def __init__(self, 
                 node_features: int,
                 edge_features: int = 10,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 output_dim: int = 1,
                 dropout_rate: float = 0.2,
                 task_type: str = 'regression'):
        super(MPNN, self).__init__()
        
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for MPNN")
            
        self.task_type = task_type
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Initial node embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # MPNN layers
        self.mp_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_features, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final prediction layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, data):  # type: ignore
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Message passing layers
        for i, mp_layer in enumerate(self.mp_layers):
            x_new = mp_layer(x, edge_index, edge_attr)
            x = x + x_new  # Residual connection
            x = self.batch_norms[i](x)
            x = F.relu(x)  # type: ignore
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)  # type: ignore
        
        # Final prediction
        x = self.classifier(x)
        
        return x


class AFP(nn.Module):  # type: ignore
    """Attentive FP (Fingerprint) for molecular property prediction"""
    
    def __init__(self, 
                 node_features: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_timesteps: int = 2,
                 output_dim: int = 1,
                 dropout_rate: float = 0.2,
                 task_type: str = 'regression'):
        super(AFP, self).__init__()
        
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for AFP")
            
        self.task_type = task_type
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        
        # Initial embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # GRU for node updates
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Graph convolutions
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)  # type: ignore
        ])
        
        # Final prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, data):  # type: ignore
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Multiple timesteps of message passing
        for _ in range(self.num_timesteps):
            # Graph convolution
            x_conv = x
            for conv in self.convs:
                x_conv = conv(x_conv, edge_index)
                x_conv = F.relu(x_conv)  # type: ignore
                x_conv = self.dropout(x_conv)
            
            # GRU update (treating each node as a sequence of length 1)
            x_gru, _ = self.gru(x.unsqueeze(1))
            x = x_gru.squeeze(1)
        
        # Attentive pooling
        batch_size = batch.max().item() + 1
        graph_representations = []
        
        for i in range(batch_size):
            # Get nodes for this graph
            mask = (batch == i)
            graph_nodes = x[mask]
            
            if graph_nodes.size(0) == 0:
                continue
                
            # Compute attention weights
            mean_node = graph_nodes.mean(dim=0, keepdim=True)
            attention_input = torch.cat([  # type: ignore
                graph_nodes, 
                mean_node.expand(graph_nodes.size(0), -1)
            ], dim=1)
            
            attention_weights = self.attention(attention_input)
            attention_weights = F.softmax(attention_weights, dim=0)  # type: ignore
            
            # Weighted sum
            graph_repr = (graph_nodes * attention_weights).sum(dim=0)
            graph_representations.append(graph_repr)
        
        x = torch.stack(graph_representations)  # type: ignore
        
        # Final prediction
        x = self.classifier(x)
        
        return x


class GraphTransformer(nn.Module):  # type: ignore
    """Graph Transformer for molecular property prediction"""
    
    def __init__(self, 
                 node_features: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 output_dim: int = 1,
                 dropout_rate: float = 0.2,
                 task_type: str = 'regression'):
        super(GraphTransformer, self).__init__()
        
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for Graph Transformer")
            
        self.task_type = task_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Initial embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerConv(  # type: ignore
                hidden_dim, hidden_dim, 
                heads=num_heads, dropout=dropout_rate,
                edge_dim=None, beta=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, data):  # type: ignore
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Transformer layers
        for i, transformer in enumerate(self.transformer_layers):
            x_new = transformer(x, edge_index)
            x = x + x_new  # Residual connection
            x = self.layer_norms[i](x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)  # type: ignore
        
        # Final prediction
        x = self.classifier(x)
        
        return x


class EnsembleGNN(nn.Module):  # type: ignore
    """Ensemble of different GNN architectures"""
        
    def __init__(self, 
                 node_features: int,
                 edge_features: int = 10,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 task_type: str = 'regression',
                 models: Optional[List[str]] = None):
        super(EnsembleGNN, self).__init__()
        
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for Ensemble GNN")
            
        if models is None:
            models = ['gcn', 'gat', 'mpnn']
            
        self.task_type = task_type
        self.models = nn.ModuleDict()
        
        # Create individual models
        for model_name in models:
            if model_name.lower() == 'gcn':
                self.models[model_name] = GCN(
                    node_features, hidden_dim, output_dim=hidden_dim, task_type=task_type
                )
            elif model_name.lower() == 'gat':
                self.models[model_name] = GAT(
                    node_features, hidden_dim, output_dim=hidden_dim, task_type=task_type
                )
            elif model_name.lower() == 'mpnn':
                self.models[model_name] = MPNN(
                    node_features, edge_features, hidden_dim, output_dim=hidden_dim, task_type=task_type
                )
            elif model_name.lower() == 'afp':
                self.models[model_name] = AFP(
                    node_features, hidden_dim, output_dim=hidden_dim, task_type=task_type
                )
            elif model_name.lower() == 'gtn':
                self.models[model_name] = GraphTransformer(
                    node_features, hidden_dim, output_dim=hidden_dim, task_type=task_type
                )
        
        # Ensemble combination layer
        self.ensemble_layer = nn.Linear(hidden_dim * len(models), output_dim)
        
    def forward(self, data):  # type: ignore
        outputs = []
        
        for model in self.models.values():
            output = model(data)
            outputs.append(output)
        
        # Concatenate outputs
        ensemble_input = torch.cat(outputs, dim=-1)  # type: ignore
        
        # Final prediction
        result = self.ensemble_layer(ensemble_input)
        
        return result


# Model factory function
def create_gnn_model(model_name: str, 
                    node_features: int,
                    edge_features: int = 10,
                    hidden_dim: int = 128,
                    num_layers: int = 3,
                    output_dim: int = 1,
                    task_type: str = 'regression',
                    **kwargs) -> nn.Module:  # type: ignore
    """
    Factory function to create GNN models
    
    Args:
        model_name: Name of the GNN model ('gcn', 'gat', 'mpnn', 'afp', 'gtn', 'ensemble')
        node_features: Number of node features
        edge_features: Number of edge features
        hidden_dim: Hidden dimension size
        num_layers: Number of layers
        output_dim: Output dimension
        task_type: 'regression' or 'classification'
        **kwargs: Additional model-specific parameters
        
    Returns:
        PyTorch GNN model
    """
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for GNN models")
    
    model_name = model_name.lower()
    
    if model_name == 'gcn':
        return GCN(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            task_type=task_type,
            **kwargs
        )
    elif model_name == 'gat':
        return GAT(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            task_type=task_type,
            **kwargs
        )
    elif model_name == 'mpnn':
        return MPNN(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            task_type=task_type,
            **kwargs
        )
    elif model_name == 'afp':
        return AFP(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            task_type=task_type,
            **kwargs
        )
    elif model_name == 'gtn':
        return GraphTransformer(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            task_type=task_type,
            **kwargs
        )
    elif model_name == 'ensemble':
        return EnsembleGNN(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            task_type=task_type,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown GNN model: {model_name}")


# Example usage
if __name__ == "__main__":
    # Test model creation
    if torch is not None:
        node_features = 150  # Example: SMILES graph node features
        edge_features = 10   # Example: SMILES graph edge features
        
        models = ['gcn', 'gat', 'mpnn', 'afp', 'gtn']
        
        for model_name in models:
            try:
                model = create_gnn_model(
                    model_name=model_name,
                    node_features=node_features,
                    edge_features=edge_features,
                    hidden_dim=64,
                    num_layers=3,
                    output_dim=1,
                    task_type='regression'
                )
                print(f"✓ Successfully created {model_name.upper()} model")
                print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            except Exception as e:
                print(f"✗ Failed to create {model_name.upper()} model: {e}")
    else:
        print("PyTorch not available, skipping model tests") 