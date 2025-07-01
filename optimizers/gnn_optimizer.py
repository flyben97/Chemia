"""
Graph Neural Network Optimizer for the CRAFT framework
Supports GCN, GAT, MPNN, AFP, GTN, and Ensemble models
"""

import os
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from rich.console import Console

from .base_optimizer import BaseOptimizer

# Conditional imports for PyTorch and related libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data, Batch
    from torch.optim import Adam, AdamW, SGD
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Data = None
    Batch = None
    Adam = None
    AdamW = None
    SGD = None
    ReduceLROnPlateau = None
    CosineAnnealingLR = None
    TORCH_AVAILABLE = False

try:
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.metrics import (
        r2_score, mean_squared_error, mean_absolute_error,
        accuracy_score, f1_score, precision_score, recall_score
    )
except ImportError:
    pass

# Import utility modules
try:
    from utils.smiles_to_graph import SmilesGraphConverter
    from utils.advanced_graph_builder import AdvancedGraphBuilder, GraphConstructionMode, CustomFusionLayer
    from models.gnn_models import create_gnn_model
except ImportError:
    SmilesGraphConverter = None
    AdvancedGraphBuilder = None
    GraphConstructionMode = None
    CustomFusionLayer = None
    create_gnn_model = None


class GNNOptimizer(BaseOptimizer):
    """Graph Neural Network optimizer for molecular property prediction"""
    
    def __init__(self, 
                 model_name: str,
                 smiles_columns: List[str],
                 n_trials: int = 100,
                 random_state: int = 42,
                 cv: Optional[int] = None,
                 task_type: str = 'regression',
                 num_classes: Optional[int] = None,
                 device: str = 'auto',
                 max_epochs: int = 100,
                 early_stopping_patience: int = 20,
                 batch_size: int = 32,
                 graph_construction_mode: str = 'batch',
                 custom_fusion_config: Optional[Dict[str, Any]] = None,
                 molecule_roles: Optional[Dict[str, str]] = None):
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and PyTorch Geometric are required for GNN models")
        
        if SmilesGraphConverter is None or create_gnn_model is None:
            raise ImportError("SMILES graph converter and GNN models are required")
        
        # Set model attributes first
        self.model_name = model_name.lower()
        self.smiles_columns = smiles_columns
        self.device = self._get_device(device)
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        
        # Initialize graph converter and advanced builder
        self.graph_converter = SmilesGraphConverter()
        
        # Setup advanced graph construction
        self.graph_construction_mode = graph_construction_mode
        self.custom_fusion_config = custom_fusion_config or {}
        self.molecule_roles = molecule_roles or {}
        
        # Initialize advanced graph builder if available
        if AdvancedGraphBuilder is not None:
            self.advanced_builder = AdvancedGraphBuilder(
                base_converter=self.graph_converter,
                construction_mode=graph_construction_mode,
                custom_fusion_config=self.custom_fusion_config
            )
        else:
            self.advanced_builder = None
        
        # Determine node and edge features based on converter
        # Use standard dimensions if methods don't exist
        try:
            self.node_features = self.graph_converter.get_node_feature_dim()
            self.edge_features = self.graph_converter.get_edge_feature_dim()
        except AttributeError:
            self.node_features = 153  # Standard SMILES node features
            self.edge_features = 10   # Standard SMILES edge features
        
        # Get parameter grid after setting model_name
        param_grid = self._get_param_grid()
        
        # Initialize base optimizer
        super().__init__(
            model_class=None,  # We create models dynamically
            param_grid=param_grid,
            n_trials=n_trials,
            random_state=random_state,
            cv=cv,
            task_type=task_type,
            num_classes=num_classes
        )
        
        self.console = Console()
        
        # Best model and parameters
        self.best_model_ = None
        self.best_params_ = None
    
    def _get_device(self, device: str) -> torch.device:  # type: ignore
        """Determine the appropriate device for training"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # type: ignore
        else:
            return torch.device(device)  # type: ignore
    
    def _get_param_grid(self) -> Dict[str, Dict[str, Any]]:
        """Get hyperparameter search space for GNN models"""
        base_params = {
            'hidden_dim': {'type': 'int', 'low': 64, 'high': 256, 'step': 32},
            'num_layers': {'type': 'int', 'low': 2, 'high': 6},
            'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-3, 'log': True},
            'batch_size': {'type': 'int', 'low': 16, 'high': 128, 'step': 16},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'adamw', 'sgd']},
            'scheduler': {'type': 'categorical', 'choices': ['plateau', 'cosine', 'none']},
        }
        
        # Add model-specific parameters
        if self.model_name == 'gat':
            base_params.update({
                'num_heads': {'type': 'int', 'low': 4, 'high': 16, 'step': 2},
                'attention_dropout': {'type': 'float', 'low': 0.0, 'high': 0.3}
            })
        elif self.model_name == 'mpnn':
            base_params.update({
                'message_hidden_dim': {'type': 'int', 'low': 64, 'high': 256, 'step': 32},
                'num_message_steps': {'type': 'int', 'low': 2, 'high': 5}
            })
        elif self.model_name == 'afp':
            base_params.update({
                'num_timesteps': {'type': 'int', 'low': 1, 'high': 3},
                'attention_hidden_dim': {'type': 'int', 'low': 64, 'high': 256, 'step': 32}
            })
        elif self.model_name == 'gtn':
            base_params.update({
                'num_heads': {'type': 'int', 'low': 4, 'high': 16, 'step': 2},
                'attention_dropout': {'type': 'float', 'low': 0.0, 'high': 0.3}
            })
        elif self.model_name == 'ensemble':
            base_params.update({
                'ensemble_models': {'type': 'categorical', 'choices': [
                    ['gcn', 'gat'], ['gcn', 'mpnn'], ['gat', 'mpnn'], 
                    ['gcn', 'gat', 'mpnn'], ['gcn', 'gat', 'afp']
                ]}
            })
        
        return base_params
    
    def _convert_smiles_to_graphs(self, X: pd.DataFrame, custom_features: Optional[Dict[int, Dict[str, Any]]] = None) -> List[Data]:  # type: ignore
        """Convert SMILES data to graph representations using advanced graph builder"""
        all_graphs = []
        
        # Use advanced builder if available
        if self.advanced_builder is not None and self.graph_construction_mode != 'batch':
            for row_idx, row in enumerate(X.itertuples()):
                # Create SMILES dictionary for this sample
                smiles_dict = {}
                for i, smiles_col in enumerate(self.smiles_columns):
                    try:
                        smiles = getattr(row, smiles_col, None)
                        if smiles is not None and not pd.isna(smiles) and str(smiles) != "":
                            smiles_dict[smiles_col] = str(smiles)
                    except AttributeError:
                        # Try index-based access
                        try:
                            smiles = row[i+1]  # +1 because itertuples includes index as first element
                            if smiles is not None and not pd.isna(smiles) and str(smiles) != "":
                                smiles_dict[smiles_col] = str(smiles)
                        except (IndexError, TypeError):
                            continue
                
                # Get custom features for this sample if provided
                sample_custom_features = custom_features.get(row_idx) if custom_features else None
                
                try:
                    # Build graph using advanced builder
                    graph = self.advanced_builder.build_graphs(
                        smiles_dict=smiles_dict,
                        custom_features=sample_custom_features,
                        molecule_roles=self.molecule_roles
                    )
                    all_graphs.append(graph)
                except Exception as e:
                    # Fallback to empty graph
                    if torch is not None and Data is not None:
                        empty_graph = Data(
                            x=torch.zeros(1, self.node_features),
                            edge_index=torch.empty(2, 0, dtype=torch.long),
                            edge_attr=None
                        )
                        all_graphs.append(empty_graph)
            
            return all_graphs
        
        # Original batch mode implementation
        for idx, row in X.iterrows():
            # Combine all SMILES columns for this sample
            sample_graphs = []
            
            for smiles_col in self.smiles_columns:
                try:
                    smiles = row[smiles_col]
                    if smiles is not None and not pd.isna(smiles) and str(smiles) != "":  # type: ignore
                        graph = self.graph_converter.smiles_to_graph(str(smiles))
                        if graph is not None:
                            sample_graphs.append(graph)
                except Exception:  # type: ignore
                    continue
            
            if sample_graphs:
                # If multiple SMILES, create a batch
                if len(sample_graphs) == 1:
                    combined_graph = sample_graphs[0]
                else:
                    combined_graph = Batch.from_data_list(sample_graphs)  # type: ignore
                
                all_graphs.append(combined_graph)
            else:
                # Create empty graph as placeholder
                empty_graph = Data(  # type: ignore
                    x=torch.zeros(1, self.node_features),  # type: ignore
                    edge_index=torch.empty(2, 0, dtype=torch.long),  # type: ignore
                    edge_attr=None
                )
                all_graphs.append(empty_graph)
        
        return all_graphs
    
    def _create_model(self, params: Dict[str, Any]) -> nn.Module:  # type: ignore
        """Create GNN model with given parameters"""
        
        model_params = {
            'node_features': self.node_features,
            'edge_features': self.edge_features,
            'hidden_dim': params.get('hidden_dim', 128),
            'num_layers': params.get('num_layers', 3),
            'output_dim': self.num_classes if self.task_type != 'regression' else 1,
            'dropout_rate': params.get('dropout_rate', 0.2),
            'task_type': self.task_type
        }
        
        # Add model-specific parameters
        if self.model_name == 'gat':
            model_params.update({
                'num_heads': params.get('num_heads', 8),
                'attention_dropout': params.get('attention_dropout', 0.1)
            })
        elif self.model_name == 'mpnn':
            model_params.update({
                'message_hidden_dim': params.get('message_hidden_dim', 128),
                'num_message_steps': params.get('num_message_steps', 3)
            })
        elif self.model_name == 'afp':
            model_params.update({
                'num_timesteps': params.get('num_timesteps', 2),
                'attention_hidden_dim': params.get('attention_hidden_dim', 128)
            })
        elif self.model_name == 'gtn':
            model_params.update({
                'num_heads': params.get('num_heads', 8),
                'attention_dropout': params.get('attention_dropout', 0.1)
            })
        elif self.model_name == 'ensemble':
            model_params.update({
                'models': params.get('ensemble_models', ['gcn', 'gat', 'mpnn'])
            })
        
        if create_gnn_model is None:
            raise RuntimeError("GNN model creation function is not available")
        
        model = create_gnn_model(self.model_name, **model_params)  # type: ignore
        return model.to(self.device)
    
    def _create_optimizer(self, model: nn.Module, params: Dict[str, Any]):  # type: ignore
        """Create optimizer with given parameters"""
        lr = params.get('learning_rate', 1e-3)
        weight_decay = params.get('weight_decay', 1e-5)
        optimizer_name = params.get('optimizer', 'adam')
        
        if optimizer_name == 'adam':
            return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # type: ignore
        elif optimizer_name == 'adamw':
            return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # type: ignore
        elif optimizer_name == 'sgd':
            return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)  # type: ignore
        else:
            return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # type: ignore
    
    def _create_scheduler(self, optimizer, params: Dict[str, Any]):  # type: ignore
        """Create learning rate scheduler"""
        scheduler_name = params.get('scheduler', 'plateau')
        
        if scheduler_name == 'plateau':
            return ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)  # type: ignore
        elif scheduler_name == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=self.max_epochs)  # type: ignore
        else:
            return None
    
    def _train_epoch(self, model: nn.Module, dataloader: DataLoader,  # type: ignore
                    optimizer, criterion) -> float:
        """Train model for one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            
            # Compute loss
            if self.task_type == 'regression':
                loss = criterion(output.squeeze(), batch.y.float())
            else:
                loss = criterion(output, batch.y.long())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _evaluate_epoch(self, model: nn.Module, dataloader: DataLoader,  # type: ignore
                       criterion) -> Tuple[float, float]:
        """Evaluate model for one epoch"""
        model.eval()
        total_loss = 0.0
        predictions: List[float] = []  # type: ignore
        targets: List[float] = []  # type: ignore
        num_batches = 0
        
        with torch.no_grad():  # type: ignore
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Forward pass
                output = model(batch)
                
                # Compute loss
                if self.task_type == 'regression':
                    loss = criterion(output.squeeze(), batch.y.float())
                    predictions.extend(output.squeeze().cpu().numpy())  # type: ignore
                else:
                    loss = criterion(output, batch.y.long())
                    predictions.extend(torch.argmax(output, dim=1).cpu().numpy())  # type: ignore
                
                targets.extend(batch.y.cpu().numpy())  # type: ignore
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute metric
        if self.task_type == 'regression':
            metric = r2_score(targets, predictions)
        else:
            avg_method = 'binary' if self.task_type == 'binary_classification' else 'weighted'
            metric = f1_score(targets, predictions, average=avg_method, zero_division='warn')
        
        return avg_loss, metric
    
    def _train_model(self, model: nn.Module, train_graphs: List[Data],  # type: ignore
                    val_graphs: List[Data], y_train: np.ndarray, y_val: np.ndarray,  # type: ignore
                    params: Dict[str, Any]) -> Tuple[nn.Module, float]:  # type: ignore
        """Train GNN model"""
        
        # Attach labels to graphs
        for i, graph in enumerate(train_graphs):
            graph.y = torch.tensor([y_train[i]], dtype=torch.float32 if self.task_type == 'regression' else torch.long)  # type: ignore
        
        for i, graph in enumerate(val_graphs):
            graph.y = torch.tensor([y_val[i]], dtype=torch.float32 if self.task_type == 'regression' else torch.long)  # type: ignore
        
        # Create data loaders
        batch_size = params.get('batch_size', self.batch_size)
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)  # type: ignore
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)  # type: ignore
        
        # Create optimizer and scheduler
        optimizer = self._create_optimizer(model, params)
        scheduler = self._create_scheduler(optimizer, params)
        
        # Loss function
        if self.task_type == 'regression':
            criterion = nn.MSELoss()  # type: ignore
        else:
            criterion = nn.CrossEntropyLoss()  # type: ignore
        
        # Training loop
        best_val_metric = -np.inf if self.task_type == 'regression' else -np.inf
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_metric = self._evaluate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            if scheduler is not None:
                if ReduceLROnPlateau is not None and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()  # type: ignore
            
            # Early stopping
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, best_val_metric
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for hyperparameter optimization"""
        try:
            # Get trial parameters
            params = self._suggest_params(trial)
            
            # Convert SMILES to graphs if not already done
            if not hasattr(self, '_graphs_converted'):
                self.console.print("[dim]Converting SMILES to graphs...[/dim]")
                train_graphs = self._convert_smiles_to_graphs(X_train)
                val_graphs = self._convert_smiles_to_graphs(X_val)
                self._graphs_converted = True
            else:
                train_graphs = self._convert_smiles_to_graphs(X_train)
                val_graphs = self._convert_smiles_to_graphs(X_val)
            
            # Handle CV if specified
            if self.cv is not None and self.cv > 1:
                return self._cv_objective(trial, params, train_graphs, y_train.ravel())
            else:
                # Single train/val split
                model = self._create_model(params)
                _, score = self._train_model(model, train_graphs, val_graphs, 
                                          y_train.ravel(), y_val.ravel(), params)
                return score
                
        except Exception as e:
            self.console.print(f"[red]Error in GNN trial {trial.number}: {e}[/red]")
            return -np.inf if self.task_type == 'regression' else 0.0
    
    def _cv_objective(self, trial, params: Dict[str, Any], 
                     graphs: List[Data], y: np.ndarray) -> float:  # type: ignore
        """Cross-validation objective function"""
        
        cv_splits = self.cv if self.cv is not None else 5
        if self.task_type == 'regression':
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        else:
            kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(graphs)), y)):
            # Split data
            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Create and train model
            model = self._create_model(params)
            _, score = self._train_model(model, train_graphs, val_graphs, 
                                      y_train_fold, y_val_fold, params)
            fold_scores.append(score)
        
        # Store fold scores for logging
        trial.set_user_attr("fold_scores", fold_scores)
        
        return float(np.mean(fold_scores))
    
    def fit(self, X_train, y_train):
        """Fit the best model"""
        if self.best_params_ is None:
            raise ValueError("Optimization has not been run. Call optimize() first.")
        
        # Convert SMILES to graphs
        train_graphs = self._convert_smiles_to_graphs(X_train)
        
        # Create and train best model
        self.best_model_ = self._create_model(self.best_params_)
        
        # For final training, use full training set as both train and val
        # (This is not ideal but matches the sklearn interface)
        self.best_model_, _ = self._train_model(
            self.best_model_, train_graphs, train_graphs, 
            y_train.ravel(), y_train.ravel(), self.best_params_
        )
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.best_model_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Convert SMILES to graphs
        graphs = self._convert_smiles_to_graphs(X)
        
        # Add dummy labels for data loader
        for graph in graphs:
            graph.y = torch.tensor([0.0], dtype=torch.float32)  # type: ignore
        
        # Create data loader
        dataloader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)  # type: ignore
        
        # Make predictions
        self.best_model_.eval()
        predictions: List[float] = []  # type: ignore
        
        with torch.no_grad():  # type: ignore
            for batch in dataloader:
                batch = batch.to(self.device)
                output = self.best_model_(batch)
                
                if self.task_type == 'regression':
                    # Handle both single and multiple predictions
                    output_np = output.squeeze().cpu().numpy()
                    if output_np.ndim == 0:  # Single prediction
                        predictions.append(float(output_np))
                    else:  # Multiple predictions
                        predictions.extend(output_np.tolist())  # type: ignore
                else:
                    predictions.extend(torch.argmax(output, dim=1).cpu().numpy())  # type: ignore
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Make probability predictions for classification"""
        if self.task_type == 'regression':
            raise ValueError("predict_proba is not available for regression tasks")
        
        if self.best_model_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Convert SMILES to graphs
        graphs = self._convert_smiles_to_graphs(X)
        
        # Add dummy labels for data loader
        for graph in graphs:
            graph.y = torch.tensor([0], dtype=torch.long)  # type: ignore
        
        # Create data loader
        dataloader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)  # type: ignore
        
        # Make predictions
        self.best_model_.eval()
        probabilities: List[np.ndarray] = []  # type: ignore
        
        with torch.no_grad():  # type: ignore
            for batch in dataloader:
                batch = batch.to(self.device)
                output = self.best_model_(batch)
                probs = F.softmax(output, dim=1)  # type: ignore
                probabilities.extend(probs.cpu().numpy())  # type: ignore
        
        return np.array(probabilities)
    
    def get_cv_predictions(self, X_train_full_for_cv, y_train_full_for_cv):
        """Get cross-validation predictions"""
        if self.best_params_ is None:
            raise ValueError("Best parameters not found. Run optimize() first.")
        
        if self.cv is None or self.cv < 2:
            self.console.print(f"CV for HPO was not used, cannot get OOF CV predictions for {self.model_name}.")
            return None
        
        # Convert SMILES to graphs
        graphs = self._convert_smiles_to_graphs(X_train_full_for_cv)
        y_ravel = y_train_full_for_cv.ravel()
        
        cv_splits = self.cv if self.cv is not None else 5
        if self.task_type == 'regression':
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        else:
            kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        
        oof_preds = np.zeros_like(y_ravel, dtype=float)
        oof_probas = None
        
        if self.task_type != 'regression':
            num_classes = self.num_classes if self.num_classes and self.num_classes >= 2 else 2
            oof_probas = np.zeros((len(y_ravel), num_classes))
        
        fold_metrics_list = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(graphs)), y_ravel)):
            self.console.print(f"  Generating predictions for CV OOF fold {fold_idx + 1}/{self.cv}...")
            
            # Split data
            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]
            y_train = y_ravel[train_idx]
            y_val = y_ravel[val_idx]
            
            # Create and train model
            model = self._create_model(self.best_params_)
            model, _ = self._train_model(model, train_graphs, val_graphs, y_train, y_val, self.best_params_)
            
            # Make predictions
            val_preds: List[float] = []  # type: ignore
            val_probs: Optional[List[np.ndarray]] = None  # type: ignore
            
            if self.task_type != 'regression':
                val_probs = []  # type: ignore
            
            # Add labels for prediction
            for i, graph in enumerate(val_graphs):
                graph.y = torch.tensor([0.0], dtype=torch.float32)  # type: ignore
            
            val_loader = DataLoader(val_graphs, batch_size=self.batch_size, shuffle=False)  # type: ignore
            
            model.eval()
            with torch.no_grad():  # type: ignore
                for batch in val_loader:
                    batch = batch.to(self.device)
                    output = model(batch)
                    
                    if self.task_type == 'regression':
                        val_preds.extend(output.squeeze().cpu().numpy())  # type: ignore
                    else:
                        val_preds.extend(torch.argmax(output, dim=1).cpu().numpy())  # type: ignore
                        probs = F.softmax(output, dim=1)  # type: ignore
                        if val_probs is not None:
                            val_probs.extend(probs.cpu().numpy())  # type: ignore
            
            oof_preds[val_idx] = val_preds
            if oof_probas is not None and val_probs is not None:
                oof_probas[val_idx, :] = val_probs
            
            # Calculate fold metrics
            fold_metrics: Dict[str, Union[int, float]] = {'fold': fold_idx + 1}  # type: ignore
            if self.task_type == 'regression':
                fold_metrics['r2'] = float(r2_score(y_val, val_preds))  # type: ignore
                fold_metrics['rmse'] = float(np.sqrt(mean_squared_error(y_val, val_preds)))  # type: ignore
                fold_metrics['mae'] = float(mean_absolute_error(y_val, val_preds))  # type: ignore
            else:
                avg = 'binary' if self.task_type == 'binary_classification' else 'weighted'
                fold_metrics['accuracy'] = float(accuracy_score(y_val, val_preds))  # type: ignore
                fold_metrics['f1'] = float(f1_score(y_val, val_preds, average=avg, zero_division='warn'))  # type: ignore
                fold_metrics['precision'] = float(precision_score(y_val, val_preds, average=avg, zero_division='warn'))  # type: ignore
                fold_metrics['recall'] = float(recall_score(y_val, val_preds, average=avg, zero_division='warn'))  # type: ignore
            
            fold_metrics_list.append(fold_metrics)
        
        oof_payload = {'y_true_oof': y_ravel, 'y_pred_oof': oof_preds, 'y_proba_oof': oof_probas}
        return {'oof_preds': oof_payload, 'fold_metrics': fold_metrics_list}


# Example usage
if __name__ == "__main__":
    # Test GNN optimizer creation
    if torch is not None:
        try:
            optimizer = GNNOptimizer(
                model_name='gcn',
                smiles_columns=['smiles'],
                n_trials=5,
                cv=3,
                task_type='regression'
            )
            print("✓ Successfully created GNN optimizer")
        except Exception as e:
            print(f"✗ Failed to create GNN optimizer: {e}")
    else:
        print("PyTorch not available, skipping GNN optimizer test") 