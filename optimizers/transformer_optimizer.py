"""
Transformer优化器
支持各种Transformer架构的超参数优化和训练
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from rich.console import Console

from .base_optimizer import BaseOptimizer

# 条件导入PyTorch相关库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import Adam, AdamW, SGD
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

try:
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.metrics import (
        r2_score, mean_squared_error, mean_absolute_error,
        accuracy_score, f1_score, precision_score, recall_score
    )
    from sklearn.preprocessing import StandardScaler
except ImportError:
    pass

# 导入Transformer模型
try:
    from models.transformer_models import create_transformer_model
except ImportError:
    create_transformer_model = None


class TransformerOptimizer(BaseOptimizer):
    """Transformer模型优化器"""

    def __init__(self,
                 model_type: str,
                 smiles_columns: Optional[List[str]] = None,
                 feature_columns: Optional[List[str]] = None,
                 n_trials: int = 100,
                 random_state: int = 42,
                 cv: Optional[int] = None,
                 task_type: str = 'regression',
                 num_classes: Optional[int] = None,
                 device: str = 'auto',
                 max_epochs: int = 100,
                 early_stopping_patience: int = 20,
                 batch_size: int = 32,
                 vocab_size: int = 1000,
                 max_length: int = 512):

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer models")

        if create_transformer_model is None:
            raise ImportError("Transformer models are not available")

        # 设置模型属性
        self.model_type = model_type.lower()
        self.smiles_columns = smiles_columns or []
        self.feature_columns = feature_columns or []
        self.device = self._get_device(device)
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_length = max_length

        # 验证模型类型和输入
        self._validate_inputs()

        # 获取参数网格
        param_grid = self._get_param_grid()

        # 初始化基础优化器
        super().__init__(
            model_class=None,  # 动态创建模型
            param_grid=param_grid,
            n_trials=n_trials,
            random_state=random_state,
            cv=cv,
            task_type=task_type,
            num_classes=num_classes
        )

        self.console = Console()

        # 最佳模型和参数
        self.best_model_ = None
        self.best_params_ = None
        self.scaler_ = None

    def _get_device(self, device: str) -> torch.device:
        """确定训练设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)

    def _validate_inputs(self):
        """验证输入参数"""
        valid_types = ['smiles', 'feature', 'pretrained', 'multimodal']
        if self.model_type not in valid_types:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be one of {valid_types}")

        if self.model_type in ['smiles', 'pretrained'] and not self.smiles_columns:
            raise ValueError(f"model_type '{self.model_type}' requires smiles_columns")

        if self.model_type == 'feature' and not self.feature_columns:
            raise ValueError("model_type 'feature' requires feature_columns")

        if self.model_type == 'multimodal' and (not self.smiles_columns or not self.feature_columns):
            raise ValueError("model_type 'multimodal' requires both smiles_columns and feature_columns")

    def _get_param_grid(self) -> Dict[str, Dict[str, Any]]:
        """获取超参数搜索空间"""
        base_params = {
            'd_model': {'type': 'categorical', 'choices': [128, 256, 512]},
            'nhead': {'type': 'categorical', 'choices': [4, 8, 16]},
            'num_layers': {'type': 'int', 'low': 2, 'high': 8},
            'dim_feedforward': {'type': 'categorical', 'choices': [512, 1024, 2048]},
            'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-3, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
            'optimizer_type': {'type': 'categorical', 'choices': ['adam', 'adamw', 'sgd']},
            'scheduler_type': {'type': 'categorical', 'choices': ['plateau', 'cosine', 'step', 'none']},
        }

        # 模型特定参数
        if self.model_type == 'pretrained':
            base_params.update({
                'model_name': {'type': 'categorical', 'choices': [
                    'bert-base-uncased', 'roberta-base', 'distilbert-base-uncased'
                ]},
                'freeze_backbone': {'type': 'categorical', 'choices': [True, False]}
            })

        elif self.model_type == 'multimodal':
            base_params.update({
                'fusion_method': {'type': 'categorical', 'choices': ['concat', 'attention', 'gated', 'add']}
            })

        return base_params

    def _prepare_data(self, X: pd.DataFrame) -> Tuple[List[str], Optional[torch.Tensor]]:
        """准备输入数据"""
        smiles_data = None
        feature_data = None

        # 提取SMILES数据
        if self.smiles_columns:
            if len(self.smiles_columns) == 1:
                smiles_data = X[self.smiles_columns[0]].tolist()
            else:
                # 多个SMILES列，拼接处理
                smiles_data = []
                for _, row in X.iterrows():
                    combined_smiles = ' '.join([str(row[col]) for col in self.smiles_columns if pd.notna(row[col])])
                    smiles_data.append(combined_smiles)

        # 提取特征数据
        if self.feature_columns:
            feature_data = torch.tensor(X[self.feature_columns].values, dtype=torch.float32)

            # 标准化特征
            if self.scaler_ is None:
                self.scaler_ = StandardScaler()
                feature_data = torch.tensor(self.scaler_.fit_transform(feature_data), dtype=torch.float32)
            else:
                feature_data = torch.tensor(self.scaler_.transform(feature_data), dtype=torch.float32)

        return smiles_data, feature_data

    def _create_model(self, params: Dict[str, Any]) -> nn.Module:
        """创建Transformer模型"""
        model_params = {
            'd_model': params.get('d_model', 256),
            'nhead': params.get('nhead', 8),
            'num_layers': params.get('num_layers', 4),
            'dim_feedforward': params.get('dim_feedforward', 1024),
            'dropout': params.get('dropout', 0.1),
            'max_length': self.max_length,
            'output_dim': self.num_classes if self.task_type != 'regression' else 1,
            'task_type': self.task_type
        }

        # 模型特定参数
        if self.model_type in ['smiles', 'multimodal']:
            model_params['vocab_size'] = self.vocab_size

        if self.model_type in ['feature', 'multimodal']:
            model_params['input_dim'] = len(self.feature_columns)
            if self.model_type == 'multimodal':
                model_params['feature_dim'] = len(self.feature_columns)

        if self.model_type == 'pretrained':
            model_params['model_name'] = params.get('model_name', 'bert-base-uncased')
            model_params['freeze_backbone'] = params.get('freeze_backbone', False)

        if self.model_type == 'multimodal':
            model_params['fusion_method'] = params.get('fusion_method', 'concat')

        model = create_transformer_model(self.model_type, **model_params)
        return model.to(self.device)

    def _create_optimizer(self, model: nn.Module, params: Dict[str, Any]):
        """创建优化器"""
        lr = params.get('learning_rate', 1e-3)
        weight_decay = params.get('weight_decay', 1e-5)
        optimizer_type = params.get('optimizer_type', 'adam')

        if optimizer_type == 'adam':
            return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _create_scheduler(self, optimizer, params: Dict[str, Any]):
        """创建学习率调度器"""
        scheduler_type = params.get('scheduler_type', 'plateau')

        if scheduler_type == 'plateau':
            return ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        elif scheduler_type == 'step':
            return StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            return None

    def _train_epoch(self, model: nn.Module, dataloader: DataLoader, optimizer, criterion) -> float:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_data in dataloader:
            optimizer.zero_grad()

            # 准备输入
            if self.model_type == 'smiles' or self.model_type == 'pretrained':
                smiles_batch, targets = batch_data
                outputs = model(smiles_batch)
            elif self.model_type == 'feature':
                features, targets = batch_data
                features = features.to(self.device)
                outputs = model(features)
            elif self.model_type == 'multimodal':
                smiles_batch, features, targets = batch_data
                features = features.to(self.device)
                outputs = model(smiles_batch, features)

            targets = targets.to(self.device)

            # 计算损失
            if self.task_type == 'regression':
                loss = criterion(outputs.squeeze(), targets.float())
            else:
                loss = criterion(outputs, targets.long())

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _evaluate_epoch(self, model: nn.Module, dataloader: DataLoader, criterion) -> Tuple[float, float]:
        """评估一个epoch"""
        model.eval()
        total_loss = 0.0
        predictions = []
        targets_list = []
        num_batches = 0

        with torch.no_grad():
            for batch_data in dataloader:
                # 准备输入
                if self.model_type == 'smiles' or self.model_type == 'pretrained':
                    smiles_batch, targets = batch_data
                    outputs = model(smiles_batch)
                elif self.model_type == 'feature':
                    features, targets = batch_data
                    features = features.to(self.device)
                    outputs = model(features)
                elif self.model_type == 'multimodal':
                    smiles_batch, features, targets = batch_data
                    features = features.to(self.device)
                    outputs = model(smiles_batch, features)

                targets = targets.to(self.device)

                # 计算损失
                if self.task_type == 'regression':
                    loss = criterion(outputs.squeeze(), targets.float())
                    predictions.extend(outputs.squeeze().cpu().numpy())
                else:
                    loss = criterion(outputs, targets.long())
                    predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

                targets_list.extend(targets.cpu().numpy())
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # 计算指标
        if self.task_type == 'regression':
            metric = r2_score(targets_list, predictions)
        else:
            avg_method = 'binary' if self.task_type == 'binary_classification' else 'weighted'
            metric = f1_score(targets_list, predictions, average=avg_method, zero_division=0)

        return avg_loss, metric

    def _create_dataloader(self, smiles_data: Optional[List[str]],
                          feature_data: Optional[torch.Tensor],
                          targets: np.ndarray,
                          batch_size: int,
                          shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""

        if self.model_type == 'smiles' or self.model_type == 'pretrained':
            # 自定义数据集类用于SMILES
            class SMILESDataset:
                def __init__(self, smiles_list, targets):
                    self.smiles_list = smiles_list
                    self.targets = torch.tensor(targets, dtype=torch.float32)

                def __len__(self):
                    return len(self.smiles_list)

                def __getitem__(self, idx):
                    return self.smiles_list[idx], self.targets[idx]

            dataset = SMILESDataset(smiles_data, targets)

        elif self.model_type == 'feature':
            dataset = TensorDataset(feature_data, torch.tensor(targets, dtype=torch.float32))

        elif self.model_type == 'multimodal':
            # 多模态数据集
            class MultiModalDataset:
                def __init__(self, smiles_list, features, targets):
                    self.smiles_list = smiles_list
                    self.features = features
                    self.targets = torch.tensor(targets, dtype=torch.float32)

                def __len__(self):
                    return len(self.smiles_list)

                def __getitem__(self, idx):
                    return self.smiles_list[idx], self.features[idx], self.targets[idx]

            dataset = MultiModalDataset(smiles_data, feature_data, targets)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _train_model(self, model: nn.Module,
                    train_smiles: Optional[List[str]], train_features: Optional[torch.Tensor], y_train: np.ndarray,
                    val_smiles: Optional[List[str]], val_features: Optional[torch.Tensor], y_val: np.ndarray,
                    params: Dict[str, Any]) -> Tuple[nn.Module, float]:
        """训练模型"""

        # 创建数据加载器
        batch_size = params.get('batch_size', self.batch_size)
        train_loader = self._create_dataloader(train_smiles, train_features, y_train, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_smiles, val_features, y_val, batch_size, shuffle=False)

        # 创建优化器和调度器
        optimizer = self._create_optimizer(model, params)
        scheduler = self._create_scheduler(optimizer, params)

        # 损失函数
        if self.task_type == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_val_metric = -np.inf
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # 训练
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)

            # 验证
            val_loss, val_metric = self._evaluate_epoch(model, val_loader, criterion)

            # 更新调度器
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # 早停
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                break

        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, best_val_metric

    def objective(self, trial, X_train, y_train, X_val, y_val):
        """目标函数用于超参数优化"""
        try:
            # 获取试验参数
            params = self._suggest_params(trial)

            # 准备数据
            train_smiles, train_features = self._prepare_data(X_train)
            val_smiles, val_features = self._prepare_data(X_val)

            # 处理交叉验证
            if self.cv is not None and self.cv > 1:
                return self._cv_objective(trial, params, X_train, y_train.ravel())
            else:
                # 单次训练验证
                model = self._create_model(params)
                _, score = self._train_model(
                    model, train_smiles, train_features, y_train.ravel(),
                    val_smiles, val_features, y_val.ravel(), params
                )
                return score

        except Exception as e:
            self.console.print(f"[red]Error in Transformer trial {trial.number}: {e}[/red]")
            return -np.inf if self.task_type == 'regression' else 0.0

    def _cv_objective(self, trial, params: Dict[str, Any], X: pd.DataFrame, y: np.ndarray) -> float:
        """交叉验证目标函数"""

        cv_splits = self.cv if self.cv is not None else 5
        if self.task_type == 'regression':
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        else:
            kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            # 拆分数据
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # 准备数据
            train_smiles, train_features = self._prepare_data(X_train_fold)
            val_smiles, val_features = self._prepare_data(X_val_fold)

            # 创建和训练模型
            model = self._create_model(params)
            _, score = self._train_model(
                model, train_smiles, train_features, y_train_fold,
                val_smiles, val_features, y_val_fold, params
            )
            fold_scores.append(score)

        # 存储折叠得分
        trial.set_user_attr("fold_scores", fold_scores)

        return float(np.mean(fold_scores))

    def fit(self, X_train, y_train):
        """训练最佳模型"""
        if self.best_params_ is None:
            raise ValueError("Optimization has not been run. Call optimize() first.")

        # 准备数据
        train_smiles, train_features = self._prepare_data(X_train)

        # 创建和训练最佳模型
        self.best_model_ = self._create_model(self.best_params_)

        # 使用全部训练数据进行最终训练
        self.best_model_, _ = self._train_model(
            self.best_model_, train_smiles, train_features, y_train.ravel(),
            train_smiles, train_features, y_train.ravel(), self.best_params_
        )

        return self

    def predict(self, X):
        """进行预测"""
        if self.best_model_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # 准备数据
        smiles_data, feature_data = self._prepare_data(X)

        # 创建数据加载器
        dummy_targets = np.zeros(len(X))  # 虚拟目标值
        dataloader = self._create_dataloader(smiles_data, feature_data, dummy_targets, self.batch_size, shuffle=False)

        # 进行预测
        self.best_model_.eval()
        predictions = []

        with torch.no_grad():
            for batch_data in dataloader:
                # 准备输入
                if self.model_type == 'smiles' or self.model_type == 'pretrained':
                    smiles_batch, _ = batch_data
                    outputs = self.best_model_(smiles_batch)
                elif self.model_type == 'feature':
                    features, _ = batch_data
                    features = features.to(self.device)
                    outputs = self.best_model_(features)
                elif self.model_type == 'multimodal':
                    smiles_batch, features, _ = batch_data
                    features = features.to(self.device)
                    outputs = self.best_model_(smiles_batch, features)

                if self.task_type == 'regression':
                    predictions.extend(outputs.squeeze().cpu().numpy())
                else:
                    predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X):
        """预测概率（分类任务）"""
        if self.task_type == 'regression':
            raise ValueError("predict_proba is not available for regression tasks")

        if self.best_model_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # 准备数据
        smiles_data, feature_data = self._prepare_data(X)

        # 创建数据加载器
        dummy_targets = np.zeros(len(X))
        dataloader = self._create_dataloader(smiles_data, feature_data, dummy_targets, self.batch_size, shuffle=False)

        # 进行预测
        self.best_model_.eval()
        probabilities = []

        with torch.no_grad():
            for batch_data in dataloader:
                # 准备输入
                if self.model_type == 'smiles' or self.model_type == 'pretrained':
                    smiles_batch, _ = batch_data
                    outputs = self.best_model_(smiles_batch)
                elif self.model_type == 'feature':
                    features, _ = batch_data
                    features = features.to(self.device)
                    outputs = self.best_model_(features)
                elif self.model_type == 'multimodal':
                    smiles_batch, features, _ = batch_data
                    features = features.to(self.device)
                    outputs = self.best_model_(smiles_batch, features)

                probs = F.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def get_cv_predictions(self, X_train_full_for_cv, y_train_full_for_cv):
        """获取交叉验证预测"""
        if self.best_params_ is None:
            raise ValueError("Best parameters not found. Run optimize() first.")

        if self.cv is None or self.cv < 2:
            self.console.print(f"CV for HPO was not used, cannot get OOF CV predictions for Transformer.")
            return None

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

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full_for_cv, y_ravel)):
            self.console.print(f"  Generating predictions for CV OOF fold {fold_idx + 1}/{self.cv}...")

            # 拆分数据
            X_train_fold = X_train_full_for_cv.iloc[train_idx]
            X_val_fold = X_train_full_for_cv.iloc[val_idx]
            y_train_fold = y_ravel[train_idx]
            y_val_fold = y_ravel[val_idx]

            # 准备数据
            train_smiles, train_features = self._prepare_data(X_train_fold)
            val_smiles, val_features = self._prepare_data(X_val_fold)

            # 创建和训练模型
            model = self._create_model(self.best_params_)
            model, _ = self._train_model(
                model, train_smiles, train_features, y_train_fold,
                val_smiles, val_features, y_val_fold, self.best_params_
            )

            # 预测验证集
            dummy_targets = np.zeros(len(X_val_fold))
            val_loader = self._create_dataloader(val_smiles, val_features, dummy_targets, self.batch_size, shuffle=False)

            val_preds = []
            val_probs = [] if self.task_type != 'regression' else None

            model.eval()
            with torch.no_grad():
                for batch_data in val_loader:
                    # 准备输入
                    if self.model_type == 'smiles' or self.model_type == 'pretrained':
                        smiles_batch, _ = batch_data
                        outputs = model(smiles_batch)
                    elif self.model_type == 'feature':
                        features, _ = batch_data
                        features = features.to(self.device)
                        outputs = model(features)
                    elif self.model_type == 'multimodal':
                        smiles_batch, features, _ = batch_data
                        features = features.to(self.device)
                        outputs = model(smiles_batch, features)

                    if self.task_type == 'regression':
                        val_preds.extend(outputs.squeeze().cpu().numpy())
                    else:
                        val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                        probs = F.softmax(outputs, dim=1)
                        val_probs.extend(probs.cpu().numpy())

            oof_preds[val_idx] = val_preds
            if oof_probas is not None and val_probs is not None:
                oof_probas[val_idx, :] = val_probs

        return {
            'oof_predictions': oof_preds,
            'oof_probabilities': oof_probas,
            'cv_folds': self.cv
        }
