# optimizers/ann_optimizer.py
import torch
import torch.nn as nn
import torch.optim as optim
from .base_optimizer import BaseOptimizer
from models.ann import ComplexANN
import numpy as np
from sklearn.metrics import r2_score, f1_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
import copy
from typing import Dict, Union

class ANNOptimizer(BaseOptimizer):
    def __init__(self, n_trials=100, random_state=42, cv=None, task_type='regression', num_classes=None):
        # --- FIXED: Only include base parameters, layer-specific parameters will be suggested dynamically ---
        param_grid = {
            'n_layers': {'type': 'int', 'low': 1, 'high': 6 },
            'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.6},
            'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True},
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'epochs': {'type': 'int', 'low': 200, 'high': 8000 },
            'patience': {'type': 'categorical', 'choices': [20, 50, 200, 500]}
        }
        super().__init__(ComplexANN, param_grid, n_trials, random_state, cv, task_type, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hpo_trained_model = None  # Store model trained during HPO for train_valid_test mode
        print(f"ANNOptimizer using device: {self.device} for task: {self.task_type}")
        if self.task_type == 'binary_classification': self.ann_output_size, self.criterion = 1, nn.BCEWithLogitsLoss()
        elif self.task_type == 'multiclass_classification':
            if self.num_classes is None or self.num_classes < 2: raise ValueError("num_classes must be specified and >= 2 for multiclass_classification.")
            self.ann_output_size, self.criterion = self.num_classes, nn.CrossEntropyLoss()
        elif self.task_type == 'regression': self.ann_output_size, self.criterion = 1, nn.MSELoss()
        else: raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _suggest_layer_params(self, trial, n_layers):
        """Dynamically suggest layer parameters based on the actual number of layers."""
        # Layer configs by position: L0 (first layer) should be widest, then gradually narrow
        layer_configs = {
            0: {'low': 8, 'high': 2048},  # L0: widest layer (receives input features)
            1: {'low': 8, 'high': 1024},   # L1: large
            2: {'low': 8, 'high': 512},    # L2: medium-large
            3: {'low': 8, 'high': 256},    # L3: medium
            4: {'low': 8, 'high': 64},      # L4: small
            5: {'low': 8, 'high': 32},      # L5: smallest (closer to output)
        }
        
        hidden_layers = []
        last_units = float('inf')
        
        for i in range(n_layers):
            config = layer_configs.get(i, {'low': 8, 'high': 32})  # Default for layers beyond 5
            low, high = config['low'], min(config['high'], last_units)
            if low > high: low = high
            
            param_name = f'n_units_l{i}'
            n_units = trial.suggest_int(param_name, low, high, step=8)
            hidden_layers.append(n_units)
            last_units = n_units
            
        return hidden_layers

    def _create_hidden_layers_config(self, trial_or_params):
        """Create hidden layers configuration, handling both trial objects and parameter dictionaries."""
        is_trial = not isinstance(trial_or_params, dict)
        
        if is_trial:
            # During optimization: suggest n_layers first, then suggest layer-specific parameters
            n_layers = trial_or_params.suggest_int('n_layers', self.param_grid['n_layers']['low'], self.param_grid['n_layers']['high'])
            hidden_layers = self._suggest_layer_params(trial_or_params, n_layers)
        else:
            # Using saved parameters: only use parameters for the actual number of layers
            n_layers = trial_or_params['n_layers']
            hidden_layers = []
            for i in range(n_layers):
                param_name = f'n_units_l{i}'
                if param_name in trial_or_params:
                    hidden_layers.append(trial_or_params[param_name])
                else:
                    # Fallback if parameter is missing (shouldn't happen with proper saved params)
                    hidden_layers.append(128)  # Default value
        
        return hidden_layers

    def objective(self, trial, X_train, y_train, X_val, y_val):
        # --- FIXED: Get base parameters first, then create layer config ---
        base_params = {}
        for param_name, param_info in self.param_grid.items():
            if param_name != 'n_layers':  # n_layers is handled in _create_hidden_layers_config
                if param_info['type'] == 'int':
                    base_params[param_name] = trial.suggest_int(param_name, param_info['low'], param_info['high'])
                elif param_info['type'] == 'float':
                    base_params[param_name] = trial.suggest_float(param_name, param_info['low'], param_info['high'], log=param_info.get('log', False))
                elif param_info['type'] == 'categorical':
                    base_params[param_name] = trial.suggest_categorical(param_name, param_info['choices'])
        
        hidden_layers_config = self._create_hidden_layers_config(trial)
        dropout_rate, weight_decay, learning_rate, epochs, patience = base_params['dropout_rate'], base_params['weight_decay'], base_params['learning_rate'], base_params['epochs'], base_params['patience']
        y_train_prepared, y_target_type_torch, y_val_prepared = self._prepare_y_data(y_train, y_val)
        fold_scores_list = []
        trial_model = None  # Store the model from this trial
        
        def train_and_evaluate_fold(model, optimizer, X_train_t, y_train_t, X_val_t, y_val_np):
            best_val_loss, patience_counter, best_model_state = float('inf'), 0, None
            for epoch in range(epochs):
                model.train(); optimizer.zero_grad(); outputs = model(X_train_t); loss = self.criterion(outputs, y_train_t); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t); val_loss = self.criterion(val_outputs, torch.tensor(y_val_np, dtype=y_target_type_torch).to(self.device))
                if val_loss < best_val_loss: best_val_loss, patience_counter, best_model_state = val_loss, 0, copy.deepcopy(model.state_dict())
                else: patience_counter += 1
                if patience_counter >= patience: break
            if best_model_state: model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                y_pred_logits = model(X_val_t).cpu()
                if self.task_type == 'regression': score = r2_score(y_val_np, y_pred_logits.numpy())
                elif self.task_type == 'binary_classification': score = f1_score(y_val_np, (torch.sigmoid(y_pred_logits).numpy() > 0.5).astype(int), average='binary', zero_division='warn')
                else: score = f1_score(y_val_np, torch.argmax(y_pred_logits, dim=1).numpy(), average='weighted', zero_division='warn')
            return score, model
            
        if self.cv is not None and self.cv > 1:
            kf = self._get_cv_splitter(); y_for_kf_split = y_train_prepared.ravel() if self.task_type != 'regression' else y_train_prepared
            for train_idx, val_idx in kf.split(X_train, y_for_kf_split):
                X_fold_train, X_fold_val, y_fold_train_np, y_fold_val_np = X_train[train_idx], X_train[val_idx], y_train_prepared[train_idx], y_train_prepared[val_idx]
                model = ComplexANN(X_fold_train.shape[1], hidden_layers_config, self.ann_output_size, self.task_type, dropout_rate).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                X_fold_train_t, y_fold_train_t, X_fold_val_t = torch.tensor(X_fold_train, dtype=torch.float32).to(self.device), torch.tensor(y_fold_train_np, dtype=y_target_type_torch).to(self.device), torch.tensor(X_fold_val, dtype=torch.float32).to(self.device)
                score, _ = train_and_evaluate_fold(model, optimizer, X_fold_train_t, y_fold_train_t, X_fold_val_t, y_fold_val_np)
                fold_scores_list.append(score)
        else:
            # No CV path - train on full train set, validate on validation set
            model = ComplexANN(X_train.shape[1], hidden_layers_config, self.ann_output_size, self.task_type, dropout_rate).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            X_train_t, y_train_t, X_val_t = torch.tensor(X_train, dtype=torch.float32).to(self.device), torch.tensor(y_train_prepared, dtype=y_target_type_torch).to(self.device), torch.tensor(X_val, dtype=torch.float32).to(self.device)
            score, trained_model = train_and_evaluate_fold(model, optimizer, X_train_t, y_train_t, X_val_t, y_val_prepared)
            trial_model = trained_model  # Store this model for potential reuse
            fold_scores_list.append(score)
            
        mean_score = np.mean(fold_scores_list) if fold_scores_list else (-np.inf if self.task_type == 'regression' else 0.0)
        
        # --- FIXED: Store the best trial model for reuse in train_valid_test mode ---
        if trial_model is not None and (not hasattr(self, '_best_trial_score') or mean_score > self._best_trial_score):
            self._best_trial_score = mean_score
            self.hpo_trained_model = trial_model
            
        trial.set_user_attr("fold_scores", fold_scores_list); return mean_score

    def fit(self, X_train, y_train):
        if self.best_params_ is None: raise ValueError("Optimization has not been run. Call optimize() first.")
        
        # --- FIXED: Check if we have a model from HPO phase that can be reused ---
        if self.hpo_trained_model is not None and self.cv is None:
            # In train_valid_test mode, reuse the model trained during HPO
            self.best_model_ = self.hpo_trained_model
            return
            
        # Otherwise, train a new model (this happens in CV mode)
        params = self.best_params_; hidden_layers_config = self._create_hidden_layers_config(params)
        dropout_rate, weight_decay, learning_rate, epochs = params['dropout_rate'], params['weight_decay'], params['learning_rate'], params['epochs']
        y_train_prepared, y_target_type_torch, _ = self._prepare_y_data(y_train)
        self.best_model_ = ComplexANN(X_train.shape[1], hidden_layers_config, self.ann_output_size, self.task_type, dropout_rate).to(self.device)
        optimizer = optim.Adam(self.best_model_.parameters(), lr=learning_rate, weight_decay=weight_decay)
        X_tensor, y_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device), torch.tensor(y_train_prepared, dtype=y_target_type_torch).to(self.device)
        self.best_model_.train()
        for _ in range(epochs): optimizer.zero_grad(); outputs = self.best_model_(X_tensor); loss = self.criterion(outputs, y_tensor); loss.backward(); optimizer.step()

    def _prepare_y_data(self, y_train, y_val=None):
        y_train_prepared, y_val_prepared = y_train.copy(), y_val.copy() if y_val is not None else None
        if self.task_type == 'regression':
            if y_train_prepared.ndim == 1: y_train_prepared = y_train_prepared.reshape(-1, 1)
            if y_val_prepared is not None and y_val_prepared.ndim == 1: y_val_prepared = y_val_prepared.reshape(-1, 1)
            y_target_type_torch = torch.float32
        elif self.task_type == 'binary_classification':
            if y_train_prepared.ndim == 1: y_train_prepared = y_train_prepared.reshape(-1, 1)
            if y_val_prepared is not None and y_val_prepared.ndim == 1: y_val_prepared = y_val_prepared.reshape(-1, 1)
            y_target_type_torch = torch.float32
        else:
            y_train_prepared = y_train_prepared.ravel()
            if y_val_prepared is not None: y_val_prepared = y_val_prepared.ravel()
            y_target_type_torch = torch.long
        return y_train_prepared, y_target_type_torch, y_val_prepared

    def _get_cv_splitter(self):
        cv_splits = self.cv if self.cv is not None else 5
        return KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state) if self.task_type == 'regression' else StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

    def predict(self, X):
        if self.best_model_ is None: raise ValueError("Model has not been fitted. Call fit() first.")
        self.best_model_.eval()
        with torch.no_grad():
            logits = self.best_model_(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu()
            if self.task_type == 'regression': return logits.numpy()
            elif self.task_type == 'binary_classification': return (torch.sigmoid(logits).numpy() > 0.5).astype(int)
            else: return torch.argmax(logits, dim=1).numpy()
            
    def predict_proba(self, X):
        if self.best_model_ is None: raise ValueError("Model has not been fitted. Call fit() first.")
        if not (self.task_type == 'binary_classification' or self.task_type == 'multiclass_classification'): raise AttributeError("predict_proba is only for classification tasks.")
        self.best_model_.eval()
        with torch.no_grad():
            logits = self.best_model_(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu()
            if self.task_type == 'binary_classification': probs = torch.sigmoid(logits).numpy(); return np.hstack([1 - probs, probs])
            elif self.task_type == 'multiclass_classification': return torch.softmax(logits, dim=1).numpy()
            return None
            
    def get_cv_predictions(self, X_train_full_for_cv, y_train_full_for_cv):
        if self.best_params_ is None: raise ValueError("Best parameters not found. Run optimize() first.")
        if self.cv is None or self.cv < 2:
            print("CV was not used, cannot get CV predictions for ANN."); return None
        
        params = self.best_params_; hidden_layers_config = self._create_hidden_layers_config(params)
        dropout_rate, weight_decay, learning_rate, epochs, patience = params['dropout_rate'], params['weight_decay'], params['learning_rate'], params['epochs'], params['patience']
        y_train_prepared, y_target_type_torch, _ = self._prepare_y_data(y_train_full_for_cv)
        kf = self._get_cv_splitter()

        oof_preds_shape = (len(y_train_prepared),) if self.task_type != 'regression' else (len(y_train_prepared), self.ann_output_size)
        oof_preds = np.zeros(oof_preds_shape)
        oof_probas, y_true_oof = None, np.zeros_like(y_train_prepared, dtype=y_train_prepared.dtype)
        if self.task_type != 'regression':
            num_target_classes = self.num_classes if self.num_classes and self.num_classes >= 2 else 2
            if self.task_type == 'binary_classification': num_target_classes = 2
            oof_probas = np.zeros((len(y_train_prepared), num_target_classes))
        
        # --- MODIFICATION: Store metrics for each fold ---
        fold_metrics_list = []
        y_for_kf_split = y_train_prepared.ravel() if self.task_type != 'regression' else y_train_prepared

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full_for_cv, y_for_kf_split)):
            print(f"  Generating ANN CV OOF predictions for fold {fold_idx + 1}/{self.cv}...")
            X_fold_train, X_fold_val = X_train_full_for_cv[train_idx], X_train_full_for_cv[val_idx]
            y_fold_train_np, y_fold_val_np = y_train_prepared[train_idx], y_train_prepared[val_idx]
            y_true_oof[val_idx] = y_fold_val_np

            fold_model = ComplexANN(X_fold_train.shape[1], hidden_layers_config, self.ann_output_size, self.task_type, dropout_rate).to(self.device)
            optimizer = optim.Adam(fold_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            X_fold_train_t, y_fold_train_t, X_fold_val_t = torch.tensor(X_fold_train, dtype=torch.float32).to(self.device), torch.tensor(y_fold_train_np, dtype=y_target_type_torch).to(self.device), torch.tensor(X_fold_val, dtype=torch.float32).to(self.device)
            
            best_val_loss, patience_counter, best_model_state = float('inf'), 0, None
            for _ in range(epochs):
                fold_model.train(); optimizer.zero_grad(); outputs = fold_model(X_fold_train_t); loss = self.criterion(outputs, y_fold_train_t); loss.backward(); optimizer.step()
                fold_model.eval()
                with torch.no_grad():
                    val_outputs = fold_model(X_fold_val_t); val_loss = self.criterion(val_outputs, torch.tensor(y_fold_val_np, dtype=y_target_type_torch).to(self.device))
                if val_loss < best_val_loss: best_val_loss, patience_counter, best_model_state = val_loss, 0, copy.deepcopy(fold_model.state_dict())
                else: patience_counter += 1
                if patience_counter >= patience: break
            if best_model_state: fold_model.load_state_dict(best_model_state)
            
            fold_model.eval()
            with torch.no_grad():
                logits = fold_model(X_fold_val_t).cpu()
                fold_metrics: Dict[str, Union[int, float]] = {'fold': fold_idx + 1}  # type: ignore
                if self.task_type == 'regression':
                    y_pred_fold = logits.numpy()
                    oof_preds[val_idx] = y_pred_fold.reshape(-1, self.ann_output_size)
                    fold_metrics['r2'] = float(r2_score(y_fold_val_np, y_pred_fold))  # type: ignore
                    fold_metrics['rmse'] = float(np.sqrt(mean_squared_error(y_fold_val_np, y_pred_fold)))  # type: ignore
                    fold_metrics['mae'] = float(mean_absolute_error(y_fold_val_np, y_pred_fold))  # type: ignore
                else:
                    if self.task_type == 'binary_classification':
                        y_pred_fold = (torch.sigmoid(logits).numpy() > 0.5).astype(int).ravel()
                        probas_sigmoid = torch.sigmoid(logits).numpy()
                        if oof_probas is not None:
                            try:
                                combined_probas = np.hstack([1 - probas_sigmoid, probas_sigmoid])
                                if combined_probas is not None and len(combined_probas) > 0:
                                    oof_probas[val_idx, :] = combined_probas
                            except Exception as e:
                                print(f"Warning: Could not generate binary probabilities for fold {fold_idx}: {e}")
                        avg_method = 'binary'
                    else: # multiclass
                        y_pred_fold = torch.argmax(logits, dim=1).numpy().ravel()
                        if oof_probas is not None:
                            try:
                                softmax_probas = torch.softmax(logits, dim=1).numpy()
                                if softmax_probas is not None and len(softmax_probas) > 0:
                                    oof_probas[val_idx, :] = softmax_probas
                            except Exception as e:
                                print(f"Warning: Could not generate multiclass probabilities for fold {fold_idx}: {e}")
                        avg_method = 'weighted'
                    oof_preds[val_idx] = y_pred_fold
                    fold_metrics['accuracy'] = float(accuracy_score(y_fold_val_np, y_pred_fold))  # type: ignore
                    fold_metrics['f1'] = float(f1_score(y_fold_val_np, y_pred_fold, average=avg_method, zero_division='warn'))  # type: ignore
                    fold_metrics['precision'] = float(precision_score(y_fold_val_np, y_pred_fold, average=avg_method, zero_division='warn'))  # type: ignore
                    fold_metrics['recall'] = float(recall_score(y_fold_val_np, y_pred_fold, average=avg_method, zero_division='warn'))  # type: ignore
                fold_metrics_list.append(fold_metrics)

        oof_payload = {'y_true_oof': y_true_oof.ravel(), 'y_pred_oof': oof_preds.ravel(), 'y_proba_oof': oof_probas}
        return {'oof_preds': oof_payload, 'fold_metrics': fold_metrics_list}