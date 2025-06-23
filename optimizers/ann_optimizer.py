# optimizers/ann_optimizer.py
import torch
import torch.nn as nn
import torch.optim as optim
from .base_optimizer import BaseOptimizer
from models.ann import ComplexANN
import numpy as np
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import copy # For deep copying the best model state

class ANNOptimizer(BaseOptimizer):

    def __init__(self,
                 n_trials=100,
                 random_state=42,
                 cv=None,
                 task_type='regression',
                 num_classes=None):
        
        param_grid = {
            'n_layers': {
                'type': 'int',
                'low': 1,
                'high': 6
            },
            'n_units_l0': {'type': 'int', 'low': 128, 'high': 2048},
            'n_units_l1': {'type': 'int', 'low': 64, 'high': 1024},
            'n_units_l2': {'type': 'int', 'low': 32, 'high': 512},
            'n_units_l3': {'type': 'int', 'low': 32, 'high': 256},
            'n_units_l4': {'type': 'int', 'low': 8, 'high': 64},
            'n_units_l5': {'type': 'int', 'low': 8, 'high': 32},
            
            'dropout_rate': {
                'type': 'float',
                'low': 0.0,
                'high': 0.6
            },
            'weight_decay': {
                'type': 'float',
                'low': 1e-6,
                'high': 1e-2,
                'log': True
            },
            'learning_rate': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-2,
                'log': True
            },
            'epochs': {
                'type': 'int',
                'low': 200,
                'high': 2000
            },
            'patience': {
                'type': 'categorical',
                'choices': [20, 30, 50]
            }
        }
        
        super().__init__(ComplexANN, param_grid, n_trials, random_state, cv,
                         task_type, num_classes)
        
        # This line automatically detects and sets the device to GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        
        print(
            f"ANNOptimizer using device: {self.device} for task: {self.task_type}"
        )

        if self.task_type == 'binary_classification':
            self.ann_output_size = 1
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.task_type == 'multiclass_classification':
            if self.num_classes is None or self.num_classes < 2:
                raise ValueError("num_classes must be specified and >= 2 for multiclass_classification.")
            self.ann_output_size = self.num_classes
            self.criterion = nn.CrossEntropyLoss()
        elif self.task_type == 'regression':
            self.ann_output_size = 1
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _create_hidden_layers_config(self, trial_or_params):
        hidden_layers = []
        is_trial = not isinstance(trial_or_params, dict)
        if is_trial:
            n_layers = trial_or_params.suggest_int('n_layers', self.param_grid['n_layers']['low'], self.param_grid['n_layers']['high'])
        else:
            n_layers = trial_or_params['n_layers']
        last_units = float('inf')
        for i in range(n_layers):
            param_name = f'n_units_l{i}'
            low = self.param_grid[param_name]['low']
            high = self.param_grid[param_name]['high']
            high = min(high, last_units)
            if low > high: low = high
            if is_trial:
                n_units = trial_or_params.suggest_int(param_name, low, high, step=8)
            else:
                n_units = trial_or_params[param_name]
            hidden_layers.append(n_units)
            last_units = n_units
        return hidden_layers

    def objective(self, trial, X_train, y_train, X_val, y_val):
        params = self._suggest_params(trial)
        hidden_layers_config = self._create_hidden_layers_config(params)
        
        dropout_rate = params['dropout_rate']
        weight_decay = params['weight_decay']
        learning_rate = params['learning_rate']
        epochs = params['epochs']
        patience = params['patience']
        
        y_train_prepared, y_target_type_torch, y_val_prepared = self._prepare_y_data(y_train, y_val)

        fold_scores_list = []
        
        def train_and_evaluate_fold(model, optimizer, X_train_t, y_train_t, X_val_t, y_val_np):
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = self.criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t)
                    val_y_tensor = torch.tensor(y_val_np, dtype=y_target_type_torch).to(self.device)
                    val_loss = self.criterion(val_outputs, val_y_tensor)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break
            
            if best_model_state:
                model.load_state_dict(best_model_state)

            model.eval()
            with torch.no_grad():
                y_pred_logits = model(X_val_t).cpu()
                if self.task_type == 'regression':
                    score = r2_score(y_val_np, y_pred_logits.numpy())
                elif self.task_type == 'binary_classification':
                    y_pred_class = (torch.sigmoid(y_pred_logits).numpy() > 0.5).astype(int)
                    score = f1_score(y_val_np, y_pred_class, average='binary', zero_division=0)
                else:
                    y_pred_class = torch.argmax(y_pred_logits, dim=1).numpy()
                    score = f1_score(y_val_np, y_pred_class, average='weighted', zero_division=0)
            return score

        if self.cv is not None and self.cv > 1:
            kf = self._get_cv_splitter()
            y_for_kf_split = y_train_prepared.ravel() if self.task_type != 'regression' else y_train_prepared
            for train_idx, val_idx in kf.split(X_train, y_for_kf_split):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train_np, y_fold_val_np = y_train_prepared[train_idx], y_train_prepared[val_idx]

                model = ComplexANN(X_fold_train.shape[1], hidden_layers_config, self.ann_output_size, self.task_type, dropout_rate).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                X_fold_train_t = torch.tensor(X_fold_train, dtype=torch.float32).to(self.device)
                y_fold_train_t = torch.tensor(y_fold_train_np, dtype=y_target_type_torch).to(self.device)
                X_fold_val_t = torch.tensor(X_fold_val, dtype=torch.float32).to(self.device)
                
                score = train_and_evaluate_fold(model, optimizer, X_fold_train_t, y_fold_train_t, X_fold_val_t, y_fold_val_np)
                fold_scores_list.append(score)
        else:
            model = ComplexANN(X_train.shape[1], hidden_layers_config, self.ann_output_size, self.task_type, dropout_rate).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            y_train_t = torch.tensor(y_train_prepared, dtype=y_target_type_torch).to(self.device)
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)

            score = train_and_evaluate_fold(model, optimizer, X_train_t, y_train_t, X_val_t, y_val_prepared)
            fold_scores_list.append(score)

        mean_score = np.mean(fold_scores_list) if fold_scores_list else (-np.inf if self.task_type == 'regression' else 0.0)
        trial.set_user_attr("fold_scores", fold_scores_list)
        return mean_score

    def fit(self, X_train, y_train):
        if self.best_params_ is None: raise ValueError("Optimization has not been run. Call optimize() first.")

        params = self.best_params_
        hidden_layers_config = self._create_hidden_layers_config(params)
        dropout_rate = params['dropout_rate']
        weight_decay = params['weight_decay']
        learning_rate = params['learning_rate']
        epochs = params['epochs']
        
        y_train_prepared, y_target_type_torch, _ = self._prepare_y_data(y_train)

        self.best_model_ = ComplexANN(X_train.shape[1], hidden_layers_config, self.ann_output_size, self.task_type, dropout_rate).to(self.device)
        optimizer = optim.Adam(self.best_model_.parameters(), lr=learning_rate, weight_decay=weight_decay)

        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train_prepared, dtype=y_target_type_torch).to(self.device)

        self.best_model_.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self.best_model_(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    def _prepare_y_data(self, y_train, y_val=None):
        y_train_prepared = y_train.copy()
        y_val_prepared = y_val.copy() if y_val is not None else None
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
        if self.task_type == 'regression':
            return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            return StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

    def predict(self, X):
        if self.best_model_ is None: raise ValueError("Model has not been fitted. Call fit() first.")
        self.best_model_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.best_model_(X_tensor).cpu()
            if self.task_type == 'regression': return logits.numpy()
            elif self.task_type == 'binary_classification': return (torch.sigmoid(logits).numpy() > 0.5).astype(int)
            elif self.task_type == 'multiclass_classification': return torch.argmax(logits, dim=1).numpy()
            
    def predict_proba(self, X):
        if self.best_model_ is None: raise ValueError("Model has not been fitted. Call fit() first.")
        if not (self.task_type == 'binary_classification' or self.task_type == 'multiclass_classification'):
            raise AttributeError("predict_proba is only for classification tasks.")
        self.best_model_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.best_model_(X_tensor).cpu()
            if self.task_type == 'binary_classification':
                probs = torch.sigmoid(logits).numpy()
                return np.hstack([1 - probs, probs])
            elif self.task_type == 'multiclass_classification':
                return torch.softmax(logits, dim=1).numpy()
            return None
            
    def get_cv_predictions(self, X_train_full_for_cv, y_train_full_for_cv):
        if self.best_params_ is None: raise ValueError("Best parameters not found. Run optimize() first.")
        if self.cv is None or self.cv < 2:
            print("CV was not used, cannot get CV predictions for ANN.")
            return None
        
        params = self.best_params_
        hidden_layers_config = self._create_hidden_layers_config(params)
        dropout_rate = params['dropout_rate']
        weight_decay = params['weight_decay']
        learning_rate = params['learning_rate']
        epochs = params['epochs']
        patience = params['patience']

        y_train_prepared, y_target_type_torch, _ = self._prepare_y_data(y_train_full_for_cv)
        kf = self._get_cv_splitter()

        oof_preds_shape = (len(y_train_prepared),) if self.task_type != 'regression' else (len(y_train_prepared), self.ann_output_size)
        oof_preds = np.zeros(oof_preds_shape, dtype=float if self.task_type == 'regression' else int)
        oof_probas = None
        if self.task_type != 'regression':
            num_target_classes = self.num_classes if self.num_classes and self.num_classes >= 2 else 2
            if self.task_type == 'binary_classification': num_target_classes = 2
            oof_probas = np.zeros((len(y_train_prepared), num_target_classes), dtype=float)
        y_true_oof = np.zeros_like(y_train_prepared, dtype=y_train_prepared.dtype)

        y_for_kf_split = y_train_prepared.ravel() if self.task_type != 'regression' else y_train_prepared

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full_for_cv, y_for_kf_split)):
            print(f"  Generating ANN CV OOF predictions for fold {fold_idx + 1}/{self.cv}...")
            X_fold_train, X_fold_val = X_train_full_for_cv[train_idx], X_train_full_for_cv[val_idx]
            y_fold_train_np = y_train_prepared[train_idx]
            y_true_oof[val_idx] = y_train_prepared[val_idx]

            fold_model = ComplexANN(X_fold_train.shape[1], hidden_layers_config, self.ann_output_size, self.task_type, dropout_rate).to(self.device)
            optimizer = optim.Adam(fold_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # Re-using the training logic from 'objective' for consistency.
            # This ensures OOF predictions are generated using the same robust process.
            X_fold_train_t = torch.tensor(X_fold_train, dtype=torch.float32).to(self.device)
            y_fold_train_t = torch.tensor(y_fold_train_np, dtype=y_target_type_torch).to(self.device)
            X_fold_val_t = torch.tensor(X_fold_val, dtype=torch.float32).to(self.device)
            
            # --- Inline early stopping logic for this fold ---
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            for _ in range(epochs):
                fold_model.train()
                optimizer.zero_grad()
                outputs = fold_model(X_fold_train_t)
                loss = self.criterion(outputs, y_fold_train_t)
                loss.backward()
                optimizer.step()

                fold_model.eval()
                with torch.no_grad():
                    val_outputs = fold_model(X_fold_val_t)
                    val_y_tensor = torch.tensor(y_train_prepared[val_idx], dtype=y_target_type_torch).to(self.device)
                    val_loss = self.criterion(val_outputs, val_y_tensor)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(fold_model.state_dict())
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
            
            if best_model_state:
                fold_model.load_state_dict(best_model_state)
            
            # --- End of inline early stopping logic ---

            fold_model.eval()
            with torch.no_grad():
                logits = fold_model(X_fold_val_t).cpu()
                if self.task_type == 'regression':
                    oof_preds[val_idx] = logits.numpy().reshape(-1, self.ann_output_size)
                elif self.task_type == 'binary_classification':
                    oof_preds[val_idx] = (torch.sigmoid(logits).numpy() > 0.5).astype(int).ravel()
                    probas_sigmoid = torch.sigmoid(logits).numpy()
                    oof_probas[val_idx, :] = np.hstack([1 - probas_sigmoid, probas_sigmoid])
                else: # multiclass
                    oof_preds[val_idx] = torch.argmax(logits, dim=1).numpy().ravel()
                    oof_probas[val_idx, :] = torch.softmax(logits, dim=1).numpy()

        return {'y_true_oof': y_true_oof.ravel(), 'y_pred_oof': oof_preds.ravel(), 'y_proba_oof': oof_probas}