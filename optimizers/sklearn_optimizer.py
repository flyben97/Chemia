# optimizers/sklearn_optimizer.py
from .base_optimizer import BaseOptimizer
import optuna
from utils.auto_cpu_optimizer import get_auto_cpu_optimizer, auto_optimize_model, auto_optimize_data
from models.sklearn_models import (
    XGBoostRegressor, CatBoostRegressor, AdaBoostRegressor, DecisionTreeRegressor,
    HistGradientBoostingRegressor, KNeighborsRegressor, KernelRidge,
    LGBMRegressor, RandomForestRegressor, Ridge, SVR,
    XGBoostClassifier, CatBoostClassifier, AdaBoostClassifier, DecisionTreeClassifier,
    HistGradientBoostingClassifier, KNeighborsClassifier, LogisticRegression,
    LGBMClassifier, RandomForestClassifier, SVC,
    # New algorithms
    GBDTRegressor, GBDTClassifier, ExtraTreesRegressor, ExtraTreesClassifier,
    ElasticNet, Lasso, BayesianRidge, SGDRegressor, SGDClassifier,
    GPRegressor, GPClassifier, TabNetRegressor, TabNetClassifier
)
from sklearn.metrics import r2_score, f1_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import lightgbm
import xgboost as xgb
import warnings
import os
from typing import Optional, Dict, Union
import torch

class SklearnOptimizer(BaseOptimizer):
    model_run_output_dir: Optional[str]

    TABNET_FIT_PARAMS = ['max_epochs', 'patience', 'batch_size', 'virtual_batch_size']

    def __init__(self, model_name, n_trials=100, random_state=42, cv=None,
                 task_type='regression', num_classes=None, sampler_type='tpe', pruner_type='median'):

        self.model_name_orig = model_name
        self.task_type = task_type
        self.num_classes = num_classes
        self.hpo_trained_model = None

        regressor_classes = {
            'xgboost': XGBoostRegressor, 'catboost': CatBoostRegressor, 'adaboost': AdaBoostRegressor,
            'decisiontree': DecisionTreeRegressor, 'histgradientboosting': HistGradientBoostingRegressor,
            'kneighbors': KNeighborsRegressor, 'kernelridge': KernelRidge, 'lgbm': LGBMRegressor,
            'randomforest': RandomForestRegressor, 'ridge': Ridge, 'svr': SVR,
            'gbdt': GBDTRegressor, 'extratrees': ExtraTreesRegressor, 'elasticnet': ElasticNet,
            'lasso': Lasso, 'bayesianridge': BayesianRidge, 'sgd': SGDRegressor,
            'gpr': GPRegressor, 'tabnet': TabNetRegressor
        }
        classifier_classes = {
            'xgboost': XGBoostClassifier, 'catboost': CatBoostClassifier, 'adaboost': AdaBoostClassifier,
            'decisiontree': DecisionTreeClassifier, 'histgradientboosting': HistGradientBoostingClassifier,
            'kneighbors': KNeighborsClassifier, 'logisticregression': LogisticRegression,
            'lgbm': LGBMClassifier, 'randomforest': RandomForestClassifier, 'svc': SVC,
            'gbdt': GBDTClassifier, 'extratrees': ExtraTreesClassifier, 'sgd': SGDClassifier,
            'gpc': GPClassifier, 'tabnet': TabNetClassifier
        }

        self.model_name_for_params = model_name
        if task_type == 'regression':
            model_class = regressor_classes.get(model_name)
        elif task_type in ['binary_classification', 'multiclass_classification']:
            if model_name == 'ridge': self.model_name_for_params = 'logisticregression'
            elif model_name == 'svr': self.model_name_for_params = 'svc'
            elif model_name == 'kernelridge': self.model_name_for_params = 'svc'
            model_class = classifier_classes.get(self.model_name_for_params)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        if model_class is None:
            raise ValueError(f"Unsupported model_name '{self.model_name_orig}' for task_type '{task_type}'.")

        param_grids = {
            'xgboost': {
                'n_estimators': {'type': 'categorical', 'choices': [100, 300, 500, 600,800, 1200]},
                'learning_rate': {'type': 'float', 'low': 0.005, 'high': 0.2, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'min_child_weight': {'type': 'int', 'low': 1, 'high': 7},
                'gamma': {'type': 'float', 'low': 0, 'high': 0.5},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True},
                'reg_lambda': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True},
            },
            'catboost': {
                'iterations': {'type': 'categorical', 'choices': [100, 200, 500, 600, 800, 1200]},
                'depth': {'type': 'int', 'low': 4, 'high': 8},
                'learning_rate': {'type': 'loguniform', 'low': 1e-3, 'high': 5e-2},
                'l2_leaf_reg': {'type': 'loguniform', 'low': 1e-2, 'high': 1e1},
                'od_wait': {'type': 'categorical', 'choices': [20, 30, 40]}
            },
            'adaboost': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 1.0, 'log': True},
                **({'algorithm': {'type': 'categorical', 'choices': ['SAMME', 'SAMME.R']}} if 'classification' in task_type else \
                {'loss': {'type': 'categorical', 'choices': ['linear', 'square', 'exponential']}})
            },
            'decisiontree': {
                'max_depth': {'type': 'int', 'low': 1, 'high': 30, 'none_is_valid': True},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy']} if 'classification' in task_type else \
                            {'type': 'categorical', 'choices': ['squared_error', 'friedman_mse', 'absolute_error']}
            },
            'histgradientboosting': {
                'max_iter': {'type': 'categorical', 'choices': [100, 200, 300, 500, 800]},
                'max_depth': {'type': 'int', 'low': 5, 'high': 20, 'none_is_valid':True},
                'learning_rate': {'type': 'loguniform', 'low': 1e-3, 'high': 1e-1},
                'l2_regularization': {'type': 'loguniform', 'low': 1e-6, 'high': 1e-2},
                'n_iter_no_change': {'type': 'categorical', 'choices': [10, 20, 30]},
            },
            'kneighbors': {
                'n_neighbors': {'type': 'int', 'low': 1, 'high': 20},
                'weights': {'type': 'categorical', 'choices': ['uniform', 'distance']}
            },
            'kernelridge': {
                'alpha': {'type': 'float', 'low': 1e-2, 'high': 1e3, 'log': True},
                'kernel': {'type': 'categorical', 'choices': ['linear', 'rbf', 'poly']},
                'gamma': {'type': 'float', 'low': 1e-4, 'high': 1e2, 'log': True}
            },
            'lgbm': {
                'n_estimators': {'type': 'categorical', 'choices': [100, 200, 300, 500]},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'num_leaves': {'type': 'int', 'low': 10, 'high': 50},
                'min_child_samples': {'type': 'int', 'low': 5, 'high': 30},
                'max_depth': {'type': 'int', 'low': 3, 'high': 8},
                'subsample': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'min_split_gain': {'type': 'float', 'low': 0.0, 'high': 0.1},
            },
            'randomforest': {
                'n_estimators': {'type': 'categorical', 'choices': [50, 100, 200, 500]},
                'max_depth': {'type': 'int', 'low': 3, 'high': 15, 'none_is_valid': True},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2']},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 5},
                'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy']} if 'classification' in task_type else \
                            {'type': 'categorical', 'choices': ['squared_error', 'absolute_error']}
            },
            'ridge': {
                'alpha': {'type': 'float', 'low': 1e-3, 'high': 1000, 'log': True}
            },
            'svr': {
                'C': {'type': 'float', 'low': 0.1, 'high': 100, 'log':True},
                'epsilon': {'type': 'float', 'low': 0.01, 'high': 0.2},
                'kernel': {'type': 'categorical', 'choices': ['linear', 'rbf']},
                'gamma': {'type': 'float', 'low': 1e-3, 'high': 10, 'log': True},
                'max_iter': {'type': 'int', 'low': 1000, 'high': 10000}
            },
            'logisticregression': {
                'C': {'type': 'float', 'low': 0.01, 'high': 100.0, 'log': True},
                'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga']},
            },
            'svc': {
                'C': {'type': 'float', 'low': 1e-2, 'high': 1e3, 'log': True},
                'kernel': {'type': 'categorical', 'choices': ['linear', 'poly', 'rbf', 'sigmoid']},
                'gamma': {'type': 'float', 'low': 1e-4, 'high': 1e2, 'log': True},
                'probability': {'type':'categorical', 'choices':[True]}
            },
            'gbdt': {
                'n_estimators': {'type': 'categorical', 'choices': [100, 200, 300, 500]},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            },
            'extratrees': {
                'n_estimators': {'type': 'categorical', 'choices': [50, 100, 200, 500]},
                'max_depth': {'type': 'int', 'low': 3, 'high': 15, 'none_is_valid': True},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 5},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2']}
            },
            'elasticnet': {
                'alpha': {'type': 'float', 'low': 1e-4, 'high': 10.0, 'log': True},
                'l1_ratio': {'type': 'float', 'low': 0.1, 'high': 0.9}
            },
            'lasso': {
                'alpha': {'type': 'float', 'low': 1e-4, 'high': 10.0, 'log': True}
            },
            'bayesianridge': {
                'alpha_1': {'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True},
                'alpha_2': {'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True},
                'lambda_1': {'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True},
                'lambda_2': {'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True}
            },
            'sgd': {
                'alpha': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'log': True},
                'learning_rate': {'type': 'categorical', 'choices': ['constant', 'optimal', 'invscaling', 'adaptive']},
                'eta0': {'type': 'float', 'low': 0.001, 'high': 1.0, 'log': True}
            },
            'gpr': { 'alpha': {'type': 'float', 'low': 1e-10, 'high': 1e-1, 'log': True} },
            'gpc': { 'max_iter_predict': {'type': 'int', 'low': 100, 'high': 1000} },
            'tabnet': {
                'n_d': {'type': 'categorical', 'choices': [8, 16, 32, 64]},
                'n_a': {'type': 'categorical', 'choices': [8, 16, 32, 64]},
                'n_steps': {'type': 'int', 'low': 3, 'high': 10},
                'gamma': {'type': 'float', 'low': 1.0, 'high': 2.0},
                'n_independent': {'type': 'int', 'low': 1, 'high': 5},
                'n_shared': {'type': 'int', 'low': 1, 'high': 5},
                'lambda_sparse': {'type': 'float', 'low': 1e-6, 'high': 1e-3, 'log': True},
                'max_epochs': {'type': 'categorical', 'choices': [50, 100, 200]},
                'patience': {'type': 'categorical', 'choices': [10, 15, 20]},
                'batch_size': {'type': 'categorical', 'choices': [256, 512, 1024]},
                'virtual_batch_size': {'type': 'categorical', 'choices': [128, 256, 512]}
            },

        }
        current_param_grid = param_grids.get(self.model_name_for_params, {})
        super().__init__(model_class, current_param_grid, n_trials, random_state, cv, task_type, num_classes)

        # 设置采样器和剪枝器类型
        self.sampler_type = sampler_type
        self.pruner_type = pruner_type

        self.models_without_random_state = ['kneighbors', 'kernelridge', 'ridge', 'svr', 'svc', 'logisticregression', 'tabnet', 'bayesianridge', 'elasticnet', 'lasso', 'gpr', 'gpc']
        self.model_run_output_dir = None
        self.scaler = StandardScaler() if self.model_name_orig in ['svr', 'svc', 'kernelridge', 'kneighbors', 'sgd', 'gpr', 'gpc', 'ridge', 'elasticnet', 'lasso', 'bayesianridge'] else None
        self.data_is_scaled = False

    def _prepare_model_kwargs(self, params_from_trial, for_cv_fold=False):
        kwargs = params_from_trial.copy()

        # General cleanup of helper keys
        keys_to_remove_general = [k for k in kwargs if k.endswith("_is_none")]
        if hasattr(self, 'current_trial_number_for_cleanup'):
            keys_to_remove_general.extend([k for k in kwargs if f"_trial_{self.current_trial_number_for_cleanup}" in k])
        for k in keys_to_remove_general:
            if k in kwargs: del kwargs[k]

        if self.model_name_orig == 'catboost':
            kwargs['verbose'] = 0
            if hasattr(self, 'model_run_output_dir') and self.model_run_output_dir:
                catboost_train_files_dir = os.path.join(self.model_run_output_dir, 'catboost_training_artefacts')
                os.makedirs(catboost_train_files_dir, exist_ok=True)
                kwargs['train_dir'] = catboost_train_files_dir

        if self.model_name_orig not in self.models_without_random_state:
             kwargs.setdefault('random_state', self.random_state)

        if self.model_name_orig == 'xgboost':
            if for_cv_fold:
                kwargs['early_stopping_rounds'] = 20

        if self.task_type == 'binary_classification':
            if self.model_name_orig == 'xgboost': kwargs.setdefault('objective', 'binary:logistic')
            if self.model_name_orig == 'lgbm': kwargs.setdefault('objective', 'binary')
            if self.model_name_orig == 'catboost': kwargs.setdefault('loss_function', 'Logloss')
        elif self.task_type == 'multiclass_classification':
            if self.model_name_orig == 'xgboost': kwargs.setdefault('objective', 'multi:softprob')
            if self.model_name_orig == 'lgbm': kwargs.setdefault('objective', 'multiclass')
            if self.model_name_orig == 'catboost': kwargs.setdefault('loss_function', 'MultiClass')
            if self.model_name_orig in ['xgboost', 'lgbm'] and self.num_classes: kwargs.setdefault('num_class', self.num_classes)
            elif self.model_name_orig == 'catboost' and self.num_classes: kwargs.setdefault('classes_count', self.num_classes)
        elif self.task_type == 'regression':
            if self.model_name_orig == 'xgboost': kwargs.setdefault('objective', 'reg:squarederror')
            if self.model_name_orig == 'lgbm': kwargs.setdefault('objective', 'regression')
            if self.model_name_orig == 'catboost': kwargs.setdefault('loss_function', 'RMSE')

        # 🚀 自动CPU优化集成 - 移除手动设置的系统参数，让自动优化器处理
        # 不再手动设置 n_jobs, thread_count, num_threads 等系统级参数
        if self.model_name_orig == 'catboost':
            kwargs['verbose'] = 0
        elif self.model_name_orig == 'ridge':
            kwargs.setdefault('solver', 'auto')
        elif self.model_name_orig == 'svr':
            kwargs.setdefault('max_iter', 50000)
        elif self.model_name_orig == 'xgboost':
            kwargs['verbosity'] = 0
        elif self.model_name_orig == 'lgbm':
            # LightGBM特殊设置以避免警告
            kwargs.setdefault('verbose', -1)
            kwargs.setdefault('force_col_wise', True)
            kwargs.setdefault('min_data_in_leaf', 5)
            kwargs.setdefault('feature_fraction_bynode', 0.8)

        if self.task_type != 'regression' and self.model_name_orig == 'xgboost':
            if xgb.__version__ >= "1.3.0": kwargs['use_label_encoder'] = False

        if self.model_name_for_params == 'svc': kwargs.setdefault('probability', True)
        if self.model_name_orig == 'histgradientboosting' and 'criterion' in kwargs: del kwargs['criterion']

        if self.model_name_orig == 'tabnet':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            kwargs.update({'device_name': device, 'verbose': 0})
            kwargs.setdefault('seed', self.random_state)

            # Set PyTorch random states for better reproducibility
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
                torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            kwargs.setdefault('optimizer_fn', torch.optim.Adam)
            kwargs.setdefault('optimizer_params', dict(lr=2e-2))
            kwargs.setdefault('scheduler_fn', torch.optim.lr_scheduler.StepLR)
            kwargs.setdefault('scheduler_params', {"step_size": 10, "gamma": 0.9})
            for param in self.TABNET_FIT_PARAMS:
                if param in kwargs:
                    del kwargs[param]

        return kwargs

    def _preprocess_data_for_training(self, X_train, X_val=None, fit_scaler=True):
        if self.scaler is None: return X_train, X_val
        if fit_scaler:
            if not self.data_is_scaled:
                X_train_scaled = self.scaler.fit_transform(X_train)
                self.data_is_scaled = True
            else:
                X_train_scaled = self.scaler.transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        return X_train_scaled, X_val_scaled

    def objective(self, trial, X_train, y_train, X_val, y_val):
        self.current_trial_number_for_cleanup = trial.number
        params_from_trial = self._suggest_params(trial)

        if self.model_name_for_params in ['kernelridge', 'svc'] and params_from_trial.get('kernel') == 'poly':
            params_from_trial.setdefault('degree', trial.suggest_int('degree', 2, 5))

        if self.model_name_for_params == 'logisticregression':
            solver = params_from_trial.get('solver')
            valid_penalties = {'liblinear': ['l1', 'l2'], 'saga': ['l1', 'l2', 'elasticnet', 'none']}.get(solver)
            if valid_penalties is None: raise ValueError(f"Solver '{solver}' not recognized.")
            penalty_param_name = f'penalty_for_solver_{solver}_trial_{trial.number}'
            penalty = trial.suggest_categorical(penalty_param_name, valid_penalties)
            params_from_trial['penalty'] = None if penalty == 'none' else penalty
            if penalty == 'elasticnet':
                params_from_trial.setdefault('l1_ratio', trial.suggest_float('l1_ratio', 0.0, 1.0))

        _y_train, _y_val = y_train.ravel(), y_val.ravel()

        # 🚀 自动数据优化
        X_train_processed, X_val_processed = self._preprocess_data_for_training(X_train, X_val, fit_scaler=True)
        X_train_processed, _y_train = auto_optimize_data(X_train_processed, _y_train)
        if X_val_processed is not None:
            X_val_processed, _y_val = auto_optimize_data(X_val_processed, _y_val)

        tabnet_fit_params = {}
        if self.model_name_orig == 'tabnet':
            X_train_processed = X_train_processed.astype(np.float32)
            if X_val_processed is not None: X_val_processed = X_val_processed.astype(np.float32)
            if self.task_type == 'regression':
                _y_train = _y_train.reshape(-1, 1).astype(np.float32)
                if _y_val is not None: _y_val = _y_val.reshape(-1, 1).astype(np.float32)
            else:
                _y_train = _y_train.astype(np.int64)
                if _y_val is not None: _y_val = _y_val.astype(np.int64)
            for param in self.TABNET_FIT_PARAMS:
                if param in params_from_trial: tabnet_fit_params[param] = params_from_trial[param]

        fold_scores_list, trial_model = [], None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model_kwargs = self._prepare_model_kwargs(params_from_trial, for_cv_fold=(self.cv is not None and self.cv > 1))

            if self.cv is None or self.cv <= 1:
                model = self.model_class(**model_kwargs)
                # 🚀 自动模型优化
                model = auto_optimize_model(model, self.model_name_orig)
                fit_params = {}

                if self.model_name_orig == 'xgboost':
                    fit_params['eval_set'] = [(X_val_processed, _y_val)]
                    fit_params['verbose'] = False
                elif self.model_name_orig == 'lgbm':
                    fit_params['callbacks'] = [lightgbm.early_stopping(stopping_rounds=20, verbose=False)]
                    fit_params['eval_set'] = [(X_val_processed, _y_val)]
                elif self.model_name_orig == 'catboost':
                    fit_params['eval_set'] = [(X_val_processed, _y_val)]
                    fit_params['early_stopping_rounds'] = params_from_trial.get('od_wait', 20)
                elif self.model_name_orig == 'tabnet':
                    fit_params.update(tabnet_fit_params)
                    eval_metric = ['rmse'] if self.task_type == 'regression' else ['accuracy']
                    fit_params.update({'eval_set': [(X_val_processed, _y_val)], 'eval_name': ['valid'], 'eval_metric': eval_metric})

                model.fit(X_train_processed, _y_train, **fit_params)
                trial_model = model
                y_pred = model.predict(X_val_processed)
                if self.task_type == 'regression': mean_score = r2_score(_y_val, y_pred.ravel())
                else: mean_score = f1_score(_y_val.ravel(), y_pred.ravel(), average='binary' if self.task_type == 'binary_classification' else 'weighted')
                fold_scores_list.append(mean_score)
            else:
                kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state) if self.task_type != 'regression' else KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                for train_idx, val_idx in kf.split(X_train_processed, _y_train):
                    X_fold_train, X_fold_val = X_train_processed[train_idx], X_train_processed[val_idx]
                    y_fold_train, y_fold_val = _y_train[train_idx], _y_train[val_idx]

                    model = self.model_class(**model_kwargs)
                    # 🚀 自动模型优化
                    model = auto_optimize_model(model, self.model_name_orig)
                    fit_params = {}

                    if self.model_name_orig == 'xgboost':
                        fit_params['eval_set'] = [(X_fold_val, y_fold_val)]
                        fit_params['verbose'] = False
                    elif self.model_name_orig == 'lgbm':
                        fit_params['callbacks'] = [lightgbm.early_stopping(stopping_rounds=20, verbose=False)]
                        fit_params['eval_set'] = [(X_fold_val, y_fold_val)]
                    elif self.model_name_orig == 'catboost':
                        fit_params['eval_set'] = [(X_fold_val, y_fold_val)]
                        fit_params['early_stopping_rounds'] = params_from_trial.get('od_wait', 20)
                    elif self.model_name_orig == 'tabnet':
                        _y_fold_train_tabnet = y_fold_train.reshape(-1, 1).astype(np.float32) if self.task_type == 'regression' else y_fold_train.astype(np.int64)
                        _y_fold_val_tabnet = y_fold_val.reshape(-1, 1).astype(np.float32) if self.task_type == 'regression' else y_fold_val.astype(np.int64)
                        fit_params.update(tabnet_fit_params)
                        eval_metric = ['rmse'] if self.task_type == 'regression' else ['accuracy']
                        fit_params.update({'eval_set': [(X_fold_val, _y_fold_val_tabnet)], 'eval_name': ['valid'], 'eval_metric': eval_metric})
                        y_fold_train = _y_fold_train_tabnet

                    model.fit(X_fold_train, y_fold_train, **fit_params)
                    y_pred_fold = model.predict(X_fold_val)

                    if self.task_type == 'regression': score = r2_score(y_fold_val, y_pred_fold.ravel())
                    else: score = f1_score(y_fold_val.ravel(), y_pred_fold.ravel(), average='binary' if self.task_type == 'binary_classification' else 'weighted')
                    fold_scores_list.append(score)
                mean_score = np.mean(fold_scores_list) if fold_scores_list else 0.0

        if trial_model is not None and (not hasattr(self, '_best_trial_score') or mean_score > getattr(self, '_best_trial_score', -np.inf)):
            self._best_trial_score = mean_score
            self.hpo_trained_model = trial_model

        trial.set_user_attr("fold_scores", fold_scores_list)
        return mean_score

    def fit(self, X_train, y_train):
        if self.best_params_ is None: raise ValueError("Optimization has not been run.")

        self.data_is_scaled = False
        X_train_processed, _ = self._preprocess_data_for_training(X_train, fit_scaler=True)

        # 🚀 自动数据优化
        X_train_processed, y_train = auto_optimize_data(X_train_processed, y_train)

        if self.hpo_trained_model is not None and self.cv is None:
            self.best_model_ = self.hpo_trained_model
            return

        final_params = self.best_params_.copy()
        model_kwargs_final = self._prepare_model_kwargs(final_params, for_cv_fold=False)

        _y_train = y_train.ravel()

        fit_params = {}
        if self.model_name_orig == 'tabnet':
            for param in self.TABNET_FIT_PARAMS:
                if param in final_params: fit_params[param] = final_params[param]
            X_train_processed = X_train_processed.astype(np.float32)
            if self.task_type == 'regression':
                _y_train = _y_train.reshape(-1, 1).astype(np.float32)
            else:
                _y_train = _y_train.astype(np.int64)

        self.best_model_ = self.model_class(**model_kwargs_final)
        # 🚀 自动模型优化
        self.best_model_ = auto_optimize_model(self.best_model_, self.model_name_orig)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.best_model_.fit(X_train_processed, _y_train, **fit_params)

    def predict(self, X):
        if self.best_model_ is None: raise ValueError("Model has not been fitted.")
        X_processed, _ = self._preprocess_data_for_training(X, fit_scaler=False)

        # 🚀 自动数据优化（仅优化X，不需要y）
        X_processed, _ = auto_optimize_data(X_processed, None)

        if self.model_name_orig == 'tabnet': X_processed = X_processed.astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = self.best_model_.predict(X_processed)
            if self.model_name_orig == 'tabnet' and self.task_type == 'regression':
                predictions = predictions.ravel()
        return predictions

    def get_cv_predictions(self, X_train_full_for_cv, y_train_full_for_cv):
        if self.best_params_ is None: raise ValueError("Best parameters not found.")
        if self.cv is None or self.cv < 2: return None

        params = self.best_params_.copy()
        model_kwargs = self._prepare_model_kwargs(params, for_cv_fold=True)
        y_ravel = y_train_full_for_cv.ravel()

        tabnet_fit_params = {}
        if self.model_name_orig == 'tabnet':
            for param in self.TABNET_FIT_PARAMS:
                if param in params:
                    tabnet_fit_params[param] = params[param]

        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state) if self.task_type != 'regression' else KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        X_scaled, _ = self._preprocess_data_for_training(X_train_full_for_cv, fit_scaler=True)

        # 🚀 自动数据优化
        X_scaled, y_ravel = auto_optimize_data(X_scaled, y_ravel)

        oof_preds = np.zeros_like(y_ravel, dtype=float)
        oof_probas = np.zeros((len(y_ravel), self.num_classes or 2)) if self.task_type != 'regression' else None
        fold_metrics_list = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_ravel)):
                self.console.print(f"  Generating predictions for CV OOF fold {fold_idx + 1}/{self.cv}...")
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_ravel[train_idx], y_ravel[val_idx]

                model = self.model_class(**model_kwargs.copy())
                # 🚀 自动模型优化
                model = auto_optimize_model(model, self.model_name_orig)

                fit_params = {}
                if self.model_name_orig == 'xgboost':
                    fit_params['eval_set'] = [(X_val, y_val)]
                    fit_params['verbose'] = False
                elif self.model_name_orig == 'lgbm':
                    fit_params['callbacks'] = [lightgbm.early_stopping(stopping_rounds=20, verbose=False)]
                    fit_params['eval_set'] = [(X_val, y_val)]
                elif self.model_name_orig == 'catboost':
                    fit_params['eval_set'] = [(X_val, y_val)]
                    fit_params['early_stopping_rounds'] = params.get('od_wait', 20)
                elif self.model_name_orig == 'tabnet':
                    _y_train_tabnet = y_train.reshape(-1, 1).astype(np.float32) if self.task_type == 'regression' else y_train.astype(np.int64)
                    _y_val_tabnet = y_val.reshape(-1, 1).astype(np.float32) if self.task_type == 'regression' else y_val.astype(np.int64)
                    fit_params.update(tabnet_fit_params)
                    fit_params.update({'eval_set': [(X_val, _y_val_tabnet)], 'eval_metric': ['rmse' if self.task_type == 'regression' else 'accuracy']})
                    y_train = _y_train_tabnet

                model.fit(X_train, y_train, **fit_params)
                y_pred_fold = model.predict(X_val)
                oof_preds[val_idx] = y_pred_fold.ravel()

                if self.task_type != 'regression' and hasattr(model, 'predict_proba') and oof_probas is not None:
                    oof_probas[val_idx, :] = model.predict_proba(X_val)

                fold_metrics: Dict[str, Union[int, float]] = {'fold': fold_idx + 1}
                if self.task_type == 'regression':
                    fold_metrics['r2'] = r2_score(y_val, y_pred_fold)
                    fold_metrics['rmse'] = np.sqrt(mean_squared_error(y_val, y_pred_fold))
                    fold_metrics['mae'] = mean_absolute_error(y_val, y_pred_fold)
                else:
                    avg = 'binary' if self.task_type == 'binary_classification' else 'weighted'
                    fold_metrics['accuracy'] = accuracy_score(y_val, y_pred_fold)
                    fold_metrics['f1'] = f1_score(y_val, y_pred_fold, average=avg, zero_division=0)
                    fold_metrics['precision'] = precision_score(y_val, y_pred_fold, average=avg, zero_division=0)
                    fold_metrics['recall'] = recall_score(y_val, y_pred_fold, average=avg, zero_division=0)

                fold_metrics_list.append(fold_metrics)

        oof_payload = {'y_true_oof': y_ravel, 'y_pred_oof': oof_preds, 'y_proba_oof': oof_probas}
        return {'oof_preds': oof_payload, 'fold_metrics': fold_metrics_list}

    def optimize(self, X_train, y_train, X_val, y_val):
        """重写 optimize 方法以支持自定义采样器和剪枝器"""
        import numpy as np

        # 创建采样器
        if self.sampler_type == 'tpe':
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        elif self.sampler_type == 'cmaes':
            sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        elif self.sampler_type == 'random':
            sampler = optuna.samplers.RandomSampler(seed=self.random_state)
        elif self.sampler_type == 'grid':
            # 网格搜索需要显式提供搜索空间；当前实现暂不支持动态网格生成，回退到 TPE
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        else:
            # 默认使用 TPE
            sampler = optuna.samplers.TPESampler(seed=self.random_state)

        # 创建剪枝器
        if self.pruner_type == 'median':
            pruner = optuna.pruners.MedianPruner()
        elif self.pruner_type == 'successive_halving':
            pruner = optuna.pruners.SuccessiveHalvingPruner()
        elif self.pruner_type == 'hyperband':
            pruner = optuna.pruners.HyperbandPruner()
        elif self.pruner_type == 'none':
            pruner = optuna.pruners.NopPruner()
        else:
            # 默认使用 MedianPruner
            pruner = optuna.pruners.MedianPruner()

        direction = 'maximize'
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

        progress_callback_counter = {'count': 0}
        def progress_callback(study, trial):
            progress_callback_counter['count'] += 1
            value_str = f"{trial.value:.4f}" if trial.value is not None else "N/A (Pruned/Failed)"
            fold_scores_str = ""
            if "fold_scores" in trial.user_attrs and trial.user_attrs["fold_scores"]:
                scores = trial.user_attrs["fold_scores"]
                fold_scores_str = ", Folds: [" + ", ".join([f"{s:.4f}" for s in scores]) + "]"
            log_message = f"Optuna Trial {progress_callback_counter['count']}/{self.n_trials} (Optuna TrialID: {trial.number}) finished. Mean Score: {value_str}{fold_scores_str}"
            if hasattr(self, 'console') and self.console is not None:
                self.console.print(f"[dim]{log_message}[/dim]")
            else:
                print(log_message)

        def safe_objective_wrapper(trial):
            try:
                result = self.objective(trial, X_train, y_train, X_val, y_val)
                if result is None:
                    return -np.inf if direction == 'maximize' else np.inf
                elif np.isnan(result):
                    return -np.inf if direction == 'maximize' else np.inf
                elif np.isinf(result):
                    return float(result)
                else:
                    return float(result)
            except Exception as e:
                self.console.print(f"[red]Error in objective function for trial {trial.number}: {e}[/red]")
                return -np.inf if direction == 'maximize' else np.inf

        study.optimize(
            func=safe_objective_wrapper,
            n_trials=self.n_trials,
            callbacks=[progress_callback]
        )

        if study.best_trial is None:
            log_message_best_trial = "Warning: Optuna study did not find a best trial. Defaulting parameters."
            if hasattr(self, 'console') and self.console is not None:
                self.console.print(f"[yellow]{log_message_best_trial}[/yellow]")
            else:
                print(log_message_best_trial)
            self.best_params_ = {}
            self.best_score_ = -np.inf if direction == 'maximize' else np.inf
            self.best_trial_fold_scores_ = []
        else:
            self.best_params_ = study.best_trial.params
            self.best_score_ = study.best_trial.value
            params_to_remove_suffixes = [f"_is_none_for_trial_{study.best_trial.number}"]
            if 'solver' in self.best_params_:
                params_to_remove_suffixes.append(f"_for_solver_{self.best_params_['solver']}_trial_{study.best_trial.number}")
            cleaned_best_params = {}
            for k, v in self.best_params_.items():
                is_helper_param = False
                for suffix in params_to_remove_suffixes:
                    if k.endswith(suffix):
                        is_helper_param = True
                        break
                if not is_helper_param:
                    cleaned_best_params[k] = v
            self.best_params_ = cleaned_best_params
            self.optimize_best_trial_number_placeholder = study.best_trial.number
            self.best_trial_fold_scores_ = study.best_trial.user_attrs.get("fold_scores", [])
        return self.best_params_, self.best_score_
