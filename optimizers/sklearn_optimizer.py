# optimizers/sklearn_optimizer.py
from .base_optimizer import BaseOptimizer
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
    GPRegressor, GPClassifier
)
from sklearn.metrics import r2_score, f1_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import lightgbm 
import xgboost as xgb 
import warnings 
import os
from typing import Optional, Dict, Union

class SklearnOptimizer(BaseOptimizer):
    model_run_output_dir: Optional[str]
    
    def __init__(self, model_name, n_trials=100, random_state=42, cv=None, 
                 task_type='regression', num_classes=None):
        
        self.model_name_orig = model_name 
        self.task_type = task_type
        self.num_classes = num_classes
        self.hpo_trained_model = None  # Store model trained during HPO for train_valid_test mode

        regressor_classes = {
            'xgboost': XGBoostRegressor, 'catboost': CatBoostRegressor, 'adaboost': AdaBoostRegressor,
            'decisiontree': DecisionTreeRegressor, 'histgradientboosting': HistGradientBoostingRegressor,
            'kneighbors': KNeighborsRegressor, 'kernelridge': KernelRidge, 'lgbm': LGBMRegressor,
            'randomforest': RandomForestRegressor, 'ridge': Ridge, 'svr': SVR,
            # New algorithms
            'gbdt': GBDTRegressor, 'extratrees': ExtraTreesRegressor, 'elasticnet': ElasticNet,
            'lasso': Lasso, 'bayesianridge': BayesianRidge, 'sgd': SGDRegressor,
            'gpr': GPRegressor
        }
        classifier_classes = {
            'xgboost': XGBoostClassifier, 'catboost': CatBoostClassifier, 'adaboost': AdaBoostClassifier,
            'decisiontree': DecisionTreeClassifier, 'histgradientboosting': HistGradientBoostingClassifier,
            'kneighbors': KNeighborsClassifier, 'logisticregression': LogisticRegression, 
            'lgbm': LGBMClassifier, 'randomforest': RandomForestClassifier, 'svc': SVC,
            # New algorithms 
            'gbdt': GBDTClassifier, 'extratrees': ExtraTreesClassifier, 'sgd': SGDClassifier,
            'gpc': GPClassifier
        }

        self.model_name_for_params = model_name 
        if task_type == 'regression':
            model_class = regressor_classes.get(model_name)
            if model_name in ['logisticregression', 'svc']:
                 raise ValueError(f"Model {model_name} is a classifier, cannot be used for regression task.")
        elif task_type in ['binary_classification', 'multiclass_classification']:
            if model_name == 'ridge': self.model_name_for_params = 'logisticregression'
            elif model_name == 'svr': self.model_name_for_params = 'svc'
            elif model_name == 'kernelridge': 
                print(f"Warning: KernelRidge is regression-only. Using SVC for classification instead of '{self.model_name_orig}'.")
                self.model_name_for_params = 'svc'
            model_class = classifier_classes.get(self.model_name_for_params)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        if model_class is None:
            raise ValueError(f"Unsupported model_name '{self.model_name_orig}' for task_type '{task_type}'. "
                             f"Effective name for params: '{self.model_name_for_params}'")
        
        param_grids = {
            'xgboost': { 
                'n_estimators': {'type': 'categorical', 'choices': [100, 300, 500, 800, 1200]},
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
                'iterations': {'type': 'categorical', 'choices': [100, 200, 500, 800, 1200]}, 
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
                'alpha': {'type': 'float', 'low': 1e-3, 'high': 1e2, 'log': True},
                'kernel': {'type': 'categorical', 'choices': ['linear', 'rbf', 'poly']},
                'gamma': {'type': 'float', 'low': 1e-4, 'high': 1e2, 'log': True}
            },
            'lgbm': { 
                'n_estimators': {'type': 'categorical', 'choices': [100, 300, 500, 800, 1200]},
                'learning_rate': {'type': 'loguniform', 'low': 1e-3, 'high': 5e-2},
                'num_leaves': {'type': 'int', 'low': 20, 'high': 100}, 
                'min_child_samples': {'type': 'int', 'low': 10, 'high': 50},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'reg_alpha': {'type': 'loguniform', 'low': 1e-2, 'high': 0.5},
                'reg_lambda': {'type': 'loguniform', 'low': 1e-2, 'high': 0.5},
            },
            'randomforest': {
                'n_estimators': {'type': 'categorical', 'choices': [100, 200, 300, 500]},
                'max_depth': {'type': 'int', 'low': 5, 'high': 30, 'none_is_valid':True}, 
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 0.6, 0.8]},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}, 
                'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy']} if 'classification' in task_type else \
                            {'type': 'categorical', 'choices': ['squared_error', 'absolute_error']}
            },
            'ridge': { 
                'alpha': {'type': 'float', 'low': 1e-5, 'high': 100, 'log': True}
            },
            'svr': { 
                'C': {'type': 'float', 'low': 1e-2, 'high': 1e3, 'log':True},
                'epsilon': {'type': 'float', 'low': 0.01, 'high': 0.5, 'log':True},
                'kernel': {'type': 'categorical', 'choices': ['linear', 'poly', 'rbf']},
                'gamma': {'type': 'float', 'low': 1e-4, 'high': 1e2, 'log': True}
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
            # New algorithms parameter grids
            'gbdt': {
                'n_estimators': {'type': 'categorical', 'choices': [100, 200, 300, 500]},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            },
            'extratrees': {
                'n_estimators': {'type': 'categorical', 'choices': [100, 200, 300, 500]},
                'max_depth': {'type': 'int', 'low': 5, 'high': 30, 'none_is_valid': True},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 0.6, 0.8]}
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
            'gpr': {
                'alpha': {'type': 'float', 'low': 1e-10, 'high': 1e-1, 'log': True}
            },
            'gpc': {
                'max_iter_predict': {'type': 'int', 'low': 100, 'high': 1000}
            }
        }
        current_param_grid = param_grids.get(self.model_name_for_params, {}) 
        if not current_param_grid and self.model_name_for_params not in ['ridge', 'kernelridge']: 
            print(f"Warning: Param grid for '{self.model_name_for_params}' might be missing or incomplete.")

        super().__init__(model_class, current_param_grid, n_trials, random_state, cv, task_type, num_classes)
        self.models_without_random_state = ['kneighbors', 'kernelridge', 'ridge', 'svr', 'svc', 'logisticregression']
        self.model_run_output_dir = None  # Will be set by trainer_setup.py for CatBoost

    def _prepare_model_kwargs(self, params_from_trial, for_cv_fold=False): 

        kwargs = params_from_trial.copy()
        
        keys_to_remove = [k for k in kwargs if k.endswith("_is_none_choice_for_" + str(self.current_trial_number_for_cleanup))]
        if hasattr(self, 'current_trial_number_for_cleanup'): 
             for k in keys_to_remove:
                if k in kwargs: del kwargs[k]

        keys_to_remove_general = [k for k in kwargs if k.endswith("_is_none")]
        for k in keys_to_remove_general:
            if k in kwargs: del kwargs[k]

        if self.model_name_orig == 'catboost':
            kwargs['verbose'] = 0 
            if hasattr(self, 'model_run_output_dir') and self.model_run_output_dir:
                catboost_train_files_dir = os.path.join(self.model_run_output_dir, 'catboost_training_artefacts')
                os.makedirs(catboost_train_files_dir, exist_ok=True)
                kwargs['train_dir'] = catboost_train_files_dir
            else:
                self.console.print(f"[yellow]Warning: model_run_output_dir not set for CatBoost. 'catboost_info' may be created in CWD.[/yellow]")
        if self.model_name_orig not in self.models_without_random_state:
             if 'random_state' not in kwargs: 
                kwargs['random_state'] = self.random_state

        if self.model_name_orig == 'xgboost' and for_cv_fold:
            if 'early_stopping_rounds' not in kwargs: 
                 kwargs['early_stopping_rounds'] = 20 
        
        if self.task_type == 'binary_classification':
            if self.model_name_orig == 'xgboost' and 'objective' not in kwargs: kwargs['objective'] = 'binary:logistic'
            if self.model_name_orig == 'lgbm' and 'objective' not in kwargs: kwargs['objective'] = 'binary'
            if self.model_name_orig == 'catboost' and 'loss_function' not in kwargs: kwargs['loss_function'] = 'Logloss'
        elif self.task_type == 'multiclass_classification':
            if self.model_name_orig == 'xgboost' and 'objective' not in kwargs: kwargs['objective'] = 'multi:softprob'
            if self.model_name_orig == 'lgbm' and 'objective' not in kwargs: kwargs['objective'] = 'multiclass'
            if self.model_name_orig == 'catboost' and 'loss_function' not in kwargs: kwargs['loss_function'] = 'MultiClass'
            if self.model_name_orig in ['xgboost', 'lgbm'] and 'num_class' not in kwargs and self.num_classes:
                kwargs['num_class'] = self.num_classes
            elif self.model_name_orig == 'catboost' and 'classes_count' not in kwargs and self.num_classes:
                kwargs['classes_count'] = self.num_classes
        elif self.task_type == 'regression':
            if self.model_name_orig == 'xgboost' and 'objective' not in kwargs: kwargs['objective'] = 'reg:squarederror'
            if self.model_name_orig == 'lgbm' and 'objective' not in kwargs: kwargs['objective'] = 'regression'
            if self.model_name_orig == 'catboost' and 'loss_function' not in kwargs: kwargs['loss_function'] = 'RMSE'

        if self.model_name_orig == 'catboost': kwargs['verbose'] = 0 
        elif self.model_name_orig == 'kneighbors': kwargs['n_jobs'] = -1
        elif self.model_name_orig == 'lgbm': kwargs['n_jobs'] = -1; kwargs['verbose'] = -100 
        elif self.model_name_orig == 'randomforest': kwargs['n_jobs'] = -1
        elif self.model_name_orig == 'xgboost': kwargs['verbosity'] = 0 
        
        if self.task_type != 'regression' and self.model_name_orig == 'xgboost':
            try: 
                if xgb.__version__ >= "1.3.0": kwargs['use_label_encoder'] = False
            except AttributeError: pass 

        if self.model_name_for_params == 'svc' and 'probability' not in kwargs:
             kwargs['probability'] = True 

        if self.model_name_orig == 'histgradientboosting' and 'criterion' in kwargs: 
            del kwargs['criterion']
            
        return kwargs

    def objective(self, trial, X_train, y_train, X_val, y_val):
        self.current_trial_number_for_cleanup = trial.number

        params_from_trial = self._suggest_params(trial) 
        
        if self.model_name_for_params in ['kernelridge', 'svc'] and params_from_trial.get('kernel') == 'poly':
            if 'degree' not in params_from_trial : 
                params_from_trial['degree'] = trial.suggest_int('degree', 2, 5)

        if self.model_name_for_params == 'logisticregression':
            solver = params_from_trial.get('solver') 
            valid_penalties_for_solver = []
            if solver == 'liblinear': valid_penalties_for_solver = ['l1', 'l2']
            elif solver == 'saga': valid_penalties_for_solver = ['l1', 'l2', 'elasticnet', 'none'] 
            if not valid_penalties_for_solver: 
                raise ValueError(f"Solver '{solver}' for LogisticRegression is not recognized for penalty selection.")
            
            penalty_param_name = f'penalty_for_solver_{solver}_trial_{trial.number}'
            suggested_penalty = trial.suggest_categorical(penalty_param_name, valid_penalties_for_solver)
            params_from_trial['penalty'] = None if suggested_penalty == 'none' else suggested_penalty

            if params_from_trial['penalty'] == 'elasticnet' and solver == 'saga':
                 if 'l1_ratio' not in params_from_trial: 
                    params_from_trial['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            elif 'l1_ratio' in params_from_trial: 
                 del params_from_trial['l1_ratio']
        
        _y_train = y_train.ravel() 
        _y_val = y_val.ravel()

        fold_scores_list = []
        trial_model = None  # Store the model from this trial

        # --- MODIFICATION START: Suppress warnings during HPO CV ---
        with warnings.catch_warnings(): 
            warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
            warnings.filterwarnings("ignore", category=FutureWarning) 
            
            if self.cv is not None and self.cv > 1:
                model_kwargs = self._prepare_model_kwargs(params_from_trial, for_cv_fold=True)
                if self.task_type == 'regression':
                    kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                else: 
                    kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, _y_train)):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = _y_train[train_idx], _y_train[val_idx]
                    
                    current_model_kwargs_for_fit = model_kwargs.copy() 
                    model = self.model_class(**current_model_kwargs_for_fit) 
                    fit_params = {} 
                    
                    if self.model_name_orig == 'xgboost':
                        fit_params['eval_set'] = [(X_fold_val, y_fold_val)]
                        fit_params['verbose'] = False 
                    elif self.model_name_orig == 'lgbm':
                        lgbm_es_rounds = params_from_trial.get('early_stopping_round', 20) 
                        if 'early_stopping_round' in current_model_kwargs_for_fit:
                            del current_model_kwargs_for_fit['early_stopping_round'] 
                            model = self.model_class(**current_model_kwargs_for_fit) 
                        fit_params['eval_set'] = [(X_fold_val, y_fold_val)]
                        fit_params['callbacks'] = [lightgbm.early_stopping(stopping_rounds=lgbm_es_rounds, verbose=False)]
                    elif self.model_name_orig == 'catboost':
                        fit_params['eval_set'] = [(X_fold_val, y_fold_val)]
                    elif self.model_name_orig == 'histgradientboosting':
                        # HistGradientBoosting uses X_val, y_val instead of validation_data
                        fit_params['X_val'] = X_fold_val
                        fit_params['y_val'] = y_fold_val

                    try:
                        model.fit(X_fold_train, y_fold_train, **fit_params)
                    except Exception as e:
                        print(f"Error fitting {self.model_name_orig} (task: {self.task_type}) in CV fold {fold_idx}: {e}")
                        print(f"  Model class: {self.model_class}, Model_kwargs: {current_model_kwargs_for_fit}, Fit_params: {fit_params}")
                        fold_scores_list.append(-np.inf if self.task_type == 'regression' else 0.0) 
                        continue

                    y_pred_fold = model.predict(X_fold_val)
                    if self.task_type == 'regression':
                        score = r2_score(y_fold_val, y_pred_fold)
                    else: 
                        avg_method = 'binary' if self.task_type == 'binary_classification' else 'weighted'
                        score = f1_score(y_fold_val, y_pred_fold, average=avg_method, zero_division='warn')
                    fold_scores_list.append(score)
                
                mean_score = np.mean(fold_scores_list) if fold_scores_list else (-np.inf if self.task_type == 'regression' else 0.0)

            else: # No CV path - train on full train set, validate on validation set
                model_kwargs = self._prepare_model_kwargs(params_from_trial, for_cv_fold=False) 
                model = self.model_class(**model_kwargs)
                model.fit(X_train, _y_train) 
                trial_model = model  # Store this model for potential reuse
                
                y_pred = model.predict(X_val)
                if self.task_type == 'regression':
                    mean_score = r2_score(_y_val, y_pred)
                else: 
                    avg_method = 'binary' if self.task_type == 'binary_classification' else 'weighted'
                    mean_score = f1_score(_y_val, y_pred, average=avg_method, zero_division='warn')
                fold_scores_list.append(mean_score)
        # --- MODIFICATION END ---

        # --- FIXED: Store the best trial model for reuse in train_valid_test mode ---
        if trial_model is not None and (not hasattr(self, '_best_trial_score') or mean_score > self._best_trial_score):
            self._best_trial_score = mean_score
            self.hpo_trained_model = trial_model

        trial.set_user_attr("fold_scores", fold_scores_list)
        return mean_score

    def fit(self, X_train, y_train): 
        if self.best_params_ is None: raise ValueError("Optimization has not been run. Call optimize() first.")
        
        # --- FIXED: Check if we have a model from HPO phase that can be reused ---
        if self.hpo_trained_model is not None and self.cv is None:
            # In train_valid_test mode, reuse the model trained during HPO
            self.best_model_ = self.hpo_trained_model
            return
        
        # Otherwise, train a new model (this happens in CV mode)
        final_params = self.best_params_.copy()
        model_kwargs_final = self._prepare_model_kwargs(final_params, for_cv_fold=False)
        
        if self.model_name_orig == 'xgboost' and 'early_stopping_rounds' in model_kwargs_final:
            del model_kwargs_final['early_stopping_rounds']
        if self.model_name_orig == 'lgbm' and 'early_stopping_round' in model_kwargs_final: 
            del model_kwargs_final['early_stopping_round']
            
        _y_train = y_train.ravel()
        self.best_model_ = self.model_class(**model_kwargs_final)
        
        # --- MODIFICATION START: Suppress warnings during final fit ---
        with warnings.catch_warnings(): 
            warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.best_model_.fit(X_train, _y_train)
        # --- MODIFICATION END ---

    def predict(self, X):
        if self.best_model_ is None: raise ValueError("Model has not been fitted. Call fit() first.")
        
        # --- MODIFICATION START: Suppress warnings during prediction ---
        with warnings.catch_warnings(): 
            warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
            predictions = self.best_model_.predict(X)
        # --- MODIFICATION END ---
        return predictions

    def get_cv_predictions(self, X_train_full_for_cv, y_train_full_for_cv):
        if self.best_params_ is None: raise ValueError("Best parameters not found. Run optimize() first.")
        if self.cv is None or self.cv < 2:
            print(f"CV for HPO was not used, cannot get OOF CV predictions for {self.model_name_orig}.")
            return None

        params = self.best_params_.copy()
        if self.model_name_for_params == 'logisticregression':
            for key in list(params.keys()):
                if key.startswith('penalty_for_solver_'):
                    params['penalty'] = None if params.pop(key) == 'none' else params.get(key)
                    break
        
        model_kwargs = self._prepare_model_kwargs(params, for_cv_fold=True)
        y_ravel = y_train_full_for_cv.ravel()
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state) if self.task_type != 'regression' else KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        oof_preds = np.zeros_like(y_ravel, dtype=float)
        oof_probas = None
        num_classes = self.num_classes if self.num_classes and self.num_classes >= 2 else 2
        if self.task_type != 'regression':
            oof_probas = np.zeros((len(y_ravel), num_classes))

        # --- MODIFICATION: Store metrics for each fold ---
        fold_metrics_list = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", FutureWarning)
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full_for_cv, y_ravel)):
                print(f"  Generating predictions for CV OOF fold {fold_idx + 1}/{self.cv}...")
                X_train, X_val = X_train_full_for_cv[train_idx], X_train_full_for_cv[val_idx]
                y_train, y_val = y_ravel[train_idx], y_ravel[val_idx]

                model, fit_params = self.model_class(**model_kwargs.copy()), {}
                # ... (fit params logic is the same)
                if self.model_name_orig == 'xgboost': fit_params.update({'eval_set': [(X_val, y_val)], 'verbose': False})
                elif self.model_name_orig == 'lgbm':
                    if 'early_stopping_round' in model_kwargs: model = self.model_class(**{k:v for k,v in model_kwargs.items() if k != 'early_stopping_round'})
                    fit_params.update({'eval_set': [(X_val, y_val)], 'callbacks': [lightgbm.early_stopping(params.get('early_stopping_round', 20), verbose=False)]})
                elif self.model_name_orig == 'catboost': fit_params['eval_set'] = [(X_val, y_val)]
                
                if self.model_name_orig == 'histgradientboosting':
                    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                else:
                    model.fit(X_train, y_train, **fit_params)
                
                y_pred_fold = model.predict(X_val)
                oof_preds[val_idx] = y_pred_fold
                if self.task_type != 'regression' and hasattr(model, 'predict_proba') and oof_probas is not None:
                    try:
                        proba_preds = model.predict_proba(X_val)
                        if proba_preds is not None and len(proba_preds) > 0:
                            oof_probas[val_idx, :] = proba_preds
                    except Exception as e:
                        print(f"Warning: Could not get probabilities for fold {fold_idx}: {e}")
                
                # --- MODIFICATION: Calculate and store metrics for this fold ---
                fold_metrics: Dict[str, Union[int, float]] = {'fold': fold_idx + 1}  # type: ignore
                if self.task_type == 'regression':
                    fold_metrics['r2'] = float(r2_score(y_val, y_pred_fold))  # type: ignore
                    fold_metrics['rmse'] = float(np.sqrt(mean_squared_error(y_val, y_pred_fold)))  # type: ignore
                    fold_metrics['mae'] = float(mean_absolute_error(y_val, y_pred_fold))  # type: ignore
                else:
                    avg = 'binary' if self.task_type == 'binary_classification' else 'weighted'
                    fold_metrics['accuracy'] = float(accuracy_score(y_val, y_pred_fold))  # type: ignore
                    fold_metrics['f1'] = float(f1_score(y_val, y_pred_fold, average=avg, zero_division='warn'))  # type: ignore
                    fold_metrics['precision'] = float(precision_score(y_val, y_pred_fold, average=avg, zero_division='warn'))  # type: ignore
                    fold_metrics['recall'] = float(recall_score(y_val, y_pred_fold, average=avg, zero_division='warn'))  # type: ignore
                fold_metrics_list.append(fold_metrics)
        
        oof_payload = {'y_true_oof': y_ravel, 'y_pred_oof': oof_preds, 'y_proba_oof': oof_probas}
        return {'oof_preds': oof_payload, 'fold_metrics': fold_metrics_list}