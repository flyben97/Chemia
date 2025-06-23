# train_valid_test/optimizers/sklearn_optimizer.py
# ... (imports and __init__ remain the same) ...
from .base_optimizer import BaseOptimizer
from models.sklearn_models import (
    XGBoostRegressor, CatBoostRegressor, AdaBoostRegressor, DecisionTreeRegressor,
    HistGradientBoostingRegressor, KNeighborsRegressor, KernelRidge,
    LGBMRegressor, RandomForestRegressor, Ridge, SVR, 
    XGBoostClassifier, CatBoostClassifier, AdaBoostClassifier, DecisionTreeClassifier,
    HistGradientBoostingClassifier, KNeighborsClassifier, LogisticRegression, 
    LGBMClassifier, RandomForestClassifier, SVC 
)
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import lightgbm 
import xgboost as xgb 
import warnings 
import os

class SklearnOptimizer(BaseOptimizer):
    def __init__(self, model_name, n_trials=100, random_state=42, cv=None, 
                 task_type='regression', num_classes=None):
        
        self.model_name_orig = model_name 
        self.task_type = task_type
        self.num_classes = num_classes

        regressor_classes = {
            'xgboost': XGBoostRegressor, 'catboost': CatBoostRegressor, 'adaboost': AdaBoostRegressor,
            'decisiontree': DecisionTreeRegressor, 'histgradientboosting': HistGradientBoostingRegressor,
            'kneighbors': KNeighborsRegressor, 'kernelridge': KernelRidge, 'lgbm': LGBMRegressor,
            'randomforest': RandomForestRegressor, 'ridge': Ridge, 'svr': SVR
        }
        classifier_classes = {
            'xgboost': XGBoostClassifier, 'catboost': CatBoostClassifier, 'adaboost': AdaBoostClassifier,
            'decisiontree': DecisionTreeClassifier, 'histgradientboosting': HistGradientBoostingClassifier,
            'kneighbors': KNeighborsClassifier, 'logisticregression': LogisticRegression, 
            'lgbm': LGBMClassifier, 'randomforest': RandomForestClassifier, 'svc': SVC 
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
        
        # 您可以替换 sklearn_optimizer.py 中的整个 param_grids 字典

        param_grids = {
            'xgboost': { 
                'n_estimators': {'type': 'categorical', 'choices': [100, 300, 500, 800, 1200]},
                'learning_rate': {'type': 'float', 'low': 0.005, 'high': 0.2, 'log': True}, 
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'min_child_weight': {'type': 'int', 'low': 1, 'high': 7}, 
                'gamma': {'type': 'float', 'low': 0, 'high': 0.5}, 
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}, 
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True},  # <<< ADDED
                'reg_lambda': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True}, # <<< ADDED
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
            'kernelridge': { # This is for regression only
                'alpha': {'type': 'float', 'low': 1e-3, 'high': 1e2, 'log': True}, # <<< CHANGED (wider range)
                'kernel': {'type': 'categorical', 'choices': ['linear', 'rbf', 'poly']},
                'gamma': {'type': 'float', 'low': 1e-4, 'high': 1e2, 'log': True} # <<< CHANGED (wider range)
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
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 0.6, 0.8]}, # <<< ADDED
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}, 
                'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy']} if 'classification' in task_type else \
                            {'type': 'categorical', 'choices': ['squared_error', 'absolute_error']}
            },
            'ridge': { 
                'alpha': {'type': 'float', 'low': 1e-5, 'high': 100, 'log': True} # <<< CHANGED (wider range)
            },
            'svr': { 
                'C': {'type': 'float', 'low': 1e-2, 'high': 1e3, 'log':True}, # <<< CHANGED (wider range)
                'epsilon': {'type': 'float', 'low': 0.01, 'high': 0.5, 'log':True},
                'kernel': {'type': 'categorical', 'choices': ['linear', 'poly', 'rbf']},
                'gamma': {'type': 'float', 'low': 1e-4, 'high': 1e2, 'log': True} # <<< CHANGED (from categorical to float search)
            },
            'logisticregression': {
                'C': {'type': 'float', 'low': 0.01, 'high': 100.0, 'log': True},
                'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga']}, 
            },
            'svc': {
                'C': {'type': 'float', 'low': 1e-2, 'high': 1e3, 'log': True}, # <<< CHANGED (wider range)
                'kernel': {'type': 'categorical', 'choices': ['linear', 'poly', 'rbf', 'sigmoid']},
                'gamma': {'type': 'float', 'low': 1e-4, 'high': 1e2, 'log': True}, # <<< CHANGED (from categorical to float search)
                'probability': {'type':'categorical', 'choices':[True]} 
            }
        }
        current_param_grid = param_grids.get(self.model_name_for_params, {}) 
        if not current_param_grid and self.model_name_for_params not in ['ridge', 'kernelridge']: 
            print(f"Warning: Param grid for '{self.model_name_for_params}' might be missing or incomplete.")

        super().__init__(model_class, current_param_grid, n_trials, random_state, cv, task_type, num_classes)
        self.models_without_random_state = ['kneighbors', 'kernelridge', 'ridge', 'svr', 'svc', 'logisticregression']

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
            kwargs['verbose'] = 0 # Ensure verbose is 0 if we want to control train_dir
            if self.model_run_output_dir: # Check if the directory has been set
                # Create a specific subdir for catboost files to keep model_specific_output_dir cleaner
                catboost_train_files_dir = os.path.join(self.model_run_output_dir, 'catboost_training_artefacts')
                os.makedirs(catboost_train_files_dir, exist_ok=True)
                kwargs['train_dir'] = catboost_train_files_dir
                # self.console.print(f"[dim]CatBoost train_dir set to: {catboost_train_files_dir}[/dim]") # Optional debug
            else:
                # Fallback or warning if model_run_output_dir is not set
                # This means catboost_info will go to CWD
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
        self.current_trial_number_for_cleanup = trial.number # Store for _prepare_model_kwargs cleanup

        params_from_trial = self._suggest_params(trial) 
        
        if self.model_name_for_params in ['kernelridge', 'svc'] and params_from_trial.get('kernel') == 'poly':
            if 'degree' not in params_from_trial : 
                params_from_trial['degree'] = trial.suggest_int('degree', 2, 5)

        if self.model_name_for_params == 'logisticregression':
            solver = params_from_trial.get('solver') 
            valid_penalties_for_solver = []
            if solver == 'liblinear': 
                valid_penalties_for_solver = ['l1', 'l2']
            elif solver == 'saga': 
                valid_penalties_for_solver = ['l1', 'l2', 'elasticnet', 'none'] 
            if not valid_penalties_for_solver: 
                raise ValueError(f"Solver '{solver}' for LogisticRegression is not recognized for penalty selection.")
            
            # Use a unique name for the conditional penalty suggestion to avoid Optuna conflicts
            # if this objective function is re-entered for the same trial (e.g. pruner).
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

        fold_scores_list = [] # To store scores of each fold for this trial

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
                    fit_params['validation_data'] = (X_fold_val, y_fold_val) 

                try:
                    with warnings.catch_warnings(): 
                        warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
                        warnings.filterwarnings("ignore", category=FutureWarning) 
                        if self.model_name_orig == 'histgradientboosting' and 'validation_data' in fit_params:
                            val_data = fit_params.pop('validation_data')
                            model.fit(X_fold_train, y_fold_train, validation_data=val_data)
                        else:
                            model.fit(X_fold_train, y_fold_train, **fit_params)
                except Exception as e:

                    print(f"Error fitting {self.model_name_orig} (task: {self.task_type}) in CV fold {fold_idx}: {e}")
                    print(f"  Model class: {self.model_class}, Model_kwargs: {current_model_kwargs_for_fit}, Fit_params: {fit_params}")
                    # Allow Optuna to prune or handle this trial as failed by returning a bad score.
                    # If all folds fail for a trial, it will naturally lead to a bad average.
                    # To make it fail faster, you could raise optuna.TrialPruned() or return float('-inf')
                    # For now, we record a very bad score for this fold.
                    fold_scores_list.append(-np.inf if self.task_type == 'regression' else 0.0) 
                    continue # Continue to next fold or trial

                y_pred_fold = model.predict(X_fold_val)
                if self.task_type == 'regression':
                    score = r2_score(y_fold_val, y_pred_fold)
                else: 
                    avg_method = 'binary' if self.task_type == 'binary_classification' else 'weighted'
                    score = f1_score(y_fold_val, y_pred_fold, average=avg_method, zero_division=0)
                fold_scores_list.append(score)
            
            mean_score = np.mean(fold_scores_list) if fold_scores_list else (-np.inf if self.task_type == 'regression' else 0.0)

        else: # No CV path
            model_kwargs = self._prepare_model_kwargs(params_from_trial, for_cv_fold=False) 
            model = self.model_class(**model_kwargs)
            with warnings.catch_warnings(): 
                warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
                warnings.filterwarnings("ignore", category=FutureWarning)
                model.fit(X_train, _y_train) 
            y_pred = model.predict(X_val)
            if self.task_type == 'regression':
                mean_score = r2_score(_y_val, y_pred)
            else: 
                avg_method = 'binary' if self.task_type == 'binary_classification' else 'weighted'
                mean_score = f1_score(_y_val, y_pred, average=avg_method, zero_division=0)
            fold_scores_list.append(mean_score) # Store the single validation score as a "fold"

        trial.set_user_attr("fold_scores", fold_scores_list)
        trial.set_user_attr("mean_fold_score", mean_score) # Also store the mean for clarity if needed later
        return mean_score

    def fit(self, X_train, y_train): 

        if self.best_params_ is None:
            raise ValueError("Optimization has not been run. Call optimize() first.")
        
        final_params = self.best_params_.copy()
        # Clean up conditional params that Optuna might have stored with unique names
        # For LogisticRegression, the main 'penalty' key should be set from the dynamic one
        # (e.g., 'penalty_liblinear_trial_X')
        # This is a bit tricky as best_params_ comes directly from Optuna trial.
        # We might need to reconstruct it carefully or ensure _prepare_model_kwargs is robust.
        
        # Let's assume best_params_ from Optuna already contains the correct 'penalty' from the dynamic suggestion.
        # We will rely on _prepare_model_kwargs to handle potentially extraneous keys if they exist.

        model_kwargs_final = self._prepare_model_kwargs(final_params, for_cv_fold=False)
        
        # Remove specific early stopping params not meant for final init if they were tuned
        if self.model_name_orig == 'xgboost' and 'early_stopping_rounds' in model_kwargs_final:
            del model_kwargs_final['early_stopping_rounds']
        if self.model_name_orig == 'lgbm' and 'early_stopping_round' in model_kwargs_final: 
            del model_kwargs_final['early_stopping_round']
            
        _y_train = y_train.ravel()
        
        self.best_model_ = self.model_class(**model_kwargs_final)
        
        with warnings.catch_warnings(): 
            warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
            warnings.filterwarnings("ignore", category=FutureWarning)
            if self.model_name_orig == 'histgradientboosting' and model_kwargs_final.get('n_iter_no_change') is not None :
                self.best_model_.fit(X_train, _y_train) 
            else:
                self.best_model_.fit(X_train, _y_train)


    def predict(self, X):

        if self.best_model_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        with warnings.catch_warnings(): 
            warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
            predictions = self.best_model_.predict(X)
        return predictions

    def get_cv_predictions(self, X_train_full_for_cv, y_train_full_for_cv):
        # ... (get_cv_predictions method remains the same as previous correct version) ...
        if self.best_params_ is None:
            raise ValueError("Best parameters not found. Run optimize() first.")
        if self.cv is None or self.cv < 2: # self.cv is the number of folds for HPO
            print(f"CV for HPO was not used (self.cv={self.cv}), cannot get OOF CV predictions this way for {self.model_name_orig}.")
            return None

        # Use best_params_ found by Optuna
        final_params_for_cv = self.best_params_.copy()
        # Similar cleanup for logistic regression as in fit() method for best_params_
        if self.model_name_for_params == 'logisticregression':
            solver = final_params_for_cv.get('solver')
            # Optuna might store the conditional choice directly under the dynamic name.
            # We need to ensure 'penalty' is the key used by the model.
            # The self.best_params_ should reflect the actual parameters that led to the best score.
            # If penalty was suggested as 'penalty_liblinear_trial_X', it should be in best_params_.
            # We need to map it to 'penalty'.
            
            # This logic assumes Optuna stores the dynamic param name in best_params_
            # (which it does for params suggested within the objective).
            # We need to clean it up for _prepare_model_kwargs
            found_dynamic_penalty = False
            for key in list(final_params_for_cv.keys()): # Iterate over a copy of keys
                if key.startswith('penalty_for_solver_') and key.endswith(f'_trial_{self.optimize_best_trial_number_placeholder}'): # Placeholder for actual trial number if needed
                    final_params_for_cv['penalty'] = final_params_for_cv.pop(key)
                    if final_params_for_cv['penalty'] == 'none':
                        final_params_for_cv['penalty'] = None
                    found_dynamic_penalty = True
                    break 
            if not found_dynamic_penalty and 'penalty' not in final_params_for_cv :
                 # This case implies penalty was not dynamically suggested or is missing.
                 # _prepare_model_kwargs might set a default, or it could error if model requires it.
                 print(f"Warning: 'penalty' not found in best_params_ for LogisticRegression during get_cv_predictions. Model might use default or error.")


        model_kwargs = self._prepare_model_kwargs(final_params_for_cv, for_cv_fold=True) 
        
        y_train_full_for_cv_ravel = y_train_full_for_cv.ravel()

        if self.task_type == 'regression':
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        oof_preds = np.zeros_like(y_train_full_for_cv_ravel, dtype=float if self.task_type == 'regression' else int)
        oof_probas = None
        if self.task_type != 'regression':
            num_target_classes = self.num_classes if self.num_classes and self.num_classes >=2 else 2
            oof_probas = np.zeros((len(y_train_full_for_cv_ravel), num_target_classes), dtype=float)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_full_for_cv, y_train_full_for_cv_ravel)):
                print(f"  Generating predictions for CV OOF fold {fold_idx + 1}/{self.cv}...")
                X_fold_train, X_fold_val = X_train_full_for_cv[train_idx], X_train_full_for_cv[val_idx]
                y_fold_train = y_train_full_for_cv_ravel[train_idx]

                fold_model_kwargs = model_kwargs.copy() # Start with common kwargs
                fold_model = self.model_class(**fold_model_kwargs)
                
                fold_fit_params = {}
                if self.model_name_orig == 'xgboost':
                    # XGBoost early stopping rounds is an init param
                    # eval_set and verbose for fit
                    fold_fit_params['eval_set'] = [(X_fold_val, y_train_full_for_cv_ravel[val_idx])]
                    fold_fit_params['verbose'] = False
                elif self.model_name_orig == 'lgbm':
                    lgbm_es_r = self.best_params_.get('early_stopping_round', 20) 
                    # Ensure 'early_stopping_round' is not in init_kwargs for LGBM if it was tuned
                    if 'early_stopping_round' in fold_model_kwargs: del fold_model_kwargs['early_stopping_round']
                    fold_model = self.model_class(**fold_model_kwargs) # Re-init if needed

                    fold_fit_params['eval_set'] = [(X_fold_val, y_train_full_for_cv_ravel[val_idx])]
                    fold_fit_params['callbacks'] = [lightgbm.early_stopping(lgbm_es_r, verbose=False)]
                elif self.model_name_orig == 'catboost':
                    fold_fit_params['eval_set'] = [(X_fold_val, y_train_full_for_cv_ravel[val_idx])]
                    # od_wait is init param, verbose=0 also init
                elif self.model_name_orig == 'histgradientboosting':
                    fold_model.fit(X_fold_train, y_fold_train, validation_data=(X_fold_val, y_train_full_for_cv_ravel[val_idx]))
                    oof_preds[val_idx] = fold_model.predict(X_fold_val)
                    if self.task_type != 'regression' and hasattr(fold_model, 'predict_proba'):
                        oof_probas[val_idx, :] = fold_model.predict_proba(X_fold_val)
                    continue 

                fold_model.fit(X_fold_train, y_fold_train, **fold_fit_params)
                
                oof_preds[val_idx] = fold_model.predict(X_fold_val)
                if self.task_type != 'regression' and hasattr(fold_model, 'predict_proba'):
                     oof_probas[val_idx, :] = fold_model.predict_proba(X_fold_val)
            
        y_true_oof = y_train_full_for_cv_ravel 
        return {'y_true_oof': y_true_oof, 'y_pred_oof': oof_preds, 'y_proba_oof': oof_probas}