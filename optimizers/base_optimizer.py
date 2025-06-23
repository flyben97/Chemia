# optimizers/base_optimizer.py
import optuna
import numpy as np
from abc import ABC, abstractmethod
# from utils.metrics import compute_metrics # REMOVE TOP-LEVEL IMPORT
import logging 
import warnings 
from rich.console import Console
default_console_opt = Console()

class BaseOptimizer(ABC):
    # ... (__init__, objective, _suggest_params, optimize methods are unchanged) ...
    def __init__(self, model_class, param_grid, n_trials=100, random_state=42, cv=None, 
                 task_type='regression', num_classes=None):
        self.model_class = model_class
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.random_state = random_state
        self.cv = cv 
        self.task_type = task_type
        self.num_classes = num_classes 
        self.best_params_ = None
        self.best_score_ = None 
        self.best_model_ = None 
        self.best_trial_fold_scores_ = [] 
        self.console = default_console_opt

    @abstractmethod
    def objective(self, trial, X_train, y_train, X_val, y_val):
        pass

    def _suggest_params(self, trial):
        params = {}
        for param_name, param_info in self.param_grid.items():
            param_type = param_info['type']
            none_is_valid = param_info.get('none_is_valid', False)
            helper_choice_name = f"{param_name}_is_none_for_trial_{trial.number}"
            if none_is_valid:
                is_none_choice = trial.suggest_categorical(helper_choice_name, [True, False, False, False, False, False, False, False])
                if is_none_choice:
                    params[param_name] = None
                    continue
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, param_info['low'], param_info['high'])
            elif param_type == 'float':
                low, high = param_info['low'], param_info['high']
                if param_info.get('log', False) and low <= 0:
                    if high > 1e-9:
                        low = min(1e-9, high / 1000)
                    else:
                        self.console.print(f"[yellow]Warning: param '{param_name}' has log=True but low ({param_info['low']}) is not positive. Optuna might error. Trying to adjust.[/yellow]")
                        low = 1e-9
                        if high <= low: high = low * 1000
                if low >= high and param_info.get('log', False):
                    self.console.print(f"[yellow]Warning: param '{param_name}' has log=True but low ({low}) >= high ({high}). Using fixed value or non-log.[/yellow]")
                    params[param_name] = low
                elif low >= high:
                    params[param_name] = low
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high, log=param_info.get('log', False))
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_info['choices'])
            elif param_type == 'loguniform':
                low, high = param_info['low'], param_info['high']
                if low <= 0: low = 1e-9
                if high <= low: high = low * 100
                params[param_name] = trial.suggest_float(param_name, low, high, log=True)
            elif param_type == 'uniform':
                params[param_name] = trial.suggest_float(param_name, param_info['low'], param_info['high'], log=False)
        return params

    def optimize(self, X_train, y_train, X_val, y_val):
        direction = 'maximize'
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=self.random_state))
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
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=self.n_trials, n_jobs=-1, callbacks=[progress_callback])
        if study.best_trial is None:
            log_message_best_trial = "Warning: Optuna study did not find a best trial. Defaulting parameters."
            if hasattr(self, 'console') and self.console is not None: self.console.print(f"[yellow]{log_message_best_trial}[/yellow]")
            else: print(log_message_best_trial)
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
        
    def evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, console=None):
            # --- 延迟导入 ---
            from utils.metrics import compute_metrics
            # --- 延迟导入结束 ---

            effective_console = console if console is not None else self.console

            if self.best_model_ is None:
                if self.best_params_ is not None and self.best_params_: 
                    effective_console.print("[dim]Best model not fitted yet for evaluation. Fitting with best parameters found...[/dim]")
                    self.fit(X_train, y_train)
                else:
                    raise ValueError("Model has not been optimized or best_params are empty, cannot evaluate.")

            y_train_pred_proba, y_val_pred_proba, y_test_pred_proba = None, None, None
            y_val_pred = None

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
                
                if self.task_type == 'regression':
                    y_train_pred = self.predict(X_train)
                    y_test_pred = self.predict(X_test) if X_test is not None and X_test.size > 0 else np.array([])
                    if X_val is not None and X_val.size > 0:
                        y_val_pred = self.predict(X_val)
                else: 
                    y_train_pred = self.predict(X_train) 
                    y_test_pred = self.predict(X_test) if X_test is not None and X_test.size > 0 else np.array([])
                    if X_val is not None and X_val.size > 0:
                        y_val_pred = self.predict(X_val)
                        
                    if hasattr(self, "predict_proba") and callable(getattr(self, "predict_proba")):
                        try:
                            y_train_pred_proba = self.predict_proba(X_train)
                            if X_test is not None and X_test.size > 0:
                                y_test_pred_proba = self.predict_proba(X_test)
                            if X_val is not None and X_val.size > 0:
                                y_val_pred_proba = self.predict_proba(X_val)
                        except (AttributeError, NotImplementedError) as e:
                            effective_console.print(f"[yellow]Warning: predict_proba call failed for {self.__class__.__name__}: {e}[/yellow]")
            
            return compute_metrics(y_train_true=y_train, y_train_pred=y_train_pred,
                                y_test_true=y_test, y_test_pred=y_test_pred,
                                task_type=self.task_type, num_classes=self.num_classes,
                                y_train_pred_proba=y_train_pred_proba, y_test_pred_proba=y_test_pred_proba,
                                y_val_true=y_val, y_val_pred=y_val_pred, y_val_pred_proba=y_val_pred_proba,
                                console=effective_console)

    @abstractmethod
    def predict(self, X): 
        pass
    
    def predict_proba(self, X): 
        if not (self.task_type == 'binary_classification' or self.task_type == 'multiclass_classification'):
            raise AttributeError(f"predict_proba is not applicable for task_type '{self.task_type}'")
        if self.best_model_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        if not hasattr(self.best_model_, "predict_proba"):
            effective_console = self.console
            effective_console.print(f"[yellow]Warning: Model {self.best_model_.__class__.__name__} may not have a predict_proba method or it's not enabled (e.g. SVC with probability=False).[/yellow]")
            raise NotImplementedError(f"The model {self.best_model_.__class__.__name__} does not have a predict_proba method or it's not configured for probabilities.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
            proba = self.best_model_.predict_proba(X)
        return proba

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def get_cv_predictions(self, X_train_full_for_cv, y_train_full_for_cv):
        pass