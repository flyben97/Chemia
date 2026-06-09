# core/trainer_setup.py
import time
from rich.console import Console
from rich.panel import Panel
import numpy as np

console = Console(width=120)




def run_all_models_on_data(X_train_main, y_train, X_val, y_val, X_test, y_test,
                           experiment_models_dir, experiment_run_name, config):
    """
    Manages the training and evaluation loop for all specified models.
    """
    from optimizers.ann_optimizer import ANNOptimizer
    from optimizers.sklearn_optimizer import SklearnOptimizer
    
    # --- MODIFICATION START: Import centralized utilities ---
    from utils.io_handler import (ensure_model_specific_directory, save_model_artifact,
                                  save_hyperparameters, log_results, save_predictions,
                                  save_cv_fold_predictions, get_full_model_name)
    # --- MODIFICATION END ---

    task_type = config['task_type']
    training_cfg = config['training']
    split_cfg = config['split_config']
    specific_models_to_run_raw = training_cfg['models_to_run']
    num_optuna_trials = training_cfg['n_trials']
    use_cv_for_hpo = (split_cfg.get('split_mode', '') == 'cross_validation')
    cv_folds_for_hpo = split_cfg.get('cross_validation', {}).get('n_splits') if use_cv_for_hpo else None

    # --- MODIFICATION START: Remove local aliases and use centralized function ---
    if specific_models_to_run_raw:
        # Normalize all model names to their full names
        specific_models_to_run = [get_full_model_name(m) for m in specific_models_to_run_raw]
    else:
        specific_models_to_run = []
    # --- MODIFICATION END ---

    common = ['xgboost', 'catboost', 'adaboost', 'decisiontree', 'histgradientboosting', 'kneighbors', 'lgbm', 'randomforest', 'gbdt', 'extratrees', 'sgd']

    reg_models = common + ['kernelridge', 'ridge', 'svr', 'ann', 'gpr', 'tabnet', 'elasticnet', 'lasso', 'bayesianridge']
    cls_models = common + ['logisticregression', 'svc', 'ann', 'gpr', 'tabnet']

    model_map = {'regression': reg_models, 'binary_classification': cls_models, 'multiclass_classification': cls_models}
    available_models = model_map.get(task_type)
    if available_models is None: raise ValueError(f"Unsupported task_type: {task_type}")

    models_to_run = [m for m in specific_models_to_run if m in available_models] if specific_models_to_run else available_models
    if not models_to_run:
        console.print("[yellow]Warning: No valid models to run based on your configuration. Exiting.[/yellow]")
        return []

    data_shapes = {
        'train': X_train_main.shape,
        'test': X_test.shape if X_test is not None and X_test.size > 0 else (0, X_train_main.shape[1]),
        'hpo_method': 'CV HPO' if use_cv_for_hpo else 'Hold-out Val HPO',
    }
    if use_cv_for_hpo:
        data_shapes['cv_folds'] = cv_folds_for_hpo
    else:
        data_shapes['val'] = X_val.shape if X_val is not None and X_val.size > 0 else (0, X_train_main.shape[1])

    console.print(f"Models to be run for {task_type}: {', '.join(models_to_run)}")
    results_list = []
    num_classes = len(np.unique(y_train)) if task_type != 'regression' else None

    for model_name in models_to_run:
        model_start_time = time.time()
        model_dir = ensure_model_specific_directory(experiment_models_dir, model_name, console)
        console.print(Panel(f"Running: {model_name.upper()}\nTask: {task_type}, HPO Method: {'CV' if use_cv_for_hpo else 'Hold-out'}, Trials: {num_optuna_trials}",
                          title="Model Optimization", expand=False, border_style="yellow"))

        is_ann = (model_name == 'ann')
        is_tabnet = (model_name == 'tabnet')

        if is_ann:
            OptimizerClass = ANNOptimizer
            opt_config = {'n_trials': num_optuna_trials, 'cv': cv_folds_for_hpo, 'task_type': task_type, 'num_classes': num_classes}
        else:
            OptimizerClass = SklearnOptimizer
            opt_config = {'model_name': model_name, 'n_trials': num_optuna_trials, 'cv': cv_folds_for_hpo, 'task_type': task_type, 'num_classes': num_classes}

        optimizer = OptimizerClass(**opt_config)
        optimizer.console = console
        if isinstance(optimizer, SklearnOptimizer) and hasattr(optimizer, 'model_name_orig') and optimizer.model_name_orig == 'catboost':
            setattr(optimizer, 'model_run_output_dir', model_dir)

        if not use_cv_for_hpo:
            if X_val is None or X_val.size == 0:
                from sklearn.model_selection import train_test_split
                X_train_main, X_val_hpo, y_train, y_val_hpo = train_test_split(
                    X_train_main, y_train, test_size=0.15, random_state=42,
                    stratify=y_train if task_type != 'regression' and len(np.unique(y_train)) > 1 else None
                )
            else:
                X_val_hpo, y_val_hpo = X_val, y_val
        else:
            X_val_hpo, y_val_hpo = X_train_main[:1], y_train[:1]

        best_params, best_score = optimizer.optimize(X_train_main, y_train, X_val_hpo, y_val_hpo)
        best_trial_fold_scores = getattr(optimizer, 'best_trial_fold_scores_', [])

        if use_cv_for_hpo:
            console.print(f"[dim]Retraining {model_name} on full training set with best hyperparameters...[/dim]")
            optimizer.fit(X_train_main, y_train)
        else:
            console.print(f"[dim]Using {model_name} model from HPO phase (trained on train set, validated on validation set)...[/dim]")

        metrics = optimizer.evaluate(X_train_main, y_train, X_val, y_val, X_test, y_test, console=console)
        model_runtime = time.time() - model_start_time

        save_model_artifact(optimizer.best_model_, f"{model_name}_model", model_dir, model_name=model_name, is_pytorch_model=(is_ann or is_tabnet), console=console)

        save_hyperparameters(best_params, model_name, model_dir, console=console)

        cv_fold_metrics = None
        if use_cv_for_hpo and cv_folds_for_hpo:
            cv_results = optimizer.get_cv_predictions(X_train_main, y_train)
            if cv_results:
                save_cv_fold_predictions(cv_results.get('oof_preds'), model_name, model_dir, console=console)
                cv_fold_metrics = cv_results.get('fold_metrics')

        log_results(model_name, best_params, best_score, metrics, model_dir, task_type,
                    best_trial_fold_scores, console, experiment_run_name, model_runtime,
                    config, data_shapes=data_shapes, cv_fold_metrics=cv_fold_metrics)

        save_predictions(metrics, model_name, model_dir, y_train, y_test, y_val, console=console)

        # --- MODIFICATION START: Dynamically select metric for summary ---
        results_list.append({
            'model_name': model_name,
            'task_type': task_type,
            'best_params': best_params,
            'best_optuna_score': best_score,
            'best_trial_fold_scores': best_trial_fold_scores,
            'data_shapes': data_shapes,
            'cv_fold_metrics': cv_fold_metrics,
            **metrics
        })

        val_metric_str, test_metric_str = "", ""
        if task_type == 'regression':
            val_r2 = metrics.get('val_r2')
            if val_r2 is not None: val_metric_str = f"  Validation R²: {val_r2:.4f}\n"
            test_metric_str = f"  Test R²: {metrics.get('test_r2', float('nan')):.4f}"
        else:
            val_f1 = metrics.get('val_f1')
            if val_f1 is not None: val_metric_str = f"  Validation F1: {val_f1:.4f}\n"
            test_metric_str = f"  Test F1: {metrics.get('test_f1', float('nan')):.4f}"
        # --- MODIFICATION END ---

        console.print(Panel(
            f"Model: {model_name.upper()}\n"
            f"  Best HPO Score (Mean CV/Val): {best_score:.4f}\n"
            f"{val_metric_str}{test_metric_str}",
            title=f"Key Results for {model_name.upper()}", expand=False, border_style="green"
        ))

    return results_list
