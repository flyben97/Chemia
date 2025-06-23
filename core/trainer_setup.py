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
    from utils.io import ensure_model_specific_directory, save_model_artifact, save_hyperparameters, log_results, save_predictions, save_cv_fold_predictions

    # Extract parameters from config dictionary
    task_type = config['task_type']
    training_cfg = config['training']
    split_cfg = config['split_config']
    
    specific_models_to_run = training_cfg['models_to_run']
    num_optuna_trials = training_cfg['n_trials']
    
    use_cv_for_hpo = (config['split_mode'] == 'cross_validation')
    cv_folds_for_hpo = split_cfg['cross_validation']['n_splits'] if use_cv_for_hpo else None

    # ... (其余逻辑与之前类似，但都从 config 中获取信息)
    model_aliases = {
        'rf': 'randomforest', 'dt': 'decisiontree', 'knn': 'kneighbors',
        'lr': 'logisticregression', 'svm': 'svc', 'krr':'kernelridge',
        'xgb': 'xgboost', 'lgbm': 'lgbm', 'cat': 'catboost', 'hgb':'histgradientboosting',
        'ann': 'ann', 'ridge': 'ridge', 'svr': 'svr', 'adab': 'adaboost'
    }
    
    if specific_models_to_run:
        specific_models_to_run = [model_aliases.get(m.lower(), m.lower()) for m in specific_models_to_run]

    common = ['xgboost', 'catboost', 'adaboost', 'decisiontree', 'histgradientboosting', 'kneighbors', 'lgbm', 'randomforest']
    reg_models = common + ['kernelridge', 'ridge', 'svr', 'ann']
    cls_models = common + ['logisticregression', 'svc', 'ann']
    
    model_map = {'regression': reg_models, 'binary_classification': cls_models, 'multiclass_classification': cls_models}
    available_models = model_map.get(task_type)
    if available_models is None: raise ValueError(f"Unsupported task_type: {task_type}")

    models_to_run = [m for m in specific_models_to_run if m in available_models] if specific_models_to_run else available_models
    if not models_to_run:
        console.print("[yellow]Warning: No valid models to run. Exiting.[/yellow]")
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
        OptimizerClass = ANNOptimizer if is_ann else SklearnOptimizer
        opt_config = {'n_trials': num_optuna_trials, 'cv': cv_folds_for_hpo, 'task_type': task_type, 'num_classes': num_classes}
        if not is_ann: opt_config['model_name'] = model_name
        
        optimizer = OptimizerClass(**opt_config)
        optimizer.console = console
        if isinstance(optimizer, SklearnOptimizer) and optimizer.model_name_orig == 'catboost':
            optimizer.model_run_output_dir = model_dir

        X_val_hpo, y_val_hpo = (X_val, y_val)
        if use_cv_for_hpo:
            X_val_hpo, y_val_hpo = (X_train_main[:1], y_train[:1])

        best_params, best_score = optimizer.optimize(X_train_main, y_train, X_val_hpo, y_val_hpo)
        best_trial_fold_scores = getattr(optimizer, 'best_trial_fold_scores_', [])
        
        optimizer.fit(X_train_main, y_train)
        
        metrics = optimizer.evaluate(
            X_train_main, y_train, X_val, y_val,
            X_test, y_test, console=console
        )
        
        model_runtime = time.time() - model_start_time
        save_model_artifact(optimizer.best_model_, f"{model_name}_model", model_dir, is_pytorch_model=is_ann, console=console)
        save_hyperparameters(best_params, model_name, model_dir, console=console)
        # 传递 config 字典而不是 args 对象
        log_results(model_name, best_params, best_score, metrics, model_dir, task_type,
                    best_trial_fold_scores, console, experiment_run_name, model_runtime,
                    config, data_shapes=data_shapes)
        save_predictions(metrics, model_name, model_dir, y_train, y_test, y_val, console=console)
        if use_cv_for_hpo and cv_folds_for_hpo:
            cv_preds = optimizer.get_cv_predictions(X_train_main, y_train)
            if cv_preds: save_cv_fold_predictions(cv_preds, model_name, model_dir, console=console)
        
        results_list.append({'model_name': model_name, 'task_type': task_type, 'best_params': best_params,
                             'best_optuna_score': best_score, 'data_shapes': data_shapes, **metrics})
        
        val_metric_str = ""
        if task_type == 'regression' and 'val_r2' in metrics:
            val_metric_str = f"  Validation R²: {metrics['val_r2']:.4f}\n"
        elif task_type != 'regression' and 'val_f1' in metrics:
            val_metric_str = f"  Validation F1: {metrics['val_f1']:.4f}\n"
        
        test_metric_str = ""
        if task_type == 'regression':
            test_metric_str = f"  Test R²: {metrics.get('test_r2', float('nan')):.4f}"
        else:
            test_metric_str = f"  Test F1: {metrics.get('test_f1', float('nan')):.4f}"
            
        console.print(Panel(
            f"Model: {model_name.upper()}\n"
            f"  Best HPO Score (Mean CV/Val): {best_score:.4f}\n"
            f"{val_metric_str}{test_metric_str}",
            title=f"Key Results for {model_name.upper()}", expand=False, border_style="green"
        ))

    return results_list