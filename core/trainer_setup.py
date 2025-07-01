# core/trainer_setup.py
import time
from rich.console import Console
from rich.panel import Panel
import numpy as np

console = Console(width=120)

def _has_smiles_columns(config):
    """Check if the configuration specifies SMILES columns for GNN processing"""
    features_config = config.get('features', {})
    molecular_config = features_config.get('molecular', {})
    
    # Check if any molecular component has a smiles_column defined
    for component, component_config in molecular_config.items():
        if isinstance(component_config, dict) and 'smiles_column' in component_config:
            return True
    
    return False

def _get_smiles_columns(config):
    """Get list of SMILES column names from configuration"""
    features_config = config.get('features', {})
    molecular_config = features_config.get('molecular', {})
    
    smiles_columns = []
    for component, component_config in molecular_config.items():
        if isinstance(component_config, dict) and 'smiles_column' in component_config:
            smiles_columns.append(component_config['smiles_column'])
    
    return smiles_columns

def run_all_models_on_data(X_train_main, y_train, X_val, y_val, X_test, y_test,
                           experiment_models_dir, experiment_run_name, config):
    """
    Manages the training and evaluation loop for all specified models.
    """
    from optimizers.ann_optimizer import ANNOptimizer
    from optimizers.sklearn_optimizer import SklearnOptimizer
    try:
        from optimizers.gnn_optimizer import GNNOptimizer
        gnn_available = True
    except (ImportError, AttributeError) as e:  # 用元组捕获多个异常
        gnn_available = False
        console.print(f"[yellow]Warning: GNN models not available (PyTorch Geometric required)[/yellow], 发生错误: {type(e).__name__} - {e}")
    from utils.io_handler import ensure_model_specific_directory, save_model_artifact, save_hyperparameters, log_results, save_predictions, save_cv_fold_predictions

    task_type = config['task_type']
    training_cfg = config['training']
    split_cfg = config['split_config']
    specific_models_to_run = training_cfg['models_to_run']
    num_optuna_trials = training_cfg['n_trials']
    use_cv_for_hpo = (config['split_mode'] == 'cross_validation')
    cv_folds_for_hpo = split_cfg['cross_validation']['n_splits'] if use_cv_for_hpo else None

    model_aliases = {
        'rf': 'randomforest', 'dt': 'decisiontree', 'knn': 'kneighbors',
        'lr': 'logisticregression', 'svm': 'svc', 'krr':'kernelridge',
        'xgb': 'xgboost', 'lgbm': 'lgbm', 'cat': 'catboost', 'hgb':'histgradientboosting',
        'ann': 'ann', 'ridge': 'ridge', 'svr': 'svr', 'adab': 'adaboost',
        # GNN aliases
        'gcn': 'gcn', 'gat': 'gat', 'mpnn': 'mpnn', 'afp': 'afp', 
        'gtn': 'graph_transformer', 'ensemble_gnn': 'ensemble_gnn'
    }
    
    if specific_models_to_run:
        specific_models_to_run = [model_aliases.get(m.lower(), m.lower()) for m in specific_models_to_run]

    common = ['xgboost', 'catboost', 'adaboost', 'decisiontree', 'histgradientboosting', 'kneighbors', 'lgbm', 'randomforest']
    
    # GNN models (available if SMILES columns are present and PyTorch Geometric is installed)
    gnn_models = []
    if gnn_available and _has_smiles_columns(config):
        gnn_models = ['gcn', 'gat', 'mpnn', 'afp', 'graph_transformer', 'ensemble_gnn']
    
    reg_models = common + ['kernelridge', 'ridge', 'svr', 'ann'] + gnn_models
    cls_models = common + ['logisticregression', 'svc', 'ann'] + gnn_models
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
        is_gnn = model_name in ['gcn', 'gat', 'mpnn', 'afp', 'graph_transformer', 'ensemble_gnn']
        
        if is_gnn:
            if not gnn_available:
                console.print(f"[red]Skipping {model_name}: GNN models require PyTorch Geometric[/red]")
                continue
            OptimizerClass = GNNOptimizer
            smiles_columns = _get_smiles_columns(config)
            opt_config = {
                'model_name': model_name,
                'smiles_columns': smiles_columns,
                'n_trials': num_optuna_trials, 
                'cv': cv_folds_for_hpo, 
                'task_type': task_type, 
                'num_classes': num_classes
            }
        elif is_ann:
            OptimizerClass = ANNOptimizer
            opt_config = {'n_trials': num_optuna_trials, 'cv': cv_folds_for_hpo, 'task_type': task_type, 'num_classes': num_classes}
        else:
            OptimizerClass = SklearnOptimizer
            opt_config = {'model_name': model_name, 'n_trials': num_optuna_trials, 'cv': cv_folds_for_hpo, 'task_type': task_type, 'num_classes': num_classes}
        
        optimizer = OptimizerClass(**opt_config)
        optimizer.console = console
        if isinstance(optimizer, SklearnOptimizer) and hasattr(optimizer, 'model_name_orig') and optimizer.model_name_orig == 'catboost':
            setattr(optimizer, 'model_run_output_dir', model_dir)

        X_val_hpo, y_val_hpo = (X_val, y_val) if not use_cv_for_hpo else (X_train_main[:1], y_train[:1])

        best_params, best_score = optimizer.optimize(X_train_main, y_train, X_val_hpo, y_val_hpo)
        best_trial_fold_scores = getattr(optimizer, 'best_trial_fold_scores_', [])
        
        # --- FIXED: Proper training flow based on split mode ---
        if use_cv_for_hpo:
            # For cross-validation mode: retrain on full training set with best hyperparameters
            console.print(f"[dim]Retraining {model_name} on full training set with best hyperparameters...[/dim]")
            optimizer.fit(X_train_main, y_train)
        else:
            # For train_valid_test mode: keep the model trained during HPO (trained on train set, validated on val set)
            console.print(f"[dim]Using {model_name} model from HPO phase (trained on train set, validated on validation set)...[/dim]")
            # The model is already trained during the optimize() phase on X_train_main
            # No need to retrain - this preserves the correct train/val/test separation
        
        metrics = optimizer.evaluate(X_train_main, y_train, X_val, y_val, X_test, y_test, console=console)
        model_runtime = time.time() - model_start_time
        
        save_model_artifact(optimizer.best_model_, f"{model_name}_model", model_dir, model_name=model_name, is_pytorch_model=(is_ann or is_gnn), console=console)
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
        
        # --- MODIFICATION: Add best_trial_fold_scores to the results dictionary ---
        results_list.append({
            'model_name': model_name, 
            'task_type': task_type, 
            'best_params': best_params,
            'best_optuna_score': best_score, 
            'best_trial_fold_scores': best_trial_fold_scores, # <<< ADDED THIS LINE
            'data_shapes': data_shapes, 
            'cv_fold_metrics': cv_fold_metrics, 
            **metrics
        })
        
        val_metric_str, test_metric_str = "", ""
        if task_type == 'regression':
            if 'val_r2' in metrics: val_metric_str = f"  Validation R²: {metrics['val_r2']:.4f}\n"
            test_metric_str = f"  Test R²: {metrics.get('test_r2', float('nan')):.4f}"
        else:
            if 'val_f1' in metrics: val_metric_str = f"  Validation F1: {metrics['val_f1']:.4f}\n"
            test_metric_str = f"  Test F1: {metrics.get('test_f1', float('nan')):.4f}"
            
        console.print(Panel(
            f"Model: {model_name.upper()}\n"
            f"  Best HPO Score (Mean CV/Val): {best_score:.4f}\n"
            f"{val_metric_str}{test_metric_str}",
            title=f"Key Results for {model_name.upper()}", expand=False, border_style="green"
        ))

    return results_list