# CANCELOUT_model.py
import os
import copy
import time
import torch
import optuna
import datetime
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

from sqlalchemy.pool import QueuePool
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


from evaluation_utils import evaluate_feature_set
from utils import get_activation, format_time, parse_list_arg, resolve_hidden_dims





def CANCELOUT_fs_with_evaluation(data, label, model_params, training_params, name, 
                               digits, folder_name, 
                               logger, feature_prediction, n_iters=20,
                               feature_step=500, n_folds=3):
    """
    execute CANCELOUT feature selection and evaluation
    
    Args:
        data: input features
        label: target values
        model_params: model parameters dictionary
        training_params: training parameters dictionary
        name: dataset name
        digits: feature dimension divisor
        folder_name: result save directory
        logger: logger
        feature_prediction: upper limit on the number of features to evaluate
        n_iters: number of iterations, default 20
        feature_step: feature evaluation step size, default 500
        n_folds: number of cross-validation folds, default 3

    """

    
    total_start_time = time.time()
    logger.info(f"Starting CANCELOUT_fs_with_evaluation at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    x_train = data['X_train']; x_test = data['X_test']
    y_train = label['y_train']; y_test = label['y_test']
    device = x_train.device if isinstance(x_train, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    total_features = x_train.shape[1]
    max_features = min(feature_prediction, total_features)



    eval_folder = os.path.join(folder_name, 'evaluation')
    os.makedirs(eval_folder, exist_ok=True)
    checkpoint_folder = os.path.join(eval_folder, 'checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)
    iter_folder = os.path.join(eval_folder, 'iterations')
    os.makedirs(iter_folder, exist_ok=True)
    feature_folder = os.path.join(eval_folder, 'feature_evaluations')
    os.makedirs(feature_folder, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_folder, 'checkpoint.npz')
    


    feature_sequence = np.arange(1, max_features + 1, feature_step)  
    logger.info(f" evaluate Top 1-{feature_sequence.max()} features with step {feature_step}")

    
    feature_sequence = np.unique(feature_sequence)  
    feature_sequence = feature_sequence[feature_sequence <= max_features] 
    n_steps = len(feature_sequence)
    

    start_iter = 0


    results = np.full((n_steps, n_iters), np.nan)
    cv_val_losses = np.full((n_steps, n_iters), np.nan)
    results_weights = []
    results_indices = []
    all_best_params = []
    seeds = np.random.choice(range(1000), n_iters, replace=False)
    all_report_metrics = [[None for _ in range(n_iters)] for __ in range(n_steps)]
    
    logger.info(f"Total features: {total_features}")
    logger.info(f"Evaluation settings:")
    logger.info(f"- Max features to evaluate: {max_features}")
    logger.info(f"- Feature sequence: {feature_sequence}")
    logger.info(f"- Number of evaluation points: {n_steps}")
    logger.info(f"- Number of cross-validation folds: {n_folds}")
    

    dropout_probs = [0.1, 0.3, 0.7, 0.9]
    l2_lambdas = [0.0001, 0.001, 0.01, 0.1]
    
    for iter in range(n_iters):
        model_iter_dir = os.path.join(eval_folder, f'model_iter_{iter}')
        os.makedirs(model_iter_dir, exist_ok=True)
        
        if training_params.get('do_parameter_search', False):
            optuna_iter_dir = os.path.join(eval_folder, f'optuna_iter_{iter}')
            os.makedirs(optuna_iter_dir, exist_ok=True)


    for iter in range(start_iter, n_iters):
        iter_start_time = time.time()
        logger.info(f'\nstart iteration {iter+1}/{n_iters}, seed {seeds[iter]}')
        
        try:
            current_model_params = model_params.copy()
            current_training_params = training_params.copy()
            current_model_params['input_size'] = x_train.shape[1]
            current_model_params['n_layers'] = int(current_model_params.get('n_layers', 2))
            current_model_params['start_digit'] = int(digits)
            current_model_params['shrink_ratio'] = float(current_model_params.get('shrink_ratio', current_training_params.get('shrink_ratio', 2.0)))
            current_model_params['digits_list'] = [
                max(1, int(round(current_model_params['start_digit'] * (current_model_params['shrink_ratio'] ** i))))
                for i in range(current_model_params['n_layers'])
            ]
            current_model_params['hidden_dims'] = [
                max(4, int(x_train.shape[1] / max(1, int(d))))
                for d in current_model_params['digits_list']
            ]
            current_model_params['digits'] = int(digits)
            current_training_params['device'] = device
            current_training_params['seed'] = seeds[iter]
            
            if current_training_params.get('do_parameter_search', False):
                logger.info(f"start hyperparameter optimization using Optuna...")
                logger.info(
                    f"current fixed CANCELOUT architecture before Optuna: n_layers={current_model_params['n_layers']}, "
                    f"start_digit={current_model_params.get('start_digit')}, shrink_ratio={current_model_params.get('shrink_ratio')}, "
                    f"generated_digits_list={current_model_params.get('digits_list')}, hidden_dims={current_model_params.get('hidden_dims')}"
                )
                importance_scores, feature_indices, trained_results = optuna_search_CANCELOUT(
                    x_train, y_train, 
                    current_model_params, 
                    current_training_params,
                    os.path.join(eval_folder, f'optuna_iter_{iter}'),
                    logger
                )
                if isinstance(trained_results, dict) and 'best_params' in trained_results:
                    all_best_params.append(trained_results['best_params'])
                logger.info(
                    f"Optuna-selected CANCELOUT architecture: n_layers={trained_results.get('best_architecture', {}).get('n_layers') if isinstance(trained_results, dict) else None}, "
                    f"start_digit={trained_results.get('best_architecture', {}).get('start_digit') if isinstance(trained_results, dict) else None}, "
                    f"shrink_ratio={trained_results.get('best_architecture', {}).get('shrink_ratio') if isinstance(trained_results, dict) else None}, "
                    f"digits_list={trained_results.get('best_architecture', {}).get('digits_list') if isinstance(trained_results, dict) else None}, "
                    f"hidden_dims={trained_results.get('best_architecture', {}).get('hidden_dims') if isinstance(trained_results, dict) else None}"
                )
            else:

                logger.info(f"start training model with fixed parameters...")


                importance_scores, feature_indices, trained_results = feature_extraction(
                    x_train, y_train,
                    current_model_params, 
                    current_training_params,
                    os.path.join(eval_folder, f'model_iter_{iter}'),
                    logger
                )
            
            results_weights.append(importance_scores)
            results_indices.append(feature_indices)
            

            feature_ranking_file = os.path.join(iter_folder, f'feature_ranking_iter_{iter}.npz')
            np.savez(feature_ranking_file, 
                    feature_indices=feature_indices, 
                    weights=importance_scores,
                    iter=iter, 
                    seed=seeds[iter])
            logger.info(f"feature ranking saved to : {feature_ranking_file}")
            feature_steps_range = range(n_steps)
            for step in feature_steps_range:
                num_features = int(feature_sequence[step])
                if num_features > len(feature_indices):
                    logger.warning(f"The requested number of features {num_features} exceeds the available number of features {len(feature_indices)}, skipping")
                    continue

                logger.info(f"\nEvaluating the top {num_features} features")

                try:
                    selected_features = feature_indices[:num_features]
                    logger.info(f"selected_features: {selected_features}...")
                
                    if isinstance(x_train, torch.Tensor):
                        x_train_selected = x_train.index_select(1, torch.tensor(selected_features, device=x_train.device))
                        x_test_selected = x_test.index_select(1, torch.tensor(selected_features, device=x_test.device))
                    else:
                        x_train_selected = x_train[:, selected_features]
                        x_test_selected = x_test[:, selected_features]
                    
                    final_score, best_dropout, best_l2, best_cv_val_loss, report_metrics = evaluate_feature_set(
                        x_train_selected, y_train,
                        x_test_selected, y_test,
                        num_features, model_params,
                        dropout_probs, l2_lambdas,
                        n_folds, device
                    )
                    

                    results[step, iter] = float(final_score) if final_score is not None else np.nan
                    cv_val_losses[step, iter] = float(best_cv_val_loss) if best_cv_val_loss is not None else np.nan
                    all_report_metrics[step][iter] = report_metrics 
                    task_type = model_params['task_type']
                    score_name = {
                        'binary': 'AUC',
                        'multiclass': 'AUC_macro',
                        'regression': 'PearsonR'
                    }.get(task_type, 'score')                 
                    logger.info(f"The number of features {num_features}:")
                    logger.info(f"CV val loss: {best_cv_val_loss:.4f}")
                    logger.info(f"Test {score_name}: {final_score:.6f}")
                    logger.info(f"Best parameters: dropout={best_dropout}, l2={best_l2}")

                    
                    feature_eval_file = os.path.join(
                        feature_folder, 
                        f'features_{num_features}_iter_{iter}.npz'
                    )
                    np.savez(feature_eval_file, 
                            num_features=num_features,
                            iter=iter,
                            selected_features=selected_features,
                            test_score=final_score,
                            best_cv_val_loss=best_cv_val_loss,
                            best_dropout=best_dropout,
                            best_l2=best_l2, 
                            report_metrics=report_metrics)
                    logger.info(f"The results for the number of features {num_features} have been saved to: {feature_eval_file}")
                    

                    checkpoint_data = {
                        'last_feature_step': step,
                        'current_iter': iter,
                        'current_feature': num_features,
                        'results': results,
                        'cv_val_losses': cv_val_losses,
                        'results_indices': np.array(results_indices, dtype=object),
                        'results_weights': np.array(results_weights, dtype=object),
                        'all_best_params': np.array(all_best_params, dtype=object) if all_best_params else None,
                        'seeds': seeds,
                        'feature_sequence': feature_sequence,
                        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                        'per_step_report_metrics': np.array(all_report_metrics, dtype=object),
                    }
                    np.savez(checkpoint_file, **checkpoint_data)
                    logger.info(f"Checkpoint updated, completed evaluation for {num_features} features ({step+1}/{n_steps}), current iteration {iter+1}/{n_iters}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {num_features} features: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            iter_results_file = os.path.join(iter_folder, f'full_results_iter_{iter}.npz')
            report_metrics_iter_column = [all_report_metrics[s][iter] for s in range(n_steps)]
            np.savez(iter_results_file, 
                    iter=iter,
                    test_results=results[:, iter],
                    cv_val_losses=cv_val_losses[:, iter],
                    feature_indices=feature_indices,
                    feature_weights=importance_scores,
                    feature_sequence=feature_sequence,
                    seed=seeds[iter],
                    timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                    report_metrics_column=report_metrics_iter_column)  
            logger.info(f"Iteration {iter+1} full results have been saved to: {iter_results_file}")
            
            checkpoint_data = {
                'last_completed_iter': iter,
                'last_feature_step': n_steps - 1,  
                'results': results,
                'cv_val_losses': cv_val_losses,
                'results_indices': np.array(results_indices, dtype=object),
                'results_weights': np.array(results_weights, dtype=object),
                'all_best_params': np.array(all_best_params, dtype=object) if all_best_params else None,
                'seeds': seeds,
                'feature_sequence': feature_sequence,
                'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                'per_step_report_metrics': np.array(all_report_metrics, dtype=object),
            }
            np.savez(checkpoint_file, **checkpoint_data)
            logger.info(f"Checkpoint updated, completed iteration {iter+1}/{n_iters}")


            iter_time = time.time() - iter_start_time
            logger.info(f"Iteration {iter+1} completed, time taken: {format_time(iter_time)} ({iter_time:.2f} seconds)")

        except Exception as e:
            logger.error(f"Iteration {iter} error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            continue
    

    total_time = time.time() - total_start_time
    logger.info(f"\nAll iterations completed, total time taken: {format_time(total_time)} ({total_time:.2f} seconds)")
    

    mean_test_results = np.nanmean(results, axis=1)
    std_test_results = np.nanstd(results, axis=1)
    mean_cv_val_losses = np.nanmean(cv_val_losses, axis=1)
    std_cv_val_losses = np.nanstd(cv_val_losses, axis=1)

    final_results = {
        'test_results': results,
        'cv_val_losses': cv_val_losses,
        'weights': np.array(results_weights, dtype=object),
        'indices': np.array(results_indices, dtype=object),
        'best_params': np.array(all_best_params, dtype=object) if all_best_params else None,
        'mean_test_results': mean_test_results,
        'std_test_results': std_test_results,
        'mean_cv_val_losses': mean_cv_val_losses,
        'std_cv_val_losses': std_cv_val_losses,
        'feature_sequence': feature_sequence,
        'total_features': total_features,
        'max_features': max_features,
        'n_steps': n_steps,
        'seeds': seeds,
        'config': {
            'n_folds': n_folds,
            'n_iters': n_iters,
            'dropout_probs': dropout_probs,
            'l2_lambdas': l2_lambdas,
        },
        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        'per_step_report_metrics': np.array(all_report_metrics, dtype=object),
    }
    

    save_path = os.path.join(folder_name, f'CANCELOUT_{name}_results.npz')
    np.savez(save_path, **final_results)
    

    # timestamp_save_path = os.path.join(
    #     folder_name, 
    #     f'CANCELOUT_{name}_results_{final_results["timestamp"]}.npz'
    # )
    # np.savez(timestamp_save_path, **final_results)

    logger.info("\nFinal results summary:")
    for step, num_features in enumerate(feature_sequence):
        if step >= len(mean_test_results):
            continue
            
        num_features = int(num_features)
        logger.info(f"\n{num_features} features results:")
        logger.info(f"Average val_loss: {mean_cv_val_losses[step]:.4f} ± {std_cv_val_losses[step]:.4f}")
        logger.info(f"Average test score: {mean_test_results[step]:.4f} ± {std_test_results[step]:.4f}")

    logger.info(f"\nResults saved to: {save_path}")
    # logger.info(f"Timestamp version saved to: {timestamp_save_path}")

    return final_results


def CANCELOUT_fs(data, label, model_params, training_params, name, folder_name, logger):
    """
    Main function for CANCELOUT feature selection with optional parameter optimization
    Args:
        data: input features
        label: target values
        model_params: model architecture parameters (including digits)
        training_params: training parameters
        folder_name: directory for saving results
        logger: logging object
    """

  

    X_train = data['X_train']
    y_train = label['y_train']

    importance_scores, feature_indices, results = optuna_search_CANCELOUT(
        X_train, y_train, model_params, training_params, folder_name, logger
    )
    

    np.save(os.path.join(folder_name, f'CANCELOUT_{name}_weights.npy'), 
            importance_scores)
    np.save(os.path.join(folder_name, f'CANCELOUT_{name}_idx.npy'), 
            feature_indices)
    
    logger.info(f"\nFiles saved to folder: {folder_name}")
    
    return importance_scores, feature_indices, results


def optuna_search_CANCELOUT(data, label, model_params, training_params, folder_name, logger):
    """
    Perform feature selection using CANCELOUT method with Optuna hyperparameter optimization
    
    Args:
        data: input features
        label: target values
        model_params: model architecture parameters (including digits)
        training_params: training parameters
        folder_name: directory for saving results
        logger: logging object
    """



    total_start_time = time.time()
    logger.info(f"optuna_search_CANCELOUT start at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    

    do_parameter_search = training_params.get('do_parameter_search', 0)
    use_cv = training_params.get('use_cv', True)
    

    task_type = model_params.get('task_type', 'regression')

    eval_metric = training_params.get('eval_metric', None)
    if eval_metric is None:
        eval_metric = 'acc' if task_type in ('binary', 'multiclass') else 'r2'
    
    minimize_metric = eval_metric in ['mse', 'mae', 'loss']

    logger.info(f"Parameter search will use '{eval_metric}' as the evaluation metric.")
    logger.info(f"During optimization, the metric will be {'minimized' if minimize_metric else 'maximized'}.")
    logger.info(f"Hyperparameter search use_cv={use_cv}")

    if do_parameter_search == 1:
        logger.info("Start Optuna hyperparameter optimization for CANCELOUT...")


        optuna_folder = os.path.join(folder_name, 'optuna_search')
        os.makedirs(optuna_folder, exist_ok=True)


        # timing_log_path = os.path.join(optuna_folder, 'timing_log_10.txt')
        # with open(timing_log_path, 'w') as f:
        #     f.write(f"CANCELOUT Optuna search timing log - Start at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        #     f.write("=" * 80 + "\n")


        k_folds = training_params.get('n_splits', 3)
        val_ratio = training_params.get('validation_split', 0.2)
        if use_cv:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=training_params['seed'])
            logger.info(f"Using {k_folds}-fold cross-validation during Optuna search.")
        else:
            logger.info(
                f"use_cv=False, so Optuna search will use a single validation split "
                f"(validation_split={val_ratio}) instead of K-fold CV."
            )


        search_start_time = time.time()


        lambda_1_min = training_params.get('lambda_1_min', 1e-5)
        lambda_1_max = training_params.get('lambda_1_max', 1e-1)
        lambda_2_min = training_params.get('lambda_2_min', 1e-5)
        lambda_2_max = training_params.get('lambda_2_max', 1e-1)
        weight_decay_min = training_params.get('weight_decay_min', 0.0)  
        weight_decay_max = training_params.get('weight_decay_max', 1e-1) 
        lr_min = training_params.get('lr_min', 1e-5)
        lr_max = training_params.get('lr_max', 1e-1)
        dropout_min = training_params.get('dropout_min', 0.1)
        dropout_max = training_params.get('dropout_max', 0.5)
        start_digit_choices = parse_list_arg(training_params.get('digits_list', '50,100,200'), dtype=int)
        shrink_ratio_choices = parse_list_arg(training_params.get('shrink_ratio_list', '1.25,1.5,2.0'), dtype=float)
        batch_size_choices = parse_list_arg(training_params.get('batch_size_list', '16,32,64,128'), dtype=int)

        min_layers = max(2, int(training_params.get('min_layers', 2)))
        max_layers = max(min_layers, int(training_params.get('max_layers', 5)))        

        trial_times = {}
        

        def objective(trial):
            trial_start_time = time.time()
            trial_id = trial.number
            lambda_1 = trial.suggest_float('lambda_1', lambda_1_min, lambda_1_max, log=True)
            lambda_2 = trial.suggest_float('lambda_2', lambda_2_min, lambda_2_max, log=True)
            weight_decay = trial.suggest_float('weight_decay', max(weight_decay_min, 1e-6), weight_decay_max, log=True)
            dropout = trial.suggest_float('dropout', dropout_min, dropout_max)
            lr = trial.suggest_float('lr', lr_min, lr_max, log=True)
            batch_size = trial.suggest_categorical('batch_size', batch_size_choices)
            n_layers = trial.suggest_int('n_layers', min_layers, max_layers)
            start_digit = trial.suggest_categorical('start_digit', start_digit_choices)
            shrink_ratio = trial.suggest_categorical('shrink_ratio', shrink_ratio_choices)
            digits_list = [
                max(1, int(round(start_digit * (shrink_ratio ** i))))
                for i in range(n_layers)
            ]
            hidden_dims = [
                max(4, int(data.shape[1] / max(1, int(d))))
                for d in digits_list
            ]



            cancelout_init = None
            if training_params.get('search_cancelout_init', 0) == 1:
                activation = model_params.get('activation', 'relu')
                if activation == 'sigmoid':
                    cancelout_init = trial.suggest_float('cancelout_init', -3.0, 3.0)
                elif activation == 'relu':
                    cancelout_init = trial.suggest_float('cancelout_init', 0.001, 3.0)
                else:
                    cancelout_init = trial.suggest_float('cancelout_init', 1.0, 3.0)

            logger.info(f"Trial {trial_id}: lambda_1={lambda_1}, lambda_2={lambda_2}, weight_decay={weight_decay}, "
                    f"dropout={dropout}, lr={lr}, batch_size={batch_size}, "
                    f"n_layers={n_layers}, start_digit={start_digit}, shrink_ratio={shrink_ratio}, "
                    f"generated_digits_list={digits_list}, generated_hidden_dims={hidden_dims}"
                    f"{', cancelout_init='+str(cancelout_init) if cancelout_init is not None else ''}")
            # logger.info(f"Trial {trial_id}: lambda_1={lambda_1}, lambda_2={lambda_2}, weight_decay={weight_decay}, "
            #            f"dropout={dropout}, lr={lr}, batch_size={batch_size}, digits={digit}"
            #            f"{', cancelout_init='+str(cancelout_init) if cancelout_init is not None else ''}")

            # with open(timing_log_path, 'a') as f:
            #     f.write(f"\nTrial {trial_id} - Start at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            #     f.write(f"Parameters: lambda_1={lambda_1}, lambda_2={lambda_2}, weight_decay={weight_decay}, dropout={dropout}, lr={lr}, "
            #             f"batch_size={batch_size}, digits={digit}"
            #             f"{', cancelout_init='+str(cancelout_init) if cancelout_init is not None else ''}\n")


            fold_scores = []
            fold_times = []

            if use_cv:
                split_iter = kf.split(data)
            else:
                all_indices = np.arange(data.shape[0])
                label_np = None
                if model_params.get('task_type', 'regression') in ('binary', 'multiclass'):
                    if isinstance(label, torch.Tensor):
                        label_np = label.detach().cpu().numpy().ravel()
                    else:
                        label_np = np.asarray(label).ravel()
                train_idx, val_idx = train_test_split(
                    all_indices,
                    test_size=val_ratio,
                    random_state=training_params['seed'],
                    stratify=label_np if label_np is not None else None
                )
                split_iter = [(train_idx, val_idx)]

            n_eval_splits = k_folds if use_cv else 1

            for fold, (train_idx, val_idx) in enumerate(split_iter):
                fold_start_time = time.time()
                logger.info(f"Trial {trial_id}, Fold {fold+1}/{n_eval_splits}")

                train_data = data[train_idx]
                train_labels = label[train_idx]
                val_data = data[val_idx]
                val_labels = label[val_idx]

                current_model_params = copy.deepcopy(model_params)
                current_model_params['n_layers'] = n_layers
                current_model_params['start_digit'] = int(start_digit)
                current_model_params['shrink_ratio'] = float(shrink_ratio)
                current_model_params['digits_list'] = [int(d) for d in digits_list]
                current_model_params['hidden_dims'] = hidden_dims
                current_model_params['dropout_prob'] = dropout
                current_model_params['digits'] = int(start_digit)
                if cancelout_init is not None:
                    current_model_params['cancelout_init'] = cancelout_init
                current_training_params = copy.deepcopy(training_params)
                current_training_params['lambda_1'] = lambda_1
                current_training_params['lambda_2'] = lambda_2
                current_training_params['weight_decay'] = weight_decay
                current_training_params['lr'] = lr
                current_training_params['batch_size'] = batch_size

                fold_result = single_fold_train(
                    train_data, train_labels, val_data, val_labels,
                    current_model_params, current_training_params, 
                    os.path.join(optuna_folder, f'trial_{trial_id}_fold_{fold}'),
                    logger
                )

                fold_time = time.time() - fold_start_time
                fold_times.append(fold_time)

                if 'loss' == eval_metric:
                    fold_score = fold_result.get('best_val_loss', fold_result['val_losses'][-1])
                else:
                    best_val_metrics = fold_result.get('best_val_metrics', None)
                    if best_val_metrics is not None and eval_metric in best_val_metrics:
                        fold_score = best_val_metrics[eval_metric]
                    else:
                        logger.warning(f"Validation metric '{eval_metric}' not found in best epoch metrics. Using fallback metric.")
                        _task = model_params.get('task_type', 'regression')
                        fold_score = (best_val_metrics or {}).get('acc' if _task in ('binary','multiclass') else 'r2', 0)

                fold_scores.append(fold_score)
                logger.info(f"Trial {trial_id}, Fold {fold+1}/{n_eval_splits} completed - Time taken: {format_time(fold_time)}")

            # Compute the average score for the current parameter combination
            avg_score = np.mean(fold_scores)

            # Record the trial completion time
            trial_total_time = time.time() - trial_start_time
            trial_times[trial_id] = trial_total_time

            # Record the trial completion time information
            # with open(timing_log_path, 'a') as f:
            #     f.write(f"Trial {trial_id} - Total time: {format_time(trial_total_time)} ({trial_total_time:.2f} seconds)\n")
            #     f.write(f"Average {eval_metric.upper()}: {avg_score:.4f}\n")
            #     f.write("-" * 40 + "\n")

            logger.info(f"Trial {trial_id} completed - Time taken: {format_time(trial_total_time)} - Average {eval_metric.upper()}: {avg_score:.4f}")

            # Optuna attempts to minimize the return value of the objective function
            return avg_score 

        # Get n_jobs parameter
        n_jobs = training_params.get('n_jobs', 1)


        try:
            db_file = os.path.join(optuna_folder, f"optuna_CANCELOUT.db")
            db_url = f"sqlite:///{db_file}"

            storage = RDBStorage(
                url=db_url,
                engine_kwargs={
                    "poolclass": QueuePool,
                    "pool_size": min(n_jobs + 1, 20),
                    "max_overflow": 10,
                    "pool_timeout": 30
                }
            )
            logger.info(f"using RDB storage to handle parallel optimization, database: {db_url}, parallel workers: {n_jobs}")
        except ImportError:
            logger.warning("cannot import RDBStorage, using default storage")
            storage = None
        sampler = TPESampler(seed=training_params['seed'])
        study = optuna.create_study(
            sampler=sampler, 
            direction='minimize' if minimize_metric else 'maximize',
            storage=storage,
            study_name=f"CANCELOUT_shrink_ratio_v3",
            load_if_exists=True
        )


        n_trials = training_params.get('n_trials', 20)
        logger.info(
            f"using {n_jobs} parallel workers to run {n_trials} Optuna optimization trials "
            f"(use_cv={use_cv})"
        )
        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        except KeyboardInterrupt:
            logger.warning("User interrupted optimization.")
        except Exception as e:
            logger.error(f"Error occurred during optimization: {e}")
            if n_jobs > 1:
                logger.warning("Parallel optimization failed. Retrying with n_jobs=1")
                study.optimize(objective, n_trials=n_trials, n_jobs=1)

        # Get best trial
        best_trial = study.best_trial
        best_params = best_trial.params
        

        search_time = time.time() - search_start_time

        logger.info(f"Optuna optimization completed, time taken {format_time(search_time)} ({search_time/60:.2f} minutes)")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best value: {best_trial.value}")

        # Record search summary time information
        # with open(timing_log_path, 'a') as f:
        #     f.write("\n" + "=" * 40 + "\n")
        #     f.write(f"Search summary - Completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        #     f.write(f"Total search time: {format_time(search_time)} ({search_time:.2f} seconds)\n")
        #     f.write(f"Number of trials: {n_trials}\n")
        #     f.write(f"Best parameters: {best_params}\n")
        #     f.write(f"Best value: {best_trial.value}\n\n")

        #     # Sort trials by duration
        #     sorted_trials = sorted(trial_times.items(), key=lambda x: x[1], reverse=True)
        #     f.write("Trial duration statistics:\n")
        #     f.write(f"  Fastest trial: {format_time(min(trial_times.values()))}\n")
        #     f.write(f"  Slowest trial: {format_time(max(trial_times.values()))}\n")
        #     f.write(f"  Average trial duration: {format_time(sum(trial_times.values()) / len(trial_times))}\n\n")

        #     f.write("Top 5 longest trials:\n")
        #     for i, (trial_id, duration) in enumerate(sorted_trials[:5]):
        #         f.write(f"  #{i+1}: Trial {trial_id} - {format_time(duration)} ({duration:.2f} seconds)\n")
        #     f.write("\n")

        logger.info("Using best parameters to train final model...")
        final_model_start_time = time.time()

        # Update model parameters and training parameters to best configuration
        final_model_params = copy.deepcopy(model_params)
        if 'n_layers' in best_params:
            best_n_layers = max(2, int(best_params['n_layers']))
            best_start_digit = int(best_params.get('start_digit', final_model_params.get('start_digit', final_model_params.get('digits', 100))))
            best_shrink_ratio = float(best_params.get('shrink_ratio', final_model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0))))
            best_digits_list = [
                max(1, int(round(best_start_digit * (best_shrink_ratio ** i))))
                for i in range(best_n_layers)
            ]
            final_model_params['n_layers'] = best_n_layers
            final_model_params['start_digit'] = best_start_digit
            final_model_params['shrink_ratio'] = best_shrink_ratio
            final_model_params['digits_list'] = best_digits_list
            final_model_params['hidden_dims'] = [
                max(4, int(data.shape[1] / max(1, d))) for d in best_digits_list
            ]
            final_model_params['digits'] = best_start_digit
        else:
            final_model_params['hidden_dims'] = resolve_hidden_dims(final_model_params)

        final_model_params['dropout_prob'] = best_params['dropout']
        if 'cancelout_init' in best_params:  
            final_model_params['cancelout_init'] = best_params['cancelout_init']        
        final_training_params = copy.deepcopy(training_params)
        final_training_params['lambda_1'] = best_params['lambda_1']
        final_training_params['lambda_2'] = best_params['lambda_2']
        final_training_params['weight_decay'] = best_params.get('weight_decay', 0.0) 
        final_training_params['lr'] = best_params['lr']
        final_training_params['batch_size'] = best_params['batch_size']

        importance_scores, feature_indices, results = feature_extraction(
            data, label, final_model_params, final_training_params, folder_name, logger
        )
        

        final_model_time = time.time() - final_model_start_time
        total_time = time.time() - total_start_time

        logger.info(f"Final model training completed in {format_time(final_model_time)} ({final_model_time/60:.2f} minutes)")
        logger.info(f"Total processing time: {format_time(total_time)} ({total_time/60:.2f} minutes)")

        # Record final timing information
        # with open(timing_log_path, 'a') as f:
        #     f.write(f"Final model training time: {format_time(final_model_time)} ({final_model_time:.2f} seconds)\n")
        #     f.write(f"Total processing time: {format_time(total_time)} ({total_time:.2f} seconds)\n")
        #     f.write("=" * 80 + "\n")
        

        with open(os.path.join(optuna_folder, 'best_params.txt'), 'w') as f:
            f.write("Best hyperparameters:\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write("\nBest architecture:\n")
            f.write(f"n_layers: {final_model_params.get('n_layers')}\n")
            f.write(f"start_digit: {final_model_params.get('start_digit')}\n")
            f.write(f"shrink_ratio: {final_model_params.get('shrink_ratio')}\n")
            f.write(f"digits_list: {final_model_params.get('digits_list')}\n")
            f.write(f"hidden_dims: {final_model_params.get('hidden_dims')}\n")
            f.write(f"\nBest value: {best_trial.value}\n")
            f.write(f"Search time: {format_time(search_time)} ({search_time:.2f} seconds)\n")
            f.write(f"Final model training time: {format_time(final_model_time)} ({final_model_time:.2f} seconds)\n")
            f.write(f"Total time: {format_time(total_time)} ({total_time:.2f} seconds)\n")

        import joblib
        joblib.dump(study, os.path.join(optuna_folder, "optuna_study.pkl"))
        
        results['best_params'] = best_params
        results['best_value'] = best_trial.value
        results['best_architecture'] = {
            'n_layers': final_model_params.get('n_layers'),
            'start_digit': final_model_params.get('start_digit'),
            'shrink_ratio': final_model_params.get('shrink_ratio'),
            'digits_list': final_model_params.get('digits_list'),
            'hidden_dims': final_model_params.get('hidden_dims'),
        }
        return importance_scores, feature_indices, results
        
    else:

        logger.info("Parameter optimization is disabled. Using single parameter values.")
        logger.info(f"Using parameters:")
        logger.info(f"  lambda_1: {training_params['lambda_1']}")
        logger.info(f"  lambda_2: {training_params['lambda_2']}")
        logger.info(f"  dropout: {model_params['dropout_prob']}")
        logger.info(f"  lr: {training_params['lr']}")
        logger.info(f"  batch_size: {training_params['batch_size']}")
        logger.info(f"  start_digit: {model_params.get('start_digit', model_params.get('digits', 100))}")
        logger.info(f"  shrink_ratio: {model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0))}")
        logger.info(f"  n_layers: {model_params.get('n_layers', 2)}")
        logger.info(f"  generated_digits_list: {model_params.get('digits_list', None)}")
        logger.info(f"  hidden_dims: {resolve_hidden_dims(model_params)}")
        logger.info("  use_cv: False (no Optuna CV because do_parameter_search=0)")
        

        single_model_start_time = time.time()
        
        importance_scores, feature_indices, results = feature_extraction(
            data, label, model_params, training_params, folder_name, 
            logger
        )
        

        single_model_time = time.time() - single_model_start_time
        total_time = time.time() - total_start_time

        logger.info(f"Single model training completed in {format_time(single_model_time)} ({single_model_time/60:.2f} minutes)")
        logger.info(f"Total processing time: {format_time(total_time)} ({total_time/60:.2f} minutes)")

        # Save timing information
        # with open(os.path.join(folder_name, 'timing_info_10.txt'), 'w') as f:
        #     f.write(f"CANCELOUT Single Model Training - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        #     f.write("=" * 80 + "\n")
        #     f.write(f"Single model training time: {format_time(single_model_time)} ({single_model_time:.2f} seconds)\n")
        #     f.write(f"Total processing time: {format_time(total_time)} ({total_time:.2f} seconds)\n")

        return importance_scores, feature_indices, results


def single_fold_train(train_data, train_labels, val_data, val_labels, model_params, training_params, folder_name,  logger):
    """
    Train model on a single fold for cross-validation
    Args:
        train_data: training features
        train_labels: training labels
        val_data: validation features
        val_labels: validation labels
        model_params: model architecture parameters
        training_params: training parameters
        folder_name: directory for saving results
        logger: logging object

    """

    task_type = model_params.get('task_type', 'regression')
    device = training_params['device']


    def _prep_xy(X, y):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=device)
        else:
            y = y.to(device)
        if task_type in ('binary','multiclass'):
            if y.dtype != torch.long:
                y = y.long()
        else:
            y = y.float()
            if y.dim() == 1:
                y = y.unsqueeze(1)  # (N,) -> (N,1)
        return X, y
    
    train_data, train_labels = _prep_xy(train_data, train_labels)
    val_data,   val_labels   = _prep_xy(val_data,   val_labels)


    train_dataset = Data.TensorDataset(train_data, train_labels)
    val_dataset   = Data.TensorDataset(val_data,   val_labels)
    
    train_loader = Data.DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, drop_last=True)
    val_loader   = Data.DataLoader(val_dataset,   batch_size=training_params['batch_size'], shuffle=False, drop_last=False)
    model = NeuralNet(model_params).to(training_params['device'])

    cancelout_params = list(model.cancelout.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('cancelout')]
    
    optimizer = torch.optim.Adam(
        [
            {'params': cancelout_params, 'weight_decay': 0.0},
            {'params': other_params, 'weight_decay': training_params.get('weight_decay', 0.0)},
        ],
        lr=training_params['lr']
    )
    

    task_type   = model_params.get('task_type', 'regression')
    # num_classes = model_params.get('num_classes', 2 if task_type=='binary' else 1)
    
    if task_type in ('binary','multiclass'):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
        

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    best_val_loss = float('inf')
    best_epoch = -1
    best_val_metrics = None
    best_model_state = None
    best_importance = None
    early_stopping_counter = 0
    

    for epoch in range(training_params['num_epochs']):
        model.train()
        epoch_loss = 0
        metrics = {'train': {}, 'val': {}}
        

        all_train_preds = []
        all_train_targets = []
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            if len(inputs) <= 1: 
                continue
            inputs = inputs.to(device=training_params['device'], dtype=torch.float32)
            targets = targets.to(device=training_params['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            weights_co = list(model.cancelout.parameters())[0]
            l1_norm = torch.norm(weights_co, 1)
            var = torch.var(weights_co)
            loss = criterion(outputs, targets) + \
                   training_params['lambda_1'] * l1_norm - \
                   training_params['lambda_2'] * var
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if task_type in ('binary','multiclass'):
                predicts = outputs.argmax(dim=1)
                train_correct += (predicts == targets).sum().item()
                train_total += targets.numel()
            else:
                all_train_preds.extend(outputs.detach().cpu().numpy().reshape(-1))
                all_train_targets.extend(targets.detach().cpu().numpy().reshape(-1))

        train_losses.append(epoch_loss / max(len(train_loader), 1))
        if task_type in ('binary','multiclass'):
            metrics['train']['acc'] = train_correct / max(train_total, 1)
        if task_type == 'regression':
            all_train_preds = np.array(all_train_preds); all_train_targets = np.array(all_train_targets)
            metrics['train']['mse'] = mean_squared_error(all_train_targets, all_train_preds)
            metrics['train']['mae'] = mean_absolute_error(all_train_targets, all_train_preds)
            metrics['train']['r2']  = r2_score(all_train_targets, all_train_preds)
        
        train_metrics.append(metrics['train'])
        
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_targets = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                if len(inputs) <= 1:  
                    continue
                inputs = inputs.to(device=training_params['device'], dtype=torch.float32)
                targets = targets.to(device=training_params['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                if task_type in ('binary','multiclass'):
                    predicts = outputs.argmax(dim=1)
                    val_correct += (predicts == targets).sum().item()
                    val_total += targets.numel()
                else:
                    all_val_preds.extend(outputs.cpu().numpy().reshape(-1))
                    all_val_targets.extend(targets.cpu().numpy().reshape(-1))

        val_losses.append(val_loss / max(len(val_loader), 1))
        if task_type in ('binary','multiclass'):
            metrics['val']['acc'] = val_correct / max(val_total, 1)
        
        if task_type == 'regression':
            all_val_preds = np.array(all_val_preds); all_val_targets = np.array(all_val_targets)
            metrics['val']['mse'] = mean_squared_error(all_val_targets, all_val_preds)
            metrics['val']['mae'] = mean_absolute_error(all_val_targets, all_val_preds)
            metrics['val']['r2']  = r2_score(all_val_targets, all_val_preds)
        
        val_metrics.append(metrics['val'])

        current_val_loss = val_losses[-1]
        if current_val_loss < best_val_loss - training_params['min_delta']:
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_val_metrics = copy.deepcopy(metrics['val'])
            early_stopping_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_importance = list(model.cancelout.parameters())[0].detach().cpu().numpy()
        else:
            early_stopping_counter += 1
        

        if early_stopping_counter >= training_params['patience']:
            break
    

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        logger.warning("single_fold_train: best_model_state is None, using final epoch model state.")

    if best_importance is None:
        logger.warning("single_fold_train: best_importance is None, using final CancelOut weights.")
        best_importance = list(model.cancelout.parameters())[0].detach().cpu().numpy()

    importance_scores = torch.sigmoid(torch.tensor(best_importance)).numpy()
    feature_indices = sorted_idx(importance_scores)

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'importance_scores': importance_scores,
        'feature_indices': feature_indices,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'best_val_metrics': best_val_metrics
    }

def feature_extraction(data, label, model_params, training_params, folder_name, logger):
    """
    Extract features using CancelOut method with specific lambda parameters
    
    Args:
        data: input features
        label: target values
        model_params: model architecture parameters
        training_params: training parameters (now includes specific lambda values)
        folder_name: directory for saving results
    """

    extraction_start_time = time.time()
    logger.info(f"start feature selection at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model_training_start = time.time()

    logger.info("start training model...")
    results = train_model(data, label, model_params, training_params, folder_name, logger)

    model_training_time = time.time() - model_training_start
    logger.info(f"model training completed in {format_time(model_training_time)} ({model_training_time/60:.2f} minutes)")

    importance_start = time.time()

    logger.info("collecting feature importance scores from training results...")
    importance_scores = results['importance_scores']
    feature_indices = results['feature_indices']

    importance_time = time.time() - importance_start
    logger.info(f"feature importance collection completed in {format_time(importance_time)}")

    total_extraction_time = time.time() - extraction_start_time
    logger.info(f"total extraction time: {format_time(total_extraction_time)} ({total_extraction_time/60:.2f} minutes)")

    return importance_scores, feature_indices, results

class CancelOut(nn.Module):
    def __init__(self, inp, activation='relu', init_value=None, *kargs, **kwargs):
        super(CancelOut, self).__init__()
        if init_value is not None:
            self.weights = nn.Parameter(torch.zeros(inp, requires_grad=True) + init_value)
        else:
            if activation == 'sigmoid':
                self.weights = nn.Parameter(torch.zeros(inp, requires_grad=True) - 3.0)
            elif activation == 'relu':
                self.weights = nn.Parameter(torch.zeros(inp, requires_grad=True) + 0.1)
            else:
                self.weights = nn.Parameter(torch.zeros(inp, requires_grad=True) + 1.0)
    
    def forward(self, x):
        return (x * torch.sigmoid(self.weights))
    




class NeuralNet(nn.Module):
    def __init__(self, model_params):
        super(NeuralNet, self).__init__()

        self.task_type = model_params.get('task_type', 'regression')
        self.num_classes = model_params.get('num_classes', 2 if self.task_type == 'binary' else 1)
        self.dropout_prob = model_params.get('dropout_prob', 0.5)
        self.activation_name = model_params.get('activation', 'relu')
        self.hidden_dims = resolve_hidden_dims(model_params)

        self.cancelout = CancelOut(
            model_params['input_size'],
            activation=self.activation_name,
            init_value=model_params.get('cancelout_init', None)
        )

        layers = []
        prev_dim = model_params['input_size']
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(self.activation_name))
            layers.append(nn.Dropout(p=self.dropout_prob))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        out_features = 1 if self.task_type == 'regression' else self.num_classes
        self.output_layer = nn.Linear(prev_dim, out_features)

    def forward(self, x):
        x = self.cancelout(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x



def sorted_idx(select_rate):  
    a = select_rate[0, :] if len(select_rate.shape) > 1 else select_rate
    idx = sorted(range(len(a)), key=lambda k: a[k], reverse=True)
    return idx


def train_model(data, label, model_params, training_params, folder_name, logger):
    """
    Train model with given parameters
    Args:
        data: input features (torch.Tensor)
        label: target values (torch.Tensor or np.array)
        model_params: model architecture parameters
        training_params: training parameters
        folder_name: directory for saving results
        logger: logging object
    """

    task_type = model_params.get('task_type', 'regression')

    device = training_params['device']
 

    
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, device=device)
    else:
        label = label.to(device)
    
    if task_type in ('binary','multiclass'):
        if label.dtype != torch.long:
            label = label.long()
    else:
        label = label.float()
        if label.dim() == 1:
            label = label.unsqueeze(1)

    dataset = Data.TensorDataset(data, label)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    
    stratify = label.view(-1).cpu().numpy() if (task_type in ('binary','multiclass')) else None
    train_indices, val_indices = train_test_split(
        indices,
        test_size=training_params['validation_split'],
        random_state=training_params.get('seed', 42),
        stratify=stratify
    )
    train_sampler = Data.Subset(dataset, train_indices)
    valid_sampler = Data.Subset(dataset, val_indices)
    
    train_loader = Data.DataLoader(train_sampler, batch_size=training_params['batch_size'],shuffle=True)
    validation_loader = Data.DataLoader(valid_sampler, batch_size=training_params['batch_size'])
    

    model = NeuralNet(model_params).to(training_params['device'])

    cancelout_params = list(model.cancelout.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('cancelout')]
    

    optimizer = torch.optim.Adam([
        {'params': cancelout_params, 'weight_decay': 0.0},  
        {'params': other_params, 'weight_decay': training_params.get('weight_decay', 0.0)}  
    ], lr=training_params['lr'])
    
    task_type   = model_params.get('task_type', 'regression')
    # num_classes = model_params.get('num_classes', 2 if task_type=='binary' else 1)
    
    if task_type in ('binary','multiclass'):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()


    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_state = None
    best_importance = None
    early_stopping_counter = 0
    

    for epoch in range(training_params['num_epochs']):
        model.train()
        epoch_loss = 0
        metrics = {'train': {}, 'val': {}}
        

        all_train_preds = []
        all_train_targets = []
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            if len(inputs) <= 1:  
                continue      

            inputs = inputs.to(device=training_params['device'], dtype=torch.float32)
            targets = targets.to(device=training_params['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            
            weights_co = list(model.cancelout.parameters())[0]
            l1_norm = torch.norm(weights_co, 1)
            var = torch.var(weights_co)
            loss = criterion(outputs, targets) + \
                   training_params['lambda_1'] * l1_norm - \
                   training_params['lambda_2'] * var
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if task_type in ('binary','multiclass'):
                predicts = outputs.argmax(dim=1)
                train_correct += torch.eq(predicts, targets).sum().float().item()
                train_total += len(targets)
            else:
                all_train_preds.extend(outputs.detach().cpu().numpy().reshape(-1))
                all_train_targets.extend(targets.detach().cpu().numpy().reshape(-1))
        
        train_losses.append(epoch_loss / max(len(train_loader), 1))
        if task_type in ('binary','multiclass'):
            metrics['train']['acc'] = train_correct / max(train_total, 1)

        if task_type in ('regression'):
            all_train_preds = np.array(all_train_preds)
            all_train_targets = np.array(all_train_targets)
            metrics['train']['mse'] = mean_squared_error(all_train_targets, all_train_preds)
            metrics['train']['mae'] = mean_absolute_error(all_train_targets, all_train_preds)
            metrics['train']['r2'] = r2_score(all_train_targets, all_train_preds)
        
        train_metrics.append(metrics['train'])
        

        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_targets = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in validation_loader:
                if len(inputs) <= 1:  
                    continue
                inputs = inputs.to(device=training_params['device'], dtype=torch.float32)
                targets = targets.to(device=training_params['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                if task_type in ('binary','multiclass'):
                    predicts = outputs.argmax(dim=1)
                    val_correct += torch.eq(predicts, targets).sum().float().item()
                    val_total += len(targets)
                else:
                    all_val_preds.extend(outputs.cpu().numpy().reshape(-1))
                    all_val_targets.extend(targets.cpu().numpy().reshape(-1))

        val_losses.append(val_loss / max(len(validation_loader), 1))
        if task_type in ('binary','multiclass'):
            metrics['val']['acc'] = val_correct / max(val_total, 1)


        if task_type in ('regression'):
            all_val_preds = np.array(all_val_preds)
            all_val_targets = np.array(all_val_targets)
            metrics['val']['mse'] = mean_squared_error(all_val_targets, all_val_preds)
            metrics['val']['mae'] = mean_absolute_error(all_val_targets, all_val_preds)
            metrics['val']['r2'] = r2_score(all_val_targets, all_val_preds)
        
        val_metrics.append(metrics['val'])
        

        if epoch % 10 == 0 or epoch == training_params['num_epochs'] - 1:
            if task_type in ('binary','multiclass'):
                logger.info(
                    f'EPOCH: {epoch} - Train Loss: {train_losses[-1]:.4f}, '
                    f'Train Acc: {metrics["train"]["acc"]:.4f}, '
                    f'Val Loss: {val_losses[-1]:.4f}, '
                    f'Val Acc: {metrics["val"]["acc"]:.4f}'
                )
            else:
                logger.info(
                    f'EPOCH: {epoch} - Train Loss: {train_losses[-1]:.4f}, '
                    f'Train MSE: {metrics["train"]["mse"]:.4f}, '
                    f'Train R2: {metrics["train"]["r2"]:.4f}, '
                    f'Val Loss: {val_losses[-1]:.4f}, '
                    f'Val MSE: {metrics["val"]["mse"]:.4f}, '
                    f'Val R2: {metrics["val"]["r2"]:.4f}'
                )
        
        current_val_loss = val_losses[-1]
        if current_val_loss < best_val_loss - training_params['min_delta']:
            best_val_loss = current_val_loss
            best_epoch = epoch
            early_stopping_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_importance = list(model.cancelout.parameters())[0].detach().cpu().numpy()
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= training_params['patience']:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        logger.warning("train_model: best_model_state is None, using final epoch model state.")

    if best_importance is None:
        logger.warning("train_model: best_importance is None, using final CancelOut weights.")
        best_importance = list(model.cancelout.parameters())[0].detach().cpu().numpy()

    importance_scores = torch.sigmoid(torch.tensor(best_importance)).numpy()
    feature_indices = sorted_idx(importance_scores)
    return {
        'model': model,
        'cancelout_weights': best_importance,
        'importance_scores': importance_scores,
        'feature_indices': feature_indices,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
    }

