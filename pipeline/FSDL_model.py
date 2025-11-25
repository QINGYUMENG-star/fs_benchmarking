import os
import time
import copy  
import torch
import optuna  
import joblib
import logging
import datetime  
import traceback
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from captum.attr import *
from sqlalchemy.pool import QueuePool
from optuna.samplers import TPESampler  
from optuna.storages import RDBStorage
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from evaluation_utils import evaluate_feature_set
from utils import get_activation, format_time,parse_list_arg


def FSDL(data,label,features_selected, folder_name, training_params, model_params,name, method, digits,logger):
    """
    Main function for Feature Selection with Deep Learning 
    
    Args:
        data: input features
        label: target values
        features_selected: pre-selected features
        folder_name: directory to save results
        training_params: training parameters dictionary
        model_params: model parameters dictionary
        method: feature attribution method (e.g., 'DeepLift', 'GradientShap', 'LRP', etc.)
        digits: divisor for feature dimensions
        logger: logging object
    """

    logger.info(f"Starting FSDL with method {method}")

    X_train = data['X_train']
    y_train = label['y_train']

    data_o = {
        'X': X_train,
        'Y': y_train,
        'selected_features': features_selected if features_selected is not None else []
    }   

    os.makedirs(folder_name, exist_ok=True)

    do_parameter_search = training_params.get('do_parameter_search', 0)
    analyzer = FeatureAttributionAnalyzer(
        data=data_o,
        name=name,
        digits=digits,
        device=training_params['device'],
        folder_name=folder_name,
        FloatTensor=training_params['FloatTensor'],
        LongTensor=training_params['LongTensor'],
        task_type=model_params['task_type'],
        logger=logger
    )
    model_class = Mlp_baseline_LRP if method == "LRP" else Mlp_baseline

    try:
        if do_parameter_search == 1:
            logger.info("Starting hyperparameter optimization...")
            optuna_folder = os.path.join(folder_name, 'optuna_search')
            os.makedirs(optuna_folder, exist_ok=True)
        
            results = optuna_search_FSDL(
                analyzer=analyzer,
                data_o=data_o,
                method=method,
                model_class=model_class, 
                model_params=model_params,
                training_params=training_params,
                folder_name=optuna_folder,
                logger=logger
            )
            
            logger.info(f"Hyperparameter optimization completed. Best params: {results['best_params']}")
        else:
            logger.info("Using predefined parameters without optimization")
            analyzer.analyze_features(
                method,
                model_class, 
                model_params,
                training_params
            )
    except Exception as e:
        logger.error(f"Error occurred while processing method {method}: {str(e)}")

        logger.error(traceback.format_exc())
        raise
        
        
        
def FSDL_with_evaluation(data, label, logger, model_params, training_params, name, 
                       features_selected, digits, folder_name, 
                       method, feature_prediction, n_iters=20, 
                       feature_step=500, n_folds=3):
    """
    Perform feature selection using FSDL method with evaluation of selected features.
    Args:
        data: input features
        label: target values
        logger: logging object
        model_params: model parameters dictionary
        training_params: training parameters dictionary
        name: dataset name
        features_selected: pre-selected features
        digits: divisor for feature dimensions
        folder_name: directory to save results
        method: feature attribution method (e.g., 'DeepLift', 'GradientShap', 'LRP', etc.)
        feature_prediction: maximum number of features to evaluate
        n_iters: number of iterations for evaluation
        feature_step: step size for feature evaluation (e.g., 1, 5, 10, 20, 50, 100, 200, 500)
        n_folds: number of folds for cross-validation
    """


    total_start_time = time.time()
    logger.info(f"FSDL_{method}_with_evaluation start at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    start_feature_step = 0


    results = np.zeros((n_steps, n_iters))  
    cv_val_losses = np.zeros((n_steps, n_iters))
    results_weights = []
    results_indices = []
    all_best_params = []
    all_report_metrics = [[None for _ in range(n_iters)] for _ in range(n_steps)]  
    seeds = np.random.choice(range(1000), n_iters, replace=False)

    
    logger.info(f"total features: {total_features}")
    logger.info(f"evaluation settings:")
    logger.info(f"- max features to evaluate: {max_features}")
    logger.info(f"- feature sequence: {feature_sequence}")
    logger.info(f"- number of evaluation points: {n_steps}")
    logger.info(f"- number of cross-validation folds: {n_folds}")
    
    dropout_probs = [0.1, 0.3, 0.7, 0.9]
    l2_lambdas = [0.0001, 0.001, 0.01, 0.1]

    model_class = Mlp_baseline_LRP if method == "LRP" else Mlp_baseline

    for iter in range(start_iter, n_iters):
        iter_start_time = time.time()
        logger.info(f'\start iteration {iter+1}/{n_iters}, seed {seeds[iter]}')
        
        try:
            current_model_params = model_params.copy()
            current_training_params = training_params.copy()
            current_training_params['device'] = device
            current_training_params['seed'] = seeds[iter]

            current_data_o = {
                'X': x_train,
                'Y': y_train,
                'selected_features': features_selected if features_selected is not None else []
            }
            analyzer = FeatureAttributionAnalyzer(
                data=current_data_o,
                name=name,
                digits=digits,
                device=current_training_params['device'],
                folder_name=iter_folder,
                FloatTensor=current_training_params['FloatTensor'],
                LongTensor=current_training_params['LongTensor'],
                task_type=current_model_params['task_type'],
                logger=logger
            )
            
            if current_training_params.get('do_parameter_search', False):

                logger.info(f"hyperparameter optimization using Optuna and train model...")
                
                results_dict = optuna_search_FSDL(
                    analyzer=analyzer,
                    data_o=current_data_o,
                    method=method,
                    model_class=model_class,
                    model_params=current_model_params,
                    training_params=current_training_params,
                    folder_name=os.path.join(iter_folder, 'optuna'),
                    logger=logger
                )
                
                all_best_params.append(results_dict['best_params'])
                logger.info(f"The best parameters found: {results_dict['best_params']}")
            else:

                logger.info(f"Using fixed parameters to train the model and perform feature attribution...")
                analyzer.analyze_features(
                    method,
                    model_class,
                    current_model_params,
                    current_training_params
                )
            
            attribution_dir = iter_folder
            feature_scores_file = None
            
            for file in os.listdir(attribution_dir):
                if file.endswith('.npy') and method in file:
                    feature_scores_file = os.path.join(attribution_dir, file)
                    break
            print(f"Feature scores file found: {feature_scores_file}")
            if feature_scores_file is None:
                logger.error(f"Feature scores file not found in {attribution_dir}, skipping iteration {iter+1}")
                continue
            


            df = np.load(feature_scores_file)
            feature_indices = df
            # feature_scores = df['scores'].values
            # results_weights.append(feature_scores)
            results_indices.append(feature_indices)
        
            feature_ranking_file = os.path.join(iter_folder, f'feature_ranking_iter_{iter}.npz')
            np.savez(feature_ranking_file, 
                    feature_indices=feature_indices, 
                    # weights=feature_scores,
                    iter=iter, 
                    seed=seeds[iter])
            logger.info(f"Feature ranking saved to: {feature_ranking_file}")




            feature_steps_range = range(n_steps)

            for step in feature_steps_range:
                num_features = int(feature_sequence[step])
                if num_features > len(feature_indices):
                    logger.warning(f"Requested number of features {num_features} exceeds available features {len(feature_indices)}, skipping")
                    continue

                logger.info(f"\nEvaluating top {num_features} features")

                try:

                    selected_features = feature_indices[:num_features]
                    x_train_selected = x_train[:, selected_features]
                    x_test_selected = x_test[:, selected_features]

                    logger.info(f"Evaluating feature set: {selected_features}")
                    final_score, best_dropout, best_l2, best_cv_val_loss, report_metrics = evaluate_feature_set(
                        x_train_selected, y_train,
                        x_test_selected, y_test,
                        num_features, model_params,
                        dropout_probs, l2_lambdas,
                        n_folds, device
                    )       
                    
                    results[step, iter] = final_score  
                    cv_val_losses[step, iter] = best_cv_val_loss 
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
                            best_l2=best_l2)
                    logger.info(f"The results for {num_features} features have been saved to: {feature_eval_file}")
                    
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
                        'per_step_report_metrics': np.array(all_report_metrics, dtype=object)
                    }
                    np.savez(checkpoint_file, **checkpoint_data)
                    logger.info(f"Checkpoint updated, completed evaluation for {num_features} features ({step+1}/{n_steps}), current iteration {iter+1}/{n_iters}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {num_features} features: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
            
            iter_results_file = os.path.join(iter_folder, f'full_results_iter_{iter}.npz')
            report_metrics_iter_column = np.array([all_report_metrics[r][iter] for r in range(n_steps)], dtype=object)
            np.savez(iter_results_file, 
                    iter=iter,
                    test_results=results[:, iter],
                    cv_val_losses=cv_val_losses[:, iter],
                    feature_indices=feature_indices,
                    # feature_weights=feature_scores,
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
            logger.info(f"Iteration {iter+1} completed, duration: {format_time(iter_time)} ({iter_time:.2f} seconds)")
        except Exception as e:
            logger.error(f"Error occurred during iteration {iter}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    

    total_time = time.time() - total_start_time
    logger.info(f"\nAll iterations completed, total time: {format_time(total_time)} ({total_time:.2f} seconds)")

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
        'per_step_report_metrics': np.array(all_report_metrics, dtype=object)  # ⭐ 新增
    }
    

    save_path = os.path.join(folder_name, f'{method}_{name}_results.npz')
    np.savez(save_path, **final_results)
    

    # timestamp_save_path = os.path.join(
    #     folder_name, 
    #     f'{method}_{name}_results_{final_results["timestamp"]}.npz'
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


        
def optuna_search_FSDL(analyzer, data_o, method, model_class, model_params, training_params, folder_name, logger):
    """
    Perform feature selection using FSDL method with Optuna hyperparameter optimization
    
    Args:
        analyzer: FeatureAttributionAnalyzer instance
        data_o: original data
        method: attribution method name
        model_class: model class to use
        model_params: model architecture parameters
        training_params: training parameters
        folder_name: directory for saving results
        logger: logging object
    """

    total_start_time = time.time()
    logger.info(f"Starting optuna_search_FSDL at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    eval_metric = training_params.get('eval_metric', None)
    if eval_metric is None:
        eval_metric = 'acc' if model_params['task_type'] in ('binary','multiclass') else 'r2'
    

    minimize_metric = eval_metric in ['mse', 'mae', 'loss']
    
    logger.info(f"Parameter search will use '{eval_metric}' as evaluation metric.")
    logger.info(f"Metric will be {'minimized' if minimize_metric else 'maximized'} during optimization.")
    

    os.makedirs(folder_name, exist_ok=True)
    # timing_log_path = os.path.join(folder_name, 'timing_log.txt')
    # with open(timing_log_path, 'w') as f:
    #     f.write(f"FSDL Optuna Search Time Log - Started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    #     f.write("=" * 80 + "\n")
    

    k_folds = training_params.get('n_splits', 3)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=training_params['seed'])
    
    search_start_time = time.time()
    
    weight_decay_min = training_params.get('weight_decay_min', 0.0)  
    weight_decay_max = training_params.get('weight_decay_max', 1e-1)  
    lr_min = training_params.get('lr_min', 1e-5)
    lr_max = training_params.get('lr_max', 1e-1)
    dropout_min = training_params.get('dropout_min', 0.1)
    dropout_max = training_params.get('dropout_max', 0.5)
    digits_choices = parse_list_arg(training_params.get('digits_list', '50,100,200'), dtype=int)
    batch_size_choices = parse_list_arg(training_params.get('batch_size_list', '16,32,64,128'), dtype=int)
    

    X = data_o['X']

    task_type = model_params.get('task_type', 'binary')
    
    if task_type == 'binary':
        y = data_o['Y'].long() 
    
    elif task_type == 'multiclass':
        y = data_o['Y'].long()
    
    else:  # regression
        y = data_o['Y'].float().view(-1, 1)
    

    trial_times = {}
    

    def objective(trial):
        trial_start_time = time.time()
        trial_id = trial.number
        weight_decay = trial.suggest_float('weight_decay', max(weight_decay_min, 1e-6), weight_decay_max, log=True)
        dropout = trial.suggest_float('dropout', dropout_min, dropout_max)
        lr = trial.suggest_float('lr', lr_min, lr_max, log=True)
        batch_size = trial.suggest_categorical('batch_size', batch_size_choices)
        digit = trial.suggest_categorical('digits', digits_choices)
        
        logger.info(f"Trial {trial_id}: weight_decay={weight_decay}, "
                  f"dropout={dropout}, lr={lr}, batch_size={batch_size}, digits={digit}")
        
        # with open(timing_log_path, 'a') as f:
        #     f.write(f"\nTrial {trial_id} - Started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        #     f.write(f"Parameters: weight_decay={weight_decay}, dropout={dropout}, lr={lr}, "
        #            f"batch_size={batch_size}, digits={digit}\n")

        fold_scores = []
        fold_times = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            fold_start_time = time.time()
            logger.info(f" Trial {trial_id}, Fold {fold+1}/{k_folds}")
            

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        

            current_model_params = copy.deepcopy(model_params)
            current_model_params['input_size'] = X.shape[1]
            current_model_params['hidden_size'] = int(X.shape[1] / digit)
            current_model_params['dropout_prob'] = dropout
            current_model_params['digits'] = digit
            

            current_training_params = copy.deepcopy(training_params)
            current_training_params['weight_decay'] = weight_decay
            current_training_params['lr'] = lr
            current_training_params['batch_size'] = batch_size
            

            fold_result = fsdl_single_fold_train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model_class=model_class,
                model_params=current_model_params,
                training_params=current_training_params,
                folder_path=None,
                logger=logger,
                eval_metric=eval_metric
            )
            

            fold_time = time.time() - fold_start_time
            fold_times.append(fold_time)
            

            # with open(timing_log_path, 'a') as f:
            #     f.write(f"  Fold {fold+1}: {format_time(fold_time)} ({fold_time:.2f} sec)\n")
                

            fold_score = fold_result['score']
            fold_scores.append(fold_score)
            
            logger.info(f" Trial {trial_id}, Fold {fold+1}/{k_folds} completed - Time: {format_time(fold_time)}")
            logger.info(f" Score: {fold_score:.4f}")
            

        avg_score = np.mean(fold_scores)
        

        trial_total_time = time.time() - trial_start_time
        trial_times[trial_id] = trial_total_time

        # with open(timing_log_path, 'a') as f:
        #     f.write(f"Trial {trial_id} - Total time: {format_time(trial_total_time)} ({trial_total_time:.2f} sec)\n")
        #     f.write(f"Average {eval_metric.upper()}: {avg_score:.4f}\n")
        #     f.write("-" * 40 + "\n")
            
        logger.info(f"Trial {trial_id} completed, time {format_time(trial_total_time)} - Average {eval_metric.upper()}: {avg_score:.4f}")
        

        return avg_score

    n_jobs = training_params.get('n_jobs', 1)

    try:

        
        db_file = os.path.join(folder_name, f"optuna_{method}.db")
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
        logger.info(f"Using RDB storage for parallel optimization, database: {db_url}, parallel workers: {n_jobs}")
    except ImportError:
        logger.warning("Could not import RDBStorage, using default storage")
        storage = None

    sampler = TPESampler(seed=training_params['seed'])
    study = optuna.create_study(
        sampler=sampler, 
        direction='minimize' if minimize_metric else 'maximize',
        storage=storage,
        study_name=f"FSDL_{method}",
        load_if_exists=True
    )
    

    n_trials = training_params.get('n_trials', 20)
    

    logger.info(f"Running {n_trials} Optuna optimization trials with {n_jobs} parallel workers")
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        logger.warning("User interrupted optimization.")
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        if n_jobs > 1:
            logger.warning("Parallel optimization failed. Retrying with n_jobs=1")
            study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_trial = study.best_trial
    best_params = best_trial.params

    search_time = time.time() - search_start_time
    
    logger.info(f"Optuna search completed, time {format_time(search_time)} ({search_time/60:.2f} minutes)")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best value: {best_trial.value}")
    

    # with open(timing_log_path, 'a') as f:
    #     f.write("\n" + "=" * 40 + "\n")
    #     f.write(f"Search Summary - Completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    #     f.write(f"Total search time: {format_time(search_time)} ({search_time:.2f} sec)\n")
    #     f.write(f"Number of trials: {n_trials}\n")
    #     f.write(f"Best parameters: {best_params}\n")
    #     f.write(f"Best value: {best_trial.value}\n\n")
        
    #     sorted_trials = sorted(trial_times.items(), key=lambda x: x[1], reverse=True)
    #     f.write("Trial time statistics:\n")
    #     f.write(f"  Fastest trial: {format_time(min(trial_times.values()))}\n")
    #     f.write(f"  Slowest trial: {format_time(max(trial_times.values()))}\n")
    #     f.write(f"  Average trial time: {format_time(sum(trial_times.values()) / len(trial_times))}\n\n")
        
    #     f.write("Top 5 longest trials:\n")
    #     for i, (trial_id, duration) in enumerate(sorted_trials[:5]):
    #         f.write(f"  #{i+1}: Trial {trial_id} - {format_time(duration)} ({duration:.2f} sec)\n")
    #     f.write("\n")
    

    logger.info("Training final model with best parameters...")
    final_model_start_time = time.time()
    

    final_model_params = copy.deepcopy(model_params)
    final_model_params['input_size'] = X.shape[1]
    final_model_params['hidden_size'] = int(X.shape[1] / best_params['digits'])
    final_model_params['dropout_prob'] = best_params['dropout']
    final_model_params['digits'] = best_params['digits']
    
    final_training_params = copy.deepcopy(training_params)
    final_training_params['weight_decay'] = best_params['weight_decay']
    final_training_params['lr'] = best_params['lr']
    final_training_params['batch_size'] = best_params['batch_size']
    

    analyzer.analyze_features(
        method,
        model_class, 
        final_model_params, 
        final_training_params
    )
    

    final_model_time = time.time() - final_model_start_time
    total_time = time.time() - total_start_time
    
    logger.info(f"Final model training completed, time {format_time(final_model_time)} ({final_model_time/60:.2f} minutes)")
    logger.info(f"Total processing time: {format_time(total_time)} ({total_time/60:.2f} minutes)")
    

    # with open(timing_log_path, 'a') as f:
    #     f.write(f"Final model training time: {format_time(final_model_time)} ({final_model_time:.2f} sec)\n")
    #     f.write(f"Total processing time: {format_time(total_time)} ({total_time:.2f} sec)\n")
    #     f.write("=" * 80 + "\n")
    

    with open(os.path.join(folder_name, 'best_params.txt'), 'w') as f:
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nSearch time: {format_time(search_time)} ({search_time:.2f} sec)\n")
        f.write(f"Final model training time: {format_time(final_model_time)} ({final_model_time:.2f} sec)\n")
        f.write(f"Total time: {format_time(total_time)} ({total_time:.2f} sec)\n")
    


    joblib.dump(study, os.path.join(folder_name, "optuna_study.pkl"))
    
    return {
        'best_params': best_params,
        'best_value': best_trial.value
    }

def fsdl_single_fold_train(X_train, y_train, X_val, y_val, model_class, model_params, training_params, folder_path, logger, eval_metric):
    """
    single fold training (supports binary / multiclass / regression), returns score for Optuna
    Args:
        X_train: training features
        y_train: training labels
        X_val: validation features
        y_val: validation labels
        model_class: model class to use     
        model_params: model architecture parameters
        training_params: training parameters
        folder_path: directory for saving results
        logger: logging object
        eval_metric: metric to evaluate model performance
    """
    device = training_params['device']
    task_type = training_params.get('task_type', model_params.get('task_type', 'regression'))

    train_dataset = MyDataset(X_train, y_train)
    val_dataset   = MyDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=training_params['batch_size'], shuffle=False)
    num_classes = model_params.get('num_classes')
    if task_type == 'binary' and (num_classes is None):
        num_classes = 2 

    model = model_class(
        feature_num=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        task_type=task_type,                        
        num_classes=(num_classes if task_type in ('multiclass','binary') else None),  # ✅
        dropout_rate=model_params['dropout_prob'],
        activation=model_params['activation']
    ).to(device)

    if task_type == 'binary':
        criterion = nn.CrossEntropyLoss()
    elif task_type == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_params['lr'],
        weight_decay=training_params.get('weight_decay', 0.0)
    )

    if not eval_metric:
        eval_metric = 'acc' if task_type in ('binary','multiclass') else 'r2'
    minimize = eval_metric in ('loss','mse','mae')
    best_val = float('inf') if minimize else -float('inf')
    min_delta = training_params.get('min_delta', 1e-4)
    patience = training_params.get('patience', 20)
    patience_counter = 0

    train_losses, val_losses, train_metrics, val_metrics = [], [], [], []

    for epoch in range(training_params['num_epochs']):
        model.train()
        tl = 0.0
        tr_logits_list, tr_targets_list = [], []

        for batch in train_loader:
            if len(batch['data']) <= 1: 
                continue
            X = batch['data'].to(device, dtype=torch.float32)
            y = batch['label']
            if task_type in ('binary', 'multiclass'):
                y = y.long().to(device)       
            else:
                y = y.float().view(-1,1).to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tl += loss.item()

            tr_logits_list.append(logits.detach().cpu())
            tr_targets_list.append(y.detach().cpu())

        tl /= max(len(train_loader),1)
        train_losses.append(tl)


        tr_logits = torch.cat(tr_logits_list,0).numpy()
        tr_tgts   = torch.cat(tr_targets_list,0).numpy()
        tr_metric = {}
        if task_type in ('binary', 'multiclass'):
            tr_pred = tr_logits.argmax(axis=1)               # ✅
            tr_metric['acc'] = (tr_pred == tr_tgts.astype(int)).mean()
        else:

            tr_metric['mse'] = mean_squared_error(tr_tgts, tr_logits)
            tr_metric['mae'] = mean_absolute_error(tr_tgts, tr_logits)
            tr_metric['r2']  = r2_score(tr_tgts, tr_logits)
        train_metrics.append(tr_metric)


        model.eval()
        vl = 0.0
        va_logits_list, va_targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                X = batch['data'].to(device,dtype=torch.float32)
                y = batch['label']
                if task_type in ('binary', 'multiclass'):
                    y = y.long().to(device)        # ✅ 分类用 long，shape=(N,)
                else:
                    y = y.float().view(-1,1).to(device)

                logits = model(X)
                loss = criterion(logits, y)
                vl += loss.item()
                va_logits_list.append(logits.detach().cpu())
                va_targets_list.append(y.detach().cpu())

        vl /= max(len(val_loader),1)
        val_losses.append(vl)

        va_logits = torch.cat(va_logits_list,0).numpy()
        va_tgts   = torch.cat(va_targets_list,0).numpy()
        va_metric = {}
        if task_type in ('binary', 'multiclass'):
            va_pred = va_logits.argmax(axis=1)               # ✅
            va_metric['acc'] = (va_pred == va_tgts.astype(int)).mean()
        else:

            va_metric['mse'] = mean_squared_error(va_tgts, va_logits)
            va_metric['mae'] = mean_absolute_error(va_tgts, va_logits)
            va_metric['r2']  = r2_score(va_tgts, va_logits)
        val_metrics.append(va_metric)


        current = vl if eval_metric == 'loss' else va_metric.get(eval_metric, vl)
        improved = (current < best_val - min_delta) if minimize else (current > best_val + min_delta)
        if improved:
            best_val = current
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    final_score = (val_losses[-1] if eval_metric == 'loss' else val_metrics[-1].get(eval_metric, val_losses[-1]))
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'score': final_score
    }






class Mlp_baseline(nn.Module):   
    """
    base MLP model with BatchNorm and Dropout
    """
    def __init__(self, feature_num, hidden_size, task_type='binary',
                 num_classes=None, dropout_rate=0.1, activation='relu'):
        super().__init__()
        self.feature_num = feature_num
        self.task_type = task_type
        self.num_classes = num_classes

        act_class = get_activation(activation)


        self.linear1 = nn.Linear(feature_num, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.act1 = copy.deepcopy(act_class)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.act2 = copy.deepcopy(act_class)
        self.dropout2 = nn.Dropout(dropout_rate)

        if task_type in ('multiclass', 'binary'):
            if task_type == 'binary' and (num_classes is None):
                num_classes = 2
            assert num_classes is not None and num_classes >= 2, \
                "num_classes need to be provided for classification tasks"
            self.output = nn.Linear(hidden_size, num_classes)
        else:
            self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        return self.output(x)

        

class Mlp_baseline_LRP(nn.Module):
    """
    base MLP model without BatchNorm, more LRP friendly
    """
    def __init__(self, feature_num, hidden_size,
                 task_type='binary', num_classes=None,
                 dropout_rate=0.1, activation='relu'):
        super().__init__()
        self.feature_num = feature_num
        self.task_type = task_type
        self.num_classes = num_classes

        act_cls = get_activation(activation)


        self.linear1 = nn.Linear(feature_num, hidden_size)
        self.act1 = copy.deepcopy(act_cls)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = copy.deepcopy(act_cls)
        self.dropout2 = nn.Dropout(dropout_rate)


        if task_type in ('multiclass', 'binary'):
            if task_type == 'binary' and (num_classes is None):
                num_classes = 2
            assert num_classes is not None and num_classes >= 2, \
                "classification tasks must provide num_classes >= 2"
            self.output = nn.Linear(hidden_size, num_classes)
        else:
            self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = x.view(x.size(0), -1) 
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        return self.output(x)


        
class MyDataset(Dataset):
    """
    define custom dataset class for PyTorch DataLoader
    """
    def __init__(self, data, label):

        self.data = data
        self.label = label

    def __getitem__(self, index):
        feature, target = self.data[index,:], self.label[index]
        sample = {'data': feature, 'label': target}
        return sample

    def __len__(self):

        return len(self.data)

def train_model(model, train_dl, valid_dl, training_params, model_params, save_path=""):
    """
    train deep learning model and evaluated performance
    """
    device = training_params['device']
    task_type = training_params.get('task_type', model_params.get('task_type', 'regression'))
    if task_type == 'binary':
        criterion = nn.CrossEntropyLoss()
    elif task_type == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_params['lr'],
        weight_decay=training_params.get('weight_decay', 0.0)
    )

    eval_metric = training_params.get('eval_metric')
    if not eval_metric:
        eval_metric = 'acc' if task_type in ('binary', 'multiclass') else 'r2'

    best_val = float('inf') if eval_metric in ('loss', 'mse', 'mae') else -float('inf')
    minimize = eval_metric in ('loss', 'mse', 'mae')
    patience_counter = 0
    min_delta = training_params.get('min_delta', 1e-4)

    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []

    for epoch in range(1, training_params['num_epochs'] + 1):
        model.train()
        epoch_train_loss = 0.0
        # 收集指标
        tr_logits_list, tr_targets_list = [], []

        for batch in train_dl:
            if len(batch['data']) <= 1: 
                continue

            X = batch['data'].to(device, dtype=torch.float32)
            y = batch['label']
            if task_type in ('binary', 'multiclass'):
                y = y.long().to(device)        # ✅ 分类用 long，shape=(N,)
            else:
                y = y.float().view(-1,1).to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            tr_logits_list.append(logits.detach().cpu())
            tr_targets_list.append(y.detach().cpu())

        epoch_train_loss /= max(len(train_dl), 1)
        train_losses.append(epoch_train_loss)


        tr_logits = torch.cat(tr_logits_list, dim=0).numpy()
        tr_tgts  = torch.cat(tr_targets_list, dim=0).numpy()
        tr_metric = {}

        if task_type in ('binary', 'multiclass'):
            tr_pred = tr_logits.argmax(axis=1)               # ✅
            tr_metric['acc'] = (tr_pred == tr_tgts.astype(int)).mean()
        else:
            tr_metric['mse'] = mean_squared_error(tr_tgts, tr_logits)
            tr_metric['mae'] = mean_absolute_error(tr_tgts, tr_logits)
            tr_metric['r2']  = r2_score(tr_tgts, tr_logits)

        train_metrics.append(tr_metric)

        model.eval()
        val_loss = 0.0
        va_logits_list, va_targets_list = [], []
        with torch.no_grad():
            for batch in valid_dl:
                X = batch['data'].to(device, dtype=torch.float32)
                y = batch['label']
                if task_type in ('binary', 'multiclass'):
                    y = y.long().to(device)       
                else:
                    y = y.float().view(-1,1).to(device)

                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item()
                va_logits_list.append(logits.detach().cpu())
                va_targets_list.append(y.detach().cpu())

        val_loss /= max(len(valid_dl), 1)
        val_losses.append(val_loss)

        va_logits = torch.cat(va_logits_list, dim=0).numpy()
        va_tgts  = torch.cat(va_targets_list, dim=0).numpy()
        va_metric = {}

        if task_type in ('binary', 'multiclass'):
            va_pred = va_logits.argmax(axis=1)               # ✅
            va_metric['acc'] = (va_pred == va_tgts.astype(int)).mean()
        else:

            va_metric['mse'] = mean_squared_error(va_tgts, va_logits)
            va_metric['mae'] = mean_absolute_error(va_tgts, va_logits)
            va_metric['r2']  = r2_score(va_tgts, va_logits)

        val_metrics.append(va_metric)


        if eval_metric == 'loss':
            current = val_loss
        else:
            current = va_metric.get(eval_metric, val_loss)  

        improved = (current < best_val - min_delta) if minimize else (current > best_val + min_delta)
        if improved:
            best_val = current
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                history = {
                    'train_losses': train_losses, 'val_losses': val_losses,
                    'train_metrics': train_metrics, 'val_metrics': val_metrics
                }
                torch.save(history, save_path + ".history")
        else:
            patience_counter += 1


        if patience_counter >= training_params.get('patience', 20):
            print(f"Early stopping at epoch {epoch}...")
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }


class FeatureAttributionAnalyzer:
    """
    Feature Attribution Analyzer for training and evaluating models with feature attribution methods
    """
    def __init__(self, data, name, digits=100, device=None, 
                 folder_name=None, FloatTensor=None, LongTensor=None, task_type='regression', logger=None):
        self.digits = digits
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.folder_name = folder_name
        self.FloatTensor = FloatTensor
        self.LongTensor = LongTensor
        self.task_type = task_type
        self.name = name 
        self.baseline_values = np.array([0])
        
        self.logger = logger or self._setup_logger()
        self.logger.info(f"Initialized baseline values: {self.baseline_values}")
        

        self.setup_data(data)

    def _setup_logger(self):
        logger = logging.getLogger(f"FeatureAttribution")
        logger.setLevel(logging.INFO)

        if not logger.handlers:

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)


            if self.folder_name:
                log_file = os.path.join(self.folder_name, f'feature_attribution.log')
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                console_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            logger.addHandler(console_handler)

        return logger
        
    def setup_data(self, data):

        self.X = data['X'].to(self.device)
        self.feature_num = self.X.shape[1]
        self.features_name = np.array([f"gene_{i+1}" for i in range(self.feature_num)])
        y_np = data['Y'].cpu().numpy()
    
        if self.task_type == 'binary':
            y_codes = pd.Categorical(y_np).codes  # -> 0/1
            self.Y = torch.from_numpy(y_codes.astype('int32'))
        elif self.task_type == 'multiclass':
            y_codes = pd.Categorical(y_np).codes  # -> 0..C-1
            self.Y = torch.from_numpy(y_codes.astype('int32'))
        else:
            self.Y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    
        self.selected_features = data.get('selected_features', data.get('global_indices', None))
        self._split_data()
            
    def _split_data(self):

        n = self.X.size(0)
        idx = np.arange(n)
    

        stratify_y = None
        if self.task_type in ('binary', 'multiclass'):
            stratify_y = self.Y.cpu().numpy()
    
        tr_idx, te_idx = train_test_split(
            idx, test_size=0.2, stratify=stratify_y
        )
    

        self.train_data = self.X[tr_idx].to(self.device)
        self.test_data  = self.X[te_idx].to(self.device)

        self.train_label = self.Y[tr_idx]
        self.test_label  = self.Y[te_idx]

    
    def __len__(self):
        return int(self.X.size(0))

    def prepare_dataloaders(self, train_batch_size=32):
        generator = torch.Generator()
        
        train_dataset = MyDataset(self.train_data, self.train_label)
        test_dataset = MyDataset(self.test_data, self.test_label)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
            drop_last=False
        )
        
        self.valid_loader = DataLoader(
            test_dataset,
            batch_size=train_batch_size,
            num_workers=0,
            generator=generator,
            shuffle=False,
            drop_last=False
        )
        
    def get_attribution_method(self, method_name):
        attribution_methods = {
            "DeepLIFT": DeepLift,
            "GradientShap": GradientShap,
            "Occlusion": Occlusion,
            "Lime": Lime,
            "FeatureAblation": FeatureAblation,
            "LRP": LRP
        }
        return attribution_methods.get(method_name)
        
    def analyze_features(self, method_name, model_class, model_params, training_params):

        start_time = time.time()
        self.logger.info(f"Starting feature analysis with {method_name}...")
        model_params['input_size'] = self.feature_num
        task_type = self.task_type
        if task_type == 'multiclass':
            if model_params.get('num_classes') is None:
                model_params['num_classes'] = int(torch.unique(self.Y).numel())
        elif task_type == 'binary':
            model_params['num_classes'] = 2    
        num_classes = model_params.get('num_classes') if task_type in ('multiclass', 'binary') else None
        model = model_class(
            feature_num=model_params['input_size'],
            hidden_size=model_params['hidden_size'],
            task_type=task_type,
            num_classes=num_classes,
            dropout_rate=model_params['dropout_prob'],
            activation=model_params['activation']
        ).to(self.device)

        self.model = model
        self.logger.info(f"Model architecture: {self.model}")
        self.model.to(self.device)
        self.prepare_dataloaders(train_batch_size=training_params['batch_size'])

        self.logger.info("Training model...")
        model_save_path = os.path.join(self.folder_name, 'model_best.pth.tar')

        training_history = train_model(
            self.model,
            self.train_loader,
            self.valid_loader,
            training_params,
            model_params,
        )


        training_time = time.time() - start_time
        self.logger.info(f"Model training completed in {format_time(training_time)}")


        if os.path.exists(model_save_path):
            self.logger.info(f"Loading best model from {model_save_path}")
            self.model.load_state_dict(torch.load(model_save_path))

        attribution_class = self.get_attribution_method(method_name)
        if attribution_class is None:
            self.logger.error(f"Unknown attribution method: {method_name}")
            return


        self.logger.info(f"Initializing {method_name} attributor...")
        deconv = attribution_class(self.model)


        # with open(os.path.join(self.folder_name, 'training_params.txt'), 'w') as f:
        #     f.write(f"Method: {method_name}\n")
        #     f.write(f"Model parameters:\n")
        #     for k, v in model_params.items():
        #         f.write(f"  {k}: {v}\n")
        #     f.write(f"\nTraining parameters:\n")
        #     for k, v in training_params.items():
        #         if k not in ['FloatTensor', 'LongTensor', 'device']:
        #             f.write(f"  {k}: {v}\n")

            # f.write("\nTraining summary:\n")
            # f.write(f"  Final train loss: {training_history['train_losses'][-1]:.4f}\n")
            # f.write(f"  Final validation loss: {training_history['val_losses'][-1]:.4f}\n")
            # if 'acc' in training_history['val_metrics'][-1]:
            #     f.write(f"  Final validation accuracy: {training_history['val_metrics'][-1]['acc']:.4f}\n")
            # if 'r2' in training_history['val_metrics'][-1]:
            #     f.write(f"  Final validation R²: {training_history['val_metrics'][-1]['r2']:.4f}\n")
            # f.write(f"  Training time: {format_time(training_time)}\n")




        self.logger.info(f"Running feature attribution with {method_name}...")
        attribution_start_time = time.time()

        if self.task_type in ("binary","multiclass"):
            self._analyze_per_class(deconv, method_name)
        else:
            self._analyze_continuous(deconv, method_name)


        attribution_time = time.time() - attribution_start_time
        total_time = time.time() - start_time

        self.logger.info(f"Feature attribution completed in {format_time(attribution_time)}")
        self.logger.info(f"Total processing time: {format_time(total_time)}")

        # with open(os.path.join(self.folder_name, 'timing_info.txt'), 'w') as f:
        #     f.write(f"{method_name} Feature Attribution - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        #     f.write("=" * 80 + "\n")
        #     f.write(f"Model training time: {format_time(training_time)} ({training_time:.2f} sec)\n")
        #     f.write(f"Feature attribution time: {format_time(attribution_time)} ({attribution_time:.2f} sec)\n")
        #     f.write(f"Total processing time: {format_time(total_time)} ({total_time:.2f} sec)\n")

    def _analyze_per_class(self, deconv, method_name):


        self.logger.info(f"Analyzing feature attribution for {self.task_type}")

        if self.task_type == 'binary':
            n_classes = 2
        else:  # multiclass
            n_classes = int(torch.unique(self.train_label).numel())
        for i in range(n_classes):
            self.logger.info(f"Processing class {i}")
            class_indices = torch.where(self.train_label == i)[0]

            if len(class_indices) == 0:
                self.logger.warning(f"No samples found for class {i}, skipping")
                continue

            class_data = self.train_data[class_indices].reshape(-1, self.feature_num)

            attribution = self._calculate_attribution(deconv, class_data, i, method_name)


            self._process_and_save_results(attribution, i, method_name)

    def _analyze_continuous(self, deconv, method_name):

        self.logger.info("Analyzing feature attribution for regression")

        # 计算特征归因
        attribution = self._calculate_attribution(deconv, self.train_data, None, method_name)

        # 处理结果并保存
        self._process_and_save_results(attribution, 0, method_name)


    def _calculate_attribution(self, deconv, class_data, class_idx, method_name):

        try:

            all_attributions = torch.zeros(len(self.baseline_values), self.feature_num).to(self.device)
            self.logger.info(f"Calculating attribution with {method_name} for {len(self.baseline_values)} baseline values")

            for baseline_idx, baseline_value in enumerate(self.baseline_values):
                self.logger.info(f"Processing baseline value: {baseline_value}")

                if method_name == "FeatureAblation":
                    attribution = self.calculate_feature_ablation(deconv, class_data, class_idx, baseline_value, self.task_type)
                elif method_name == "Occlusion":
                    attribution = self.calculate_occlusion(deconv, class_data, class_idx, baseline_value, self.task_type)
                elif method_name == "GradientShap":
                    attribution = self.calculate_gradient_shap(deconv, class_data, class_idx, baseline_value, self.task_type)
                elif method_name == "Lime":
                    attribution = self.calculate_lime_attribution(deconv, class_data, class_idx, baseline_value, self.task_type)
                else: 
                    attribution = self.calculate_default(deconv, class_data, class_idx, baseline_value, self.task_type)

                all_attributions[baseline_idx] = attribution.squeeze()

            return all_attributions

        except Exception as e:
            self.logger.error(f"Error in _calculate_attribution: {str(e)}")
            raise


    def _process_and_save_results(self, attribution, class_idx, method_name):

        self.logger.info(f"Processing attribution results (shape: {attribution.shape})")


        results_dir =self.folder_name# os.path.join(self.folder_name, 'attribution_results')
        os.makedirs(results_dir, exist_ok=True)

        all_dfs = []


        for baseline_idx in range(attribution.shape[0]):

            current_attribution = attribution[baseline_idx]
            v, ind = current_attribution.sort(dim=0, descending=True)

            df = pd.DataFrame({
                "index": ind.cpu().numpy().astype("int64"),
                "scores": v.detach().cpu().numpy(),
                "rank": np.arange(len(self.features_name))
            })
    
            if self.task_type in ('binary','multiclass'):
                if class_idx is None:
                    raise ValueError("classification must provide class_idx")
                anchor_exp = self.train_data[self.train_label == class_idx]
                other_exp  = self.train_data[self.train_label != class_idx]
                logFC = torch.mean(anchor_exp.to(self.device, dtype=torch.float32), dim=0) - torch.mean(other_exp.to(self.device, dtype=torch.float32), dim=0)
            else:
                logFC = torch.mean(self.train_data, dim=0)


            df = df.sort_values(by=['index'], ascending=True)
            df.index = self.features_name
            df["logFC"] = logFC.cpu().numpy()
            df["baseline"] = baseline_idx
            df = df.sort_values(by=['scores'], ascending=False)

            all_dfs.append(df)

            csv_path = os.path.join(results_dir, f"{method_name}_{self.name}_{class_idx}_.csv")
            # df.to_csv(csv_path)

            npz_path = os.path.join(results_dir, f"{method_name}_{self.name}_{class_idx}_idx.npy")
            np.save(npz_path,df.values[:,0])

            self.logger.info(f"Saved results for baseline {baseline_idx} to {npz_path}")

            # self._evaluate_results(df, class_idx)

        if len(all_dfs) > 1:
            self.logger.info("Calculating average results across all baselines")


            combined_df = pd.concat(all_dfs)


            avg_df = combined_df.groupby(combined_df.index)['scores'].mean().reset_index()


            final_df = pd.DataFrame(index=avg_df['index'])
            final_df['scores'] = avg_df['scores']
            final_df['logFC'] = all_dfs[0]['logFC'] 


            final_df = final_df.sort_values(by=['scores'], ascending=False)

            indices = np.array([int(np.where(self.features_name == idx)[0][0]) for idx in final_df.index])
            final_df["index"] = indices
            final_df["rank"] = np.arange(len(self.features_name))

            avg_csv_path = os.path.join(results_dir, f"{method_name}_{self.digits}_{class_idx}_avg_baseline.csv")
            final_df.to_csv(avg_csv_path)

            self.logger.info(f"Saved average results to {avg_csv_path}")

            # self._evaluate_results(final_df, class_idx)


    def _evaluate_results(self, df, class_idx):

        task_type = getattr(self, "task_type", "regression")
        sortaa = df['index'].to_numpy()

        results_dir = os.path.join(self.folder_name, 'evaluation_results')
        os.makedirs(results_dir, exist_ok=True)
        tag = f"class_{class_idx}" if task_type in ("binary", "multiclass") else "reg"
        result_file = os.path.join(results_dir, f"evaluation_{tag}.txt")
    
        with open(result_file, 'w') as f:
            f.write(f"Evaluation Results for {tag}\n")
            f.write("=" * 50 + "\n\n")

            if 'logFC' in df.columns:
                try:
                    corr = np.corrcoef(df['scores'], df['logFC'])[0, 1]
                    self.logger.info(f"Correlation between attribution scores and logFC: {corr:.4f}")
                    f.write(f"Correlation between attribution scores and logFC: {corr:.4f}\n")
                except Exception as e:
                    self.logger.warning(f"Correlation computation failed: {e}")
                    f.write("Correlation computation failed.\n")
            else:
                self.logger.info("`logFC` not found in df; skip correlation.")
                f.write("`logFC` not found; skip correlation.\n")
  
          
    def calculate_gradient_shap(self, deconv, class_data, class_idx, baseline_value, task_type):

        return self.calculate_batched_attribution(
            deconv, class_data, class_idx, baseline_value, task_type
        )
    def calculate_feature_ablation(self, deconv, class_data, class_idx, baseline_value, task_type):
        is_classification = (self.task_type in ('binary','multiclass'))
        target = int(class_idx) if (is_classification and class_idx is not None) else None
        inputs = class_data.to(self.device, dtype=torch.float32)
        baselines = torch.ones_like(inputs, dtype=torch.float32, device=self.device) * float(baseline_value)
        attribution = deconv.attribute(
            inputs,
            baselines=baselines,
            target=target
        )
        return attribution.mean(dim=0, keepdim=True)

    def calculate_occlusion(self, deconv, class_data, class_idx, baseline_value, task_type):
        """处理Occlusion方法"""
        is_classification = (self.task_type in ('binary','multiclass'))
        target = int(class_idx) if (is_classification and class_idx is not None) else None
        inputs = class_data.to(self.device, dtype=torch.float32)
        baselines = torch.ones_like(inputs, dtype=torch.float32, device=self.device) * float(baseline_value)
        attribution = deconv.attribute(
            inputs,
            baselines=baselines,
            target=target,
            sliding_window_shapes=(1,)
        )
        return attribution.mean(dim=0, keepdim=True)

    def calculate_default(self, deconv, class_data, class_idx, baseline_value, task_type):
        method_name = type(deconv).__name__.lower()
    
        is_classification = (self.task_type in ('binary','multiclass'))
        target = int(class_idx) if (is_classification and class_idx is not None) else None
    
        if 'lrp' in method_name:
            batch_size = 100
            n_samples = class_data.size(0)

            attribution = torch.zeros(1, self.feature_num, dtype=torch.float32, device=self.device)
            for j in tqdm(range(0, n_samples, batch_size), desc='LRP'):

                input_data = class_data[j:j+min(batch_size, n_samples-j), :].to(self.device, dtype=torch.float32)
                input_data.requires_grad_(True)

                batch_attr = deconv.attribute(input_data, target=target)
                attribution += batch_attr.mean(dim=0, keepdim=True)
            return attribution
    
        elif 'gradients' in method_name and 'shap' in method_name:

            batch_size = 100
            n_samples = class_data.size(0)

            attribution = torch.zeros(1, self.feature_num, dtype=torch.float32, device=self.device)
            for j in tqdm(range(0, n_samples, batch_size), desc='GradientShap'):
                input_data = class_data[j:j+min(batch_size, n_samples-j), :].to(self.device, dtype=torch.float32)
                baselines = torch.ones_like(input_data, dtype=torch.float32, device=self.device) * float(baseline_value)
                input_data.requires_grad_(True)

                batch_attr = deconv.attribute(
                    input_data, baselines=baselines, target=target,
                    n_samples=50, stdevs=0.01  
                )
                attribution += batch_attr.mean(dim=0, keepdim=True)
            return attribution
    
        else:

            return self.calculate_batched_attribution(
                deconv, class_data, class_idx, baseline_value, task_type
            )

    def calculate_batched_attribution(self, deconv, class_data, class_idx, baseline_value, task_type):
        batch_size = 100
        n_samples = class_data.size(0)  

        attribution = torch.zeros(1, self.feature_num, dtype=torch.float32, device=self.device)

        is_classification = (self.task_type in ('binary','multiclass'))
        target = int(class_idx) if (is_classification and class_idx is not None) else None
    
        for j in tqdm(range(0, n_samples, batch_size), desc=f'Baseline {baseline_value}'):
            current_batch_size = min(batch_size, n_samples - j)
            input_data = class_data[j:j+current_batch_size, :].to(self.device, dtype=torch.float32)
            input_data.requires_grad_(True)  
            baselines = torch.ones_like(input_data, dtype=torch.float32, device=self.device) * float(baseline_value)
            batch_attr = deconv.attribute(
                input_data,
                baselines=baselines,
                target=target
            )
    
            attribution += batch_attr.mean(dim=0, keepdim=True)
    
        return attribution
        
    def calculate_lime_attribution(self, deconv, class_data, class_idx, baseline_value, task_type, n_samples=500, batch_size=100):

        n_samples_total = class_data.size(0) 
        attribution = torch.zeros(1, self.feature_num, dtype=torch.float32, device=self.device)

        for j in tqdm(range(0, n_samples_total, batch_size), 
                     desc=f'Lime baseline {baseline_value}'):
            current_batch_size = min(batch_size, n_samples_total - j)
            input_data = class_data[j:j+current_batch_size, :].to(self.device, dtype=torch.float32)

            is_classification = (self.task_type in ('binary','multiclass'))
        

            batch_attr = deconv.attribute(
                input_data.to(self.device),
                target = int(class_idx) if (is_classification and class_idx is not None) else None,
                n_samples=n_samples
            )


            attribution += batch_attr.sum(dim=0, keepdim=True)


        return attribution



