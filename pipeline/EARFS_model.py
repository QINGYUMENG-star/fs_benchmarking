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
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


from evaluation_utils import evaluate_feature_set
from utils import get_activation, format_time, parse_list_arg  

def EARFS_fs(data, label, model_params, training_params, name,  folder_name, logger, causal_variants):
    """
    Main function for EARFS feature selection with optional parameter optimization
    data: input X data
    label: target values
    model_params: model parameter dictionary
    training_params: training parameter dictionary
    folder_name: result save directory
    logger: logger
    causal_variants: pre-selected feature list
    """

    do_parameter_search = training_params.get('do_parameter_search', 0)

    x_train = data['X_train']; y_train = label['y_train']
    if do_parameter_search == 1:
        results = optuna_search_EARFS(
            x_train, y_train, model_params, training_params, folder_name,
            causal_variants, logger
        )
    else:
        results = EARFS_model_train(
            x_train, y_train, model_params, training_params, folder_name,
            logger, causal_variants
        )

    
    np.save(os.path.join(folder_name, f'EARFS_{name}_weights.npy'),
            results['select_rate'])

    np.save(os.path.join(folder_name, f'EARFS_{name}_idx.npy'),
            results['feature_indices'])

    # torch.save(results['model'].state_dict(),
    #            os.path.join(folder_name, f'EARFS_final_model.pth'))
    # np.save(os.path.join(folder_name, f'EARFS_train_loss_history.npy'), 
    #         np.array(results['train_losses']))
    # np.save(os.path.join(folder_name, f'EARFS_val_loss_history.npy'), 
    #         np.array(results['val_losses']))
    # np.save(os.path.join(folder_name, f'EARFS_train_metrics_history.npy'), 
    #         results['train_metrics'])
    # np.save(os.path.join(folder_name, f'EARFS_val_metrics_history.npy'), 
    #         results['val_metrics'])
    
    print(f"\nFiles saved to folder: {folder_name}")
    
    return results

    
def EARFS_fs_with_evaluation(data, label, model_params, training_params, name, 
                           causal_variants, digits, folder_name, 
                           logger, feature_prediction, n_iters=20,
                           feature_step=500, n_folds=3):
    """
    Perform EARFS feature selection with evaluation across varying feature counts.
    
    Args:
        data: input X data
        label: target values
        model_params: model parameter dictionary
        training_params: training parameter dictionary
        name: dataset name
        causal_variants: pre-selected feature list
        digits: feature dimension divisor
        folder_name: result save directory
        logger: logger
        feature_prediction: upper limit on the number of features to evaluate
        n_iters: number of iterations, default 20
        feature_step: feature evaluation step size, default 500
        n_folds: number of cross-validation folds, default 3
        
    """

    total_start_time = time.time()
    logger.info(f" EARFS_fs_with_evaluation start {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    x_train = data['X_train']; x_test = data['X_test']
    y_train = label['y_train']; y_test = label['y_test']

    

    device = x_train.device if isinstance(x_train, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")

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
    logger.info(f"using device: {device}")
    
    for iter in range(start_iter, n_iters):
        iter_start_time = time.time()
        logger.info(f'\nstart iteration {iter+1}/{n_iters}, seed {seeds[iter]}')
        
        try:   
            current_model_params = model_params.copy()
            current_training_params = training_params.copy()
            current_model_params['input_size'] = x_train.shape[1]
            current_model_params['hidden_size'] = int(x_train.shape[1] / digits)            
            
            

            current_training_params['device'] = device
            current_training_params['seed'] = seeds[iter]
            

            if current_training_params.get('do_parameter_search', False):

                logger.info(f"using Optuna for hyperparameter optimization and training...")
                earfs_results = optuna_search_EARFS(
                    x_train, y_train, 
                    current_model_params, 
                    current_training_params,
                    os.path.join(eval_folder, f'optuna_iter_{iter}'),
                    f"{iter}", 
                    causal_variants, 
                    logger
                )


                if isinstance(earfs_results, dict) and 'model' not in earfs_results:

                    best_params = earfs_results
                    logger.info(f"get best params: {best_params}")
                    all_best_params.append(best_params)


                    if 'dropout' in best_params:
                        current_model_params['dropout_prob'] = best_params['dropout']
                    if 'digits' in best_params:
                        current_model_params['hidden_size'] = int(x_train.shape[1] / best_params['digits'])
                    if 'lambda_fs' in best_params:
                        current_training_params['lambda_fs'] = best_params['lambda_fs']
                    if 'weight_decay' in best_params:
                        current_training_params['weight_decay'] = best_params['weight_decay']
                    if 'lr' in best_params:
                        current_training_params['lr'] = best_params['lr']
                    if 'batch_size' in best_params:
                        current_training_params['batch_size'] = best_params['batch_size']


                    earfs_results = EARFS_model_train(
                        x_train, y_train,
                        current_model_params, 
                        current_training_params,
                        os.path.join(eval_folder, f'model_iter_{iter}'),
                        f"{iter}", 
                        logger, 
                        causal_variants
                    )
            else:
                logger.info(f"using fixed parameters to train model...")
                earfs_results = EARFS_model_train(
                    x_train, y_train,
                    current_model_params, 
                    current_training_params,
                    os.path.join(eval_folder, f'model_iter_{iter}'),
                    logger, 
                    causal_variants
                )

            feature_ranking = earfs_results['feature_indices']


            results_weights.append(earfs_results['model'].feature_selection_MLP.get_selection_rate().detach().cpu().numpy())
            results_indices.append(feature_ranking)


            feature_ranking_file = os.path.join(iter_folder, f'feature_ranking_iter_{iter}.npz')
            np.savez(feature_ranking_file, 
                    feature_indices=feature_ranking, 
                    weights=results_weights[-1],
                    iter=iter, 
                    seed=seeds[iter])
            logger.info(f"feature ranking saved to: {feature_ranking_file}")

            feature_steps_range = range(n_steps)

            for step in feature_steps_range:
                num_features = int(feature_sequence[step])
                if num_features > len(feature_ranking):
                    logger.warning(f"requested number of features {num_features} exceeds available features {len(feature_ranking)}, skipping")
                    continue

                logger.info(f"\nevaluating top {num_features} features")

                try:

                    selected_features = feature_ranking[:num_features]
                    logger.info(f"selected_features前几个: {selected_features}...")

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
                            best_l2=best_l2,
                            report_metrics=report_metrics)
                    logger.info(f"The evaluation results for the number of features {num_features} have been saved to: {feature_eval_file}")
                    
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
                    logger.info(f"Checkpoint updated, completed evaluation for the number of features {num_features} ({step+1}/{n_steps}), current iteration {iter+1}/{n_iters}")
                    
                except Exception as e:
                    logger.error(f"Error occurred while evaluating the number of features {num_features}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())

                    continue
            
            iter_results_file = os.path.join(iter_folder, f'full_results_iter_{iter}.npz')
            report_metrics_iter_column = np.array([all_report_metrics[r][iter] for r in range(n_steps)], dtype=object)

            np.savez(iter_results_file, 
                    iter=iter,
                    test_results=results[:, iter],
                    cv_val_losses=cv_val_losses[:, iter],
                    feature_indices=feature_ranking,
                    feature_weights=results_weights[-1] if len(results_weights) > 0 else None,
                    feature_sequence=feature_sequence,
                    seed=seeds[iter],
                    timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                    report_metrics_column=report_metrics_iter_column)  
            logger.info(f"Iteration {iter+1} full results saved to: {iter_results_file}")
            

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
            logger.error(f"Error occurred while iterating {iter}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            continue
    

    total_time = time.time() - total_start_time
    logger.info(f"\n All iteration finished, total time taken: {format_time(total_time)} ({total_time:.2f} seconds)")
    

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
        'per_step_report_metrics': np.array(all_report_metrics, dtype=object)  
    }

    save_path = os.path.join(folder_name, f'EARFS_{name}_results.npz')
    np.savez(save_path, **final_results)
    

    # timestamp_save_path = os.path.join(
    #     folder_name, 
    #     f'EARFS_{name}_results_{final_results["timestamp"]}.npz'
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
    # logger.info(f"Timestamped version saved to: {timestamp_save_path}")

    return final_results




def optuna_search_EARFS(data, label, model_params, training_params, folder_name, causal_variants, logger):
    """
    Perform feature selection using EARFS method with Optuna hyperparameter optimization
    
    Args:
        data: input features
        label: target values
        model_params: model architecture parameters (including digits)
        training_params: training parameters
        folder_name: directory for saving results
        causal_variants: pre-selected features for evaluation
        logger: logging object
    Returns:
        dict with training results and the trained model or best hyperparameters if only searching  
    """

    

    total_start_time = time.time()
    logger.info(f"optuna_search_EARFS start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    

    do_parameter_search = training_params.get('do_parameter_search', 0)

    eval_metric = training_params.get('eval_metric', None)
    if eval_metric is None:
        eval_metric = 'acc' if model_params['task_type'] in ('binary','multiclass') else 'r2'
    

    minimize_metric = eval_metric in ['mse', 'mae', 'loss']

    logger.info(f"Parameter search will use '{eval_metric}' as the evaluation metric.")
    logger.info(f"During optimization, the metric will be {'minimized' if minimize_metric else 'maximized'}.")

    if do_parameter_search == 1:
        logger.info("Start Optuna hyperparameter optimization for EARFS...")
        optuna_folder = os.path.join(folder_name, 'optuna_search')
        os.makedirs(optuna_folder, exist_ok=True)

        # timing_log_path = os.path.join(optuna_folder, 'timing_log.txt')
        # with open(timing_log_path, 'w') as f:
        #     f.write(f"EARFS Optuna search timing log - Start at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        #     f.write("=" * 80 + "\n")
        k_folds = training_params.get('n_splits', 3)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=training_params['seed'])

        search_start_time = time.time()
        lambda_fs_min = training_params.get('lambda_fs_min', 1e-5)
        lambda_fs_max = training_params.get('lambda_fs_max', 1e-1)
        weight_decay_min = training_params.get('weight_decay_min', 0.0)
        weight_decay_max = training_params.get('weight_decay_max', 1e-1)
        lr_min = training_params.get('lr_min', 1e-5)
        lr_max = training_params.get('lr_max', 1e-1)
        dropout_min = training_params.get('dropout_min', 0.1)
        dropout_max = training_params.get('dropout_max', 0.5)
        digits_choices = parse_list_arg(training_params.get('digits_list', '50,100,200'), dtype=int)
        batch_size_choices = parse_list_arg(training_params.get('batch_size_list', '16,32,64,128'), dtype=int)
        

        trial_times = {}
        def objective(trial):
            trial_start_time = time.time()
            trial_id = trial.number
            lambda_fs = trial.suggest_float('lambda_fs', lambda_fs_min, lambda_fs_max, log=True)
            weight_decay = trial.suggest_float('weight_decay', max(weight_decay_min, 1e-6), weight_decay_max, log=True)
            dropout = trial.suggest_float('dropout', dropout_min, dropout_max)
            lr = trial.suggest_float('lr', lr_min, lr_max, log=True)
            batch_size = trial.suggest_categorical('batch_size', batch_size_choices)
            digit = trial.suggest_categorical('digits', digits_choices)

            logger.info(f"Trial {trial_id}: lambda_fs={lambda_fs}, weight_decay={weight_decay}, "
                       f"dropout={dropout}, lr={lr}, batch_size={batch_size}, digits={digit}")

            # with open(timing_log_path, 'a') as f:
            #     f.write(f"\nTrial {trial_id} - Start at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            #     f.write(f"Parameters: lambda_fs={lambda_fs}, weight_decay={weight_decay}, dropout={dropout}, lr={lr}, "
            #             f"batch_size={batch_size}, digits={digit}\n")


            fold_scores = []
            fold_times = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
                fold_start_time = time.time()
                logger.info(f"Trial {trial_id}, Fold {fold+1}/{k_folds}")
                ti = torch.as_tensor(train_idx, dtype=torch.long, device=data.device if isinstance(data, torch.Tensor) else 'cpu')
                vi = torch.as_tensor(val_idx, dtype=torch.long, device=data.device if isinstance(data, torch.Tensor) else 'cpu')
                if isinstance(data, torch.Tensor):
                    train_data  = data[ti]
                    train_labels= label[ti]
                    val_data    = data[vi]
                    val_labels  = label[vi]
                else:
                    train_data  = data[train_idx]
                    train_labels= label[train_idx]
                    val_data    = data[val_idx]
                    val_labels  = label[val_idx]
                current_model_params = copy.deepcopy(model_params)
                current_model_params['hidden_size'] = int(data.shape[1] / digit)
                current_model_params['dropout_prob'] = dropout
                current_model_params['digits'] = digit
                current_training_params = copy.deepcopy(training_params)
                current_training_params['lambda_fs'] = lambda_fs
                current_training_params['weight_decay'] = weight_decay
                current_training_params['lr'] = lr
                current_training_params['batch_size'] = batch_size

                fold_result = EARFS_single_fold_train(
                    train_data, train_labels, val_data, val_labels,
                    current_model_params, current_training_params, 
                    os.path.join(optuna_folder, f'trial_{trial_id}_fold_{fold}'),logger
                )


                fold_time = time.time() - fold_start_time
                fold_times.append(fold_time)

                # with open(timing_log_path, 'a') as f:
                #     f.write(f"  Fold {fold+1}: {format_time(fold_time)} ({fold_time:.2f} seconds)\n")


                if 'loss' == eval_metric:

                    fold_score = fold_result['val_losses'][-1]
                else:

                    if len(fold_result['val_metrics']) > 0 and eval_metric in fold_result['val_metrics'][-1]:
                        fold_score = fold_result['val_metrics'][-1][eval_metric]
                    else:
                        logger.warning(f"Validation metric '{eval_metric}' not found. Using default metric.")
                        if model_params['task_type'] in ('binary','multiclass'):
                            fold_score = fold_result['val_metrics'][-1].get('acc', 0)
                        else:
                            fold_score = fold_result['val_metrics'][-1].get('r2', 0)

                fold_scores.append(fold_score)
                logger.info(f" Experiment {trial_id}, Fold {fold+1}/{k_folds} completed - Time taken: {format_time(fold_time)}")


            avg_score = np.mean(fold_scores)


            trial_total_time = time.time() - trial_start_time
            trial_times[trial_id] = trial_total_time


            # with open(timing_log_path, 'a') as f:
            #     f.write(f"Experiment {trial_id} - Total time: {format_time(trial_total_time)} ({trial_total_time:.2f} seconds)\n")
            #     f.write(f"Average {eval_metric.upper()}: {avg_score:.4f}\n")
            #     f.write("-" * 40 + "\n")

            logger.info(f"Experiment {trial_id} completed - Time taken: {format_time(trial_total_time)} - Average {eval_metric.upper()}: {avg_score:.4f}")


            return avg_score

        # Get n_jobs parameter
        n_jobs = training_params.get('n_jobs', 1)
        try:
            db_file = os.path.join(optuna_folder, f"optuna_EARFS.db")
            db_url = f"sqlite:///{db_file}"

            # Create RDB storage
            storage = RDBStorage(
                url=db_url,
                engine_kwargs={
                    "poolclass": QueuePool,
                    "pool_size": min(n_jobs + 1, 20),  
                    "max_overflow": 10,
                    "pool_timeout": 30
                }
            )
            logger.info(f"Use RDB storage to handle parallel optimization, database: {db_url}, parallel workers: {n_jobs}")
        except ImportError:
            logger.warning("Unable to import RDBStorage, using default storage")
            storage = None


        sampler = TPESampler(seed=training_params['seed'])
        study = optuna.create_study(
            sampler=sampler, 
            direction='minimize' if minimize_metric else 'maximize',
            storage=storage,
            study_name=f"EARFS",
            load_if_exists=True
        )


        n_trials = training_params.get('n_trials', 20)
        logger.info(f"Use {n_jobs} parallel workers to run {n_trials} Optuna optimization trials")
        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        except KeyboardInterrupt:
            logger.warning("User interrupted optimization.")
        except Exception as e:
            logger.error(f"Error occurred during optimization: {e}")
            if n_jobs > 1:
                logger.warning("Parallel optimization failed. Retrying with n_jobs=1")
                study.optimize(objective, n_trials=n_trials, n_jobs=1)


        best_trial = study.best_trial
        best_params = best_trial.params

        search_time = time.time() - search_start_time

        logger.info(f"Optuna search completed, time taken {format_time(search_time)} ({search_time/60:.2f} minutes)")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best value: {best_trial.value}")
        # with open(timing_log_path, 'a') as f:
        #     f.write("\n" + "=" * 40 + "\n")
        #     f.write(f"Search summary - Completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        #     f.write(f"Total search time: {format_time(search_time)} ({search_time:.2f} seconds)\n")
        #     f.write(f"Number of trials: {n_trials}\n")
        #     f.write(f"Best parameters: {best_params}\n")
        #     f.write(f"Best value: {best_trial.value}\n\n")

        #     sorted_trials = sorted(trial_times.items(), key=lambda x: x[1], reverse=True)
        #     f.write("Trial duration statistics:\n")
        #     f.write(f"  Fastest trial: {format_time(min(trial_times.values()))}\n")
        #     f.write(f"  Slowest trial: {format_time(max(trial_times.values()))}\n")
        #     f.write(f"  Average trial duration: {format_time(sum(trial_times.values()) / len(trial_times))}\n\n")

        #     f.write("Top 5 longest trials:\n")
        #     for i, (trial_id, duration) in enumerate(sorted_trials[:5]):
        #         f.write(f"  #{i+1}: Trial {trial_id} - {format_time(duration)} ({duration:.2f} seconds)\n")
        #     f.write("\n")


        logger.info("Training final model with best parameters...")
        final_model_start_time = time.time()


        final_model_params = copy.deepcopy(model_params)
        final_model_params['hidden_size'] = int(data.shape[1] / best_params['digits'])
        final_model_params['dropout_prob'] = best_params['dropout']
        final_model_params['digits'] = best_params['digits']
        
        final_training_params = copy.deepcopy(training_params)
        final_training_params['lambda_fs'] = best_params['lambda_fs']
        final_training_params['weight_decay'] = best_params['weight_decay']
        final_training_params['lr'] = best_params['lr']
        final_training_params['batch_size'] = best_params['batch_size']

        results = EARFS_model_train(
            data, label, final_model_params, final_training_params, folder_name,  logger, causal_variants
        )


        final_model_time = time.time() - final_model_start_time
        total_time = time.time() - total_start_time

        logger.info(f"Final model training completed, time taken {format_time(final_model_time)} ({final_model_time/60:.2f} minutes)")
        logger.info(f"Total processing time: {format_time(total_time)} ({total_time/60:.2f} minutes)")


        # with open(timing_log_path, 'a') as f:
        #     f.write(f"Final model training time: {format_time(final_model_time)} ({final_model_time:.2f} seconds)\n")
        #     f.write(f"Total processing time: {format_time(total_time)} ({total_time:.2f} seconds)\n")
        #     f.write("=" * 80 + "\n")


        with open(os.path.join(optuna_folder, 'best_params.txt'), 'w') as f:
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write(f"\nSearch time: {format_time(search_time)} ({search_time:.2f} seconds)\n")
            f.write(f"Final model training time: {format_time(final_model_time)} ({final_model_time:.2f} seconds)\n")
            f.write(f"Total time: {format_time(total_time)} ({total_time:.2f} seconds)\n")

        import joblib
        joblib.dump(study, os.path.join(optuna_folder, "optuna_study.pkl"))
        
        return results
        
    else:

        logger.info("Parameter optimization is disabled. Using single parameter values.")
        logger.info(f"Using parameters:")
        logger.info(f"  lambda_fs: {training_params['lambda_fs']}")
        logger.info(f"  weight_decay: {training_params.get('weight_decay', 0.0)}")
        logger.info(f"  dropout: {model_params['dropout_prob']}")
        logger.info(f"  lr: {training_params['lr']}")
        logger.info(f"  batch_size: {training_params['batch_size']}")
        logger.info(f"  digits: {model_params.get('digits', 100)}")

        single_model_start_time = time.time()
        
        results = EARFS_model_train(
            data, label, model_params, training_params, folder_name,  logger, causal_variants
        )

        single_model_time = time.time() - single_model_start_time
        total_time = time.time() - total_start_time

        logger.info(f"Single model training completed, time taken {format_time(single_model_time)} ({single_model_time/60:.2f} minutes)")
        logger.info(f"Total processing time: {format_time(total_time)} ({total_time/60:.2f} minutes)")


        with open(os.path.join(folder_name, 'timing_info.txt'), 'w') as f:
            f.write(f"EARFS Single Model Training - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Single model training time: {format_time(single_model_time)} ({single_model_time:.2f} seconds)\n")
            f.write(f"Total processing time: {format_time(total_time)} ({total_time:.2f} seconds)\n")

        return results

def EARFS_model_train(data, label, model_params, training_params, folder_name, logger, causal_variants):
    """
    Train full EARFS model with given parameters
    data: input features
    label: target values
    model_params: model architecture parameters
    training_params: training parameters
    folder_name: directory for saving results
    logger: logging object
    causal_variants: pre-selected features for evaluation
    Returns:
        dict with training results and the trained model
    """
    if model_params['task_type'] in ('binary','multiclass') and label.dtype != torch.long:
        label = label.long()    
    dataset = Data.TensorDataset(data, label)
    dataset_size = len(dataset)
    
    indices = list(range(dataset_size))
        

    strata = None
    if model_params['task_type'] in ('binary','multiclass'):
        if isinstance(label, torch.Tensor):
            strata = label.detach().cpu().numpy().ravel()
        else:
            strata = np.asarray(label).ravel()
    
    train_indices, val_indices = train_test_split(
        indices,
        test_size=training_params['validation_split'],
        random_state=training_params.get('seed', 42),
        stratify=strata
    )
    train_sampler = Data.Subset(dataset, train_indices)
    valid_sampler = Data.Subset(dataset, val_indices)
    
    train_loader = Data.DataLoader(train_sampler, batch_size=training_params['batch_size'], shuffle=True, drop_last=True)
    validation_loader = Data.DataLoader(valid_sampler, batch_size=training_params['batch_size'], drop_last=False)
    
    model = Model(model_params).to(training_params['device'])
    fs_params = list(model.feature_selection_MLP.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('feature_selection_MLP')]


    optimizer = torch.optim.Adam([
        {'params': fs_params, 'weight_decay': 0.0}, 
        {'params': other_params, 'weight_decay': training_params.get('weight_decay', 0.0)}  
    ], lr=training_params['lr'])
    
    
    if model_params['task_type'] in ('binary', 'multiclass'):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    

    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    for epoch in range(training_params['num_epochs']):
        model.train()
        epoch_loss = 0
        metrics = {'train': {}, 'val': {}}
        

        all_train_preds = []
        all_train_targets = []
        correct, total = 0, 0
        for inputs, targets in train_loader:
            if len(inputs) <= 1: 
                continue
            inputs = inputs.to(device=training_params['device'], dtype=torch.float32)
            targets = targets.to(device=training_params['device'])

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss += training_params['lambda_fs'] / torch.sum(
                (model.feature_selection_MLP.get_selection_rate() - 0.5) ** 2
            )
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            


            if model_params['task_type'] in ('binary','multiclass'):
                predicts = outputs.argmax(dim=1)
                correct += (predicts == targets).sum().item()
                total += targets.numel()
            else:
                all_train_preds.extend(outputs.view(-1).detach().cpu().numpy())
                all_train_targets.extend(targets.view(-1).detach().cpu().numpy())
        
        train_losses.append(epoch_loss / len(train_loader))
        if model_params['task_type'] in ('binary','multiclass'):

            metrics['train']['acc'] = correct / max(total, 1)
        
        elif model_params['task_type'] == 'regression':
            all_train_preds = np.array(all_train_preds)
            all_train_targets = np.array(all_train_targets)
            metrics['train']['mse'] = mean_squared_error(all_train_targets, all_train_preds)
            metrics['train']['mae'] = mean_absolute_error(all_train_targets, all_train_preds)
            metrics['train']['r2']  = r2_score(all_train_targets, all_train_preds)
        
        train_metrics.append(metrics['train'])


        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_targets = []
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                if len(inputs) <= 1: 
                    continue
                inputs = inputs.to(device=training_params['device'], dtype=torch.float32)
                targets = targets.to(device=training_params['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                if model_params['task_type'] in ('binary','multiclass'):
                    predicts = outputs.argmax(dim=1)
                    correct += (predicts == targets).sum().item()
                    total += targets.numel()
                else:
                    all_val_preds.extend(outputs.view(-1).cpu().numpy())
                    all_val_targets.extend(targets.view(-1).cpu().numpy())
        

        val_losses.append(val_loss / len(validation_loader))
        if model_params['task_type'] in ('binary','multiclass'):
            metrics['val']['acc'] = correct / max(total, 1)
        else:
            all_val_preds   = np.array(all_val_preds)
            all_val_targets = np.array(all_val_targets)
            metrics['val']['mse'] = mean_squared_error(all_val_targets, all_val_preds)
            metrics['val']['mae'] = mean_absolute_error(all_val_targets, all_val_preds)
            metrics['val']['r2']  = r2_score(all_val_targets, all_val_preds)
        
        val_metrics.append(metrics['val'])
        
        if epoch % 10 == 0 or epoch == training_params['num_epochs'] - 1:
            if model_params['task_type'] in ('binary','multiclass'):
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
        
        if val_losses[-1] < best_val_loss - training_params['min_delta']:
            best_val_loss = val_losses[-1]
            early_stopping_counter = 0
            
            # save_path = os.path.join(folder_name, f'EARFS_best_model.pth')
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # torch.save(model.state_dict(), save_path)

        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= training_params['patience']:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break
    

    select_rate = model.feature_selection_MLP.get_selection_rate().detach().cpu().numpy()
    feature_indices = sorted_idx(select_rate)
    

    return {
        'model': model,

        'select_rate': select_rate,
        'feature_indices': feature_indices,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }

def EARFS_single_fold_train(train_data, train_labels, val_data, val_labels, model_params, training_params, folder_name, logger):
    """
    Train EARFS model on a single fold for cross-validation
    Args:
        train_data: training features
        train_labels: training target values
        val_data: validation features
        val_labels: validation target values
        model_params: model architecture parameters
        training_params: training parameters
        folder_name: directory for saving results
        logger: logging object
    """

    if model_params['task_type'] in ('binary','multiclass'):
        if train_labels.dtype != torch.long: train_labels = train_labels.long()
        if val_labels.dtype   != torch.long: val_labels   = val_labels.long()
            
    train_dataset = Data.TensorDataset(train_data, train_labels)
    val_dataset = Data.TensorDataset(val_data, val_labels)
    
    train_loader = Data.DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, drop_last=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=training_params['batch_size'], drop_last=False)
    

    model = Model(model_params).to(training_params['device'])
    

    fs_params = list(model.feature_selection_MLP.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('feature_selection_MLP')]
    

    optimizer = torch.optim.Adam([
        {'params': fs_params, 'weight_decay': 0.0},  
        {'params': other_params, 'weight_decay': training_params.get('weight_decay', 0.0)} 
    ], lr=training_params['lr'])
    

    if model_params['task_type'] in ('binary', 'multiclass'):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    best_val_loss = float('inf')
    best_model_state = None
    best_select_rate = None
    early_stopping_counter = 0
    

    for epoch in range(training_params['num_epochs']):
        model.train()
        epoch_loss = 0
        metrics = {'train': {}, 'val': {}}
        

        all_train_preds = []
        all_train_targets = []
        correct, total = 0, 0
        for inputs, targets in train_loader:
            if len(inputs) <= 1:  
                continue
            inputs = inputs.to(device=training_params['device'], dtype=torch.float32)
            targets = targets.to(device=training_params['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss += training_params['lambda_fs'] / torch.sum(
                (model.feature_selection_MLP.get_selection_rate() - 0.5) ** 2
            )
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
                
            if model_params['task_type'] in ('binary', 'multiclass'):
                predicts = outputs.argmax(dim=1)
                correct += (predicts == targets).sum().item()
                total   += targets.numel()
            else:
                all_train_preds.extend(outputs.view(-1).detach().cpu().numpy())
                all_train_targets.extend(targets.view(-1).detach().cpu().numpy())
        

        train_losses.append(epoch_loss / max(len(train_loader), 1))
        if model_params['task_type'] in ('binary', 'multiclass'):
            metrics['train']['acc'] = correct / max(total, 1)
        else:
            all_train_preds   = np.array(all_train_preds)
            all_train_targets = np.array(all_train_targets)
            metrics['train']['mse'] = mean_squared_error(all_train_targets, all_train_preds)
            metrics['train']['mae'] = mean_absolute_error(all_train_targets, all_train_preds)
            metrics['train']['r2']  = r2_score(all_train_targets, all_train_preds)
        train_metrics.append(metrics['train'])
        

        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_targets = []
        correct, total = 0, 0  
        with torch.no_grad():
            for inputs, targets in val_loader:
                if len(inputs) <= 1:  
                    continue
                inputs = inputs.to(device=training_params['device'], dtype=torch.float32)
                targets = targets.to(device=training_params['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                if model_params['task_type'] in ('binary', 'multiclass'):
                    predicts = outputs.argmax(dim=1)
                    correct += (predicts == targets).sum().item()
                    total   += targets.numel()
                else:
                    all_val_preds.extend(outputs.view(-1).cpu().numpy())
                    all_val_targets.extend(targets.view(-1).cpu().numpy())
        
        val_losses.append(val_loss / max(len(val_loader), 1))
        if model_params['task_type'] in ('binary', 'multiclass'):
            metrics['val']['acc'] = correct / max(total, 1)
        else:
            all_val_preds   = np.array(all_val_preds)
            all_val_targets = np.array(all_val_targets)
            metrics['val']['mse'] = mean_squared_error(all_val_targets, all_val_preds)
            metrics['val']['mae'] = mean_absolute_error(all_val_targets, all_val_preds)
            metrics['val']['r2']  = r2_score(all_val_targets, all_val_preds)
        val_metrics.append(metrics['val'])
        
        current_val_loss = val_losses[-1]
        if current_val_loss < best_val_loss - training_params['min_delta']:
            best_val_loss = current_val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()
            best_select_rate = model.feature_selection_MLP.get_selection_rate().detach().cpu().numpy()
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= training_params['patience']:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_select_rate is None:
        best_select_rate = model.feature_selection_MLP.get_selection_rate().detach().cpu().numpy()
    select_rate = best_select_rate
    feature_indices = sorted_idx(select_rate)
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'select_rate': select_rate,
        'feature_indices': feature_indices
    }


class FeatureSelectionMLP(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.selection_rate = nn.Parameter(
            torch.ones([1, input_size])
        )

    def get_selection_rate(self):
        return torch.sigmoid(self.selection_rate)

    def forward(self, x):
        return x * torch.sigmoid(self.selection_rate)


class Classifier(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size, dropout_prob=0.5, activation='sigmoid') -> None:
        super().__init__()
        self.activation_func = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            self.activation_func,
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            self.activation_func,
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, n_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class Regressor(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.5, activation='sigmoid') -> None:
        super().__init__()
        self.activation_func = get_activation(activation)
        self.regressor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            self.activation_func,
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            self.activation_func,
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.regressor(x)

class Model(nn.Module):
    def __init__(self, model_params) -> None:
        super().__init__()
        self.task_type = model_params['task_type']  
        if self.task_type in ('binary', 'multiclass'):
            self.predictor = Classifier(
                model_params['input_size'],
                model_params['num_classes'],  
                model_params['hidden_size'],
                model_params['dropout_prob'],
                model_params.get('activation', 'sigmoid')
            )
        else:
            self.predictor = Regressor(
                model_params['input_size'],
                model_params['hidden_size'],
                model_params['dropout_prob'],
                model_params.get('activation', 'sigmoid')
            )
        self.feature_selection_MLP = FeatureSelectionMLP(model_params['input_size'])

        
    def forward(self, x):
        return self.predictor(
            self.feature_selection_MLP(x)
        )
def sorted_idx(select_rate):
    """sort feature indices based on selection rate"""
    a = select_rate[0, :]
    idx = sorted(range(len(a)), key=lambda k: a[k], reverse=True)
    return idx



