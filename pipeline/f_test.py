import numpy as np
import torch
import os
import time
import datetime
import traceback
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from evaluation_utils import evaluate_feature_set
from sklearn.feature_selection import f_classif, f_regression
from utils import format_time





def FTEST_fs(data, label, model_params, training_params,name, folder_name, logger, causal_variants=None):

    os.makedirs(folder_name, exist_ok=True)
    

    start_time = time.time()
    logger.info(f"开始 FTEST_fs 于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    

    seed = training_params['seed']
    

    X = data.numpy() if isinstance(data, torch.Tensor) else data
    y = label.numpy() if isinstance(label, torch.Tensor) else label
    

    x_train = data['X_train']; y_train = label['y_train']
    

    task_type = model_params.get('task_type', 'binary')  # 'binary' | 'multiclass' | 'regression'

    

    logger.info("F-Test...")
    try:

        if task_type in ['binary', 'multiclass']:
            F, p = f_classif(x_train, y_train if y_train.ndim == 1 else np.argmax(y_train, axis=1))
        else:
            # regression
            y_reg = y_train.reshape(-1) if y_train.ndim > 1 else y_train
            F, p = f_regression(x_train, y_reg)
        

        feature_indices = np.argsort(-F)
        feature_scores = F[feature_indices]

        logger.info(f"F-Test completed。The total number of features: {len(feature_indices)}")
        logger.info(f"The F values of the top 10 features: {feature_scores[:10]}")
        logger.info(f"The indices of the top 10 features: {feature_indices[:10]}")



        np.save(os.path.join(folder_name, f'FTEST_{name}_weights.npy'), 
                F)
        np.save(os.path.join(folder_name, f'FTEST_{name}_idx.npy'), 
                feature_indices)
        # result_file = os.path.join(folder_name, f"ftest_features.npz")

        # np.savez(
        #     result_file,
        #     feature_indices=feature_indices,
        #     F_statistics=F,
        #     p_values=p,
        #     seed=seed,
        #     timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # )
        # logger.info(f"feature ranking saved to: {result_file}")
        
        

        elapsed_time = time.time() - start_time
        logger.info(f"F-Test feature selection completed, elapsed time: {format_time(elapsed_time)} ({elapsed_time:.2f} seconds)")

        return feature_indices
        
    except Exception as e:
        logger.error(f"F-Test feature selection error: {str(e)}")
        logger.error(traceback.format_exc())
        raise e


def FTEST_fs_with_evaluation(data, label, model_params, training_params, name, 
                       features_selected, digits, folder_name, 
                       logger, feature_prediction, n_iters=20,
                       feature_step=500, n_folds=3):


    total_start_time = time.time()
    logger.info(f" FTEST_with_evaluation start at  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

    

    results = np.zeros((n_steps, n_iters)) 
    cv_val_losses = np.zeros((n_steps, n_iters))  
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

    x_train = x_train.numpy() if isinstance(x_train, torch.Tensor) else x_train
    x_test = x_test.numpy() if isinstance(x_test, torch.Tensor) else x_test
    y_train = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
    y_test = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
    task_type = model_params.get('task_type', 'binary')
    

    for iter in range(start_iter, n_iters):
        iter_start_time = time.time()
        logger.info(f'\start iteration {iter+1}/{n_iters}, seed {seeds[iter]}')
        
        try:
            

            logger.info("F-Test...")

            try:
                F, p = f_classif(x_train, y_train)

                feature_indices = np.argsort(-F)

                logger.info(f"successfully completed F-Test feature ranking, top 10 features F values: {F[feature_indices[:10]]}")
                logger.info(f"Top 10 feature indices: {feature_indices[:10]}")
            except Exception as e:
                logger.error(f"F-Test feature selection error: {str(e)}")
                logger.error(traceback.format_exc())
                raise e
            

            results_indices.append(feature_indices)

            ftest_results_file = os.path.join(iter_folder, f'feature_ranking_iter_{iter}.npz')
            np.savez(ftest_results_file, 
                    feature_indices=feature_indices, 
                    F_stats=F,
                    p_values=p,
                    iter=iter, 
                    seed=seeds[iter])
            logger.info(f"feature ranking saved to: {ftest_results_file}")
            

            feature_steps_range = range(n_steps)

            

            for step in feature_steps_range:
                num_features = int(feature_sequence[step])
                if num_features > len(feature_indices):
                    logger.warning(f"Requested number of features {num_features} exceeds available features {len(feature_indices)}, skipping")
                    continue

                logger.info(f"\nEvaluating top {num_features} features")

                try:

                    selected_features = feature_indices[:num_features]
                    logger.info(f"selected_features: {selected_features}...")
                    

                    x_train_selected = x_train[:, selected_features]
                    x_test_selected = x_test[:, selected_features]
                    

                    y_train_np = np.asarray(y_train)
                    y_test_np  = np.asarray(y_test)
                    
                    model_params_local = model_params.copy()
                    
                    if task_type == 'binary':

                        y_train_np = y_train_np.astype(np.float32).reshape(-1, 1)
                        y_test_np  = y_test_np.astype(np.float32).reshape(-1, 1)
                    
                    elif task_type == 'multiclass':

                        if y_train_np.ndim > 1 and y_train_np.shape[1] > 1:
                            y_train_np = np.argmax(y_train_np, axis=1)
                        if y_test_np.ndim > 1 and y_test_np.shape[1] > 1:
                            y_test_np = np.argmax(y_test_np, axis=1)

                        classes, y_all_idx = np.unique(
                            np.concatenate([y_train_np.ravel(), y_test_np.ravel()]),
                            return_inverse=True
                        )
                        num_classes = len(classes)
                        y_train_np = y_all_idx[: len(y_train_np)].astype(np.int64)
                        y_test_np  = y_all_idx[len(y_train_np):].astype(np.int64)
                        model_params_local['num_classes'] = num_classes
                    
                    else:  # regression
                        y_train_np = y_train_np.astype(np.float32).reshape(-1, 1)
                        y_test_np  = y_test_np.astype(np.float32).reshape(-1, 1)
                    

                    final_score, best_dropout, best_l2, best_cv_val_loss, report_metrics = evaluate_feature_set(
                        x_train_selected, y_train_np,
                        x_test_selected,  y_test_np,
                        num_features, model_params_local,
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
                    logger.info(f"The results for feature count {num_features} have been saved to: {feature_eval_file}")
                    

                    checkpoint_data = {
                        'last_feature_step': step,
                        'current_iter': iter,
                        'current_feature': num_features,
                        'results': results,
                        'cv_val_losses': cv_val_losses,
                        'results_indices': np.array(results_indices, dtype=object),
                        'seeds': seeds,
                        'feature_sequence': feature_sequence,
                        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),

                        'per_step_report_metrics': np.array(all_report_metrics, dtype=object) 
                    }
                    np.savez(checkpoint_file, **checkpoint_data)
                    logger.info(f"Checkpoint updated, completed evaluation for feature count {num_features} ({step+1}/{n_steps}), current iteration {iter+1}/{n_iters}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating feature count {num_features}: {str(e)}")
                    logger.error(traceback.format_exc())

                    continue

            iter_results_file = os.path.join(iter_folder, f'full_results_iter_{iter}.npz')
            report_metrics_iter_column = np.array([all_report_metrics[r][iter] for r in range(n_steps)], dtype=object)
            np.savez(iter_results_file, 
                    iter=iter,
                    test_results=results[:, iter],
                    cv_val_losses=cv_val_losses[:, iter],
                    feature_indices=feature_indices,
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
            logger.error(f"Error occurred in iteration {iter}: {str(e)}")
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
        'indices': np.array(results_indices, dtype=object),
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

    save_path = os.path.join(folder_name, f'FTEST_{name}_results.npz')
    np.savez(save_path, **final_results)

    # timestamp_save_path = os.path.join(
    #     folder_name, 
    #     f'FTEST_{name}_results_{final_results["timestamp"]}.npz'
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