# bcor_evaluation.py
import os
import numpy as np
import torch
import time
import datetime
import subprocess
import logging
from sklearn.model_selection import train_test_split
from evaluation_utils import evaluate_feature_set
import sys
from utils import format_time

def to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)
def _normalize_y_for_bcor(y, task_type: str):
    y = to_numpy(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)

    if task_type == 'regression':
        if y.ndim > 1 and y.shape[1] > 1:
            y = y[:, 0]
        y = y.astype(np.float64).reshape(-1)
        return y


    if y.ndim > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)


    y = y.reshape(-1)


    y = y.astype(np.int64)


    uniq = np.unique(y)
    remap = {c: i for i, c in enumerate(np.sort(uniq))}
    y = np.vectorize(remap.get, otypes=[np.int64])(y)

    if task_type == 'binary':
        if len(uniq) != 2:
            raise ValueError(f"binary require two classes but now detected {len(uniq)} : {uniq}")

        return y.astype(np.int64)

    if task_type == 'multiclass':
        if len(uniq) < 3:
            raise ValueError(f"multiclass require at least three classes but now detected {len(uniq)} : {uniq}")
        return y.astype(np.int64)


    raise ValueError(f"unknown task_type: {task_type}")

    
def BCOR_with_evaluation(data, label, model_params, training_params, name, 
                       features_selected, digits, folder_name, 
                       logger, feature_prediction, n_iters=20,
                       feature_step=500, n_folds=3):
    """
    execute feature selection and evaluation based on Ball correlation, with checkpointing and intermediate result saving
    
    Args:
        data: input feature data
        label: target values
        model_params: model parameter dictionary
        training_params: train model dictionary
        name: dataset name
        features_selected: pre-selected feature list (usually the indices of causal features)
        digits: feature dimension divisor (for neural network hidden layers)
        folder_name: result save directory
        logger: logger
        feature_prediction: upper limit on the number of features to evaluate
        n_iters: number of iterations, default 20
        feature_step: feature evaluation step size, default 500
        n_folds: number of cross-validation folds, default 3
    """

    total_start_time = time.time()
    logger.info(f"开始 BCOR_with_evaluation 于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

        
    start_iter = 0

    

    logger.info(f"total features: {total_features}")
    logger.info(f"evaluation settings:")
    logger.info(f"- max features to evaluate: {max_features}")
    logger.info(f"- feature sequence: {feature_sequence}")
    logger.info(f"- number of evaluation points: {n_steps}")
    logger.info(f"- number of cross-validation folds: {n_folds}")
    

    dropout_probs = [0.1, 0.3, 0.7, 0.9]
    l2_lambdas = [0.0001, 0.001, 0.01, 0.1]
    task_type = model_params['task_type']


    temp_data_dir = os.path.join(eval_folder, 'temp_data')
    os.makedirs(temp_data_dir, exist_ok=True)
    r_output_dir = os.path.join(eval_folder, 'r_output')
    os.makedirs(r_output_dir, exist_ok=True)
    
    for iter in range(n_iters):
        model_iter_dir = os.path.join(eval_folder, f'model_iter_{iter}')
        os.makedirs(model_iter_dir, exist_ok=True)
    

    for iter in range(start_iter, n_iters):
        iter_start_time = time.time()
        logger.info(f'\start iteration {iter+1}/{n_iters}, seed {seeds[iter]}')
        
        try:


            temp_data_file = os.path.join(temp_data_dir, f'data_iter_{iter}.npz')
            y_train_norm = _normalize_y_for_bcor(y_train, task_type)
            np.savez(temp_data_file, X=to_numpy(x_train), Y=y_train_norm)
            bcor_output_dir = os.path.join(r_output_dir, f'iter_{iter}')
            os.makedirs(bcor_output_dir, exist_ok=True)

            logger.info(f"executing R script...")


            r_script_path = 'bcor.R'  
            env = os.environ.copy()
            env['RETICULATE_PYTHON'] = sys.executable

            cmd = ['Rscript', r_script_path, temp_data_file, bcor_output_dir, task_type, name]

            bcor_start_time = time.time()
            process = subprocess.run(cmd, capture_output=True, text=True, env=env)
            bcor_time = time.time() - bcor_start_time
            
            if process.returncode != 0:
                logger.error("R script failed to run")
                logger.error("---- R STDERR ----\n" + (process.stderr or "<empty>"))
                logger.error("---- R STDOUT ----\n" + (process.stdout or "<empty>"))
                raise RuntimeError("R script failed to run ( stdout/stderr)")

            logger.info(f"R script completed successfully, time taken: {format_time(bcor_time)} ({bcor_time:.2f} seconds)")
            logger.info(f"R script standard output:\n{process.stdout}")
            if process.stderr:
                logger.info(f"R script standard error:\n{process.stderr}")

            

            result_file = os.path.join(bcor_output_dir, f'BCOR_{name}_idx.npy')
            if not os.path.exists(result_file):
                logger.error(f"cannot find R script result file: {result_file}")
                available_files = os.listdir(bcor_output_dir)
                logger.error(f"Files available in directory: {available_files}")

                for file in os.listdir(bcor_output_dir):
                    if file.startswith('bcor_idx') and file.endswith('.npz'):
                        result_file = os.path.join(bcor_output_dir, file)
                        logger.info(f"Found alternative result file: {result_file}")
                        break
            
            if not os.path.exists(result_file):
                raise FileNotFoundError(f"cannot find R script result file: {result_file}")


            bcor_results = np.load(result_file, allow_pickle=True)
            feature_indices = bcor_results#['selected_features']
            

            if np.min(feature_indices) > 0 and np.max(feature_indices) <= total_features:
                logger.info("Detected feature indices starting from 1, converting to 0-based...")
                feature_indices = feature_indices - 1


            results_indices.append(feature_indices)


            feature_ranking_file = os.path.join(iter_folder, f'feature_ranking_iter_{iter}.npz')
            np.savez(feature_ranking_file, 
                    feature_indices=feature_indices, 
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
                    logger.info(f"selected_features: {selected_features}")
                    
                    # 准备包含选定特征的数据
                    x_train_selected = x_train[:, selected_features]
                    x_test_selected  = x_test[:, selected_features]
                    

                    y_train_np = np.asarray(y_train)
                    y_test_np  = np.asarray(y_test)
                    
                    model_params_local = model_params.copy()
                    
                    if task_type == 'binary':
                        if y_train_np.ndim > 1 and y_train_np.shape[1] > 1:
                            y_train_np = np.argmax(y_train_np, axis=1)
                        if y_test_np.ndim > 1 and y_test_np.shape[1] > 1:
                            y_test_np = np.argmax(y_test_np, axis=1)
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
                        y_train_np = y_all_idx[:len(y_train_np)].astype(np.int64)
                        y_test_np  = y_all_idx[len(y_train_np):].astype(np.int64)
                        model_params_local['num_classes'] = num_classes

                    
                    else:  # regression

                        if y_train_np.ndim == 1: y_train_np = y_train_np.reshape(-1, 1)
                        if y_test_np.ndim  == 1: y_test_np  = y_test_np.reshape(-1, 1)

                    

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
                    import traceback
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
            logger.info(f"The full results for iteration {iter+1} have been saved to: {iter_results_file}")
            

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
            logger.info(f"Iteration {iter+1} completed, duration: {format_time(iter_time)} ({iter_time:.2f} seconds)")

        except Exception as e:
            logger.error(f"Error in iteration {iter}: {str(e)}")
            import traceback
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
        'per_step_report_metrics': np.array(all_report_metrics, dtype=object)  # ⭐ 新增
    }
    

    save_path = os.path.join(folder_name, f'BCOR_{name}_results.npz')
    np.savez(save_path, **final_results)

    # timestamp_save_path = os.path.join(
    #     folder_name, 
    #     f'BCOR_{name}_results_{final_results["timestamp"]}.npz'
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


