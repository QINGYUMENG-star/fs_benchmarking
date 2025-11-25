# lasso_evaluation.py
import os
import sys
import time
import torch
import numpy as np
import datetime
import subprocess
from utils import format_time
from evaluation_utils import evaluate_feature_set

def _to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def _normalize_y_for_lasso(y, task_type: str):
    y = _to_numpy(y)

    # 统一把 (N,1)->(N,), one-hot/proba -> argmax
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.reshape(-1)
    elif y.ndim > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)

    if task_type == 'binary':
        # 映射到 {0,1}
        uniq = np.unique(y)
        mapping = {c: i for i, c in enumerate(sorted(uniq))}
        y = np.vectorize(mapping.get)(y).astype(np.int64).reshape(-1)
        if len(np.unique(y)) != 2:
            raise ValueError(f"binary expects 2 classes after mapping, got {np.unique(y)}")
        return y

    if task_type == 'multiclass':
        # 映射到 0..K-1
        classes, y = np.unique(y, return_inverse=True)
        return y.astype(np.int64).reshape(-1)

    if task_type == 'regression':
        return y.astype(np.float64).reshape(-1)

    raise ValueError(f"Unknown task_type: {task_type}")

def _data_type_for_r(task_type: str) -> str:
    if task_type == 'binary':
        return 'binary'       # glmnet family = "binomial"
    if task_type == 'regression':
        return 'continuous'   # glmnet family = "gaussian"
    if task_type == 'multiclass':
        return 'multinomial'  # glmnet family = "multinomial"
    raise NotImplementedError(
        "Only 'binary' and 'regression' are supported by lasso.R currently."
    )

def _rank_features_from_lasso_path(coefs_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    从 R 脚本导出的 Lasso path 系数矩阵生成特征排序：
    - coefs_matrix 形状通常为 (p, n_lambda) 或 (n_features, n_lambda)
    - 评分规则：沿路径对每个特征取 max(|coef|) 作为 score
    返回：
      feature_indices: 按 score 降序的特征下标（0-based）
      feature_scores:  与之对应的分数（max|coef|）
    """
    if coefs_matrix.ndim != 2:
        raise ValueError(f"Expected 2D coefs matrix, got shape {coefs_matrix.shape}")
    # glmnet beta: rows=features, cols=λ序列
    abs_max = np.max(np.abs(coefs_matrix), axis=1)  # shape: (p,)
    order = np.argsort(-abs_max)                    # 降序
    return order.astype(np.int64), abs_max

def LASSO_with_evaluation(
    data, label, model_params, training_params, name,
    features_selected, digits, folder_name,
    logger, feature_prediction, n_iters=20, feature_step=500, n_folds=3
):
    """
    与 BCOR_with_evaluation 同流程的 LASSO 评估版：
    1) 每个迭代将 train 写入临时 npz
    2) 调 R 脚本跑 Lasso Path
    3) 用 max|coef| 做特征排名
    4) 做 Top-k 评估并保存所有中间/最终结果
    """
    total_start_time = time.time()
    logger.info(f"Start LASSO_with_evaluation at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    x_train = data['X_train']; x_test = data['X_test']
    y_train = label['y_train']; y_test = label['y_test']

    device = x_train.device if isinstance(x_train, torch.Tensor) \
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    logger.info(f"Evaluate Top 1–{feature_sequence.max()} features with step {feature_step}")
    feature_sequence = np.unique(feature_sequence)
    feature_sequence = feature_sequence[feature_sequence <= max_features]
    n_steps = len(feature_sequence)

    # 记录容器
    results = np.zeros((n_steps, n_iters))
    cv_val_losses = np.zeros((n_steps, n_iters))
    results_indices, results_scores = [], []
    all_best_params = []
    all_report_metrics = [[None for _ in range(n_iters)] for _ in range(n_steps)]
    seeds = np.random.choice(range(1000), n_iters, replace=False)

    logger.info(f"total features: {total_features}")
    logger.info(f"evaluation settings:")
    logger.info(f"- max features to evaluate: {max_features}")
    logger.info(f"- feature sequence: {feature_sequence}")
    logger.info(f"- number of evaluation points: {n_steps}")
    logger.info(f"- number of cross-validation folds: {n_folds}")

    # 与其它方法保持一致的轻量 sweep
    dropout_probs = [0.1, 0.3, 0.7, 0.9]
    l2_lambdas   = [0.0001, 0.001, 0.01, 0.1]
    task_type    = model_params['task_type']
    dfmax        = int(training_params.get('dfmax', 3000))

    temp_data_dir = os.path.join(eval_folder, 'temp_data')
    os.makedirs(temp_data_dir, exist_ok=True)
    r_output_dir = os.path.join(eval_folder, 'r_output')
    os.makedirs(r_output_dir, exist_ok=True)

    for it in range(n_iters):
        os.makedirs(os.path.join(eval_folder, f'model_iter_{it}'), exist_ok=True)

    for iter in range(n_iters):
        iter_start = time.time()
        logger.info(f"\n--- iteration {iter+1}/{n_iters}, seed {seeds[iter]} ---")

        try:
            # 1) 写入 R 脚本输入
            temp_data_file = os.path.join(temp_data_dir, f'data_iter_{iter}.npz')
            x_np = _to_numpy(x_train)
            y_np = _normalize_y_for_lasso(y_train, task_type)
            np.savez(temp_data_file, X=x_np, Y=y_np)

            # 2) 调用 R
            r_iter_dir = os.path.join(r_output_dir, f'iter_{iter}')
            os.makedirs(r_iter_dir, exist_ok=True)

            data_type = _data_type_for_r(task_type)
            r_script = 'lasso.R'
            cmd = ['Rscript', r_script, temp_data_file, r_iter_dir, data_type, str(dfmax),name]
            logger.info(f"Running R: {' '.join(map(str, cmd))}")

            r_start = time.time()
            env = os.environ.copy()
            env['RETICULATE_PYTHON'] = sys.executable  # 让 reticulate 认得当前 python
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            r_time = time.time() - r_start

            if proc.stdout:
                logger.info("---- R STDOUT ----\n" + proc.stdout)
            if proc.stderr:
                # glmnet 常有 warning 到 stderr，这里仅打印
                logger.warning("---- R STDERR ----\n" + proc.stderr)

            if proc.returncode != 0:
                raise RuntimeError("R LASSO script failed. See logs above.")
            logger.info(f"R finished, time: {format_time(r_time)} ({r_time:.2f}s)")

            # 3) 读取 R 输出（文件名前缀 lasso_path_）
            #    我们找该目录下最新/唯一的 lasso_path_*.npz
            lasso_npz = None
            for f in os.listdir(r_iter_dir):
                if f.startswith(f'LASSO_{name}_idx') and f.endswith('.npz'):
                    lasso_npz = os.path.join(r_iter_dir, f)
                    break
            if lasso_npz is None:
                raise FileNotFoundError(f"No LASSO_idx*.npz found in {r_iter_dir}")

            pack = np.load(lasso_npz, allow_pickle=True)
            coefs = pack['coefs']     # shape: (p, n_lambda)
            # lambdas = pack['lambdas']   # 如需分析路径，可拿出来
            # intercept = pack['intercept']

            # 4) 由路径得特征排序
            feature_indices, feature_scores = _rank_features_from_lasso_path(coefs)
            results_indices.append(feature_indices)
            results_scores.append(feature_scores)

            # 落地单次迭代的排名
            np.savez(os.path.join(iter_folder, f'feature_ranking_iter_{iter}.npz'),
                     feature_indices=feature_indices,
                     feature_scores=feature_scores,
                     iter=iter,
                     seed=seeds[iter])

            # 5) 逐个 Top-k 评估
            for step, k in enumerate(feature_sequence):
                k = int(k)
                if k > len(feature_indices):
                    logger.warning(f"Requested top-{k} > available {len(feature_indices)}, skip")
                    continue

                sel = feature_indices[:k]
                x_tr_sel = x_train[:, sel]
                x_te_sel = x_test[:,  sel]

                y_tr_np = _to_numpy(y_train)
                y_te_np = _to_numpy(y_test)

                if task_type == 'binary':
                    if y_tr_np.ndim > 1 and y_tr_np.shape[1] > 1:
                        y_tr_np = np.argmax(y_tr_np, axis=1)
                    if y_te_np.ndim > 1 and y_te_np.shape[1] > 1:
                        y_te_np = np.argmax(y_te_np, axis=1)
                    y_tr_np = y_tr_np.astype(np.float32).reshape(-1, 1)
                    y_te_np = y_te_np.astype(np.float32).reshape(-1, 1)

                    model_params_local = model_params.copy()
                    model_params_local['num_classes'] = 2
                    mp_to_use = model_params_local

                elif task_type == 'multiclass':
                    # one-hot/proba -> argmax，然后映射到连续 0..K-1，并把 num_classes 传给评估模型
                    if y_tr_np.ndim > 1 and y_tr_np.shape[1] > 1:
                        y_tr_np = np.argmax(y_tr_np, axis=1)
                    if y_te_np.ndim > 1 and y_te_np.shape[1] > 1:
                        y_te_np = np.argmax(y_te_np, axis=1)
                    classes, y_all_idx = np.unique(
                        np.concatenate([y_tr_np.ravel(), y_te_np.ravel()]),
                        return_inverse=True
                    )
                    num_classes = len(classes)
                    y_tr_np = y_all_idx[:len(y_tr_np)].astype(np.int64)
                    y_te_np  = y_all_idx[len(y_tr_np):].astype(np.int64)

                    # 针对多分类，把类别数写进一个局部的 model_params 再传给 evaluate_feature_set
                    model_params_local = model_params.copy()
                    model_params_local['num_classes'] = num_classes
                    mp_to_use = model_params_local
                else:  # regression
                    if y_tr_np.ndim == 1: y_tr_np = y_tr_np.reshape(-1, 1)
                    if y_te_np.ndim == 1: y_te_np = y_te_np.reshape(-1, 1)

                    mp_to_use = model_params

                final_score, best_dropout, best_l2, best_cv_val_loss, report_metrics = evaluate_feature_set(
                    x_tr_sel, y_tr_np,
                    x_te_sel, y_te_np,
                    k, mp_to_use,
                    dropout_probs, l2_lambdas,
                    n_folds, device
                )

                results[step, iter] = final_score
                cv_val_losses[step, iter] = best_cv_val_loss
                all_report_metrics[step][iter] = report_metrics

                score_name = {
                    'binary': 'AUC',
                    'multiclass': 'AUC_macro',
                    'regression': 'PearsonR'
                }.get(task_type, 'score')

                logger.info(f"[iter {iter+1}] Top-{k}: val_loss={best_cv_val_loss:.4f}, "
                            f"Test {score_name}={final_score:.6f}, "
                            f"best(dropout={best_dropout}, l2={best_l2})")

                np.savez(os.path.join(feature_folder, f'features_{k}_iter_{iter}.npz'),
                         num_features=k,
                         iter=iter,
                         selected_features=sel,
                         test_score=final_score,
                         best_cv_val_loss=best_cv_val_loss,
                         best_dropout=best_dropout,
                         best_l2=best_l2,
                         report_metrics=report_metrics)

                # checkpoint
                np.savez(checkpoint_file,
                         last_feature_step=step,
                         current_iter=iter,
                         current_feature=k,
                         results=results,
                         cv_val_losses=cv_val_losses,
                         results_indices=np.array(results_indices, dtype=object),
                         results_scores=np.array(results_scores, dtype=object),
                         seeds=seeds,
                         feature_sequence=feature_sequence,
                         timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                         per_step_report_metrics=np.array(all_report_metrics, dtype=object))

            # 每次迭代汇总
            report_metrics_iter_column = np.array([all_report_metrics[r][iter] for r in range(n_steps)], dtype=object)
            np.savez(os.path.join(iter_folder, f'full_results_iter_{iter}.npz'),
                     iter=iter,
                     test_results=results[:, iter],
                     cv_val_losses=cv_val_losses[:, iter],
                     feature_indices=feature_indices,
                     feature_scores=feature_scores,
                     feature_sequence=feature_sequence,
                     seed=seeds[iter],
                     timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                     report_metrics_column=report_metrics_iter_column)

            # 完成一次迭代后的 checkpoint
            np.savez(checkpoint_file,
                     last_completed_iter=iter,
                     last_feature_step=n_steps - 1,
                     results=results,
                     cv_val_losses=cv_val_losses,
                     results_indices=np.array(results_indices, dtype=object),
                     results_scores=np.array(results_scores, dtype=object),
                     seeds=seeds,
                     feature_sequence=feature_sequence,
                     timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                     per_step_report_metrics=np.array(all_report_metrics, dtype=object))

            iter_time = time.time() - iter_start
            logger.info(f"Iteration {iter+1} finished. Time: {format_time(iter_time)} ({iter_time:.2f}s)")

        except Exception as e:
            import traceback
            logger.error(f"Error in iteration {iter}: {e}")
            logger.error(traceback.format_exc())
            continue

    # 汇总并落地最终结果
    total_time = time.time() - total_start_time
    logger.info(f"\nAll iterations done. Total time: {format_time(total_time)} ({total_time:.2f}s)")

    mean_test_results = np.nanmean(results, axis=1)
    std_test_results  = np.nanstd(results, axis=1)
    mean_cv_val       = np.nanmean(cv_val_losses, axis=1)
    std_cv_val        = np.nanstd(cv_val_losses, axis=1)

    final_results = {
        'test_results': results,
        'cv_val_losses': cv_val_losses,
        'indices': np.array(results_indices, dtype=object),
        'scores': np.array(results_scores, dtype=object),
        'mean_test_results': mean_test_results,
        'std_test_results': std_test_results,
        'mean_cv_val_losses': mean_cv_val,
        'std_cv_val_losses': std_cv_val,
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
            'dfmax': dfmax,
        },
        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        'per_step_report_metrics': np.array(all_report_metrics, dtype=object),
    }

    save_path = os.path.join(folder_name, f'LASSO_{name}_results.npz')
    np.savez(save_path, **final_results)
    # ts_path = os.path.join(folder_name, f'LASSO_{name}_results_{final_results["timestamp"]}.npz')
    # np.savez(ts_path, **final_results)


    logger.info("\nFinal results summary:")
    for step, num_features in enumerate(feature_sequence):
        if step >= len(mean_test_results):
            continue
            
        num_features = int(num_features)
        logger.info(f"\n{num_features} features results:")
        logger.info(f"Average val_loss: {mean_cv_val[step]:.4f} ± {std_cv_val[step]:.4f}")
        logger.info(f"Average test score: {mean_test_results[step]:.4f} ± {std_test_results[step]:.4f}")
    logger.info(f"\nResults saved to: {save_path}")
    # logger.info(f"Timestamped copy saved to: {ts_path}")
    return final_results