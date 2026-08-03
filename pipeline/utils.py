import os
import sys
import logging
import numpy as np
import time
import secrets
import torch
import torch.nn as nn
import random
import json
from sklearn.model_selection import train_test_split


def get_activation(activation_name):
    """
    Get activation function by name
    """
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leakyrelu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
    }
    return activations.get(activation_name.lower(), nn.ReLU())

def resolve_hidden_dims(model_params):
    input_size = int(model_params['input_size'])

    if 'hidden_dims' in model_params and model_params['hidden_dims'] is not None:
        hidden_dims = [max(4, int(h)) for h in model_params['hidden_dims']]
        if len(hidden_dims) == 0:
            raise ValueError("model_params['hidden_dims'] must not be empty")
        expected_n_layers = max(2, int(model_params.get('n_layers', len(hidden_dims))))
        if len(hidden_dims) != expected_n_layers:
            raise ValueError(
                f"Length mismatch: n_layers={expected_n_layers}, but hidden_dims has {len(hidden_dims)} entries"
            )
        return hidden_dims

    if 'digits_list' in model_params and model_params['digits_list'] is not None:
        digits_list = [int(d) for d in model_params['digits_list']]
        if len(digits_list) == 0:
            raise ValueError("model_params['digits_list'] must not be empty")
        expected_n_layers = max(2, int(model_params.get('n_layers', len(digits_list))))
        if len(digits_list) != expected_n_layers:
            raise ValueError(
                f"Length mismatch: n_layers={expected_n_layers}, but digits_list has {len(digits_list)} entries"
            )
        return [max(4, input_size // max(1, d)) for d in digits_list]


    if 'start_digit' in model_params and model_params['start_digit'] is not None:
        start_digit = max(1, int(model_params['start_digit']))
        shrink_ratio = float(model_params.get('shrink_ratio', 2.0))
        n_layers = max(2, int(model_params.get('n_layers', 2)))
        digits_list = [
            max(1, int(round(start_digit * (shrink_ratio ** i))))
            for i in range(n_layers)
        ]
        return [max(4, input_size // max(1, d)) for d in digits_list]

    if 'digits' in model_params and model_params['digits'] is not None:
        digit = max(1, int(model_params['digits']))
        repeated_hidden_dim = max(4, input_size // digit)
        n_layers = max(2, int(model_params.get('n_layers', 2)))
        return [repeated_hidden_dim for _ in range(n_layers)]

    raise ValueError("Unable to resolve hidden_dims from model_params")


def setup_seed(seed):
    """
    set seed for reproducibility
    
    Args:
        seed: random seed
    """
    # set torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # set numpy seed
    np.random.seed(seed)

    # set python seed
    random.seed(seed)

    #  make sure to CUDNN behavior deterministic
    torch.backends.cudnn.deterministic = True

    # disable cudnn benchmark
    torch.backends.cudnn.benchmark = False

    # set python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)



def parse_list_arg(arg_str, dtype=float):
    """get args from string"""
    if not arg_str:
        return []
    return [dtype(x) for x in arg_str.split(',')]




def format_time(seconds):
    """Format seconds into a human-readable time string (hours, minutes, seconds)"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}" if h > 0 else f"{int(m):02d}:{int(s):02d}"


def setup_logger(out_dir: str, method: str, do_parameter_search: int, use_evaluation: int, seed: int,task_type,selected_activate):
    job_root = os.path.abspath(os.getcwd())
    log_dir = os.path.join(job_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    search_label = "param_search" if int(do_parameter_search) == 1 else "no_param_search"
    evaluation_label = "with_eval" if int(use_evaluation) == 1 else "no_eval"

    log_name = f"pipeline_{method}_{search_label}_{evaluation_label}_seed{seed}_{task_type}_{selected_activate}.log"
    log_path = os.path.join(log_dir, log_name)

    root_logger = logging.getLogger("")
    if getattr(root_logger, "_pipeline_handlers_set", False):
        return

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(fmt)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console)

    root_logger._pipeline_handlers_set = True
    root_logger.info(f"[LOG] writing to {log_path}")    



def _is_one_hot(y: np.ndarray) -> bool:
    if y.ndim != 2: return False
    if not np.all((y >= 0) & (y <= 1)): return False

    return np.allclose(y.sum(axis=1), 1.0, atol=1e-6)

def _to_class_index_from_one_hot(y: np.ndarray) -> np.ndarray:
    return np.argmax(y, axis=1).astype(np.int64)

def auto_detect_and_fix_labels(y_train, y_test, cli_task_type: str = "auto"):
    """
    according to the shape/value of y, automatically identify the task type and convert one-hot to integer labels if needed.
    Returns: task_type, y_train_fixed, y_test_fixed, n_classes
    """
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    # 1) one-hot -> multiclass
    if _is_one_hot(y_train) and _is_one_hot(y_test):
        y_train_fixed = _to_class_index_from_one_hot(y_train)
        y_test_fixed  = _to_class_index_from_one_hot(y_test)
        detected = "multiclass"
        n_classes = int(y_train.shape[1])
        print(f"[auto] detected one-hot -> multiclass (C={n_classes})")

    else:
        # try to cast near-integer floats to integers
        def try_cast_to_int(y):
            if np.issubdtype(y.dtype, np.integer):
                return y.astype(np.int64), True
            if np.allclose(y, np.round(y)):
                return np.round(y).astype(np.int64), True
            return y, False

        y_train_try, t_ok = try_cast_to_int(y_train)
        y_test_try,  v_ok = try_cast_to_int(y_test)

        if np.issubdtype(y_train_try.dtype, np.integer) and np.issubdtype(y_test_try.dtype, np.integer):
            y_train_fixed, y_test_fixed = y_train_try, y_test_try
            uniq = np.unique(np.concatenate([y_train_fixed, y_test_fixed]))
            n_classes = len(uniq)
            if n_classes <= 2 and set(uniq).issubset({0,1}):
                detected = "binary"
                print(f"[auto] detected integer labels -> binary, classes={uniq.tolist()}")
            elif n_classes >= 3:
                detected = "multiclass"
                print(f"[auto] detected integer labels -> multiclass, classes={uniq.tolist()}")
            else:
                print(f"[Error] Invalid labels. classes={uniq.tolist()}")
                sys.exit(1)
        else:
            # continuous values (or few unique floats)
            uniq = np.unique(np.concatenate([y_train, y_test]))
            if len(uniq) > 20:
                detected = "regression"
                y_train_fixed = y_train.astype(np.float32)
                y_test_fixed  = y_test.astype(np.float32)
                n_classes = 1
                print(f"[auto] detected float-continuous -> regression "
                        f"(train range {y_train.min():.4f}~{y_train.max():.4f})")
            else:
                # few unique values, but stored as float; treat as classification
                mapping = {v: i for i, v in enumerate(sorted(uniq))}
                y_train_fixed = np.vectorize(mapping.get)(y_train).astype(np.int64)
                y_test_fixed  = np.vectorize(mapping.get)(y_test).astype(np.int64)
                n_classes = len(uniq)
                detected = "binary" if n_classes == 2 else "multiclass"
                print(f"[auto] detected few unique floats -> {detected}, "
                        f"classes(mapped)={list(range(n_classes))} from uniq={uniq.tolist()}")

    # with CLI forcing consistency check
    if cli_task_type != "auto" and cli_task_type != detected:
        print(f"[Error] task_type mismatch: CLI='{cli_task_type}' but detected='{detected}'.")
        sys.exit(1)

    final_task_type = detected if cli_task_type == "auto" else cli_task_type
    return final_task_type, y_train_fixed, y_test_fixed, (n_classes if final_task_type != "regression" else 1)


def _impute_missing(X, strategy, logger):
    """
    X: numpy array, may contain NaN
    strategy: 'none' | 'mean' | 'zero' | 'mode'
    """
    if strategy == 'none':
        if np.isnan(X).any():
            raise ValueError(
                "Detected NaN in X but impute_strategy='none'. "
                "Please choose an imputation method in the web UI."
            )
        logger.info("[Impute] No NaN detected in X, skip imputation.")
        return X

    X = X.astype(np.float32, copy=True)
    logger.info(f"[Impute] Strategy = {strategy}")

    if strategy == 'zero':
        X[np.isnan(X)] = 0.0
        return X

    if strategy == 'mean':
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = col_means[inds[1]]
        return X

    if strategy == 'mode':
        n_features = X.shape[1]
        for j in range(n_features):
            col = X[:, j]
            mask = np.isnan(col)
            if not np.any(mask):
                continue
            vals = col[~mask]
            if vals.size == 0:
                fill = 0.0
            else:
                uniq, cnt = np.unique(vals, return_counts=True)
                fill = uniq[np.argmax(cnt)]
            col[mask] = fill
            X[:, j] = col
        return X

    # Shouldn't reach here theoretically
    return X

def _remove_constant_features_train_based(
    x_train,
    x_test,
    logger,
    save_path=None,
    atol=1e-12,
    y_train=None,
    remove_by_class=1,
):
    """
    Remove features based on training-set constancy.

    Modes:
    - remove_by_class=0: remove features whose variance on the whole training set is zero
      (or numerically close to zero).
    - remove_by_class=1: remove features if they are constant within ANY class in y_train.

    The same columns are removed from both train and test to avoid leakage.

    Returns:
        x_train_filtered, x_test_filtered, removed_indices, kept_indices
    """
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError(
            f"x_train and x_test must be 2D arrays, got shapes {x_train.shape} and {x_test.shape}"
        )

    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            f"x_train and x_test must have the same number of features, got {x_train.shape[1]} and {x_test.shape[1]}"
        )

    if remove_by_class not in (0, 1):
        raise ValueError(f"remove_by_class must be 0 or 1, got {remove_by_class}")

    train_var = np.var(x_train, axis=0)

    if remove_by_class == 0:
        remove_mask = np.isclose(train_var, 0.0, atol=atol)
        mode_name = "whole-train"
    else:
        if y_train is None:
            raise ValueError("y_train must be provided when remove_by_class=1")

        y_train = np.asarray(y_train).reshape(-1)
        if y_train.shape[0] != x_train.shape[0]:
            raise ValueError(
                f"y_train length must match x_train rows, got {y_train.shape[0]} and {x_train.shape[0]}"
            )

        classes = np.unique(y_train)
        remove_mask = np.zeros(x_train.shape[1], dtype=bool)
        class_constant_details = {}

        for cls in classes:
            cls_mask = (y_train == cls)
            cls_count = int(np.sum(cls_mask))
            if cls_count <= 1:
                cls_remove_mask = np.ones(x_train.shape[1], dtype=bool)
            else:
                cls_var = np.var(x_train[cls_mask], axis=0)
                cls_remove_mask = np.isclose(cls_var, 0.0, atol=atol)

            if np.any(cls_remove_mask):
                class_constant_details[str(cls)] = np.where(cls_remove_mask)[0].astype(np.int64).tolist()
            remove_mask |= cls_remove_mask

        mode_name = "per-class"

    kept_mask = ~remove_mask
    removed_indices = np.where(remove_mask)[0].astype(np.int64)
    kept_indices = np.where(kept_mask)[0].astype(np.int64)

    if removed_indices.size == 0:
        logger.info(f"[FeatureFilter] No constant features detected on training set ({mode_name} mode).")
    else:
        logger.info(
            f"[FeatureFilter] Removing {removed_indices.size} constant features based on training set "
            f"({mode_name} mode). Example removed indices: {removed_indices[:20].tolist()}"
        )
        if remove_by_class == 1 and 'class_constant_details' in locals():
            preview = {k: v[:20] for k, v in class_constant_details.items() if len(v) > 0}
            logger.info(f"[FeatureFilter] Per-class constant feature preview: {preview}")

    x_train_filtered = x_train[:, kept_mask]
    x_test_filtered = x_test[:, kept_mask]

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        payload = {
            "n_original_features": int(x_train.shape[1]),
            "n_kept_features": int(kept_indices.size),
            "n_removed_features": int(removed_indices.size),
            "removed_indices": removed_indices.tolist(),
            "kept_indices": kept_indices.tolist(),
            "train_variance": train_var.astype(float).tolist(),
            "atol": float(atol),
            "remove_by_class": int(remove_by_class),
            "mode": mode_name,
        }
        if remove_by_class == 1 and 'class_constant_details' in locals():
            payload["class_constant_details"] = class_constant_details
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"[FeatureFilter] Saved constant-feature report to {save_path}")

    return x_train_filtered, x_test_filtered, removed_indices, kept_indices

def _scale_features(x_train, x_test, scaler, logger):
    """
    scaler: 'none' | 'zscore' | 'minmax'
    """
    if scaler == 'none':
        logger.info("[Scale] scaler='none', skip scaling.")
        return x_train, x_test

    x_train = x_train.astype(np.float32, copy=True)
    x_test  = x_test.astype(np.float32, copy=True)

    logger.info(f"[Scale] Strategy = {scaler}")

    if scaler == 'zscore':
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        std_safe = std.copy()
        std_safe[std_safe == 0] = 1.0
        x_train = (x_train - mean) / std_safe
        x_test  = (x_test  - mean) / std_safe
        return x_train, x_test

    if scaler == 'minmax':
        minv = np.min(x_train, axis=0)
        maxv = np.max(x_train, axis=0)
        rangev = maxv - minv
        rangev[rangev == 0] = 1.0
        x_train = (x_train - minv) / rangev
        x_test  = (x_test  - minv) / rangev
        return x_train, x_test

    return x_train, x_test

def _is_probably_classification(y, logger):
    """
    Heuristically determine whether `y` is a classification target:
    - int/bool types -> treat as classification
    - float with few unique values -> treat as classification
    Used only to decide whether to stratify during splitting; final task type
    is determined later by `auto_detect_and_fix_labels`.
    """
    y = np.asarray(y)
    n = y.shape[0]
    uniq = np.unique(y)
    n_unique = len(uniq)

    if np.issubdtype(y.dtype, np.integer) or y.dtype == bool:
        logger.info(f"[GuessTask] y dtype={y.dtype}, treat as classification (n_classes={n_unique}).")
        return True

    # float 但是唯一值很少（比如 ≤ 50 且远小于样本数），大概率是多分类
    if n_unique <= min(50, n // 2):
        logger.info(f"[GuessTask] y seems low-cardinality (n_unique={n_unique}), treat as classification.")
        return True

    logger.info(f"[GuessTask] y seems continuous (n_unique={n_unique}), treat as regression.")
    return False

def _split_train_test(
    X,
    y,
    train_ratio,
    seed,
    logger,
    classification=None,
    drop_constant_features=False,
    constant_feature_report_path=None,
    constant_feature_atol=1e-12,
    remove_constant_by_class=1,
):
    """
    Split using sklearn.model_selection.train_test_split:
    - If `classification=True`, use `stratify=y`
    - Otherwise do not stratify
    - Optional: remove constant features based on training-set variance and save a report
    """
    n = X.shape[0]
    if n != len(y):
        raise ValueError(f"X and y sample size mismatch: {n} vs {len(y)}")

    if not (0 < train_ratio <= 1.0):
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")

    # Need a test_size
    test_size = 1.0 - train_ratio

    # If almost no test set remains or too few samples, copy train as test (no independent test set)
    if n < 2 or test_size <= 0:
        logger.warning(
            f"[Split] train_ratio={train_ratio} or n={n} → "
            "no real test set, test will be a copy of train. Evaluation will be disabled."
        )
        x_train, x_test, y_train, y_test = X, X.copy(), y, y.copy()
        if drop_constant_features:
            x_train, x_test, _, _ = _remove_constant_features_train_based(
                x_train,
                x_test,
                logger,
                save_path=constant_feature_report_path,
                atol=constant_feature_atol,
                y_train=y_train,
                remove_by_class=remove_constant_by_class,
            )
        return x_train, x_test, y_train, y_test, False

    if classification is None:
        classification = _is_probably_classification(y, logger)

    logger.info(
        f"[Split] Using sklearn.train_test_split, "
        f"train_ratio={train_ratio}, test_size={test_size:.3f}, "
        f"classification={classification}"
    )

    try:
        if classification:
            x_train, x_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=seed,
                stratify=y
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=seed,
                stratify=None
            )
    except ValueError as e:
        # For example, stratify may fail due to too few samples in a class; fallback to non-stratified split
        logger.warning(
            f"[Split] train_test_split with stratify failed ({e}), "
            "fallback to non-stratified split."
        )
        x_train, x_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            stratify=None
        )
        classification = False  # stratification not actually used

    if drop_constant_features:
        x_train, x_test, removed_indices, kept_indices = _remove_constant_features_train_based(
            x_train,
            x_test,
            logger,
            save_path=constant_feature_report_path,
            atol=constant_feature_atol,
            y_train=y_train,
            remove_by_class=remove_constant_by_class,
        )
        logger.info(
            f"[Split] Constant-feature filtering done: kept={len(kept_indices)}, removed={len(removed_indices)}, "
            f"remove_by_class={remove_constant_by_class}"
        )

    logger.info(
        f"[Split] Done: n_train={x_train.shape[0]}, n_test={x_test.shape[0]}, "
        f"n_features={x_train.shape[1]}, stratified={classification}"
    )

    return x_train, x_test, y_train, y_test, True