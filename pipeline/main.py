import numpy as np
import random
import os
import torch
import warnings
import argparse
import subprocess
import os
import shutil
import sys
import logging
import json
import logging, os


from utils import setup_logger, auto_detect_and_fix_labels,setup_seed,_split_train_test,_scale_features,_impute_missing
from EARFS_model import EARFS_fs,EARFS_fs_with_evaluation
from CANCELOUT_model import CANCELOUT_fs, CANCELOUT_fs_with_evaluation
# from lasso import lasso_fs, LASSO_with_evaluation
from GRACES import GRACES_fs, GRACES_fs_with_evaluation
from FSDL_model import FSDL,FSDL_with_evaluation
from bcor_evaluation import BCOR_with_evaluation
from f_test import FTEST_fs, FTEST_fs_with_evaluation
from RF import RF_fs, RF_fs_with_evaluation
from LASSO_evaluation import LASSO_with_evaluation


def parse_arguments():

    #####input files and output directory#####
    parser = argparse.ArgumentParser(description='Feature Selection Analysis')
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=False,
        help='Path to uploaded .npz file (container mounts /work).'
    )
    parser.add_argument(
        '--out_dir', 
        type=str, 
        default='result',
        help='Output root directory (relative to CWD, default: result).'
    )
    parser.add_argument(
        '--name', 
        type=str, 
        default='adni_brainstem',
        # choices=['ALLAML','colon','GLI_85', 'leukemia','Prostate_GE', 'SMK_CAN_187'],
        help='Dataset identifier'
    )
    parser.add_argument(
        '--method', 
        type=str, 
        default='EARFS',
        choices=['LASSO','GRACES','EARFS', 'CANCELOUT','DeepLIFT', 'GradientShap', 
                 'LRP', 'FeatureAblation', 'Occlusion', 'Lime', 'BCOR', 
                 'FTEST', 'RF'],  
        help='Feature selection method to use'
    )

    ###################Model options#######################
    # 添加命令行参数
    # ======== New preprocessing options ========
    parser.add_argument(
        '--preprocess_mode',
        type=str,
        default='auto',
        choices=['auto', 'external'],
        help=(
            "Data preprocessing mode:\n"
            "  - 'auto': backend will impute, scale and split X,y into train/test.\n"
            "  - 'external': user has already done splitting/imputation/scaling and "
            "provides x_train/x_test/y_train/y_test directly."
        )
    )    
    parser.add_argument(
        '--impute_strategy',
        type=str,
        default='none',
        choices=['none', 'mean', 'zero', 'mode'],
        help='Missing value imputation strategy for X: none / mean / zero / mode.'
    )

    parser.add_argument(
        '--scaler',
        type=str,
        default='none',
        choices=['none', 'zscore', 'minmax'],
        help='Feature scaling strategy for X: none / zscore / minmax.'
    )

    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Proportion of samples used for training (0 < train_ratio ≤ 1.0).'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--is_snp',
        type=int,
        default=0,
        choices=[0, 1],
        help='Set 1 if input is SNP data (stored as int8); set 0 for other data (stored as float32).'
    )
    
    parser.add_argument(
        '--task_type',
        type=str,
        default='auto',
        choices=['auto','binary','multiclass','regression'],
        help='Task type. Use "auto" to infer from labels (one-hot/int/float).'
    )
    
    parser.add_argument(
        '--selected_activate',
        type=str,
        default='relu',
        choices=['relu','tanh','sigmoid','leakyrelu','elu','gelu'],
        help='Activation function used in the model.'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Max training epochs (early stopping may end earlier if enabled).'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Patience for early stopping (number of epochs with no improvement).'
    )
    parser.add_argument(
        '--min_delta',
        type=float,
        default=0.0,
        help='Minimum change in the monitored metric to qualify as an improvement in early stopping.'
    )

    parser.add_argument(
        '--validation_split',
        type=float,
        default=0.2,
        help='Fraction of training data to use as validation set for early stopping.'
    )
    parser.add_argument(
        '--n_iters',
        type=int,
        default=1,
        help='Number of iterations for feature selection (some methods may use this).'
    )
    ################### Evaluation options #######################
    parser.add_argument(
        '--use_evaluation',
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            'Whether to run feature evaluation curves. '
            '0 = feature selection only; 1 = progressively add features and evaluate on the test set.'
        )
    )
    
    parser.add_argument(
        '--max_features',
        type=int,
        default=100,
        help=(
            'Maximum number of top features to evaluate (upper bound for the curve).'
        )
    )
    
    parser.add_argument(
        '--feature_step',
        type=int,
        default=100,
        help=(
            'Step size for the number of features when sweeping the evaluation curve '
            '(e.g., 10 means evaluate at 10, 20, 30, ...).'
        )
    )
    
    parser.add_argument(
        '--max_features_graces', 
        type=int,
        default=100,
        help='Maximum number of top features to select in GRACES (default: 100).'
    )


    ###################Optuna hyperparameter options#######################
 
    parser.add_argument(
        '--do_parameter_search', 
        type=int, 
        default=0,
        choices=[0, 1],
        help='Whether to perform Optuna parameter optimization (1) or use single parameter values (0)'
    )
    parser.add_argument(
        '--eval_metric', 
        type=str, default='loss',
        choices=['acc', 'r2', 'mse', 'mae', 'loss'],
        help='Evaluation metric for Optuna parameter optimization (default: acc for classification, r2 for regression)'
    )
    parser.add_argument(
        '--n_trials', 
        type=int, default=20,
        help='Number of Optuna trials to run'
    )
    parser.add_argument(
        '--n_jobs', 
        type=int, default=1,
        help='Number of parallel jobs for Optuna optimization'
    )

    #model parameters#
    parser.add_argument(
        '--batch_size_list', 
        type=str, default='8,16,32',
        help='Comma-separated list of batch size values for Optuna search'
    )
    
    parser.add_argument(
        '--digits_list',
        type=str,
        default='100,200,500,1000',
        help=(
            "Comma-separated list of divisor values for Optuna search. "
            "The hidden layer size is computed as input_dim / digits. "
            "For example, if input_dim=20,000 and digits=100, then hidden_size=200."
        )
    )
    parser.add_argument(
        '--n_splits',
         type=int, default=5,
        help='Number of folds for cross-validation during Optuna search'
    )
    parser.add_argument(
        '--lr_min',
        type=float,
        default=1e-5,
        help='Minimum value for learning rate in Optuna search'
    )
    parser.add_argument(
        '--lr_max',
        type=float,
        default=1e-1,
        help='Maximum value for learning rate in Optuna search'
    )
    parser.add_argument(
        '--weight_decay_min',
        type=float,
        default=0.0,
        help='Minimum value for weight_decay in Adam of Optuna search'
    )
    parser.add_argument(
        '--weight_decay_max',
        type=float,
        default=1e-1,
        help='Maximum value for weight_decay in Adam of Optuna search'
    )

    parser.add_argument(
        '--dropout_min',
        type=float,
        default=0.2,
        help='Minimum value for dropout in Optuna search'
    )
    parser.add_argument(
        '--dropout_max',
        type=float,
        default=0.8,
        help='Maximum value for dropout in Optuna search'
    )





    #####CANCELOUT specific hyperparameter options######
    parser.add_argument(
        '--lambda_1_min',
        type=float,
        default=1e-5,
        help='Minimum value for lambda_1 in Optuna search'
    )
    parser.add_argument(
        '--lambda_1_max',
        type=float,
        default=1e-1,
        help='Maximum value for lambda_1 in Optuna search'
    )

    parser.add_argument(
        '--lambda_2_min',
        type=float,
        default=1e-5,
        help='Minimum value for lambda_2 in Optuna search'
    )
    parser.add_argument(
        '--lambda_2_max',
        type=float,
        default=1e-1,
        help='Maximum value for lambda_2 in Optuna search'
    )

    parser.add_argument(
        '--cancelout_init',
        type=float,
        default=None,
        help='Initial value for CancelOut weights (if None, will be determined by activation function)'
    )
    parser.add_argument(
        '--search_cancelout_init',
        type=int,
        default=0,
        choices=[0, 1],
        help='Whether to search for optimal CancelOut initialization (1) or use function-based default (0)'
    )

    

    
    ####EARFS specific hyperparameter options######
    parser.add_argument(
        '--lambda_fs_min', 
        type=float, 
        default=1e-5,
        help='Minimum value for lambda_fs in Optuna search for EAR-FS method'
    )
    parser.add_argument(
        '--lambda_fs_max',
        type=float,
        default=1e-1,
        help='Maximum value for lambda_fs in Optuna search for EAR-FS method'
    )


    ####GRACES specific hyperparameter options######
    parser.add_argument(
        '--alpha_min',
        type=float,
        default=0.8,
        help=("Lower bound of alpha for Optuna search. "
            "In GRACES, alpha is the threshold for constructing the cosine similarity graph: "
            "cosine similarity > alpha is set to 1 (edge), otherwise 0."))
    
    parser.add_argument(
        '--alpha_max',
        type=float,
        default=0.99,
        help=("Upper bound of alpha for Optuna search. "
            "In GRACES, alpha is the threshold for constructing the cosine similarity graph: "
            "cosine similarity > alpha is set to 1 (edge), otherwise 0."))
    
    parser.add_argument(
        '--f_correct_list',
        type=str,
        default='0,0.1,0.5,0.9',
        help=("Comma-separated list of f_correct values for GRACES. "
            "The coefficient balances the feature score derived from the graph structure "
            "with the feature score derived from the F-test."))


     ###################Fixed hyperparameter options#######################
    # 训练参数
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for training'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )

    parser.add_argument(
        '--digits',
        type=int,
        default=100,
        help='Number to divide feature dimension by for hidden size'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout probability'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay (L2 regularization) for middle layers'
    )

    #LASSO specifiy parameter
    parser.add_argument(
        '--initial_max_iter',
        type=int,
        default=1000,
        help='Initial maximum iterations for LASSO'
    )
    parser.add_argument(
        '--max_attempts',
        type=int,
        default=1,
        help='Maximum attempts for LASSO'
    )
    parser.add_argument(
        '--iter_multiplier',
        type=int,
        default=2,
        help='Iteration multiplier for LASSO'
    )
    parser.add_argument('--dfmax', 
                        type=int, 
                        default=3000, 
                        help='glmnet dfmax limit for LASSO (R)'
    )
    
    # EARFS 特定参数
    parser.add_argument(
        '--lambda_fs',
        type=float,
        default=1e-2,
        help='Feature selection regularization for EARFS'
    )
    
    # CANCELOUT 特定参数
    parser.add_argument(
        '--lambda_1',
        type=float,
        default=0.001,
        help='L1 regularization for CANCELOUT'
    )
    parser.add_argument(
        '--lambda_2',
        type=float,
        default=0.001,
        help='Variance regularization for CANCELOUT'
    )

    
    parser.add_argument(
        '--f_correct',
        type=float,
        default=0.5,
        help=(
            "Balancing coefficient used by GRACES when computing the final feature score: "
            "it trades off the graph-derived feature score against the F-test feature score. "
            "Same semantics as values in --f_correct_list."
        )
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.95,
        help=(
            "Cosine-similarity threshold for building the graph in GRACES (same semantics as "
            "--alpha_min/--alpha_max). Pairs with cosine similarity > alpha are set to 1 (edge), "
            "otherwise 0."
        )
    )
    
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.0,
        help=(
            "Variance of the Gaussian noise matrix introduced in GRACES. "
            "This controls the strength of noise injection when computing feature scores. "
            "Set to 0.0 to disable noise."
        )
    )
    
    parser.add_argument(
        '--n_dropouts',
        type=int,
        default=10,
        help=(
            "Number of stochastic dropout passes used when computing gradient-based feature scores. "
            "Larger values stabilize the estimate but increase runtime. Typical range: 5–20."
        )
    )
    
    parser.add_argument(
        '--q',
        type=int,
        default=2,
        help=(
            "q-norm applied to gradients when aggregating per-feature importance in GRACES. "
            "Examples: q=1 (L1), q=2 (L2), large q approximates L∞ (max)."
        )
    )
    
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        default_n_jobs = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', 1)) - 2)
        parser.set_defaults(n_jobs=default_n_jobs) 


    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_logger(args.out_dir, args.method, args.do_parameter_search, args.use_evaluation)
    logger = logging.getLogger(f"pipeline_{args.method}_{args.do_parameter_search}_{args.use_evaluation}")
    logger.info("Starting pipeline...")


    class _CompatLogger:
        def info(self, *a, **k): logger.info(*a, **k)
        def error(self, *a, **k): logger.error(*a, **k)
        def warning(self, *a, **k): logger.warning(*a, **k)
        def debug(self, *a, **k): logger.debug(*a, **k)

    compat_logger = _CompatLogger()



    compat_logger.info("Starting pipeline...")

    # environment settings
    cpu_cores = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
    compat_logger.info(f"The number of CPU: {cpu_cores}")



    # set random seed
    seed = args.seed  # or any other base seed you prefer
    setup_seed(seed)
    
    # get current path and parent folder name
    current_path = os.getcwd()
    parent_folder = os.path.basename(current_path)




    out_root = os.path.abspath(args.out_dir)  # -> /work/result

    folder_name = os.path.join(
        out_root,
        f"{args.method}_{args.name}_{args.selected_activate}",
    )
    # clean up old folder and create a new one
    if os.path.exists(folder_name):
        compat_logger.info(f"folder {folder_name} exists, deleting...")
        shutil.rmtree(folder_name)
        compat_logger.info(f"folder {folder_name} deleted")
    os.makedirs(folder_name, exist_ok=True)
    compat_logger.info(f"folder {folder_name} created")


    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")



    # start data processing
    compat_logger.info(f"Processing the dataset {args.name}")


    # Use --input_path if provided. else, use the default dataset path.
    if args.input_path:
        split_path = args.input_path
    else:
        split_path = os.path.join("./data", f"{args.name}.npz")

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"dataset not found: {split_path}")

    # ############ Load data from .npz file ############
    if not split_path.lower().endswith(".npz"):
        raise ValueError(f"only .npz files are supported, got: {split_path}")

    pack = np.load(split_path, allow_pickle=True)
    files = set(pack.files)

    # -------------------------------------------------
    #  优先：新格式  —  只包含 X, y（前端要求）
    #  兼容：旧格式  —  x_train,y_train,x_test,y_test
    # -------------------------------------------------

    if args.preprocess_mode == 'auto':    
        if ('X' in files) and ('y' in files):
            compat_logger.info("[Load] Detected new format with keys: X, y")

            X = pack['X']
            y = pack['y']

            # y 展平到 1D（大多数情况）
            if y.ndim > 1 and y.shape[1] == 1:
                y = y.reshape(-1)
            # multi-output 的情况交给 auto_detect_and_fix_labels 处理

            compat_logger.info(f"[Load] X shape: {X.shape}, dtype: {X.dtype}")
            compat_logger.info(f"[Load] y shape: {y.shape}, dtype: {y.dtype}")

            # ------- 缺失值插补 -------
            if np.isnan(X).any():
                compat_logger.info(
                    f"[Check] Detected NaN in X, using impute_strategy={args.impute_strategy}"
                )
                X = _impute_missing(X, args.impute_strategy, compat_logger)
            else:
                if args.impute_strategy != 'none':
                    compat_logger.info(
                        "[Impute] No NaN detected in X, imputation skipped "
                        f"(impute_strategy={args.impute_strategy})."
                    )

            # ------- 划分 train/test -------
            x_train, x_test, y_train, y_test, original_has_test = _split_train_test(
                X, y, args.train_ratio, seed, compat_logger
            )

            # ------- 标准化（只用 train 计算参数） -------
            if args.is_snp and args.scaler != 'none':
                compat_logger.warning(
                    "[Scale] is_snp=1 且 scaler!=none，通常不建议对 SNP 做连续型缩放，将跳过缩放。"
                )
            else:
                x_train, x_test = _scale_features(
                    x_train, x_test, args.scaler, compat_logger
                )

            # 如果没有真实 test 集，则禁用 evaluation
            if not original_has_test and args.use_evaluation == 1:
                compat_logger.warning(
                    "[Eval] train_ratio=1 或样本太少，无法构造独立测试集，强制 use_evaluation=0。"
                )
                args.use_evaluation = 0

        else:
            # ============ 兼容旧格式：x_train/y_train(/x_test/y_test) ============
            compat_logger.info(
                "[Load] Fallback to legacy format: x_train,y_train,(x_test,y_test)"
            )

            has_train = ('x_train' in files) and ('y_train' in files)
            has_test  = ('x_test'  in files) and ('y_test'  in files)

            if not has_train:
                missing = []
                if 'x_train' not in files: missing.append('x_train')
                if 'y_train' not in files: missing.append('y_train')
                raise KeyError(
                    "Legacy format: .npz file is missing required keys: "
                    + ', '.join(missing)
                )

            if has_train and not has_test:
                x_train = pack['x_train']; y_train = pack['y_train']
                x_test, y_test = x_train, y_train

                original_has_test = False
                if getattr(args, 'use_evaluation', 0) != 0:
                    compat_logger.warning(
                        "Legacy format only has x_train / y_train, no x_test / y_test. "
                        "Forced use_evaluation=0."
                    )
                args.use_evaluation = 0
                compat_logger.info(
                    "Legacy format: only x_train / y_train, "
                    "so we can only perform feature selection, no final evaluation."
                )

            else:
                x_train = pack['x_train']; y_train = pack['y_train']
                x_test  = pack['x_test'];  y_test  = pack['y_test']
                original_has_test = True

                if getattr(args, 'use_evaluation', 0) == 0:
                    compat_logger.info(
                        "Legacy format detected 4 keys (x_train/y_train/x_test/y_test) "
                        "but use_evaluation=0, evaluation will be skipped."
                    )

            compat_logger.info(f"[Load] x_train shape: {x_train.shape}, dtype: {x_train.dtype}")
            compat_logger.info(f"[Load] y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
            if original_has_test:
                compat_logger.info(f"[Load] x_test shape: {x_test.shape}, dtype: {x_test.dtype}")
                compat_logger.info(f"[Load] y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
    else:
        compat_logger.info("[Preprocess] preprocess_mode='external': skip internal split/impute/scale.")
        
        # 这里我们直接要求 legacy 格式：
        has_train = ('x_train' in files) and ('y_train' in files)
        has_test  = ('x_test' in files) and ('y_test' in files)

        if not has_train:
            raise KeyError(
                "[Preprocess=external] Expect keys x_train,y_train(,x_test,y_test) "
                "because user claims data are already split and preprocessed."
            )

        x_train = pack['x_train']; y_train = pack['y_train']

        if args.use_evaluation == 1:
            if not has_test:
                raise KeyError(
                    "[Preprocess=external] use_evaluation=1 but no x_test/y_test found. "
                    "Please provide an independent test set or set use_evaluation=0."
                )
            x_test = pack['x_test']; y_test = pack['y_test']
            original_has_test = True
        else:
            # 没有 test 也没关系，eval 本来就关着
            x_test, y_test = x_train, y_train
            original_has_test = False

        compat_logger.info(
            "[Preprocess=external] Using user-provided preprocessed splits "
            "(no further imputation or scaling will be applied)."
        )        
    # -------------------------------------------------
    # after reading and branching logic: construct auxiliary flags
    # -------------------------------------------------
    # original_has_test = ('x_test' in files) and ('y_test' in files)
    
    causal_variants = np.array([0, 1]) ##not import here, just for placeholder

    # —— automatic label detection/fixing (requires y_train / y_test, both guaranteed to exist upstream)
    # If no test set is available, we previously set it as a copy of the train set for placeholder

    args.task_type, y_train, y_test, num_label = auto_detect_and_fix_labels(
        y_train, y_test, cli_task_type=args.task_type
    )
    compat_logger.info(f"The number of classes {num_label}")
    
    # data checking
    if args.task_type == "binary":
        compat_logger.info(f"[Binary] positives - train: {int(y_train.sum())}"
              + (f", test: {int(y_test.sum())}" if original_has_test else ""))
    elif args.task_type == "multiclass":
        ut, ct = np.unique(y_train, return_counts=True)
        compat_logger.info(f"[Multiclass] train: {dict(zip(ut.tolist(), ct.tolist()))}")
        if original_has_test:
            uv, cv = np.unique(y_test, return_counts=True)
            compat_logger.info(f"[Multiclass] test: {dict(zip(uv.tolist(), cv.tolist()))}")
    else:
        tr_min, tr_max = float(np.min(y_train)), float(np.max(y_train))
        line = f"[Regression] train range: {tr_min:.4f} ~ {tr_max:.4f}"
        if original_has_test:
            te_min, te_max = float(np.min(y_test)), float(np.max(y_test))
            line += f"\n[Regression] test  range: {te_min:.4f} ~ {te_max:.4f}"
        compat_logger.info(line)

    compat_logger.info(f"[Split] name={args.name}")
    compat_logger.info(f"The shape of x_train: {x_train.shape}, the dtype of x_train: {x_train.dtype}")
    compat_logger.info(f"The shape of y_train: {y_train.shape}, the dtype of y_train: {y_train.dtype}")
    if original_has_test:
        compat_logger.info(f"The shape of x_test: {x_test.shape}, the dtype of x_test: {x_test.dtype}")
        compat_logger.info(f"The shape of y_test: {y_test.shape}, the dtype of y_test: {y_test.dtype}")


    # -------------------------------------------------
    # set data to tensors on the target device,
    # -------------------------------------------------
    if args.task_type in ('binary', 'multiclass'):
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
        y_test_t  = torch.tensor(y_test,  dtype=torch.long, device=device)
    else:
        # continue regression: change to column vector
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
        y_test_t  = torch.tensor(y_test,  dtype=torch.float32, device=device).view(-1, 1)
    

    # caution⚠️: This is only for snp or some data that are originally integers to save memory
    if args.is_snp:
        x_train_t = torch.tensor(x_train, dtype=torch.int8, device=device)
        x_test_t  = torch.tensor(x_test,  dtype=torch.int8, device=device)
    else:
        x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        x_test_t  = torch.tensor(x_test,  dtype=torch.float32, device=device)
    

    data = {
        'X_train': x_train_t,
        'X_test':  x_test_t,   # Note: if no original test set, this is just a copy of train
    }
    label = {
        'y_train': y_train_t,
        'y_test':  y_test_t,   # Same as above
    }
    print(x_train_t.shape)
    # get model input dimension
    model_input_dim = x_train_t.shape[1]
    feature_prediction = getattr(args, 'max_features', 2500)
    compat_logger.info(f"Model input dimension: {model_input_dim}")
    if args.use_evaluation == 1:
        compat_logger.info(f"Feature evaluation upper limit: {feature_prediction}")

    compat_logger.info(f"Using Method: {args.method}")


    if args.method == 'CANCELOUT':
        # model parameters
        model_params = {
            'input_size': model_input_dim,
            'hidden_size': int(model_input_dim / args.digits),
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'is_snp':args.is_snp,
            'digits': args.digits  
        }

        # training parameters
        training_params = {
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'validation_split': args.validation_split if hasattr(args, 'validation_split') else 0.2,
            'patience': args.patience if hasattr(args, 'patience') else 10,
            'min_delta': args.min_delta if hasattr(args, 'min_delta') else 0.0,
            'device': device,
            'seed': seed,
            'lambda_1': args.lambda_1,
            'lambda_2': args.lambda_2,
            'weight_decay': args.weight_decay,  
            'n_splits': args.n_splits,
            'do_parameter_search': args.do_parameter_search, 
            'eval_metric': args.eval_metric,
            'n_trials': args.n_trials,
            'n_jobs': args.n_jobs  
        }


        # if user want to do parameter search, add the search space to training_params
        if args.do_parameter_search == 1:
            compat_logger.info(f"  n_jobs: {args.n_jobs} (parallelism)")

            # additional log if n_jobs is set from SLURM_CPUS_PER_TASK
            if 'SLURM_CPUS_PER_TASK' in os.environ and args.n_jobs == int(os.environ.get('SLURM_CPUS_PER_TASK')):
                compat_logger.info(f"  n_jobs was automatically set from SLURM_CPUS_PER_TASK")

            training_params.update({
                'lambda_1_min': args.lambda_1_min,
                'lambda_1_max': args.lambda_1_max,
                'lambda_2_min': args.lambda_2_min,
                'lambda_2_max': args.lambda_2_max,
                'weight_decay_min': args.weight_decay_min,
                'weight_decay_max': args.weight_decay_max,
                'lr_min': args.lr_min,
                'lr_max': args.lr_max,
                'dropout_min': args.dropout_min,
                'dropout_max': args.dropout_max,
                'batch_size_list': args.batch_size_list,
                'digits_list': args.digits_list,
                'search_cancelout_init': args.search_cancelout_init,
                'cancelout_init': args.cancelout_init if args.cancelout_init is not None else 0.1,
            })
            compat_logger.info("Optuna parameter optimization enabled.")
            compat_logger.info(f"Parameter search settings:")
            compat_logger.info(f"  n_trials: {args.n_trials}")
            compat_logger.info(f"  lambda_1 range: [{args.lambda_1_min}, {args.lambda_1_max}]")
            compat_logger.info(f"  lambda_2 range: [{args.lambda_2_min}, {args.lambda_2_max}]")
            compat_logger.info(f"  weight_decay range: [{args.weight_decay_min}, {args.weight_decay_max}]")  
            compat_logger.info(f"  lr range: [{args.lr_min}, {args.lr_max}]")
            compat_logger.info(f"  dropout range: [{args.dropout_min}, {args.dropout_max}]")
            compat_logger.info(f"  batch_size options: {args.batch_size_list}")
            compat_logger.info(f"  digits options: {args.digits_list}")
            compat_logger.info(f"  n_splits: {args.n_splits}")
            compat_logger.info(f"  eval_metric: {args.eval_metric} (default if None)")
            compat_logger.info(f"  search_cancelout_init: {args.search_cancelout_init}")  
            if args.search_cancelout_init == 1:
                compat_logger.info(f"  cancelout_init (default if search_cancelout_init=1): {args.cancelout_init}")
        else:
            compat_logger.info("Parameter optimization disabled. Using single parameter values.")
            compat_logger.info(f"Using parameters:")
            compat_logger.info(f"  lambda_1: {args.lambda_1}")
            compat_logger.info(f"  lambda_2: {args.lambda_2}")
            compat_logger.info(f"  weight_decay: {args.weight_decay}")  
            compat_logger.info(f"  dropout: {args.dropout}")
            compat_logger.info(f"  lr: {args.lr}")
            compat_logger.info(f"  batch_size: {args.batch_size}")
            compat_logger.info(f"  digits: {args.digits}")


        compat_logger.info(f"Model parameters for method {args.method}:")
        for key, value in model_params.items():
            compat_logger.info(f"  {key}: {value}")

        compat_logger.info(f"Training parameters for method {args.method}:")
        for key, value in training_params.items():
            compat_logger.info(f"  {key}: {value}")

        # check if user want to use evaluation mode
        if getattr(args, 'use_evaluation', False):
            compat_logger.info(f"Using CANCELOUT with evaluation mode. Maximum features to evaluate: {feature_prediction}")

            CANCELOUT_fs_with_evaluation(
                data, label, 
                model_params, 
                training_params,
                args.name,  
                causal_variants,                 
                args.digits,              
                folder_name,              
                compat_logger,                 
                feature_prediction,       
                n_iters=getattr(args, 'n_iters', 1),           
                feature_step=getattr(args, 'feature_step', 500), 
                n_folds=args.n_splits,  

            )
        else:

            compat_logger.info("Using standard CANCELOUT feature selection (without evaluation).")
            CANCELOUT_fs(data, label, model_params, training_params,args.name,  folder_name, 
                        causal_variants, compat_logger)



    elif args.method == 'LASSO':
        model_params = {
            'input_size': model_input_dim,
            'hidden_size': int(model_input_dim / args.digits),
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'is_snp': args.is_snp,
            'digits': args.digits,
        }

        training_params = {
            'initial_max_iter': getattr(args, 'initial_max_iter', 1000),
            'max_attempts': getattr(args, 'max_attempts', 1),
            'iter_multiplier': getattr(args, 'iter_multiplier', 2),
            'seed': seed,
            'n_splits': args.n_splits,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'validation_split': args.validation_split,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'device': device,
            'weight_decay': args.weight_decay,
            'dfmax': getattr(args, 'dfmax', 100),  
        }

        compat_logger.info(f"Model parameters for method {args.method}:")
        for k, v in model_params.items():
            compat_logger.info(f"  {k}: {v}")
        compat_logger.info(f"Training parameters for method {args.method}:")
        for k, v in training_params.items():
            compat_logger.info(f"  {k}: {v}")


        if getattr(args, 'use_evaluation', 0) == 1:

            compat_logger.info(f"Using LASSO with evaluation mode. Maximum features to evaluate: {feature_prediction}")




            LASSO_with_evaluation(
                data, label,
                model_params,
                training_params,
                args.name,
                causal_variants,         
                args.digits,
                folder_name,
                compat_logger,
                feature_prediction,
                n_iters=getattr(args, 'n_iters', 1),
                feature_step=getattr(args, 'feature_step', 500),
                n_folds=args.n_splits,
            )

        else:

            compat_logger.info("Using standard LASSO feature selection (without evaluation).")

   
            X_t = data['X_train']
            y_t = label['y_train']


            x_np = X_t.detach().cpu().numpy() if isinstance(X_t, torch.Tensor) else np.asarray(X_t)
            y_np = y_t.detach().cpu().numpy() if isinstance(y_t, torch.Tensor) else np.asarray(y_t)


            if y_np.ndim > 1 and y_np.shape[1] == 1:
                y_np = y_np.reshape(-1)
            elif y_np.ndim > 1 and y_np.shape[1] > 1:
                y_np = np.argmax(y_np, axis=1)

            if args.task_type == 'binary':
                data_type = 'binary'          

                uniq = np.unique(y_np)
                if not np.all(np.isin(uniq, [0, 1])):
                    _, y_np = np.unique(y_np, return_inverse=True) 
                if len(np.unique(y_np)) != 2:
                    raise ValueError(f"binary expects 2 classes, got {np.unique(y_np)}")

            elif args.task_type == 'multiclass':
                data_type = 'multinomial'   

                classes, y_np = np.unique(y_np, return_inverse=True)
                y_np = y_np.astype(np.int64)

            elif args.task_type == 'regression':
                data_type = 'continuous'    
                y_np = y_np.astype(np.float64).reshape(-1)

            else:
                raise ValueError(f"Unknown task_type for LASSO: {args.task_type}")


            temp_data_file = os.path.join(folder_name, 'input_data.npz')
            np.savez(temp_data_file, X=x_np, Y=y_np)
            compat_logger.info(f"Saved LASSO input to: {temp_data_file}")


            dfmax = int(training_params.get('dfmax', 3000))
            compat_logger.info(f"LASSO (R) data_type={data_type}, dfmax={dfmax}")

            r_script = 'lasso.R'  
            cmd = ['Rscript', r_script, temp_data_file, folder_name, data_type, str(dfmax), args.name]
            compat_logger.info(f"Running R script: {' '.join(map(str, cmd))}")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.stdout:
                compat_logger.info("---- R STDOUT ----\n" + result.stdout)
            if result.stderr:
                compat_logger.warning("---- R STDERR ----\n" + result.stderr)

            if result.returncode != 0:
                raise RuntimeError("R LASSO script failed. See logs above.")

            compat_logger.info("Standard LASSO (R) finished successfully.")

    elif args.method == 'EARFS':
        model_params = {
            'input_size': model_input_dim,
            'hidden_size': int(model_input_dim / args.digits),
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'digits': args.digits,
            'is_snp':args.is_snp,
        }

        training_params = {
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'validation_split': args.validation_split,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'device': device,
            'seed': seed,
            'lambda_fs': args.lambda_fs,
            'weight_decay': args.weight_decay, 
            'n_splits': args.n_splits,
            'do_parameter_search': args.do_parameter_search,
            'eval_metric': args.eval_metric,
            'n_trials': args.n_trials,
            'n_jobs': args.n_jobs
        }

        if args.do_parameter_search == 1:
            compat_logger.info(f"  n_jobs: {args.n_jobs} (parallelism)")

            if 'SLURM_CPUS_PER_TASK' in os.environ and args.n_jobs == int(os.environ.get('SLURM_CPUS_PER_TASK')):
                compat_logger.info(f"  n_jobs was automatically set from SLURM_CPUS_PER_TASK")

            training_params.update({
                'lambda_fs_min': args.lambda_fs_min,
                'lambda_fs_max': args.lambda_fs_max,
                'weight_decay_min': args.weight_decay_min,
                'weight_decay_max': args.weight_decay_max,
                'lr_min': args.lr_min,
                'lr_max': args.lr_max,
                'dropout_min': args.dropout_min,
                'dropout_max': args.dropout_max,
                'batch_size_list': args.batch_size_list,
                'digits_list': args.digits_list,
            })

            compat_logger.info("Optuna parameter optimization enabled.")
            compat_logger.info(f"Parameter search settings:")
            compat_logger.info(f"  n_trials: {args.n_trials}")
            compat_logger.info(f"  lambda_fs range: [{args.lambda_fs_min}, {args.lambda_fs_max}]")
            compat_logger.info(f"  weight_decay range: [{args.weight_decay_min}, {args.weight_decay_max}]")
            compat_logger.info(f"  lr range: [{args.lr_min}, {args.lr_max}]")
            compat_logger.info(f"  dropout range: [{args.dropout_min}, {args.dropout_max}]")
            compat_logger.info(f"  batch_size options: {args.batch_size_list}")
            compat_logger.info(f"  digits options: {args.digits_list}")
            compat_logger.info(f"  n_splits: {args.n_splits}")
            compat_logger.info(f"  eval_metric: {args.eval_metric } (default if None)")
        else:
            compat_logger.info("Parameter optimization disabled. Using single parameter values.")
            compat_logger.info(f"Using parameters:")
            compat_logger.info(f"  lambda_fs: {args.lambda_fs}")
            compat_logger.info(f"  weight_decay: {args.weight_decay}") 
            compat_logger.info(f"  dropout: {args.dropout}")
            compat_logger.info(f"  lr: {args.lr}")
            compat_logger.info(f"  batch_size: {args.batch_size}")
            compat_logger.info(f"  digits: {args.digits}")


        compat_logger.info(f"Model parameters for method {args.method}:")
        for key, value in model_params.items():
            compat_logger.info(f"  {key}: {value}")
        compat_logger.info(f"Training parameters for method {args.method}:")
        for key, value in training_params.items():
            compat_logger.info(f"  {key}: {value}")

         # check if user want to use evaluation mode
        if getattr(args, 'use_evaluation', False):
            compat_logger.info(f"Using EARFS with evaluation mode. Maximum features to evaluate: {feature_prediction}")

            EARFS_fs_with_evaluation(
                data, label, 
                model_params, 
                training_params,
                args.name,  
                causal_variants,               
                args.digits,         
                folder_name,            
                compat_logger,                 
                feature_prediction,  
                n_iters=getattr(args, 'n_iters', 1),          
                feature_step=getattr(args, 'feature_step', 500), 
                n_folds=args.n_splits,   
            )
        else:

            compat_logger.info("Using standard EARFS feature selection (without evaluation).")
            EARFS_fs(data, label, model_params, training_params,args.name,  folder_name, 
                    compat_logger, causal_variants)


    elif args.method == 'GRACES':
        model_params = {
            'input_size': model_input_dim,
            'hidden_size': [int(model_input_dim / args.digits), int(model_input_dim /args.digits )],
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'alpha': args.alpha,
            'q': args.q if hasattr(args, 'q') else 2,
            'f_correct': args.f_correct,
            'digits': args.digits,
            'max_features': args.max_features,
            'is_snp':args.is_snp,
            'sigma': args.sigma,
            'n_dropouts': args.n_dropouts,
            'max_features_graces': getattr(args, 'max_features_graces', 100),
        }

        training_params = {
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'validation_split': args.validation_split,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'device': device,
            'seed': seed,
            'weight_decay': args.weight_decay,       
            'n_splits': args.n_splits,        
            'do_parameter_search': args.do_parameter_search,  
            'eval_metric': args.eval_metric,  
            'n_trials': args.n_trials,        
            'n_jobs': args.n_jobs,             
        }


        if args.do_parameter_search == 1:
            compat_logger.info(f"GRACES method will use Optuna for hyperparameter optimization")
            compat_logger.info(f"  n_jobs: {args.n_jobs} (parallel jobs)")


            if 'SLURM_CPUS_PER_TASK' in os.environ and args.n_jobs == int(os.environ.get('SLURM_CPUS_PER_TASK')):
                compat_logger.info(f"  n_jobs value automatically obtained from SLURM_CPUS_PER_TASK")


            training_params.update({
                'alpha_min': args.alpha_min if hasattr(args, 'alpha_min') else 0.8,
                'alpha_max': args.alpha_max if hasattr(args, 'alpha_max') else 0.99,
                'f_correct_list': args.f_correct_list if hasattr(args, 'f_correct_list') else '0,0.1,0.5,0.9',
                'lr_min': args.lr_min,
                'lr_max': args.lr_max,
                'dropout_min': args.dropout_min,
                'dropout_max': args.dropout_max,
                'batch_size_list': args.batch_size_list,
                'digits_list': args.digits_list,
            })

            compat_logger.info("Start Optuna hyperparameter optimization.")
            compat_logger.info(f"Parameter search settings:")
            compat_logger.info(f"  n_trials: {args.n_trials}")
            compat_logger.info(f"  alpha range: [{training_params['alpha_min']}, {training_params['alpha_max']}]")
            compat_logger.info(f"  f_correct options: {training_params['f_correct_list']}")
            compat_logger.info(f"  lr range: [{args.lr_min}, {args.lr_max}]")
            compat_logger.info(f"  dropout range: [{args.dropout_min}, {args.dropout_max}]")
            compat_logger.info(f"  batch_size options: {args.batch_size_list}")
            compat_logger.info(f"  hidden_divisor options: {args.digits_list}")
            compat_logger.info(f"  n_splits: {args.n_splits}")
            compat_logger.info(f"  eval_metric: {args.eval_metric}")
        else:
            compat_logger.info("Parameter optimization is disabled. Using single parameter values.")
            compat_logger.info(f"Using parameters:")
            compat_logger.info(f"  alpha: {args.alpha}")
            compat_logger.info(f"  q: {model_params['q']}")
            compat_logger.info(f"  f_correct: {args.f_correct}")
            compat_logger.info(f"  weight_decay: {args.weight_decay}")
            compat_logger.info(f"  dropout: {args.dropout}")
            compat_logger.info(f"  lr: {args.lr}")
            compat_logger.info(f"  batch_size: {args.batch_size}")
            compat_logger.info(f"  digits: {args.digits}")


        compat_logger.info(f"Model parameters for method {args.method}:")
        for key, value in model_params.items():
            compat_logger.info(f"  {key}: {value}")
        compat_logger.info(f"Training parameters for method {args.method}:")
        for key, value in training_params.items():
            compat_logger.info(f"  {key}: {value}")

        compat_logger.info(f"The number of features to select: {args.max_features_graces} (max_features_graces)")

        if getattr(args, 'use_evaluation', False):
            compat_logger.info(f"Using GRACES with evaluation mode. Maximum features to evaluate: {feature_prediction}")


            from GRACES import GRACES_fs_with_evaluation

            GRACES_fs_with_evaluation(
                data, label, 
                model_params, 
                training_params,
                args.name, 
                causal_variants, 
                args.digits,  
                folder_name, 
                compat_logger,  
                feature_prediction,  
                n_iters=getattr(args, 'n_iters', 1), 
                feature_step=getattr(args, 'feature_step', 500),  
                n_folds=args.n_splits, 
            )
        else:

            compat_logger.info("Using standard GRACES feature selection (without evaluation).")
            GRACES_fs(data, label, model_params, training_params,args.name, 
                      causal_variants, args.digits, folder_name, compat_logger)
                  
    elif args.method in ['DeepLIFT', 'GradientShap', 'LRP', 'FeatureAblation', 'Occlusion', 'Lime']:
        model_params = {
            'input_size': model_input_dim,
            'hidden_size': int(model_input_dim / args.digits),
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'digits': args.digits,
            'is_snp':args.is_snp,
        }


        training_params = {
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'validation_split': args.validation_split,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'FloatTensor': torch.cuda.FloatTensor if cuda else torch.FloatTensor,
            'LongTensor': torch.cuda.LongTensor if cuda else torch.LongTensor,
            'task_type': args.task_type,
            'device': device,
            'seed': seed,
            'weight_decay': args.weight_decay,  
            'do_parameter_search': args.do_parameter_search,
            'eval_metric': args.eval_metric,
            'n_trials': args.n_trials,
            'n_jobs': args.n_jobs,
            'n_splits': args.n_splits
        }


        if args.do_parameter_search == 1:
            compat_logger.info(f"  n_jobs: {args.n_jobs} (parallelism)")


            if 'SLURM_CPUS_PER_TASK' in os.environ and args.n_jobs == int(os.environ.get('SLURM_CPUS_PER_TASK')):
                compat_logger.info(f"  n_jobs was automatically set from SLURM_CPUS_PER_TASK")

            training_params.update({
                'weight_decay_min': args.weight_decay_min,
                'weight_decay_max': args.weight_decay_max,
                'lr_min': args.lr_min,
                'lr_max': args.lr_max,
                'dropout_min': args.dropout_min,
                'dropout_max': args.dropout_max,
                'batch_size_list': args.batch_size_list,
                'digits_list': args.digits_list
            })

            compat_logger.info(f"Optuna parameter optimization enabled for {args.method}.")
            compat_logger.info(f"Parameter search settings:")
            compat_logger.info(f"  n_trials: {args.n_trials}")
            compat_logger.info(f"  weight_decay range: [{args.weight_decay_min}, {args.weight_decay_max}]")
            compat_logger.info(f"  lr range: [{args.lr_min}, {args.lr_max}]")
            compat_logger.info(f"  dropout range: [{args.dropout_min}, {args.dropout_max}]")
            compat_logger.info(f"  batch_size options: {args.batch_size_list}")
            compat_logger.info(f"  digits options: {args.digits_list}")
            compat_logger.info(f"  n_splits: {args.n_splits}")
            compat_logger.info(f"  eval_metric: {args.eval_metric} (default if None)")
        else:
            compat_logger.info(f"Parameter optimization disabled for {args.method}. Using single parameter values.")
            compat_logger.info(f"Using parameters:")
            compat_logger.info(f"  weight_decay: {args.weight_decay}")
            compat_logger.info(f"  dropout: {args.dropout}")
            compat_logger.info(f"  lr: {args.lr}")
            compat_logger.info(f"  batch_size: {args.batch_size}")
            compat_logger.info(f"  digits: {args.digits}")


        compat_logger.info(f"Model parameters for method {args.method}:")
        for key, value in model_params.items():
            compat_logger.info(f"  {key}: {value}")
        compat_logger.info(f"Training parameters for method {args.method}:")
        for key, value in training_params.items():
            compat_logger.info(f"  {key}: {value}") 


        if getattr(args, 'use_evaluation', False):
            compat_logger.info(f"Using {args.method} with evaluation mode. Maximum features to evaluate: {feature_prediction}")


            FSDL_with_evaluation(
                data, label, 
                compat_logger, 
                model_params, 
                training_params,
                args.name,  
                causal_variants,        
                args.digits,            
                folder_name,           
                args.method,            
                feature_prediction,      
                n_iters=getattr(args, 'n_iters', 1),         
                feature_step=getattr(args, 'feature_step', 500), 
                n_folds=args.n_splits,    
            )
        else:

            compat_logger.info(f"Using standard {args.method} feature selection (without evaluation).")
            FSDL(data, label, causal_variants, folder_name, training_params, model_params,args.name, args.method, args.digits, compat_logger)
    elif args.method == 'BCOR':

        model_params = {
            'input_size': model_input_dim,
            'hidden_size': int(model_input_dim / args.digits),
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'is_snp':args.is_snp,
        }


        training_params = {
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'validation_split': args.validation_split,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'device': device,
            'seed': seed,
            'n_splits': args.n_splits,
        }



        if getattr(args, 'use_evaluation', False):
            compat_logger.info(f"Using BCOR with evaluation mode. Maximum features to evaluate: {feature_prediction}")


            temp_data_dir = os.path.join(folder_name, 'temp_data')
            os.makedirs(temp_data_dir, exist_ok=True)

            from bcor_evaluation import BCOR_with_evaluation

            BCOR_with_evaluation(
                data, label, 
                model_params, 
                training_params,
                args.name,       
                causal_variants,   
                args.digits,        
                folder_name,       
                compat_logger,         
                feature_prediction, 
                n_iters=getattr(args, 'n_iters', 1),         
                feature_step=getattr(args, 'feature_step', 500),
                n_folds=args.n_splits, 
            )
        else:

            compat_logger.info("Using standard BCOR feature selection (without evaluation).")


            temp_data_file = os.path.join(folder_name, 'input_data.npz')

            x_np = data['X_train'].detach().cpu().numpy() if isinstance(data['X_train'], torch.Tensor) else data['X_train']
            y_np = label['y_train'].detach().cpu().numpy() if isinstance(label['y_train'], torch.Tensor) else label['y_train']
            

            if y_np.ndim > 1 and y_np.shape[1] == 1:
                y_np = y_np.reshape(-1)
            
            temp_data_file = os.path.join(folder_name, 'input_data.npz')
            np.savez(temp_data_file, X=x_np, Y=y_np)



            cmd = ['Rscript', 'bcor.R', temp_data_file, folder_name, args.task_type, args.name]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.stdout:
                compat_logger.info("---- R STDOUT ----\n%s", result.stdout)

            # 仅当非 0 退出码时视为错误；否则把 stderr 当作非致命提示
            if result.returncode != 0:
                compat_logger.error("R exited with code %d. STDERR:\n%s", result.returncode, result.stderr)
                raise RuntimeError("R BCOR script failed. See logs above.")
            else:
                if result.stderr.strip():
                    compat_logger.warning("---- R STDERR (non-fatal) ----\n%s", result.stderr)

            compat_logger.info("Standard BCOR finished successfully.")



    elif args.method == 'FTEST':

        model_params = {
            'input_size': model_input_dim,
            'hidden_size': int(model_input_dim/ args.digits),
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'is_snp':args.is_snp,
        }

        training_params = {
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'validation_split': args.validation_split,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'device': device,
            'seed': seed,
            'n_splits': args.n_splits,
        }


        if getattr(args, 'use_evaluation', False):
            compat_logger.info(f"Using F-Test with evaluation mode. Maximum features to evaluate: {feature_prediction}")

            FTEST_fs_with_evaluation(
                data, label, 
                model_params, 
                training_params,
                args.name,  
                causal_variants,  
                args.digits, 
                folder_name, 
                compat_logger,  
                feature_prediction,  
                n_iters=getattr(args, 'n_iters', 1),  
                feature_step=getattr(args, 'feature_step', 500), 
                n_folds=args.n_splits,  
            )
        else:

            compat_logger.info("Using standard F-Test feature selection (without evaluation).")
            FTEST_fs(data, label, model_params, training_params,args.name,  folder_name, 
                      compat_logger, causal_variants)

    elif args.method == 'RF':

        model_params = {
            'input_size': model_input_dim,
            'hidden_size': int(model_input_dim / args.digits),
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'is_snp':args.is_snp,
        }

        training_params = {
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'validation_split': args.validation_split,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'device': device,
            'seed': seed,
            'n_splits': args.n_splits,
        }



        if getattr(args, 'use_evaluation', False):
            compat_logger.info(f"Using Random Forest with evaluation mode. Maximum features to evaluate: {feature_prediction}")

            RF_fs_with_evaluation(
                data, label, 
                model_params, 
                training_params,
                args.name,  
                causal_variants,    
                args.digits,  
                folder_name,  
                compat_logger,  
                feature_prediction,  
                n_iters=getattr(args, 'n_iters', 1),  
                feature_step=getattr(args, 'feature_step', 500),  
                n_folds=args.n_splits,  
            )
        else:

            compat_logger.info("Using standard Random Forest feature selection (without evaluation).")
            RF_fs(data, label, model_params, training_params, args.name,  folder_name, 
                  compat_logger, causal_variants)
        
    else:
        compat_logger.warning(f"Method {args.method} not recognized or not implemented.")

    # summary = {
    #     "name": args.name,
    #     "method": args.method,
    #     "task_type": args.task_type,
    #     "input_path": os.path.abspath(split_path),
    #     "out_dir": os.path.abspath(folder_name),
    #     "x_train_shape": tuple(x_train.shape),
    #     "x_test_shape": tuple(x_test.shape),
    #     "seed": seed
    # }
    # with open(os.path.join(folder_name, "summary.json"), "w") as f:
    #     json.dump(summary, f, indent=2)        
    # compat_logger.info(f"Feature selection completed. Results saved to {folder_name}")