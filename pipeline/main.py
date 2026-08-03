import numpy as np
import torch
import argparse
import shutil
import logging, os
import gc

from utils import setup_logger, auto_detect_and_fix_labels,setup_seed,_split_train_test,_scale_features,_impute_missing
from EARFS_model import EARFS_fs,EARFS_fs_with_evaluation
from CANCELOUT_model import CANCELOUT_fs, CANCELOUT_fs_with_evaluation

from GRACES import GRACES_fs, GRACES_fs_with_evaluation
from FSDL_model import FSDL, FSDL_with_evaluation
from STG_model import STG_fs, STG_fs_with_evaluation
from CAE_model import CAE_fs, CAE_fs_with_evaluation, is_CAE_result_complete


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
        default='test',
        # choices=['ALLAML','colon','GLI_85', 'leukemia','Prostate_GE', 'SMK_CAN_187'],
        help='Dataset identifier'
    )
    parser.add_argument(
        '--method', 
        type=str, 
        default='EARFS',
        choices=['GRACES', 'EARFS', 'CANCELOUT', 'STG', 'CAE','DeepLIFT', 'GradientShap',
                 'FeatureAblation', 'Occlusion', 'Lime'],  
        help='Feature selection method to use'
    )

    ###################Model options#######################

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
        default='zscore',
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
        '--drop_constant_features',
        type=int,
        default=1,
        choices=[0, 1],
        help='Whether to remove features that have zero variance on the training set after splitting. 1 = remove, 0 = keep.'
    )

    parser.add_argument(
        '--constant_feature_atol',
        type=float,
        default=1e-12,
        help='Absolute tolerance used when deciding whether a training-set feature variance is effectively zero.'
    )

    parser.add_argument(
        '--constant_feature_report_name',
        type=str,
        default='constant_feature_report.json',
        help='Filename used to save the train-based constant-feature removal report under the method output folder.'
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
        default=5,
        help=(
            'Step size for the number of features when sweeping the evaluation curve '
            '(e.g., 10 means evaluate at 10, 20, 30, ...).'
        )
    )
    
    parser.add_argument(
        '--max_features_graces', 
        type=int,
        default=200,
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
        '--shrink_ratio',
        type=float,
        default=2.0,
        help='Fixed shrink ratio used to generate progressively narrower hidden layers in fixed-parameter mode.'
    )
    parser.add_argument(
        '--shrink_ratio_list',
        type=str,
        default='1',
        help='Comma-separated list of shrink ratios for Optuna search. Used together with start_digit and n_layers to generate progressively narrower hidden layers.'
    )
    parser.add_argument(
        '--n_layers',
        type=int,
        default=2,
        help=(
            'Fixed number of hidden layers for methods that use configurable MLP depth. '
            'When parameter search is disabled, this value is used directly.'
        )
    )

    parser.add_argument(
        '--min_layers',
        type=int,
        default=2,
        help=(
            'Minimum number of hidden layers considered during Optuna search '
            'for methods that support variable-depth MLP architectures.'
        )
    )
    parser.add_argument(
        '--max_layers',
        type=int,
        default=2,
        help=(
            'Maximum number of hidden layers considered during Optuna search '
            'for methods that support variable-depth MLP architectures.'
        )
    )
    parser.add_argument(
        '--n_splits',
         type=int, default=5,
        help='Number of folds for cross-validation during Optuna search'
    )
    parser.add_argument(
        '--use_cv',
        type=int,
        default=1,
        choices=[0, 1],
        help='Whether to use cross-validation during hyperparameter search. 1 = use CV, 0 = use a single validation split.'
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
    
    parser.add_argument(
        '--graph_layer_mode',
        type=str,
        default='one_sage_then_mlp',
        choices=['all_sage', 'one_sage_then_mlp'],
        help=(
            "Graph layer layout used by GRACES. "
            "'all_sage' keeps all hidden transitions after the input layer as SAGEConv; "
            "'one_sage_then_mlp' uses only one SAGEConv between the first and second hidden layers, "
            "and all later hidden transitions are standard Linear MLP layers."
        )
    )




    ####STG specific hyperparameter options######
    parser.add_argument(
        '--stg_sigma_min',
        type=float,
        default=0.1,
        help='Minimum sigma value for STG Optuna search'
    )
    parser.add_argument(
        '--stg_sigma_max',
        type=float,
        default=1.0,
        help='Maximum sigma value for STG Optuna search'
    )
    parser.add_argument(
        '--stg_lam_min',
        type=float,
        default=1e-4,
        help='Minimum lam value for STG Optuna search'
    )
    parser.add_argument(
        '--stg_lam_max',
        type=float,
        default=1.0,
        help='Maximum lam value for STG Optuna search'
    )

    ####CAE specific hyperparameter options######
    parser.add_argument(
        '--cat_start_temp_min',
        type=float,
        default=5.0,
        help='Minimum start temperature for CAE Optuna search'
    )
    parser.add_argument(
        '--cat_start_temp_max',
        type=float,
        default=20.0,
        help='Maximum start temperature for CAE Optuna search'
    )
    parser.add_argument(
        '--cat_min_temp_min',
        type=float,
        default=0.01,
        help='Minimum min temperature for CAE Optuna search'
    )
    parser.add_argument(
        '--cat_min_temp_max',
        type=float,
        default=1.0,
        help='Maximum min temperature for CAE Optuna search'
    )


     ###################Fixed hyperparameter options#######################
    # Training Paramters
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


    # EARFS specific fixed parameters
    parser.add_argument(
        '--lambda_fs',
        type=float,
        default=1e-2,
        help='Feature selection regularization for EARFS'
    )
    
    # CANCELOUT specific fixed parameters
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

    # STG specific fixed parameters
    parser.add_argument(
        '--stg_sigma',
        type=float,
        default=0.5,
        help='Sigma used by STG feature selection'
    )
    parser.add_argument(
        '--stg_lam',
        type=float,
        default=0.1,
        help='Lambda sparsity penalty used by STG feature selection'
    )

    # CAE specific fixed parameters60
    parser.add_argument(
        '--cat_k_select',
        type=str,
        default='26,130,260,390,520,650,779,909,1039',#'26,130,260,390,520,650,779,909,1039',
        help='Comma-separated list of K values for the concrete autoencoder selector, e.g. "1,5,10,20".'
    )
    parser.add_argument(
        '--cat_start_temp',
        type=float,
        default=10.0,
        help='Start temperature for CAE feature selection'
    )
    parser.add_argument(
        '--cat_min_temp',
        type=float,
        default=0.1,
        help='Minimum temperature for CAE feature selection'
    )
    parser.add_argument(
        '--cat_tryout_limit',
        type=int,
        default=1,
        help='Tryout limit for CAE feature selection'
    )
    parser.add_argument(
        '--cat_selector_mode',
        type=str,
        default='supervised',
        choices=['unsupervised', 'supervised'],
        help='Selector mode for CAE: unsupervised reconstruction or supervised prediction.'
    )

    
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        default_n_jobs = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', 1)) - 2)
        parser.set_defaults(n_jobs=default_n_jobs) 

    return parser.parse_args()





if __name__ == "__main__":
    args = parse_arguments()

    os.makedirs(args.out_dir, exist_ok=True)
    search_label = "param_search" if int(args.do_parameter_search) == 1 else "no_param_search"
    evaluation_label = "with_eval" if int(args.use_evaluation) == 1 else "no_eval"

    setup_logger(
        args.out_dir,
        args.method,
        args.do_parameter_search,
        args.use_evaluation,
        args.seed,
        args.task_type,
        args.selected_activate,
    )
    logger = logging.getLogger(
        f"pipeline_{args.method}_{search_label}_{evaluation_label}_seed{args.seed}_{args.task_type}_{args.selected_activate}"
    )
    # logger.info("Starting pipeline...")


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

    array_task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))  # Default to 0
    compat_logger.info(f"The ID of task: {array_task_id}")

    # set random seed
    seed = args.seed  # or any other base seed you prefer
    setup_seed(seed)
    
    # get current path and parent folder name



    out_root = os.path.abspath(args.out_dir)  # -> /work/result

    folder_name = os.path.join(
        out_root,
        f"{args.name}",
        f"result{args.selected_activate}",
        f"result_{seed}",
        f"{args.method}",
    )
    # clean up old folder and create a new one, except for CAE base folder
    if args.method == 'CAE':
        os.makedirs(folder_name, exist_ok=True)
        compat_logger.info(
            f"folder {folder_name} prepared (base CAE folder kept because per-K subfolders are managed later)"
        )
    else:
        if os.path.exists(folder_name):
            compat_logger.info(f"folder {folder_name} exists, deleting...")
            shutil.rmtree(folder_name)
            compat_logger.info(f"folder {folder_name} deleted")
        os.makedirs(folder_name, exist_ok=True)
        compat_logger.info(f"folder {folder_name} created")


    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")



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
    # Priority: new format containing only X and Y.
    # Compatibility: legacy format containing x_train, y_train, x_test, and y_test.
    # -------------------------------------------------

    if args.preprocess_mode == 'auto':    
        if ('X' in files) and ('Y' in files):
            compat_logger.info("[Load] Detected new format with keys: X, Y")

            X = pack['X']
            y = pack['Y']

            # Flatten y if it's a 2D array with a single column, to ensure it's 1D.
            if y.ndim > 1 and y.shape[1] == 1:
                y = y.reshape(-1)
            # Multi-output cases are handled by auto_detect_and_fix_labels.

            compat_logger.info(f"[Load] X shape: {X.shape}, dtype: {X.dtype}")
            compat_logger.info(f"[Load] y shape: {y.shape}, dtype: {y.dtype}")

            # ------- Missing-value imputation -------
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

            # ------- Decide whether to split train/test based on use_evaluation -------
            if args.use_evaluation == 1:
                compat_logger.info(
                    f"[Split] use_evaluation=1, splitting data with train_ratio={args.train_ratio}"
                )

                constant_feature_report_path = os.path.join(
                    folder_name,
                    args.constant_feature_report_name,
                )

                x_train, x_test, y_train, y_test, original_has_test = _split_train_test(
                    X,
                    y,
                    args.train_ratio,
                    seed,
                    compat_logger,
                    drop_constant_features=bool(args.drop_constant_features),
                    constant_feature_report_path=constant_feature_report_path,
                    constant_feature_atol=args.constant_feature_atol,
                )

                if bool(args.drop_constant_features):
                    compat_logger.info(
                        f"[FeatureFilter] drop_constant_features=1, report path: {constant_feature_report_path}"
                    )
                else:
                    compat_logger.info("[FeatureFilter] drop_constant_features=0, constant-feature filtering disabled.")

                
                # ------- Scaling: fit parameters on the training set only -------
                if args.is_snp and args.scaler != 'none':
                    compat_logger.warning(
                        "[Scale] is_snp=1 and scaler!=none. Continuous scaling is usually not recommended for SNP data, so scaling will be skipped."
                    )
                else:
                    x_train, x_test = _scale_features(
                        x_train, x_test, args.scaler, compat_logger
                    )

                # Disable evaluation if no independent test set is available.
                if not original_has_test:
                    compat_logger.warning(
                        "[Eval] train_ratio=1 or too few samples to construct an independent test set, forcing use_evaluation=0."
                    )
                    args.use_evaluation = 0
                    compat_logger.warning(
                        "[Eval] Keeping the already prepared x_train/x_test produced by _split_train_test. "
                        "No reset to the raw full X,y will be performed, so any train-based constant-feature filtering remains in effect."
                    )

            else:
                compat_logger.info(
                    "[Split] use_evaluation=0, no train/test split will be performed. "
                    "Using the full dataset for feature selection."
                )

                x_train, y_train = X, y
                x_test, y_test = X, y
                original_has_test = False

                if bool(args.drop_constant_features):
                    compat_logger.warning(
                        "[FeatureFilter] drop_constant_features=1 but use_evaluation=0, so no train/test split is performed and train-based constant-feature filtering is skipped."
                    )

                # ------- Skip train/test-based scaling -------
                if args.scaler != 'none':
                    compat_logger.warning(
                        "[Scale] use_evaluation=0 and no split is performed, "
                        "so scaler is skipped to avoid fitting on the full dataset "
                        "and creating ambiguous train/test semantics."
                    )



        else:
            # ============ Legacy format compatibility: x_train/y_train(/x_test/y_test) ============
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
        
        # Here we directly require the legacy format:
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
            # A missing test set is acceptable because evaluation is already disabled.
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
    
    # causal_variants = np.array([0, 1])  #pack['global_indices']       ##not import here, just for placeholder

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
    

    compat_logger.info(
        f"x_train_t: shape={tuple(x_train_t.shape)}, dtype={x_train_t.dtype}, device={x_train_t.device}"
    )
    compat_logger.info(
        f"x_test_t: shape={tuple(x_test_t.shape)}, dtype={x_test_t.dtype}, device={x_test_t.device}"
    )
    compat_logger.info(
        f"y_train_t: shape={tuple(y_train_t.shape)}, dtype={y_train_t.dtype}, device={y_train_t.device}"
    )
    compat_logger.info(
        f"y_test_t: shape={tuple(y_test_t.shape)}, dtype={y_test_t.dtype}, device={y_test_t.device}"
    )


    data = {
        'X_train': x_train_t,
        'X_test':  x_test_t,   # Note: if no original test set, this is just a copy of train
    }
    label = {
        'y_train': y_train_t,
        'y_test':  y_test_t,   # Same as above
    }
    compat_logger.info(f"X_train shape: {x_train_t.shape}")
    # get model input dimension
    model_input_dim = x_train_t.shape[1]
    feature_prediction = getattr(args, 'max_features', 2500)
    compat_logger.info(f"Model input dimension: {model_input_dim}")
    if args.use_evaluation == 1:
        compat_logger.info(f"Feature evaluation upper limit: {feature_prediction}")

    compat_logger.info(f"Using Method: {args.method}")


    if args.method == 'CANCELOUT':
        # model parameters
        fixed_digits_list = [
            max(1, int(round(args.digits * (args.shrink_ratio ** i))))
            for i in range(args.n_layers)
        ]
        fixed_hidden_dims = [
            max(4, int(model_input_dim / d))
            for d in fixed_digits_list
        ]

        model_params = {
            'input_size': model_input_dim,
            'hidden_dims': fixed_hidden_dims,
            'n_layers': args.n_layers,
            'digits_list': fixed_digits_list,
            'start_digit': args.digits,
            'shrink_ratio': args.shrink_ratio,
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'is_snp': args.is_snp,
            'digits': args.digits,
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
            'use_cv': bool(args.use_cv),
            'do_parameter_search': args.do_parameter_search,
            'eval_metric': args.eval_metric,
            'n_trials': args.n_trials,
            'n_jobs': args.n_jobs,
            'min_layers': args.min_layers,
            'max_layers': args.max_layers,
            'shrink_ratio': args.shrink_ratio,
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
                'shrink_ratio_list': args.shrink_ratio_list,
                'min_layers': args.min_layers,
                'max_layers': args.max_layers,
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
            compat_logger.info(f"  start_digit options: {args.digits_list}")
            compat_logger.info(f"  shrink_ratio options: {args.shrink_ratio_list}")
            compat_logger.info(f"  min_layers: {args.min_layers}")
            compat_logger.info(f"  max_layers: {args.max_layers}")
            compat_logger.info(f"  n_splits: {args.n_splits}")
            compat_logger.info(f"  use_cv: {bool(args.use_cv)}")
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
            compat_logger.info(f"  start_digit: {args.digits}")
            compat_logger.info(f"  shrink_ratio: {args.shrink_ratio}")
            compat_logger.info(f"  n_layers: {args.n_layers}")
            compat_logger.info(f"  generated_digits_list: {fixed_digits_list}")
            compat_logger.info(f"  generated_hidden_dims: {fixed_hidden_dims}")
            compat_logger.info("  use_cv: False (no Optuna CV because do_parameter_search=0)")


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
            CANCELOUT_fs(data, label, model_params, training_params,args.name,  folder_name, compat_logger)


    elif args.method == 'EARFS':
        fixed_digits_list = [
            max(1, int(round(args.digits * (args.shrink_ratio ** i))))
            for i in range(args.n_layers)
        ]
        fixed_hidden_dims = [
            max(4, int(model_input_dim / d))
            for d in fixed_digits_list
        ]

        model_params = {
            'input_size': model_input_dim,
            'hidden_dims': fixed_hidden_dims,
            'n_layers': args.n_layers,
            'digits_list': fixed_digits_list,
            'start_digit': args.digits,
            'shrink_ratio': args.shrink_ratio,
            'hidden_size': fixed_hidden_dims,
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'digits': args.digits,
            'is_snp': args.is_snp,
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
            'use_cv': bool(args.use_cv),
            'do_parameter_search': args.do_parameter_search,
            'eval_metric': args.eval_metric,
            'n_trials': args.n_trials,
            'n_jobs': args.n_jobs,
            'min_layers': args.min_layers,
            'max_layers': args.max_layers,
            'shrink_ratio': args.shrink_ratio,
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
                'shrink_ratio_list': args.shrink_ratio_list,
                'min_layers': args.min_layers,
                'max_layers': args.max_layers,
            })

            compat_logger.info("Optuna parameter optimization enabled.")
            compat_logger.info(f"Parameter search settings:")
            compat_logger.info(f"  n_trials: {args.n_trials}")
            compat_logger.info(f"  lambda_fs range: [{args.lambda_fs_min}, {args.lambda_fs_max}]")
            compat_logger.info(f"  weight_decay range: [{args.weight_decay_min}, {args.weight_decay_max}]")
            compat_logger.info(f"  lr range: [{args.lr_min}, {args.lr_max}]")
            compat_logger.info(f"  dropout range: [{args.dropout_min}, {args.dropout_max}]")
            compat_logger.info(f"  batch_size options: {args.batch_size_list}")
            compat_logger.info(f"  start_digit options: {args.digits_list}")
            compat_logger.info(f"  shrink_ratio options: {args.shrink_ratio_list}")
            compat_logger.info(f"  min_layers: {args.min_layers}")
            compat_logger.info(f"  max_layers: {args.max_layers}")
            compat_logger.info(f"  n_splits: {args.n_splits}")
            compat_logger.info(f"  use_cv: {bool(args.use_cv)}")
            compat_logger.info(f"  eval_metric: {args.eval_metric } (default if None)")
        else:
            compat_logger.info("Parameter optimization disabled. Using single parameter values.")
            compat_logger.info(f"Using parameters:")
            compat_logger.info(f"  lambda_fs: {args.lambda_fs}")
            compat_logger.info(f"  weight_decay: {args.weight_decay}")
            compat_logger.info(f"  dropout: {args.dropout}")
            compat_logger.info(f"  lr: {args.lr}")
            compat_logger.info(f"  batch_size: {args.batch_size}")
            compat_logger.info(f"  start_digit: {args.digits}")
            compat_logger.info(f"  shrink_ratio: {args.shrink_ratio}")
            compat_logger.info(f"  n_layers: {args.n_layers}")
            compat_logger.info(f"  generated_digits_list: {fixed_digits_list}")
            compat_logger.info(f"  generated_hidden_dims: {fixed_hidden_dims}")
            compat_logger.info(f"  search layer range: [{args.min_layers}, {args.max_layers}]")
            compat_logger.info("  use_cv: False (no Optuna CV because do_parameter_search=0)")


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
                    compat_logger)


    elif args.method == 'GRACES':
        # 1) Compute progressive digits/hidden_dims lists
        fixed_digits_list = [
            max(1, int(round(args.digits * (args.shrink_ratio ** i))))
            for i in range(args.n_layers)
        ]
        fixed_hidden_dims = [
            max(4, int(model_input_dim / d))
            for d in fixed_digits_list
        ]

        # 2) Model params dictionary
        model_params = {
            'input_size': model_input_dim,
            'hidden_dims': fixed_hidden_dims,
            'n_layers': args.n_layers,
            'digits_list': fixed_digits_list,
            'start_digit': args.digits,
            'shrink_ratio': args.shrink_ratio,
            'hidden_size': fixed_hidden_dims,
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'alpha': args.alpha,
            'q': args.q if hasattr(args, 'q') else 2,
            'f_correct': args.f_correct,
            'digits': args.digits,
            'max_features': args.max_features,
            'is_snp': args.is_snp,
            'sigma': args.sigma,
            'n_dropouts': args.n_dropouts,
            'max_features_graces': getattr(args, 'max_features_graces', 100),
            'graph_layer_mode': args.graph_layer_mode,
        }

        # 3) Training params dictionary
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
            'use_cv': bool(args.use_cv),
            'do_parameter_search': args.do_parameter_search,
            'eval_metric': args.eval_metric,
            'n_trials': args.n_trials,
            'n_jobs': args.n_jobs,
            'min_layers': args.min_layers,
            'max_layers': args.max_layers,
            'shrink_ratio': args.shrink_ratio,
            'graph_layer_mode': args.graph_layer_mode,
        }

        # 4) Parameter search block
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
                'shrink_ratio_list': args.shrink_ratio_list,
                'min_layers': args.min_layers,
                'max_layers': args.max_layers,
            })

            compat_logger.info("Start Optuna hyperparameter optimization.")
            compat_logger.info(f"Parameter search settings:")
            compat_logger.info(f"  n_trials: {args.n_trials}")
            compat_logger.info(f"  alpha range: [{training_params['alpha_min']}, {training_params['alpha_max']}]")
            compat_logger.info(f"  f_correct options: {training_params['f_correct_list']}")
            compat_logger.info(f"  lr range: [{args.lr_min}, {args.lr_max}]")
            compat_logger.info(f"  dropout range: [{args.dropout_min}, {args.dropout_max}]")
            compat_logger.info(f"  batch_size options: {args.batch_size_list}")
            compat_logger.info(f"  start_digit options: {args.digits_list}")
            compat_logger.info(f"  shrink_ratio options: {args.shrink_ratio_list}")
            compat_logger.info(f"  min_layers: {args.min_layers}")
            compat_logger.info(f"  max_layers: {args.max_layers}")
            compat_logger.info(f"  graph_layer_mode: {args.graph_layer_mode}")
            compat_logger.info(f"  n_splits: {args.n_splits}")
            compat_logger.info(f"  use_cv: {bool(args.use_cv)}")
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
            compat_logger.info(f"  start_digit: {args.digits}")
            compat_logger.info(f"  shrink_ratio: {args.shrink_ratio}")
            compat_logger.info(f"  n_layers: {args.n_layers}")
            compat_logger.info(f"  generated_digits_list: {fixed_digits_list}")
            compat_logger.info(f"  generated_hidden_dims: {fixed_hidden_dims}")
            compat_logger.info(f"  graph_layer_mode: {args.graph_layer_mode}")
            compat_logger.info("  use_cv: False (no Optuna CV because do_parameter_search=0)")

        compat_logger.info(f"Model parameters for method {args.method}:")
        for key, value in model_params.items():
            compat_logger.info(f"  {key}: {value}")
        compat_logger.info(f"Training parameters for method {args.method}:")
        for key, value in training_params.items():
            compat_logger.info(f"  {key}: {value}")

        compat_logger.info(f"The number of features to select: {args.max_features_graces} (max_features_graces)")

        if getattr(args, 'use_evaluation', False):
            compat_logger.info(f"Using GRACES with evaluation mode. Maximum features to evaluate: {feature_prediction}")

            GRACES_fs_with_evaluation(
                data, label,
                model_params,
                training_params,
                args.name,
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
            GRACES_fs(data, label, model_params, training_params, args.name,
                      args.digits, folder_name, compat_logger)
                  
    elif args.method == 'STG':
        fixed_digits_list = [
            max(1, int(round(args.digits * (args.shrink_ratio ** i))))
            for i in range(args.n_layers)
        ]
        fixed_hidden_dims = [
            max(4, int(model_input_dim / d))
            for d in fixed_digits_list
        ]

        model_params = {
            'input_size': model_input_dim,
            'hidden_dims': fixed_hidden_dims,
            'n_layers': args.n_layers,
            'digits_list': fixed_digits_list,
            'start_digit': args.digits,
            'shrink_ratio': args.shrink_ratio,
            'hidden_size': fixed_hidden_dims[0],
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'batch_norm': True,
            'optimizer': 'Adam',
            'digits': args.digits,
            'is_snp': args.is_snp,
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
            'sigma': args.stg_sigma,
            'lam': args.stg_lam,
            'n_splits': args.n_splits,
            'use_cv': bool(args.use_cv),
            'do_parameter_search': args.do_parameter_search,
            'eval_metric': args.eval_metric,
            'n_trials': args.n_trials,
            'n_jobs': args.n_jobs,
            'min_layers': args.min_layers,
            'max_layers': args.max_layers,
            'shrink_ratio': args.shrink_ratio,
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
                'digits_list': args.digits_list,
                'shrink_ratio_list': args.shrink_ratio_list,
                'min_layers': args.min_layers,
                'max_layers': args.max_layers,
                'sigma_min': args.stg_sigma_min,
                'sigma_max': args.stg_sigma_max,
                'lam_min': args.stg_lam_min,
                'lam_max': args.stg_lam_max,
            })

            compat_logger.info("Optuna parameter optimization enabled for STG.")
            compat_logger.info(f"Parameter search settings:")
            compat_logger.info(f"  n_trials: {args.n_trials}")
            compat_logger.info(f"  weight_decay range: [{args.weight_decay_min}, {args.weight_decay_max}]")
            compat_logger.info(f"  lr range: [{args.lr_min}, {args.lr_max}]")
            compat_logger.info(f"  dropout range: [{args.dropout_min}, {args.dropout_max}]")
            compat_logger.info(f"  batch_size options: {args.batch_size_list}")
            compat_logger.info(f"  start_digit options: {args.digits_list}")
            compat_logger.info(f"  shrink_ratio options: {args.shrink_ratio_list}")
            compat_logger.info(f"  min_layers: {args.min_layers}")
            compat_logger.info(f"  max_layers: {args.max_layers}")
            compat_logger.info(f"  sigma range: [{args.stg_sigma_min}, {args.stg_sigma_max}]")
            compat_logger.info(f"  lam range: [{args.stg_lam_min}, {args.stg_lam_max}]")
            compat_logger.info(f"  n_splits: {args.n_splits}")
            compat_logger.info(f"  use_cv: {bool(args.use_cv)}")
            compat_logger.info(f"  eval_metric: {args.eval_metric} (default if None)")
        else:
            compat_logger.info("Parameter optimization disabled for STG. Using single parameter values.")
            compat_logger.info(f"Using parameters:")
            compat_logger.info(f"  weight_decay: {args.weight_decay}")
            compat_logger.info(f"  dropout: {args.dropout}")
            compat_logger.info(f"  lr: {args.lr}")
            compat_logger.info(f"  batch_size: {args.batch_size}")
            compat_logger.info(f"  start_digit: {args.digits}")
            compat_logger.info(f"  shrink_ratio: {args.shrink_ratio}")
            compat_logger.info(f"  n_layers: {args.n_layers}")
            compat_logger.info(f"  generated_digits_list: {fixed_digits_list}")
            compat_logger.info(f"  generated_hidden_dims: {fixed_hidden_dims}")
            compat_logger.info(f"  sigma: {args.stg_sigma}")
            compat_logger.info(f"  lam: {args.stg_lam}")
            compat_logger.info("  use_cv: False (no Optuna CV because do_parameter_search=0)")

        compat_logger.info(f"Model parameters for method {args.method}:")
        for key, value in model_params.items():
            compat_logger.info(f"  {key}: {value}")
        compat_logger.info(f"Training parameters for method {args.method}:")
        for key, value in training_params.items():
            compat_logger.info(f"  {key}: {value}")

        if getattr(args, 'use_evaluation', False):
            compat_logger.info(f"Using STG with evaluation mode. Maximum features to evaluate: {feature_prediction}")

            STG_fs_with_evaluation(
                data, label,
                model_params, training_params,
                args.name,
                args.digits,
                folder_name,
                compat_logger,
                feature_prediction,
                n_iters=getattr(args, 'n_iters', 1),
                feature_step=getattr(args, 'feature_step', 500),
                n_folds=args.n_splits,
            )
        else:
            compat_logger.info("Using standard STG feature selection (without evaluation).")
            STG_fs(
                data, label,
                model_params, training_params,
                args.name,
                folder_name,
                compat_logger,
            )

    elif args.method == 'CAE':
        fixed_digits_list = [
            max(1, int(round(args.digits * (args.shrink_ratio ** i))))
            for i in range(args.n_layers)
        ]
        fixed_hidden_dims = [
            max(4, int(model_input_dim / d))
            for d in fixed_digits_list
        ]

        cat_k_select_values = []
        for _raw_k in str(args.cat_k_select).split(','):
            _raw_k = _raw_k.strip()
            if not _raw_k:
                continue
            _k = int(_raw_k)
            if _k <= 0:
                raise ValueError(f"Each cat_k_select value must be positive, got {_k}")
            cat_k_select_values.append(_k)

        if len(cat_k_select_values) == 0:
            raise ValueError("cat_k_select must contain at least one positive integer")

        model_params = {
            'input_size': model_input_dim,
            'hidden_dims': fixed_hidden_dims,
            'n_layers': args.n_layers,
            'digits_list': fixed_digits_list,
            'start_digit': args.digits,
            'shrink_ratio': args.shrink_ratio,
            'hidden_size': fixed_hidden_dims[0],
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'digits': args.digits,
            'is_snp': args.is_snp,
            'selector_mode': args.cat_selector_mode,
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
            'K_select': cat_k_select_values[0],
            'start_temp': args.cat_start_temp,
            'min_temp': args.cat_min_temp,
            'tryout_limit': args.cat_tryout_limit,
            'score_reduction': 'max',
            'n_splits': args.n_splits,
            'use_cv': bool(args.use_cv),
            'do_parameter_search': args.do_parameter_search,
            'eval_metric': args.eval_metric,
            'n_trials': args.n_trials,
            'n_jobs': args.n_jobs,
            'min_layers': args.min_layers,
            'max_layers': args.max_layers,
            'shrink_ratio': args.shrink_ratio,
            'selector_mode': args.cat_selector_mode,
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
                'digits_list': args.digits_list,
                'shrink_ratio_list': args.shrink_ratio_list,
                'min_layers': args.min_layers,
                'max_layers': args.max_layers,
                'start_temp_min': args.cat_start_temp_min,
                'start_temp_max': args.cat_start_temp_max,
                'min_temp_min': args.cat_min_temp_min,
                'min_temp_max': args.cat_min_temp_max,
            })

            compat_logger.info("Optuna parameter optimization enabled for CAE.")
            compat_logger.info(f"Parameter search settings:")
            compat_logger.info(f"  n_trials: {args.n_trials}")
            compat_logger.info(f"  weight_decay range: [{args.weight_decay_min}, {args.weight_decay_max}]")
            compat_logger.info(f"  lr range: [{args.lr_min}, {args.lr_max}]")
            compat_logger.info(f"  dropout range: [{args.dropout_min}, {args.dropout_max}]")
            compat_logger.info(f"  batch_size options: {args.batch_size_list}")
            compat_logger.info(f"  start_digit options: {args.digits_list}")
            compat_logger.info(f"  shrink_ratio options: {args.shrink_ratio_list}")
            compat_logger.info(f"  min_layers: {args.min_layers}")
            compat_logger.info(f"  max_layers: {args.max_layers}")
            compat_logger.info(f"  start_temp range: [{args.cat_start_temp_min}, {args.cat_start_temp_max}]")
            compat_logger.info(f"  min_temp range: [{args.cat_min_temp_min}, {args.cat_min_temp_max}]")
            compat_logger.info(f"  n_splits: {args.n_splits}")
            compat_logger.info(f"  use_cv: {bool(args.use_cv)}")
            if args.cat_selector_mode == 'unsupervised':
                compat_logger.info(f"  eval_metric requested: {args.eval_metric} (unsupervised CAE will internally use loss)")
            else:
                compat_logger.info(f"  eval_metric requested: {args.eval_metric} (supervised CAE will use the requested metric)")
        else:
            compat_logger.info("Parameter optimization disabled for CAE. Using single parameter values.")
            compat_logger.info(f"Using parameters:")
            compat_logger.info(f"  weight_decay: {args.weight_decay}")
            compat_logger.info(f"  dropout: {args.dropout}")
            compat_logger.info(f"  lr: {args.lr}")
            compat_logger.info(f"  batch_size: {args.batch_size}")
            compat_logger.info(f"  start_digit: {args.digits}")
            compat_logger.info(f"  shrink_ratio: {args.shrink_ratio}")
            compat_logger.info(f"  n_layers: {args.n_layers}")
            compat_logger.info(f"  generated_digits_list: {fixed_digits_list}")
            compat_logger.info(f"  generated_hidden_dims: {fixed_hidden_dims}")
            compat_logger.info(f"  K_select list: {cat_k_select_values}")
            compat_logger.info(f"  start_temp: {args.cat_start_temp}")
            compat_logger.info(f"  min_temp: {args.cat_min_temp}")
            compat_logger.info(f"  tryout_limit: {args.cat_tryout_limit}")
            compat_logger.info(f"  selector_mode: {args.cat_selector_mode}")
            compat_logger.info("  use_cv: False (no Optuna CV because do_parameter_search=0)")

        compat_logger.info(f"Model parameters for method {args.method}:")
        for key, value in model_params.items():
            compat_logger.info(f"  {key}: {value}")

        compat_logger.info(f"Base training parameters for method {args.method}:")
        for key, value in training_params.items():
            compat_logger.info(f"  {key}: {value}")

        compat_logger.info(f"CAE will run once for each K_select in: {cat_k_select_values}")

        for current_k_select in cat_k_select_values:
            current_k_select = int(current_k_select)
        
            current_training_params = training_params.copy()
            current_training_params['K_select'] = current_k_select
        
            current_folder_name = os.path.join(folder_name, str(current_k_select))
            result_file = os.path.join(current_folder_name, "feature_selection_result.npz")
        
            # ============================================================
            # CAE resume logic:
            # If the result file for the current K exists and is complete, skip it.
            # If it is missing or incomplete, delete the K-specific folder and rerun it.
            # ============================================================
        
            if is_CAE_result_complete(
                result_file=result_file,
                expected_k=current_k_select,
                logger=compat_logger,
            ):
                compat_logger.info(
                    f"[Skip] CAE K_select={current_k_select} already completed. "
                    f"Skip this K and continue to the next one."
                )
                continue
        
            if os.path.exists(current_folder_name):
                compat_logger.info(
                    f"[Rerun] CAE K_select={current_k_select} result is missing or incomplete. "
                    f"Deleting folder: {current_folder_name}"
                )
                shutil.rmtree(current_folder_name)
                compat_logger.info(f"folder {current_folder_name} deleted")
        
            os.makedirs(current_folder_name, exist_ok=True)
        
            compat_logger.info(f"Running CAE with K_select={current_k_select}")
            compat_logger.info(f"CAE output folder: {current_folder_name}")
            compat_logger.info(f"Current training parameters for K_select={current_k_select}:")
            for key, value in current_training_params.items():
                compat_logger.info(f"  {key}: {value}")
        
            if getattr(args, 'use_evaluation', False):
                compat_logger.info(
                    f"Using CAE with evaluation mode for K_select={current_k_select}. "
                    f"Maximum features argument received: {feature_prediction}"
                )
        
                CAE_fs_with_evaluation(
                    data, label,
                    model_params, current_training_params,
                    args.name,
                    args.digits,
                    current_folder_name,
                    compat_logger,
                    feature_prediction,
                    n_iters=getattr(args, 'n_iters', 1),
                    feature_step=getattr(args, 'feature_step', 500),
                    n_folds=args.n_splits,
                )
            else:
                compat_logger.info(
                    f"Using standard CAE feature selection without evaluation for K_select={current_k_select}."
                )
                CAE_fs(
                    data, label,
                    model_params, current_training_params,
                    args.name,
                    current_folder_name,
                    compat_logger,
                )
        
            gc.collect()
        
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    elif args.method in ['DeepLIFT', 'GradientShap', 'LRP', 'FeatureAblation', 'Occlusion', 'Lime']:
        fixed_digits_list = [
            max(1, int(round(args.digits * (args.shrink_ratio ** i))))
            for i in range(args.n_layers)
        ]
        fixed_hidden_dims = [
            max(4, int(model_input_dim / d))
            for d in fixed_digits_list
        ]

        model_params = {
            'input_size': model_input_dim,
            'hidden_dims': fixed_hidden_dims,
            'n_layers': args.n_layers,
            'digits_list': fixed_digits_list,
            'start_digit': args.digits,
            'shrink_ratio': args.shrink_ratio,
            'hidden_size': fixed_hidden_dims[0],
            'num_classes': num_label,
            'task_type': args.task_type,
            'dropout_prob': args.dropout,
            'activation': args.selected_activate,
            'digits': args.digits,
            'is_snp': args.is_snp,
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
            'n_splits': args.n_splits,
            'use_cv': bool(args.use_cv),
            'min_layers': args.min_layers,
            'max_layers': args.max_layers,
            'shrink_ratio': args.shrink_ratio,
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
                'digits_list': args.digits_list,
                'shrink_ratio_list': args.shrink_ratio_list,
                'min_layers': args.min_layers,
                'max_layers': args.max_layers,
            })

            compat_logger.info(f"Optuna parameter optimization enabled for {args.method}.")
            compat_logger.info(f"Parameter search settings:")
            compat_logger.info(f"  n_trials: {args.n_trials}")
            compat_logger.info(f"  weight_decay range: [{args.weight_decay_min}, {args.weight_decay_max}]")
            compat_logger.info(f"  lr range: [{args.lr_min}, {args.lr_max}]")
            compat_logger.info(f"  dropout range: [{args.dropout_min}, {args.dropout_max}]")
            compat_logger.info(f"  batch_size options: {args.batch_size_list}")
            compat_logger.info(f"  start_digit options: {args.digits_list}")
            compat_logger.info(f"  shrink_ratio options: {args.shrink_ratio_list}")
            compat_logger.info(f"  min_layers: {args.min_layers}")
            compat_logger.info(f"  max_layers: {args.max_layers}")
            compat_logger.info(f"  n_splits: {args.n_splits}")
            compat_logger.info(f"  eval_metric: {args.eval_metric} (default if None)")
            compat_logger.info(f"  use_cv: {bool(args.use_cv)}")
        else:
            compat_logger.info(f"Parameter optimization disabled for {args.method}. Using single parameter values.")
            compat_logger.info(f"Using parameters:")
            compat_logger.info(f"  weight_decay: {args.weight_decay}")
            compat_logger.info(f"  dropout: {args.dropout}")
            compat_logger.info(f"  lr: {args.lr}")
            compat_logger.info(f"  batch_size: {args.batch_size}")
            compat_logger.info(f"  start_digit: {args.digits}")
            compat_logger.info(f"  shrink_ratio: {args.shrink_ratio}")
            compat_logger.info(f"  n_layers: {args.n_layers}")
            compat_logger.info(f"  generated_digits_list: {fixed_digits_list}")
            compat_logger.info(f"  generated_hidden_dims: {fixed_hidden_dims}")
            compat_logger.info("  use_cv: False (no Optuna CV because do_parameter_search=0)")


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
            FSDL(data, label, folder_name, training_params, model_params,args.name, args.method, args.digits, compat_logger)
   
   
    else:
        compat_logger.error(f"Unsupported method: {args.method}")
        raise ValueError(f"Unsupported method: {args.method}")