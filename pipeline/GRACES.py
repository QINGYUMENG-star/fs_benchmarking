import os
import copy
import time
import json 
import torch
import optuna
import warnings
import datetime
import traceback
import numpy as np
from torch import nn
import torch_geometric.nn as gnn
from torch.nn import functional as F
from sqlalchemy.pool import QueuePool
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from evaluation_utils import evaluate_feature_set
from utils import get_activation, format_time, parse_list_arg
warnings.filterwarnings("ignore")





def GRACES_fs_with_evaluation(data, label, model_params, training_params, name, 
                           digits, folder_name, 
                           logger, feature_prediction, n_iters=20,
                           feature_step=500, n_folds=3):
    """
    execute GRACES feature selection and evaluation with checkpointing and intermediate result saving
    
    Args:
        data: input data dictionary with 'X_train' and 'X_test'
        label: target values
        model_params: model parameters dictionary
        training_params: training parameters dictionary
        name: dataset name
        digits: feature dimension divisor
        folder_name: result saving directory
        logger: logger
        feature_prediction: upper limit on the number of features to evaluate
        n_iters: number of iterations, default 20
        feature_step: feature evaluation step size, default 500
        n_folds: number of cross-validation folds, default 3
    """

    
    total_start_time = time.time()
    logger.info(f"GRACES_fs_with_evaluation start at  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    feature_selection_checkpoint_folder = os.path.join(folder_name, 'feature_selection_checkpoints')
    os.makedirs(feature_selection_checkpoint_folder, exist_ok=True)    
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
    results_indices = []
    all_best_params = []
    seeds = np.random.choice(range(1000), n_iters, replace=False)
    all_report_metrics = [[None for _ in range(n_iters)] for _ in range(n_steps)] 
    per_step_reports = [] 
    

    logger.info(f"total features: {total_features}")
    logger.info(f"evaluation settings:")
    logger.info(f"- max features to evaluate: {max_features}")
    logger.info(f"- feature sequence: {feature_sequence}")
    logger.info(f"- number of evaluation points: {n_steps}")
    logger.info(f"- number of cross-validation folds: {n_folds}")
    

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
        logger.info(f"start iteration {iter+1}/{n_iters}, seed {seeds[iter]}")
        
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
            current_model_params['hidden_size'] = current_model_params['hidden_dims']
            current_model_params['digits'] = int(digits)
            current_training_params['device'] = device
            current_training_params['seed'] = seeds[iter]

            
            

            iter_feature_selection_checkpoint_dir = os.path.join(feature_selection_checkpoint_folder, f'iter_{iter}')
            os.makedirs(iter_feature_selection_checkpoint_dir, exist_ok=True)                
            if current_training_params.get('do_parameter_search', False):
                logger.info(f"hyperparameter tuning using Optuna and training model...")
                logger.info(
                    f"current fixed GRACES architecture before Optuna: n_layers={current_model_params['n_layers']}, "
                    f"start_digit={current_model_params.get('start_digit')}, shrink_ratio={current_model_params.get('shrink_ratio')}, "
                    f"generated_digits_list={current_model_params.get('digits_list')}, hidden_dims={current_model_params.get('hidden_dims')}"
                )

                optuna_dir = os.path.join(eval_folder, f'optuna_iter_{iter}')
                graces_results = optuna_search_GRACES(
                    x_train, y_train, 
                    current_model_params, 
                    current_training_params,
                    optuna_dir, 
                    logger,
                    checkpoint_dir=iter_feature_selection_checkpoint_dir  
                )

                feature_indices = graces_results.get('feature_indices', graces_results['select_rate'])

                best_params = graces_results.get('best_params', None) if isinstance(graces_results, dict) else None
                best_architecture = graces_results.get('best_architecture', None) if isinstance(graces_results, dict) else None
                if best_params is not None:
                    all_best_params.append(best_params)
                    logger.info(f"get best params: {best_params}")
                else:
                    best_params_file = os.path.join(optuna_dir, 'best_params.json')
                    if os.path.exists(best_params_file):
                        try:
                            with open(best_params_file, 'r') as f:
                                best_params = json.load(f)
                            all_best_params.append(best_params)
                            logger.info(f"get best params: {best_params}")
                        except Exception as e:
                            logger.warning(f"failed to read best_params.json: {e}")
                if best_architecture is not None:
                    logger.info(
                        f"best GRACES architecture from Optuna: n_layers={best_architecture.get('n_layers')}, "
                        f"start_digit={best_architecture.get('start_digit')}, shrink_ratio={best_architecture.get('shrink_ratio')}, "
                        f"digits_list={best_architecture.get('digits_list')}, hidden_dims={best_architecture.get('hidden_dims')}"
                    )
            else:
                logger.info(f"train model with fix parameters...")
                
                graces_results = GRACES_model_train(
                    x_train, y_train,
                    current_model_params,
                    current_training_params,
                    iter_folder,
                    logger,
                    checkpoint_dir=iter_feature_selection_checkpoint_dir
                )

                feature_indices = graces_results.get('feature_indices', graces_results['select_rate'])

            results_indices.append(feature_indices)

            feature_ranking_file = os.path.join(iter_folder, f'feature_ranking_iter_{iter}.npz')
            np.savez(feature_ranking_file, 
                    feature_indices=feature_indices, 
                    iter=iter, 
                    seed=seeds[iter])
            logger.info(f"feature ranking has been saved to: {feature_ranking_file}")

            feature_steps_range = range(n_steps)

            for step in feature_steps_range:
                num_features = int(feature_sequence[step])
                if num_features > len(feature_indices):
                    logger.warning(f"requested number of features {num_features} exceeds available features {len(feature_indices)}, skipping")
                    continue

                logger.info(f"\nEvaluating top {num_features} features")

                try:
     
                    selected_features = feature_indices[:num_features]
                    logger.info(f"selected_features: {selected_features}...")

                    if len(selected_features) > 0:
                        latest_feature = selected_features[-1]
                        latest_feature_train_values = (
                            x_train[:, latest_feature].detach().cpu().numpy().tolist()
                            if isinstance(x_train, torch.Tensor)
                            else np.asarray(x_train[:, latest_feature]).tolist()
                        )
                        latest_feature_test_values = (
                            x_test[:, latest_feature].detach().cpu().numpy().tolist()
                            if isinstance(x_test, torch.Tensor)
                            else np.asarray(x_test[:, latest_feature]).tolist()
                        )
                        # logger.info(
                        #     f"Evaluation top-{num_features}: latest selected feature index = {latest_feature}, "
                        #     f"train values = {latest_feature_train_values}"
                        # )
                        # logger.info(
                        #     f"Evaluation top-{num_features}: latest selected feature index = {latest_feature}, "
                        #     f"test values = {latest_feature_test_values}"
                        # )
                    x_train_selected = x_train[:, selected_features]
                    x_test_selected  = x_test[:, selected_features]
                    
                    task_type = model_params.get('task_type', 'binary')
                    y_train_np = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else np.asarray(y_train)
                    y_test_np  = y_test.detach().cpu().numpy() if isinstance(y_test, torch.Tensor) else np.asarray(y_test)
                    
                    model_params_local = model_params.copy()
                    
                    if task_type == 'binary':
                        y_train_np = y_train_np.astype(np.float32).reshape(-1, 1)
                        y_test_np  = y_test_np.astype(np.float32).reshape(-1, 1)
                    elif task_type == 'multiclass':
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
                    logger.info(f"The results of the number of features {num_features} have been saved to: {feature_eval_file}")
                    
                    checkpoint_data = {
                        'last_feature_step': step,
                        'current_iter': iter,
                        'current_feature': num_features,
                        'results': results,
                        'cv_val_losses': cv_val_losses,
                        'results_indices': np.array(results_indices, dtype=object),
                        'all_best_params': np.array(all_best_params, dtype=object) if all_best_params else None,
                        'seeds': seeds,
                        'feature_sequence': feature_sequence,
                        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                        'per_step_report_metrics': np.array(all_report_metrics, dtype=object)
                    }
                    np.savez(checkpoint_file, **checkpoint_data)
                    logger.info(f"Checkpoint updated, completed evaluation of {num_features} features ({step+1}/{n_steps}), current iteration {iter+1}/{n_iters}")
                    
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
                    feature_sequence=feature_sequence,
                    seed=seeds[iter],
                    timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                    report_metrics_column=report_metrics_iter_column)  
            logger.info(f"The full results of iteration {iter+1} have been saved to: {iter_results_file}")

            checkpoint_data = {
                'last_completed_iter': iter,
                'last_feature_step': n_steps - 1, 
                'results': results,
                'cv_val_losses': cv_val_losses,
                'results_indices': np.array(results_indices, dtype=object),
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
            logger.error(f"Iteration {iter} error: {str(e)}")
            logger.error(traceback.format_exc())

            continue
    

    total_time = time.time() - total_start_time
    logger.info(f"\nALL iterations completed, total time: {format_time(total_time)} ({total_time:.2f} seconds)")
    

    mean_test_results = np.nanmean(results, axis=1)
    std_test_results = np.nanstd(results, axis=1)
    mean_cv_val_losses = np.nanmean(cv_val_losses, axis=1)
    std_cv_val_losses = np.nanstd(cv_val_losses, axis=1)
    
    final_results = {
        'test_results': results,
        'cv_val_losses': cv_val_losses,
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
    

    save_path = os.path.join(folder_name, f'GRACES_{name}_results.npz')
    np.savez(save_path, **final_results)

    # timestamp_save_path = os.path.join(
    #     folder_name,
    #     f'GRACES_{name}_results_{final_results["timestamp"]}.npz'
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


    return final_results


def resolve_hidden_size_list(hidden_size, min_layers=2, min_width=4):
    if hidden_size is None:
        hidden_size = [64, 32]
    elif isinstance(hidden_size, int):
        hidden_size = [hidden_size]
    elif isinstance(hidden_size, tuple):
        hidden_size = list(hidden_size)
    elif not isinstance(hidden_size, list):
        raise ValueError("hidden_size must be None, int, list, or tuple")

    if len(hidden_size) == 0:
        raise ValueError("hidden_size must not be empty")
    if len(hidden_size) < min_layers:
        raise ValueError(f"hidden_size must contain at least {min_layers} layers, got {len(hidden_size)}")

    normalized = []
    for h in hidden_size:
        h_int = int(h)
        if h_int <= 0:
            raise ValueError(f"Each hidden layer width must be positive, got {h}")
        normalized.append(max(min_width, h_int))
    return normalized

class GraphConvNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, alpha, activation='relu', graph_layer_mode='all_sage'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = get_activation(activation)
        self.softmax = nn.Softmax(dim=1)
        self.input = nn.Linear(self.input_size, self.hidden_size[0], bias=False)
        self.input_bn = nn.BatchNorm1d(self.hidden_size[0])

        self.alpha = alpha
        self.graph_layer_mode = graph_layer_mode
        self.hiddens = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.mlp_hiddens = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        if self.graph_layer_mode == 'one_sage_then_mlp':
            if len(self.hidden_size) >= 2:
                self.hiddens.append(gnn.SAGEConv(self.hidden_size[0], self.hidden_size[1]))
                self.bns.append(nn.BatchNorm1d(self.hidden_size[1]))
            for h in range(1, len(self.hidden_size) - 1):
                self.mlp_hiddens.append(nn.Linear(self.hidden_size[h], self.hidden_size[h + 1], bias=False))
                self.mlp_bns.append(nn.BatchNorm1d(self.hidden_size[h + 1]))
        else:
            for h in range(len(self.hidden_size) - 1):
                self.hiddens.append(gnn.SAGEConv(self.hidden_size[h], self.hidden_size[h + 1]))
                self.bns.append(nn.BatchNorm1d(self.hidden_size[h + 1]))

        self.output = nn.Linear(hidden_size[-1], output_size)
        self.is_classification = output_size > 1

    def forward(self, x, edge_index):
        x = self.input(x)
        x = self.input_bn(x)
        x = self.activation(x)

        if self.graph_layer_mode == 'one_sage_then_mlp':
            if len(self.hiddens) > 0:
                x = self.hiddens[0](x, edge_index)
                x = self.bns[0](x)
                x = self.activation(x)
            for hidden, bn in zip(self.mlp_hiddens, self.mlp_bns):
                x = hidden(x)
                x = bn(x)
                x = self.activation(x)
        else:
            for hidden, bn in zip(self.hiddens, self.bns):
                x = hidden(x, edge_index)
                x = bn(x)
                x = self.activation(x)

        x = self.output(x)
        return x


class GRACES:
    """
    GRACES feature selection class
    Reference: GRACES: Graph convolutional network-based feature selection for high-dimensional and low-sample size data

    Args:
        max_features (int): The maximum number of features to select.
        task_type (str): The type of task (e.g., "regression", "binary", "multiclass").
        hidden_size (list, optional): The size of the hidden layers.
        q (int, optional): The number of top features to select.
        n_dropouts (int, optional): The number of dropout layers.
        dropout_prob (float, optional): The dropout probability.
        batch_size (int, optional): The batch size for training.
        learning_rate (float, optional): The learning rate for the optimizer.
        epochs (int, optional): The number of training epochs.
        alpha (float, optional): The weight for the graph convolutional layer.
        sigma (float, optional): The weight for the feature selection layer.
        f_correct (float, optional): The weight for the feature correction term.
        patience (int, optional): The number of epochs to wait for improvement.
        min_delta (float, optional): The minimum change to qualify as an improvement.
        device (str, optional): The device to run the model on (e.g., "cpu", "cuda").
        activation (str, optional): The activation function to use.
        training_params (dict, optional): Additional training parameters.
    """
    def __init__(self, max_features, task_type="regression", hidden_size=None, q=2, n_dropouts=10, 
                 dropout_prob=0.5, batch_size=16, learning_rate=0.001, epochs=50, 
                 alpha=0.95, sigma=0, f_correct=0, patience=10, min_delta=0.001, 
                 device='cpu', activation='relu', training_params=None):
        self.max_features = max_features
        self.activation = activation
        self.q = q
        self.task_type = task_type
        self.hidden_size = resolve_hidden_size_list(hidden_size)
        self.n_dropouts = n_dropouts
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.sigma = sigma
        self.f_correct = f_correct
        self.S = None
        self.new = None
        self.model = None
        self.last_model = None
        self.loss_fn = None
        self.f_scores = None
        self.patience = patience
        self.min_delta = min_delta
        self.device = device
        self.training_params = training_params if training_params is not None else {}
        self.graph_layer_mode = self.training_params.get('graph_layer_mode', 'all_sage')
    @staticmethod
    def bias(x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        if not torch.all(x[:, 0] == 1):
            bias_col = torch.ones(
                (x.shape[0], 1),
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat((bias_col, x), dim=1)
        return x



    def f_test(self, x, y):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = y
    
        y_np = y_np.reshape(-1)
    
        if self.task_type in ["binary", "multiclass"]:
            slc = SelectKBest(f_classif, k=x_np.shape[1])
        else:  # regression
            slc = SelectKBest(f_regression, k=x_np.shape[1])
        slc.fit(x_np, y_np)
        return getattr(slc, 'scores_')
    
    def xavier_initialization(self):
        if self.last_model is not None:
            weight = torch.zeros(
                (self.hidden_size[0], len(self.S)),
                dtype=self.model.input.weight.dtype,
                device=self.model.input.weight.device,
            )
            nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain('relu'))
            old_s = self.S.copy()
            if self.new in old_s:
                old_s.remove(self.new)
            for i in self.S:
                if i != self.new:
                    weight[:, self.S.index(i)] = self.last_model.input.weight.data[:, old_s.index(i)]
            self.model.input.weight.data = weight
            if self.graph_layer_mode == 'one_sage_then_mlp':
                if len(self.hidden_size) >= 2 and len(self.model.hiddens) > 0 and len(self.last_model.hiddens) > 0:
                    self.model.hiddens[0].lin_l.weight.data = self.last_model.hiddens[0].lin_l.weight.data
                    self.model.hiddens[0].lin_r.weight.data = self.last_model.hiddens[0].lin_r.weight.data
                for h in range(len(self.model.mlp_hiddens)):
                    self.model.mlp_hiddens[h].weight.data = self.last_model.mlp_hiddens[h].weight.data
            else:
                for h in range(len(self.hidden_size) - 1):
                    self.model.hiddens[h].lin_l.weight.data = self.last_model.hiddens[h].lin_l.weight.data
                    self.model.hiddens[h].lin_r.weight.data = self.last_model.hiddens[h].lin_r.weight.data
            self.model.output.weight.data = self.last_model.output.weight.data

    def train(self, x, y, x_val=None, y_val=None, verbose=False, log_callback=None):

        def log_message(msg):

            if log_callback:
                log_callback(msg)
            elif verbose:
                print(msg)


        # log_message(f"The size of feature set S: {len(self.S)}")
        input_size = len(self.S)
        if self.task_type == "binary":
            output_size = 1
        elif self.task_type == "multiclass":
            output_size = len(torch.unique(y))
        else:  # regression
            output_size = 1


        # log_message(f"The size of hidden layers: {self.hidden_size}")
        # log_message(f"The size of output layer: {output_size}")

        self.model = GraphConvNet(
            input_size, output_size, self.hidden_size, self.alpha, self.activation,
            graph_layer_mode=self.graph_layer_mode
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # log_message("================ Model Architecture ================")
        # log_message(
        #     f"Graph layer mode: {self.graph_layer_mode} | "
        #     f"input_size={input_size}, hidden_size={self.hidden_size}, output_size={output_size}, activation={self.activation}"
        # )
        # log_message(str(self.model))
        # log_message(f"Total parameters: {total_params:,}")
        # log_message(f"Trainable parameters: {trainable_params:,}")
        # log_message("====================================================")

        self.xavier_initialization()
        x = x[:, self.S]

        if x_val is not None and y_val is not None:
            x_val = x_val[:, self.S]
        # log_message(f"The shape of training data: {x.shape}")
        # if x_val is not None:
        #     log_message(f"The shape of validation data: {x_val.shape}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_set = []
        for i in range(x.shape[0]):
            train_set.append([x[i, :], y[i]])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)


        best_val_loss = float('inf')
        best_epoch = -1
        best_model = None
        counter = 0
        train_losses = []
        val_losses = []
        
        if self.task_type == "binary":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.task_type == "multiclass":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

        for e in range(self.epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            for data, label in train_loader:
                if len(data) <= 1:  
                    continue
                data = data.to(device=self.training_params['device'], dtype=torch.float32)
                label = label.to(device=self.training_params['device'])

                input_0 = data.view(data.shape[0], -1)
                optimizer.zero_grad()
                edge_index_0 = self.create_edge_index(input_0.float())
                output = self.model(input_0.float(), edge_index_0)
                loss = self.loss_fn(output, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)

            train_loss = train_loss / len(train_set)
            train_losses.append(train_loss)


            if x_val is not None and y_val is not None:
                val_loss = self.validate(x_val, y_val)
                val_losses.append(val_loss)


                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    best_epoch = e
                    best_model = copy.deepcopy(self.model)
                    counter = 0
                else:
                    counter += 1


                if counter >= self.patience:
                    # log_message(f"Early stopping at epoch {e+1}/{self.epochs}")
                    if best_model is not None:
                        self.model = best_model
                    break
                # if (e+1) % 10 == 0 or e == 0:
                    # log_message(f"Epoch {e+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if x_val is not None and y_val is not None and best_model is not None:
            self.model = best_model

        self.last_model = copy.deepcopy(self.model)
        return train_losses, val_losses, (best_val_loss if x_val is not None and y_val is not None else None), best_epoch
    
    def validate(self, x_val, y_val):

        self.model.eval()

        val_set = []
        for i in range(x_val.shape[0]):
            val_set.append([x_val[i, :], y_val[i]])
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(x_val), shuffle=False)
        total_loss = 0
        with torch.no_grad():
            for data, label in val_loader:
                if len(data) <= 1: 
                    continue
                data = data.to(device=self.training_params['device'], dtype=torch.float32)
                label = label.to(device=self.training_params['device'])
                
                input_0 = data.view(data.shape[0], -1)
                edge_index_0 = self.create_edge_index(input_0.float())
                output = self.model(input_0.float(), edge_index_0)
                loss = self.loss_fn(output, label)
                total_loss += loss.item() * data.size(0)
        return total_loss / len(val_set)

    def dropout(self):
        model_dp = copy.deepcopy(self.model)
        if self.graph_layer_mode == 'one_sage_then_mlp':
            if len(self.hidden_size) >= 2 and len(model_dp.hiddens) > 0:
                h_size = self.hidden_size[0]
                dropout_index = np.random.choice(range(h_size), int(h_size * self.dropout_prob), replace=False)
                target = model_dp.hiddens[0].lin_l.weight.data[:, dropout_index]
                model_dp.hiddens[0].lin_l.weight.data[:, dropout_index] = torch.zeros_like(target)
                target = model_dp.hiddens[0].lin_r.weight.data[:, dropout_index]
                model_dp.hiddens[0].lin_r.weight.data[:, dropout_index] = torch.zeros_like(target)
            for h in range(len(model_dp.mlp_hiddens)):
                h_size = model_dp.mlp_hiddens[h].weight.data.shape[1]
                dropout_index = np.random.choice(range(h_size), int(h_size * self.dropout_prob), replace=False)
                target = model_dp.mlp_hiddens[h].weight.data[:, dropout_index]
                model_dp.mlp_hiddens[h].weight.data[:, dropout_index] = torch.zeros_like(target)
        else:
            for h in range(len(self.hidden_size) - 1):
                h_size = self.hidden_size[h]
                dropout_index = np.random.choice(range(h_size), int(h_size * self.dropout_prob), replace=False)
                target = model_dp.hiddens[h].lin_l.weight.data[:, dropout_index]
                model_dp.hiddens[h].lin_l.weight.data[:, dropout_index] = torch.zeros_like(target)
                target = model_dp.hiddens[h].lin_r.weight.data[:, dropout_index]
                model_dp.hiddens[h].lin_r.weight.data[:, dropout_index] = torch.zeros_like(target)
        dropout_index = np.random.choice(range(self.hidden_size[-1]), int(self.hidden_size[-1] * self.dropout_prob), replace=False)
        target = model_dp.output.weight.data[:, dropout_index]
        model_dp.output.weight.data[:, dropout_index] = torch.zeros_like(target)
        return model_dp

    def gradient(self, x, y, model, edge_index):
        if self.task_type == "binary":
            ou_size = 1
        elif self.task_type == "multiclass":
            ou_size = len(torch.unique(y))
        else:
            ou_size = 1
        model_gr = GraphConvNet(
            x.shape[1], ou_size, self.hidden_size, self.alpha, self.activation,
            graph_layer_mode=self.graph_layer_mode
        ).to(self.device)

        temp = torch.zeros_like(model_gr.input.weight)

        if len(self.S) > model.input.weight.shape[1]:
            self.S = self.S[:model.input.weight.shape[1]]
        
        temp[:, self.S] = model.input.weight
        model_gr.input.weight.data = temp
        if self.graph_layer_mode == 'one_sage_then_mlp':
            if len(self.hidden_size) >= 2 and len(model_gr.hiddens) > 0 and len(model.hiddens) > 0:
                model_gr.hiddens[0].lin_l.weight.data = model.hiddens[0].lin_l.weight + self.sigma * torch.randn_like(model.hiddens[0].lin_l.weight)
                model_gr.hiddens[0].lin_r.weight.data = model.hiddens[0].lin_r.weight + self.sigma * torch.randn_like(model.hiddens[0].lin_r.weight)
            for h in range(len(model_gr.mlp_hiddens)):
                model_gr.mlp_hiddens[h].weight.data = model.mlp_hiddens[h].weight + self.sigma * torch.randn_like(model.mlp_hiddens[h].weight)
        else:
            for h in range(len(self.hidden_size) - 1):
                model_gr.hiddens[h].lin_l.weight.data = model.hiddens[h].lin_l.weight + self.sigma * torch.randn_like(model.hiddens[h].lin_l.weight)
                model_gr.hiddens[h].lin_r.weight.data = model.hiddens[h].lin_r.weight + self.sigma * torch.randn_like(model.hiddens[h].lin_r.weight)
        model_gr.output.weight.data = model.output.weight
        output_gr = model_gr(x.float(), edge_index)
        # if torch.isnan(output_gr).any() or torch.isinf(output_gr).any():
        #     print(
        #         f"gradient(): output_gr has invalid values | nan_count={torch.isnan(output_gr).sum().item()}, "
        #         f"posinf_count={torch.isinf(output_gr).logical_and(output_gr > 0).sum().item()}, "
        #         f"neginf_count={torch.isinf(output_gr).logical_and(output_gr < 0).sum().item()}"
        #     )
        loss_gr = self.loss_fn(output_gr, y)
        # if torch.isnan(loss_gr).any() or torch.isinf(loss_gr).any():
        #     print(
        #         f"gradient(): loss_gr is invalid | loss={loss_gr.item() if loss_gr.numel() == 1 else loss_gr}"
        #     )
        loss_gr.backward()
        input_gradient = model_gr.input.weight.grad
        # if input_gradient is None:
        #     print("gradient(): model_gr.input.weight.grad is None")
        # else:
        #     print(
        #         f"gradient(): input_gradient stats | shape={tuple(input_gradient.shape)}, "
        #         f"nan_count={torch.isnan(input_gradient).sum().item()}, "
        #         f"posinf_count={torch.isinf(input_gradient).logical_and(input_gradient > 0).sum().item()}, "
        #         f"neginf_count={torch.isinf(input_gradient).logical_and(input_gradient < 0).sum().item()}"
        #     )
        return input_gradient

    def average(self, x, y, n_average, verbose=False, log_callback=None):
        def log_message(msg):
            if log_callback:
                log_callback(msg)
            elif verbose:
                print(msg)

        grad_cache = None
        edge_index = self.create_edge_index(x.float())
        # log_message(
        #     f"average(): x shape={tuple(x.shape)}, n_average={n_average}, edge_index shape={tuple(edge_index.shape)}"
        # )
        for num in range(n_average):
            model = self.dropout()
            input_grad = self.gradient(x, y, model, edge_index)

            # input_grad_nan_count = torch.isnan(input_grad).sum().item()
            # input_grad_posinf_count = torch.isinf(input_grad).logical_and(input_grad > 0).sum().item()
            # input_grad_neginf_count = torch.isinf(input_grad).logical_and(input_grad < 0).sum().item()
            # input_grad_min = torch.nan_to_num(input_grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).min().item()
            # input_grad_max = torch.nan_to_num(input_grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).max().item()
            # log_message(
            #     f"average() pass {num + 1}/{n_average}: input_grad shape={tuple(input_grad.shape)}, "
            #     f"nan_count={input_grad_nan_count}, posinf_count={input_grad_posinf_count}, "
            #     f"neginf_count={input_grad_neginf_count}, min={input_grad_min:.6f}, max={input_grad_max:.6f}"
            # )

            if grad_cache is None:
                grad_cache = input_grad
            else:
                grad_cache += input_grad

            # grad_cache_nan_count = torch.isnan(grad_cache).sum().item()
            # grad_cache_posinf_count = torch.isinf(grad_cache).logical_and(grad_cache > 0).sum().item()
            # grad_cache_neginf_count = torch.isinf(grad_cache).logical_and(grad_cache < 0).sum().item()
            # log_message(
            #     f"average() accumulated after pass {num + 1}: nan_count={grad_cache_nan_count}, "
            #     f"posinf_count={grad_cache_posinf_count}, neginf_count={grad_cache_neginf_count}"
            # )

        averaged_grad = grad_cache / n_average
        # averaged_grad_nan_count = torch.isnan(averaged_grad).sum().item()
        # averaged_grad_posinf_count = torch.isinf(averaged_grad).logical_and(averaged_grad > 0).sum().item()
        # averaged_grad_neginf_count = torch.isinf(averaged_grad).logical_and(averaged_grad < 0).sum().item()
        # log_message(
        #     f"average() final averaged_grad: shape={tuple(averaged_grad.shape)}, nan_count={averaged_grad_nan_count}, "
        #     f"posinf_count={averaged_grad_posinf_count}, neginf_count={averaged_grad_neginf_count}"
        # )
        return averaged_grad

    def find(self, input_gradient, verbose=False, log_callback=None):
        def log_message(msg):
            if log_callback:
                log_callback(msg)
            elif verbose:
                print(msg)

        gradient_norm = input_gradient.norm(p=self.q, dim=0)

        self.f_scores = self.f_scores.to(device=gradient_norm.device, dtype=gradient_norm.dtype)

        gradient_norm_l2 = gradient_norm.norm(p=2)
        gradient_norm = gradient_norm / gradient_norm_l2

        gradient_norm[1:] = (1 - self.f_correct) * gradient_norm[1:] + self.f_correct * self.f_scores


        gradient_norm[self.S] = 0
        nan_mask = torch.isnan(gradient_norm)
        max_index = torch.argmax(gradient_norm)
        return max_index.item()

    def select(self, x, y, x_val=None, y_val=None, verbose=False, log_callback=None, checkpoint_dir=None, resume=True):

        def log_message(msg):

            if log_callback:
                log_callback(msg)
            elif verbose:
                print(msg)

        checkpoint_path = None
        if checkpoint_dir:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'graces_feature_selection_checkpoint.npz')
            log_message(f"The checkpoint will be saved at: {checkpoint_path}")

        original_feature_indices = None

        if checkpoint_path and os.path.exists(checkpoint_path) and resume:
            try:
                log_message(f"Found feature selection checkpoint file, restoring...")
                checkpoint = np.load(checkpoint_path, allow_pickle=True)
                self.S = checkpoint['S'].tolist()
                self.f_scores = torch.as_tensor(checkpoint['f_scores'], device=self.device, dtype=torch.float32)
                self.new = checkpoint['new'].item() if 'new' in checkpoint else None
                if 'original_feature_indices' in checkpoint:
                    original_feature_indices = checkpoint['original_feature_indices']
                if 'last_model_state' in checkpoint:
                    self.last_model = copy.deepcopy(self.model) if self.model else None
                    if self.last_model:
                        # 恢复模型状态
                        last_model_state = checkpoint['last_model_state'].item()
                        self.last_model.load_state_dict(last_model_state)

                log_message(f"Restored from checkpoint, currently selected {len(self.S)-1} features")

                # Check if the target number of features has been reached
                if len(self.S) >= self.max_features + 1:
                    log_message(f"Reached target number of features {self.max_features}, no need to continue selection")
                    selected_features = []
                    for feature_idx in self.S[1:]:
                        if feature_idx >= 1:
                            filtered_feature_index = feature_idx - 1
                            if original_feature_indices is not None and 0 <= filtered_feature_index < len(original_feature_indices):
                                selected_features.append(int(original_feature_indices[filtered_feature_index]))
                    return selected_features, [], []

            except Exception as e:
                log_message(f"Failed to restore from checkpoint: {str(e)}")
                log_message("Starting feature selection from scratch")
                # 重置状态
                self.S = None
                self.f_scores = None
                self.new = None
                self.last_model = None

        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        y = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
        x = x.to(self.device)
        y = y.to(self.device)
        if x_val is not None and y_val is not None:
            x_val = x_val if isinstance(x_val, torch.Tensor) else torch.as_tensor(x_val)
            y_val = y_val if isinstance(y_val, torch.Tensor) else torch.as_tensor(y_val)
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)

        if self.task_type == "regression":
            if len(y.shape) == 1:
                y = y.float().reshape(-1, 1)
            if y_val is not None and len(y_val.shape) == 1:
                y_val = y_val.float().reshape(-1, 1)
        elif self.task_type == "binary":
            y = y.float().view(-1, 1)
            if y_val is not None:
                y_val = y_val.float().view(-1, 1)
        elif self.task_type == "multiclass":
            y = y.long().view(-1)
            if y_val is not None:
                y_val = y_val.long().view(-1)

        # Remove constant features before f_test, with optional per-class filtering on train only.
        original_feature_indices = np.arange(x.shape[1])

                





        self.f_scores = torch.as_tensor(self.f_test(x, y), device=self.device, dtype=torch.float32)



        self.f_scores = torch.nan_to_num(self.f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        self.f_scores[self.f_scores < 0] = 0


        f_scores_norm = self.f_scores.norm(p=2)
        if f_scores_norm.item() == 0:
            log_message("select(): f_scores L2 is 0 after cleaning, skip normalization and keep all-zero f_scores")
        else:
            self.f_scores = self.f_scores / f_scores_norm

        x = self.bias(x)
        if x_val is not None:
            x_val = self.bias(x_val)
                

        if self.task_type == "binary":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.task_type == "multiclass":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

            if len(y.shape) == 1:
                y = y.float().reshape(-1, 1)
            if x_val is not None and y_val is not None and len(y_val.shape) == 1:
                y_val = y_val.float().reshape(-1, 1)
            

        if self.S is None:
            self.S = [0]


        # log_message(f"Loss function is: {self.loss_fn}")
        # log_message(f"Input feature count: {x.shape}")
        # if x_val is not None:
        #     log_message(f"Validation feature count: {x_val.shape}")

        selected_features = []
        train_loss_history = []
        val_loss_history = []


        if self.S and len(self.S) > 1:
            for feature_idx in self.S[1:]:
                if feature_idx >= 1:
                    filtered_feature_index = feature_idx - 1
                    if 0 <= filtered_feature_index < len(original_feature_indices):
                        selected_features.append(int(original_feature_indices[filtered_feature_index]))



        while len(self.S) < self.max_features + 1:
            log_message(f"selecting {len(self.S)-1}/{self.max_features} features")

            train_losses, val_losses, best_val_loss, best_epoch = self.train(
                x, y, x_val, y_val, verbose=verbose, log_callback=log_callback
            )
            train_loss_history.extend(train_losses)
            if x_val is not None and y_val is not None:
                val_loss_history.extend(val_losses)

            input_gradient = self.average(x, y, self.n_dropouts, verbose=verbose, log_callback=log_callback)
            self.new = self.find(input_gradient, verbose=verbose, log_callback=log_callback)
            self.S.append(self.new)


            filtered_feature_index = self.new - 1
            new_feature = original_feature_indices[filtered_feature_index].item() if filtered_feature_index >= 0 else -1
            log_message(f"New feature index: {new_feature}")
            log_message(f"Current feature set: {self.S}")
            if new_feature >= 0:
                selected_features.append(new_feature)
                selected_feature_f_score = self.f_scores[filtered_feature_index].item() if filtered_feature_index < len(self.f_scores) else None
                log_message(
                    f"Selected feature: {new_feature}, total selected: {len(selected_features)}, "
                )
                selected_feature_train_values = x[:, self.new].detach().cpu().numpy().tolist()
                # log_message(f"Selected feature {new_feature} values on train samples: {selected_feature_train_values}")

            if checkpoint_path and (len(self.S) % 10 == 0 or len(self.S) == self.max_features + 1):
                try:
                    last_model_state = self.last_model.state_dict() if self.last_model else None

                    np.savez(
                        checkpoint_path,
                        S=np.array(self.S),
                        f_scores=self.f_scores.detach().cpu().numpy(),
                        new=np.array(self.new),
                        original_feature_indices=np.array(original_feature_indices),
                        last_model_state=np.array([last_model_state]) if last_model_state else None,
                        timestamp=np.array([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]),
                        current_feature_count=np.array([len(self.S)-1]),
                        max_features=np.array([self.max_features])
                    )
                    log_message(f"Checkpoint saved, current feature count: {len(self.S)-1}")
                except Exception as e:
                    log_message(f"Checkpoint saving failed: {str(e)}")

        return selected_features, train_loss_history, val_loss_history
    
    def create_edge_index(self, x):
        x_normalized = F.normalize(x, p=2, dim=1)
        similarity_matrix = torch.mm(x_normalized, x_normalized.t())
        similarity_matrix = torch.abs(similarity_matrix)
        similarity = torch.sort(similarity_matrix.view(-1))[0]
        eps = torch.quantile(similarity, self.alpha, interpolation='nearest')
        adj_matrix = similarity_matrix >= eps
        row, col = torch.where(adj_matrix)
        edge_index = torch.cat((row.reshape(1, -1), col.reshape(1, -1)), dim=0)
        return edge_index
    
    
def GRACES_fs(data, label, model_params, training_params,name,  digits, folder_name, logger):
    """
    Main function for GRACES feature selection with optional parameter optimization
    Args:
        data (dict): Dictionary containing training data 'X_train'.
        label (dict): Dictionary containing training labels 'y_train'.
        model_params (dict): Model parameters for GRACES.
        training_params (dict): Training parameters including 'do_parameter_search'.
        digits (int): Number of digits for formatting.  
        folder_name (str): Folder to save results and checkpoints.
        logger (logging.Logger): Logger for logging information.
    """
    X_train = data['X_train']
    y_train = label['y_train']

    checkpoint_dir = os.path.join(folder_name, 'feature_selection_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    

    do_parameter_search = training_params.get('do_parameter_search', 0)
    

    if do_parameter_search == 1:
        results = optuna_search_GRACES(
            X_train, y_train, model_params, training_params, folder_name,
            logger, checkpoint_dir=checkpoint_dir
        )
    else:
        logger.info("Parameter optimization is disabled. Using single parameter values.")
        logger.info("Using parameters:")
        logger.info(f"  alpha: {model_params.get('alpha', 0.95)}")
        logger.info(f"  dropout: {model_params.get('dropout_prob', 0.5)}")
        logger.info(f"  q: {model_params.get('q', 2)}")
        logger.info(f"  f_correct: {model_params.get('f_correct', 0)}")
        logger.info(f"  lr: {training_params.get('lr', 0.001)}")
        logger.info(f"  batch_size: {training_params.get('batch_size', 32)}")
        logger.info(f"  start_digit: {model_params.get('start_digit', model_params.get('digits', 100))}")
        logger.info(f"  shrink_ratio: {model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0))}")
        logger.info(f"  n_layers: {model_params.get('n_layers', 2)}")
        logger.info(f"  generated_digits_list: {model_params.get('digits_list', None)}")
        logger.info(f"  hidden_dims: {model_params.get('hidden_dims', model_params.get('hidden_size', None))}")
        logger.info("  use_cv: False (no Optuna CV because do_parameter_search=0)")

        results = GRACES_model_train(
            X_train, y_train, model_params, training_params, folder_name,
            logger, checkpoint_dir=checkpoint_dir
        )

    np.save(os.path.join(folder_name, f'GRACES_{name}_idx.npy'),
            results.get('feature_indices', results['select_rate']))

    logger.info(f"Files saved to folder: {folder_name}")
    return results



def optuna_search_GRACES(data, label, model_params, training_params, folder_name,  logger, checkpoint_dir=None):
    """
    Perform feature selection using GRACES method with Optuna hyperparameter optimization,
    or use existing best parameters if available.
    Args:
        data : Training data features.
        label : Training data labels.
        model_params (dict): Model parameters for GRACES.
        training_params (dict): Training parameters including optimization settings.
        folder_name (str): Folder to save results and checkpoints.
        logger (logging.Logger): Logger for logging information.
        checkpoint_dir (str, optional): Directory for saving checkpoints.
    
    """
 

    total_start_time = time.time()
    logger.info(f" optuna_search_GRACES start at  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    best_params_file = os.path.join(folder_name, 'best_params.json')
    best_params_txt = os.path.join(folder_name, 'best_params.txt')
    logger.info(f"best_params_file: {best_params_file}")
    best_params = None
    

    if os.path.exists(best_params_file):
        try:
            logger.info(f"find the file of best parameters: {best_params_file}")
            with open(best_params_file, 'r') as f:
                best_params = json.load(f)
            logger.info(f"successfully load the file of best parameters: {best_params}")
        except Exception as e:
            logger.warning(f"cannot load JSON file  : {str(e)}")
            best_params = None
    
    if best_params is None and os.path.exists(best_params_txt):
        try:
            logger.info(f"try to parse best parameters from text file: {best_params_txt}")
            best_params = {}
            in_hyperparams_section = False
            with open(best_params_txt, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line == 'Best hyperparameters:':
                        in_hyperparams_section = True
                        continue
                    if line == 'Best architecture:':
                        break
                    if not in_hyperparams_section:
                        continue
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        try:
                            if '.' in value:
                                best_params[key] = float(value)
                            else:
                                best_params[key] = int(value)
                        except ValueError:
                            best_params[key] = value
            
            if best_params:
                logger.info(f"successfully parse parameters from text file: {best_params}")
                with open(best_params_file, 'w') as f:
                    json.dump(best_params, f, indent=4)
            else:
                logger.warning("cannot parse any parameters from text file")
                best_params = None
        except Exception as e:
            logger.warning(f"error occurred while parsing text parameter file: {str(e)}")
            best_params = None
    

    if best_params:
        logger.info("use existing best parameters for feature selection, skip Optuna optimization")

        final_model_start_time = time.time()

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
                max(4, int(data.shape[1] / max(1, int(d))))
                for d in best_digits_list
            ]
            final_model_params['hidden_size'] = final_model_params['hidden_dims']
            final_model_params['digits'] = int(best_start_digit)
        elif 'digits' in best_params:
            fallback_n_layers = max(2, int(final_model_params.get('n_layers', 2)))
            fallback_start_digit = int(best_params['digits'])
            fallback_shrink_ratio = float(final_model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0)))
            best_digits_list = [
                max(1, int(round(fallback_start_digit * (fallback_shrink_ratio ** i))))
                for i in range(fallback_n_layers)
            ]
            final_model_params['n_layers'] = fallback_n_layers
            final_model_params['start_digit'] = fallback_start_digit
            final_model_params['shrink_ratio'] = fallback_shrink_ratio
            final_model_params['digits_list'] = best_digits_list
            final_model_params['hidden_dims'] = [
                max(4, int(data.shape[1] / max(1, int(d))))
                for d in best_digits_list
            ]
            final_model_params['hidden_size'] = final_model_params['hidden_dims']
            final_model_params['digits'] = fallback_start_digit
        elif 'hidden_dims' in final_model_params and final_model_params.get('hidden_dims') is not None:
            final_model_params['hidden_dims'] = [max(4, int(h)) for h in final_model_params['hidden_dims']]
            final_model_params['hidden_size'] = final_model_params['hidden_dims']
        else:
            fallback_n_layers = max(2, int(final_model_params.get('n_layers', 2)))
            fallback_start_digit = int(final_model_params.get('start_digit', final_model_params.get('digits', 100)))
            fallback_shrink_ratio = float(final_model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0)))
            final_model_params['n_layers'] = fallback_n_layers
            final_model_params['start_digit'] = fallback_start_digit
            final_model_params['shrink_ratio'] = fallback_shrink_ratio
            final_model_params['digits_list'] = [
                max(1, int(round(fallback_start_digit * (fallback_shrink_ratio ** i))))
                for i in range(fallback_n_layers)
            ]
            final_model_params['hidden_dims'] = [
                max(4, int(data.shape[1] / max(1, int(d))))
                for d in final_model_params['digits_list']
            ]
            final_model_params['hidden_size'] = final_model_params['hidden_dims']

        if 'alpha' in best_params:
            final_model_params['alpha'] = best_params['alpha']
        if 'dropout' in best_params:
            final_model_params['dropout_prob'] = best_params['dropout']
        if 'q' in best_params:
            final_model_params['q'] = best_params['q']
        if 'f_correct' in best_params:
            final_model_params['f_correct'] = best_params['f_correct']

        results = GRACES_model_train(
            data, label, final_model_params, training_params, folder_name,
            logger, best_params, checkpoint_dir=checkpoint_dir
        )

        final_model_time = time.time() - final_model_start_time
        total_time = time.time() - total_start_time

        logger.info(f"using existing best parameters for model training, completed in {format_time(final_model_time)} ({final_model_time/60:.2f} minutes)")
        logger.info(f"total processing time: {format_time(total_time)} ({total_time/60:.2f} minutes)")

        results['best_params'] = copy.deepcopy(best_params)
        results['best_value'] = None
        results['final_model_params'] = copy.deepcopy(final_model_params)
        results['best_architecture'] = {
            'n_layers': final_model_params.get('n_layers'),
            'start_digit': final_model_params.get('start_digit'),
            'shrink_ratio': final_model_params.get('shrink_ratio'),
            'digits_list': final_model_params.get('digits_list'),
            'hidden_dims': final_model_params.get('hidden_dims'),
        }

        return results
    
    logger.info("cannot find existing best parameters, will perform Optuna hyperparameter optimization...")

    use_cv = training_params.get('use_cv', True)
    eval_metric = training_params.get('eval_metric', 'loss')
    if eval_metric is None:
        eval_metric = 'acc' if model_params['task_type'] in ('binary','multiclass') else 'r2'
    

    minimize_metric = eval_metric in ['mse', 'mae', 'loss']    
    logger.info(f"parameter search will use '{eval_metric}' as evaluation metric.")
    logger.info(f"during optimization, the metric will be {'minimized' if minimize_metric else 'maximized'}.")
    logger.info(f"Hyperparameter search use_cv={use_cv}")
    optuna_folder = os.path.join(folder_name, 'optuna_search')
    os.makedirs(optuna_folder, exist_ok=True)




    # timing_log_path = os.path.join(optuna_folder, 'timing_log.txt')
    # with open(timing_log_path, 'w') as f:
    #     f.write(f"GRACES Optuna search timing log - started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    #     f.write("=" * 80 + "\n")


    k_folds = training_params.get('n_splits', 3)

    # kf = KFold(n_splits=k_folds, shuffle=True, random_state=training_params['seed'])

    val_ratio = training_params.get('validation_split', 0.2)
    if use_cv:
        if model_params.get('task_type', 'regression') in ('binary', 'multiclass'):
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=training_params['seed'])
            logger.info(f"Using {k_folds}-fold StratifiedKFold during Optuna search.")
        else:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=training_params['seed'])
            logger.info(f"Using {k_folds}-fold KFold during Optuna search.")
    else:
        logger.info(
            f"use_cv=False, so Optuna search will use a single validation split "
            f"(validation_split={val_ratio}) instead of K-fold CV."
        )


    search_start_time = time.time()

    alpha_min = training_params.get('alpha_min', 0.8)
    alpha_max = training_params.get('alpha_max', 0.99)
    dropout_min = training_params.get('dropout_min', 0.1)
    dropout_max = training_params.get('dropout_max', 0.8)
    f_correct_choices = parse_list_arg(training_params.get('f_correct_list', '0,0.1,0.5,0.9'), dtype=float)
    lr_min = training_params.get('lr_min', 1e-5)
    lr_max = training_params.get('lr_max', 1e-1)
    start_digit_choices = parse_list_arg(training_params.get('digits_list', '50,100,200'), dtype=int)
    shrink_ratio_choices = parse_list_arg(training_params.get('shrink_ratio_list', '1.25,1.5,2.0'), dtype=float)
    batch_size_choices = parse_list_arg(training_params.get('batch_size_list', '16,32,64,128'), dtype=int)
    min_layers = max(2, int(training_params.get('min_layers', 2)))
    max_layers = max(min_layers, int(training_params.get('max_layers', 5)))
    q = model_params['q']
    trial_times = {}


    def objective(trial):
        trial_start_time = time.time()
        trial_id = trial.number
        alpha = trial.suggest_float('alpha', alpha_min, alpha_max)
        dropout = trial.suggest_float('dropout', dropout_min, dropout_max)
        f_correct = trial.suggest_categorical('f_correct', f_correct_choices)
        lr = trial.suggest_float('lr', lr_min, lr_max, log=True)
        batch_size = trial.suggest_categorical('batch_size', batch_size_choices)
        n_layers = trial.suggest_int('n_layers', min_layers, max_layers)
        start_digit = trial.suggest_categorical('start_digit', start_digit_choices)
        shrink_ratio = trial.suggest_categorical('shrink_ratio', shrink_ratio_choices)
        digits_list = [
            max(1, int(round(start_digit * (shrink_ratio ** i))))
            for i in range(n_layers)
        ]
        hidden_size = [
            max(4, int(data.shape[1] / max(1, int(d))))
            for d in digits_list
        ]

        logger.info(f"trial {trial_id}: alpha={alpha}, dropout={dropout}, q={q}, "
                   f"f_correct={f_correct}, lr={lr}, batch_size={batch_size}, "
                   f"n_layers={n_layers}, start_digit={start_digit}, shrink_ratio={shrink_ratio}, "
                   f"generated_digits_list={digits_list}, generated_hidden_dims={hidden_size}")

        fold_scores = []
        fold_times = []
        
        if use_cv:
            all_indices = np.arange(data.shape[0])
            if model_params.get('task_type', 'regression') in ('binary', 'multiclass'):
                if isinstance(label, torch.Tensor):
                    label_np = label.detach().cpu().numpy().ravel()
                else:
                    label_np = np.asarray(label).ravel()
                split_iter = kf.split(all_indices, label_np)
            else:
                split_iter = kf.split(all_indices)
        else:
            all_indices = np.arange(data.shape[0])
            label_np = None
            if model_params['task_type'] in ('binary', 'multiclass'):
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

            graces = GRACES(
                max_features=1,
                task_type=model_params.get('task_type', 'regression'),
                hidden_size=hidden_size,
                q=q,
                dropout_prob=dropout,
                batch_size=batch_size,
                learning_rate=lr,
                epochs=training_params.get('num_epochs', 50),
                alpha=alpha,
                sigma=training_params.get('sigma', 0),
                f_correct=f_correct,
                patience=training_params.get('patience', 10),
                min_delta=training_params.get('min_delta', 0.001),
                device=training_params.get('device', 'cpu'),
                activation=model_params.get('activation', 'relu'),
                training_params=training_params
            )

            # logger.info(f"Before bias() in Optuna train split - first column values: {train_data[:, 0].detach().cpu().numpy().tolist() if isinstance(train_data, torch.Tensor) else np.asarray(train_data[:, 0]).tolist()}")
            # logger.info(f"Before bias() in Optuna validation split - first column values: {val_data[:, 0].detach().cpu().numpy().tolist() if isinstance(val_data, torch.Tensor) else np.asarray(val_data[:, 0]).tolist()}")

            train_data_tensor = graces.bias(train_data)
            val_data_tensor = graces.bias(val_data)

            task_type = model_params.get('task_type', 'regression')
            target_device = training_params.get('device', 'cpu')
            if task_type == 'binary':
                train_labels_tensor = torch.as_tensor(train_labels, dtype=torch.float32, device=target_device).view(-1, 1)
                val_labels_tensor   = torch.as_tensor(val_labels,   dtype=torch.float32, device=target_device).view(-1, 1)
            elif task_type == 'multiclass':
                train_labels_tensor = torch.as_tensor(train_labels, dtype=torch.long, device=target_device)
                val_labels_tensor   = torch.as_tensor(val_labels,   dtype=torch.long, device=target_device)
            else:  # regression
                train_labels_tensor = torch.as_tensor(train_labels, dtype=torch.float32, device=target_device).reshape(-1, 1)
                val_labels_tensor   = torch.as_tensor(val_labels,   dtype=torch.float32, device=target_device).reshape(-1, 1)


            graces.S = list(range(train_data_tensor.shape[1]))
            task_type = model_params.get('task_type', 'regression')

            graces.f_scores = torch.ones(
                train_data_tensor.shape[1],
                dtype=train_data_tensor.dtype,
                device=train_data_tensor.device,
            )

            train_losses, val_losses, best_val_loss, best_epoch = graces.train(
                train_data_tensor, train_labels_tensor,
                val_data_tensor, val_labels_tensor,
                verbose=False,
                log_callback=logger.info
            )

            fold_score = best_val_loss if best_val_loss is not None else (val_losses[-1] if val_losses else float('inf'))
            fold_scores.append(fold_score)
            fold_time = time.time() - fold_start_time
            fold_times.append(fold_time)
            logger.info(f" Trial {trial_id}, Fold {fold+1}/{n_eval_splits} Finished - time consumed: {format_time(fold_time)}")

        avg_score = np.mean(fold_scores)
        trial_total_time = time.time() - trial_start_time
        trial_times[trial_id] = trial_total_time
        logger.info(f" Trial {trial_id} Finished - time consumed: {format_time(trial_total_time)} - average loss: {avg_score:.4f}")
        return avg_score
    
    n_jobs = training_params.get('n_jobs', 1)

    storage = None
    db_url = None
    if n_jobs > 1:
        try:
            db_file = os.path.join(optuna_folder, "optuna_GRACES.db")
            db_url = f"sqlite:///{db_file}"

            storage = RDBStorage(
                url=db_url,
                engine_kwargs={
                    "poolclass": QueuePool,
                    "pool_size": min(n_jobs + 1, 20),
                    "max_overflow": 10,
                    "pool_timeout": 30,
                },
            )
            logger.info(
                f"using RDB storage for parallel optimization, database: {db_url}, parallel workers: {n_jobs}"
            )
        except ImportError:
            logger.warning("Cannot import RDBStorage, using default in-memory storage")
            storage = None
    else:
        logger.info("n_jobs=1, using default in-memory Optuna storage to avoid SQLite/RDB issues.")

    sampler = TPESampler(seed=training_params['seed'])
    study = optuna.create_study(
        sampler=sampler, 
        direction='minimize' if minimize_metric else 'maximize',
        storage=storage,
        study_name=f"GRACES_shrink_ratio_v3",
        load_if_exists=True
    )


    n_trials = training_params.get('n_trials', 20)
    logger.info(
        f"Use {n_jobs} parallel workers to run {n_trials} Optuna optimization trials "
        f"(use_cv={use_cv})"
    )
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, catch=(Exception,))
    except KeyboardInterrupt:
        logger.warning("User interrupted optimization.")
    except Exception as e:
        logger.error(f"Error occurred during optimization: {e}")
        if n_jobs > 1:
            logger.warning("Parallel optimization failed. Retrying with n_jobs=1")
            study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=(Exception,))

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    logger.info(
        f"Optuna trial summary: total={len(study.trials)}, completed={len(completed_trials)}, "
        f"failed={len(failed_trials)}, pruned={len(pruned_trials)}"
    )

    if len(completed_trials) == 0:
        raise RuntimeError(
            "Optuna finished without any completed trials, so best_trial is unavailable. "
            "Please inspect the trial logs above for the first failure."
        )

    best_trial = study.best_trial
    best_params = best_trial.params
    

    search_time = time.time() - search_start_time

    logger.info(f"Optuna optimization finished, elapsed time {format_time(search_time)} ({search_time/60:.2f} minutes)")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best value: {best_trial.value}")
    

    # with open(timing_log_path, 'a') as f:
    #     f.write("\n" + "=" * 40 + "\n")
    #     f.write(f"Search Summary - Completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    #     f.write(f"Total Search Time: {format_time(search_time)} ({search_time:.2f} seconds)\n")
    #     f.write(f"Number of Trials: {n_trials}\n")
    #     f.write(f"Best Parameters: {best_params}\n")
    #     f.write(f"Best Value: {best_trial.value}\n\n")

    #     sorted_trials = sorted(trial_times.items(), key=lambda x: x[1], reverse=True)
    #     f.write("Trial Time Statistics:\n")
    #     f.write(f"  Fastest Trial: {format_time(min(trial_times.values()))}\n")
    #     f.write(f"  Slowest Trial: {format_time(max(trial_times.values()))}\n")
    #     f.write(f"  Average Trial Time: {format_time(sum(trial_times.values()) / len(trial_times))}\n\n")

    #     f.write("Top 5 Longest Trials:\n")
    #     for i, (trial_id, duration) in enumerate(sorted_trials[:5]):
    #         f.write(f"  #{i+1}: Trial {trial_id} - {format_time(duration)} ({duration:.2f} seconds)\n")
    #     f.write("\n")

    logger.info("Train final model with best parameters...")
    final_model_start_time = time.time()

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
            max(4, int(data.shape[1] / max(1, int(d))))
            for d in best_digits_list
        ]
        final_model_params['hidden_size'] = final_model_params['hidden_dims']
        final_model_params['digits'] = int(best_start_digit)
    elif 'digits' in best_params:
        fallback_n_layers = max(2, int(final_model_params.get('n_layers', 2)))
        fallback_start_digit = int(best_params['digits'])
        fallback_shrink_ratio = float(final_model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0)))
        best_digits_list = [
            max(1, int(round(fallback_start_digit * (fallback_shrink_ratio ** i))))
            for i in range(fallback_n_layers)
        ]
        final_model_params['n_layers'] = fallback_n_layers
        final_model_params['start_digit'] = fallback_start_digit
        final_model_params['shrink_ratio'] = fallback_shrink_ratio
        final_model_params['digits_list'] = best_digits_list
        final_model_params['hidden_dims'] = [
            max(4, int(data.shape[1] / max(1, int(d))))
            for d in best_digits_list
        ]
        final_model_params['hidden_size'] = final_model_params['hidden_dims']
        final_model_params['digits'] = fallback_start_digit
    elif 'hidden_dims' in final_model_params and final_model_params.get('hidden_dims') is not None:
        final_model_params['hidden_dims'] = [max(4, int(h)) for h in final_model_params['hidden_dims']]
        final_model_params['hidden_size'] = final_model_params['hidden_dims']
    else:
        fallback_n_layers = max(2, int(final_model_params.get('n_layers', 2)))
        fallback_start_digit = int(final_model_params.get('start_digit', final_model_params.get('digits', 100)))
        fallback_shrink_ratio = float(final_model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0)))
        final_model_params['n_layers'] = fallback_n_layers
        final_model_params['start_digit'] = fallback_start_digit
        final_model_params['shrink_ratio'] = fallback_shrink_ratio
        final_model_params['digits_list'] = [
            max(1, int(round(fallback_start_digit * (fallback_shrink_ratio ** i))))
            for i in range(fallback_n_layers)
        ]
        final_model_params['hidden_dims'] = [
            max(4, int(data.shape[1] / max(1, int(d))))
            for d in final_model_params['digits_list']
        ]
        final_model_params['hidden_size'] = final_model_params['hidden_dims']

    if 'alpha' in best_params:
        final_model_params['alpha'] = best_params['alpha']
    if 'dropout' in best_params:
        final_model_params['dropout_prob'] = best_params['dropout']
    if 'q' in best_params:
        final_model_params['q'] = best_params['q']
    if 'f_correct' in best_params:
        final_model_params['f_correct'] = best_params['f_correct']

    results = GRACES_model_train(
        data, label, final_model_params, training_params, optuna_folder,
        logger, best_params, checkpoint_dir=checkpoint_dir
    )
    final_model_time = time.time() - final_model_start_time
    total_time = time.time() - total_start_time

    logger.info(f"Final model training completed in {format_time(final_model_time)} ({final_model_time/60:.2f} minutes)")
    logger.info(f"Total processing time: {format_time(total_time)} ({total_time/60:.2f} minutes)")
    

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
        f.write(f"search time: {format_time(search_time)} ({search_time:.2f} seconds)\n")
        f.write(f"Final model training time: {format_time(final_model_time)} ({final_model_time:.2f} seconds)\n")
        f.write(f"Total time: {format_time(total_time)} ({total_time:.2f} seconds)\n")


    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=4)

    import joblib
    joblib.dump(study, os.path.join(optuna_folder, "optuna_study.pkl"))
    

    results['best_params'] = copy.deepcopy(best_params)
    results['best_value'] = float(best_trial.value)
    results['final_model_params'] = copy.deepcopy(final_model_params)
    results['best_architecture'] = {
        'n_layers': final_model_params.get('n_layers'),
        'start_digit': final_model_params.get('start_digit'),
        'shrink_ratio': final_model_params.get('shrink_ratio'),
        'digits_list': final_model_params.get('digits_list'),
        'hidden_dims': final_model_params.get('hidden_dims'),
    }
    
    return results




def GRACES_model_train(data, label, model_params, training_params, folder_name, logger, best_params=None, checkpoint_dir=None):
    """
    Train full GRACES model with given parameters or best parameters from optimization
    Args:
        data (np.ndarray or torch.Tensor): Training data features.
        label (np.ndarray or torch.Tensor): Training data labels.
        model_params (dict): Model parameters for GRACES.
        training_params (dict): Training parameters.        
        folder_name (str): Folder to save results and checkpoints.
        logger (logging.Logger): Logger for logging information.
        best_params (dict, optional): Best parameters from optimization. Defaults to None.
        checkpoint_dir (str, optional): Directory for saving checkpoints. Defaults to None.


    """
    if best_params is not None:
        alpha = best_params.get('alpha', model_params.get('alpha', 0.95))
        dropout_prob = best_params.get('dropout', model_params.get('dropout_prob', 0.5))
        q = best_params.get('q', model_params.get('q', 2))
        f_correct = best_params.get('f_correct', model_params.get('f_correct', 0))
        lr = best_params.get('lr', training_params.get('lr', 0.001))
        batch_size = best_params.get('batch_size', training_params.get('batch_size', 32))

        if 'n_layers' in best_params:
            best_n_layers = max(2, int(best_params['n_layers']))
            best_start_digit = int(best_params.get('start_digit', model_params.get('start_digit', model_params.get('digits', 100))))
            best_shrink_ratio = float(best_params.get('shrink_ratio', model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0))))
            best_digits_list = [
                max(1, int(round(best_start_digit * (best_shrink_ratio ** i))))
                for i in range(best_n_layers)
            ]
            hidden_size = [
                max(4, int(data.shape[1] / max(1, int(d))))
                for d in best_digits_list
            ]
        elif 'hidden_dims' in model_params and model_params.get('hidden_dims') is not None:
            hidden_size = [max(4, int(h)) for h in model_params['hidden_dims']]
        elif 'digits_list' in model_params and model_params.get('digits_list') is not None:
            hidden_size = [
                max(4, int(data.shape[1] / max(1, int(d))))
                for d in model_params['digits_list']
            ]
        elif 'digits' in best_params:
            fallback_n_layers = max(2, int(model_params.get('n_layers', 2)))
            fallback_start_digit = int(best_params.get('digits', 100))
            fallback_shrink_ratio = float(model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0)))
            hidden_size = [
                max(4, int(data.shape[1] / max(1, int(round(fallback_start_digit * (fallback_shrink_ratio ** i))))))
                for i in range(fallback_n_layers)
            ]
        else:
            fallback_n_layers = max(2, int(model_params.get('n_layers', 2)))
            fallback_start_digit = int(model_params.get('start_digit', model_params.get('digits', 100)))
            fallback_shrink_ratio = float(model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0)))
            hidden_size = [
                max(4, int(data.shape[1] / max(1, int(round(fallback_start_digit * (fallback_shrink_ratio ** i))))))
                for i in range(fallback_n_layers)
            ]
    else:
        alpha = model_params.get('alpha', 0.95)
        dropout_prob = model_params.get('dropout_prob', 0.5)
        q = model_params.get('q', 2)
        f_correct = model_params.get('f_correct', 0)
        lr = training_params.get('lr', 0.001)
        batch_size = training_params.get('batch_size', 32)

        if 'hidden_dims' in model_params and model_params.get('hidden_dims') is not None:
            hidden_size = [max(4, int(h)) for h in model_params['hidden_dims']]
        elif 'digits_list' in model_params and model_params.get('digits_list') is not None:
            hidden_size = [
                max(4, int(data.shape[1] / max(1, int(d))))
                for d in model_params['digits_list']
            ]
        elif isinstance(model_params.get('hidden_size'), list):
            hidden_size = [max(4, int(h)) for h in model_params['hidden_size']]
        else:
            fallback_n_layers = max(2, int(model_params.get('n_layers', 2)))
            fallback_start_digit = int(model_params.get('start_digit', model_params.get('digits', 100)))
            fallback_shrink_ratio = float(model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0)))
            hidden_size = [
                max(4, int(data.shape[1] / max(1, int(round(fallback_start_digit * (fallback_shrink_ratio ** i))))))
                for i in range(fallback_n_layers)
            ]
    

    numpy_data = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
    numpy_label = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label
    validation_split = training_params.get('validation_split', 0.2)
    if validation_split > 0:
        stratify_labels = None
        if model_params.get('task_type', 'regression') in ('binary', 'multiclass'):
            stratify_labels = np.asarray(numpy_label).ravel()
        x_train, x_val, y_train, y_val = train_test_split(
            numpy_data, numpy_label,
            test_size=validation_split,
            random_state=training_params.get('seed', 42),
            stratify=stratify_labels
        )
    else:
        x_train, y_train = numpy_data, numpy_label
        x_val, y_val = None, None


    max_features_graces = model_params.get('max_features_graces', 10)  
    epochs = training_params.get('num_epochs', 50)
    patience = training_params.get('patience', 10)
    min_delta = training_params.get('min_delta', 0.001)
    n_dropouts = training_params.get('n_dropouts', 10)
    sigma = training_params.get('sigma', 0)
    device = training_params.get('device', 'cpu')

    logger.info("Creating GRACES model...")
    logger.info(f"Parameters: max_features_graces={max_features_graces}, hidden_size={hidden_size}, q={q}, "
               f"dropout_prob={dropout_prob}, batch_size={batch_size}, "
               f"lr={lr}, epochs={epochs}, alpha={alpha}, "
               f"f_correct={f_correct}")
    logger.info(f"graph_layer_mode: {training_params.get('graph_layer_mode', 'all_sage')}")
    
    graces = GRACES(
        max_features=max_features_graces,
        task_type=model_params.get('task_type', "regression"),
        hidden_size=hidden_size, 
        q=q,
        n_dropouts=n_dropouts,
        dropout_prob=dropout_prob,
        batch_size=batch_size,
        learning_rate=lr,
        epochs=epochs,
        alpha=alpha,
        sigma=sigma,
        f_correct=f_correct,
        patience=patience,
        min_delta=min_delta,
        device=device,
        activation=model_params.get('activation', 'relu'),
        training_params=training_params,
    )
   

    logger.info(f"start feature selection...")
    start_time = time.time()

    selected_features, train_losses, val_losses = graces.select(
        x_train, y_train, x_val, y_val, 
        verbose=True, 
        log_callback=logger.info,
        checkpoint_dir=checkpoint_dir
    )

    selection_time = time.time() - start_time
    logger.info(f"Feature selection completed in: {format_time(selection_time)}")
    logger.info(f"Selected {len(selected_features)} features")



    return {
        'feature_indices': selected_features,
        'select_rate': selected_features,
        'train_losses': train_losses,
        'val_losses': val_losses if val_losses else []
    }

