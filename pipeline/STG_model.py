import os
import time
import copy
import json
import numpy as np
import torch
import datetime
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from sqlalchemy.pool import QueuePool


from evaluation_utils import evaluate_feature_set


import math

import torch.nn as nn
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

import os.path as osp

import random

import logging
import fnmatch
import re
import six
import itertools
import json
# from torch.autograd import Variable

import collections
import collections.abc
import copy
from torch.utils.data import Dataset
import torch.optim as optim

from collections import defaultdict
import h5py
from scipy.stats import norm
import os
from sklearn.datasets import make_moons

SKIP_TYPES = six.string_types

logger = logging.getLogger("my-logger")


# Helper function for hidden_dims normalization
def resolve_hidden_dims(hidden_dims, min_layers=2, min_width=4):
    if hidden_dims is None:
        hidden_dims = [400, 200]
    elif isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    elif isinstance(hidden_dims, tuple):
        hidden_dims = list(hidden_dims)
    elif not isinstance(hidden_dims, list):
        raise ValueError("hidden_dims must be None, int, list, or tuple")

    if len(hidden_dims) == 0:
        raise ValueError("hidden_dims must not be empty")
    if len(hidden_dims) < min_layers:
        raise ValueError(f"hidden_dims must contain at least {min_layers} layers, got {len(hidden_dims)}")

    normalized = []
    for h in hidden_dims:
        h_int = int(h)
        if h_int <= 0:
            raise ValueError(f"Each hidden dimension must be positive, got {h}")
        normalized.append(max(min_width, h_int))
    return normalized

class EarlyStopping(object):
    def __init__(self, patience=20, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state_dict = None
        self.stop = False

    def _is_improvement(self, score):
        if self.best_score is None:
            return True
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        elif self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            raise ValueError("mode must be 'min' or 'max'.")

    def step(self, score, model):
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self.best_state_dict = state_dict(model, cpu=True)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return self.stop

    def restore_best_weights(self, model):
        if self.best_state_dict is not None:
            load_state_dict(model, self.best_state_dict)











# __all__ = ['STG']


def _standard_truncnorm_sample(lower_bound, upper_bound, sample_shape=torch.Size()):
    r"""
    Implements accept-reject algorithm for doubly truncated standard normal distribution.
    (Section 2.2. Two-sided truncated normal distribution in [1])
    [1] Robert, Christian P. "Simulation of truncated normal variables." Statistics and computing 5.2 (1995): 121-125.
    Available online: https://arxiv.org/abs/0907.4010
    Args:
        lower_bound (Tensor): lower bound for standard normal distribution. Best to keep it greater than -4.0 for
        stable results
        upper_bound (Tensor): upper bound for standard normal distribution. Best to keep it smaller than 4.0 for
        stable results
    """
    x = torch.randn(sample_shape)
    done = torch.zeros(sample_shape).byte() 
    while not done.all():
        proposed_x = lower_bound + torch.rand(sample_shape) * (upper_bound - lower_bound)
        if (upper_bound * lower_bound).lt(0.0):  # of opposite sign
            log_prob_accept = -0.5 * proposed_x**2
        elif upper_bound < 0.0:  # both negative
            log_prob_accept = 0.5 * (upper_bound**2 - proposed_x**2)
        else:  # both positive
            assert(lower_bound.gt(0.0))
            log_prob_accept = 0.5 * (lower_bound**2 - proposed_x**2)
        prob_accept = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
        accept = torch.bernoulli(prob_accept).byte() & ~done
        if accept.any():
            accept = accept.bool()
            x[accept] = proposed_x[accept]
            accept = accept.byte()
            done |= accept
    return x


class STG(object):
    def __init__(self, 
                device,                             # 'cpu' or 'cuda'
                input_dim=784,                      # input dimension
                output_dim=10,                      # output dimension (number of classes for classification, 1 for regression)
                hidden_dims=None,                   # list of hidden layer dimensions
                activation='relu',                  # activation function for hidden layers, 'relu' or 'tanh'
                batch_norm=None,
                dropout=None,                
                sigma=0.5,                          # initial value for sigma in STG
                lam=0.1,                            # lambda for regularization in STG                  
                optimizer='Adam',                    # optimizer, 'Adam' or 'SGD'
                learning_rate=1e-5,                  # learning rate for optimizer
                batch_size=100,                      # batch size for training
                freeze_onward=None,                  # epoch number to freeze feature selection layer onward (default None, meaning never freeze)
                feature_selection=True,              # whether to use STG for feature selection or just a regular MLP
                weight_decay=1e-3,                   # weight decay for optimizer
                task_type='classification',          # 'classification', 'regression' or 'cox'
                report_maps=False,                   # whether to report feature selection maps during training (default False, meaning only report at the end of each epoch)
                random_state=1,                      # random seed for reproducibility
                extra_args=None):
        self.batch_size = batch_size
        self.activation = activation
        self.random_state = random_state
        self.set_random_state(random_state)
        self.device = self.get_device(device)
        self.report_maps = report_maps 
        self.task_type = task_type
        self.extra_args = extra_args
        self.freeze_onward = freeze_onward
        self.hidden_dims = resolve_hidden_dims(hidden_dims)
        self._model = self.build_model(
            input_dim, output_dim, self.hidden_dims,
            activation, batch_norm, dropout, sigma, lam,
            task_type, feature_selection
        )
        self._model.apply(self.init_weights)
        self._model = self._model.to(self.device)
        self._optimizer = get_optimizer(optimizer, self._model, lr=learning_rate, weight_decay=weight_decay)
        self.batch_norm = batch_norm
        self.dropout = dropout



    def get_device(self, device):
        if isinstance(device, torch.device):
            if device.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return device

        if isinstance(device, str):
            device_lower = device.lower()
            if device_lower == "cpu":
                return torch.device("cpu")
            elif device_lower.startswith("cuda"):
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        raise NotImplementedError("Only 'cpu', 'cuda', or torch.device(...) are valid options.")

    def set_random_state(self, random_state):
        if random_state is None:
            return

        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)

        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        
    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         stddev = torch.tensor(0.1)
    #         shape = m.weight.shape
    #         m.weight = nn.Parameter(_standard_truncnorm_sample(lower_bound=-2*stddev, upper_bound=2*stddev, 
    #                               sample_shape=shape))
    #         torch.nn.init.zeros_(m.bias)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            stddev = torch.tensor(0.1)
            shape = m.weight.shape
            m.weight = nn.Parameter(
                _standard_truncnorm_sample(
                    lower_bound=-2 * stddev,
                    upper_bound=2 * stddev,
                    sample_shape=shape
                )
            )
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)    

    # def build_model(self, input_dim, output_dim, hidden_dims, activation, sigma, lam, task_type, feature_selection):
    def build_model(self, input_dim, output_dim, hidden_dims, activation, batch_norm, dropout, sigma, lam, task_type, feature_selection):
        hidden_dims = resolve_hidden_dims(hidden_dims)
        if task_type in ('classification', 'binary', 'multiclass'):
            if task_type == 'binary':
                self.metric = nn.BCEWithLogitsLoss()
            else:
                self.metric = nn.CrossEntropyLoss()
            self.tensor_names = ('input','label')
            if feature_selection:
                # The models for classification based on STG and MLP
                # return STGClassificationModel(input_dim, output_dim, hidden_dims, device=self.device, activation=activation, sigma=sigma, lam=lam)
                return STGClassificationModel(
                    input_dim, output_dim, hidden_dims,
                    device=self.device,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    activation=activation,
                    sigma=sigma,
                    lam=lam
                )   
            else:
                # The models for classification based on only MLP
                # return MLPClassificationModel(input_dim, output_dim, hidden_dims, activation=activation)
                return MLPClassificationModel(
                    input_dim, output_dim, hidden_dims,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    activation=activation
                )
        elif task_type == 'regression':
            self.metric = nn.MSELoss()
            self.tensor_names = ('input','label')
            if self.extra_args is not None:
                if self.extra_args == 'l1-softthresh':
                    # The model for regression with L1 regularization implemented by soft thresholding operator
                    # return SoftThreshRegressionModel(input_dim, output_dim, hidden_dims, device=self.device, activation=activation)
                    return SoftThreshRegressionModel(
                        input_dim, output_dim, hidden_dims,
                        device=self.device,
                        batch_norm=batch_norm,
                        dropout=dropout,
                        activation=activation,
                        sigma=sigma,
                        lam=lam
                    )
                elif self.extra_args == 'l1-norm-reg':
                    # The model for regression with L1 regularization implemented by directly adding L1 norm of weights to the loss function
                    # return L1RegressionModel(input_dim, output_dim, hidden_dims, device=self.device, activation=activation)
                    return L1RegressionModel(
                        input_dim, output_dim, hidden_dims,
                        device=self.device,
                        batch_norm=batch_norm,
                        dropout=dropout,
                        activation=activation,
                        sigma=sigma,
                        lam=lam
                    )
                elif self.extra_args == 'l1-gate':
                    # The model for regression with L1 regularization implemented by gating mechanism
                    # return L1GateRegressionModel(input_dim, output_dim, hidden_dims, device=self.device, activation=activation)

                    return L1GateRegressionModel(
                        input_dim, output_dim, hidden_dims,
                        device=self.device,
                        batch_norm=batch_norm,
                        dropout=dropout,
                        activation=activation,
                        sigma=sigma,
                        lam=lam
                    )
            else:
                if feature_selection:
                    # The model for regression based on STG and MLP
                    return STGRegressionModel(
                        input_dim, output_dim, hidden_dims,
                        device=self.device,
                        batch_norm=batch_norm,
                        dropout=dropout,
                        activation=activation,
                        sigma=sigma,
                        lam=lam
                    )
                else:
                    # The model for regression based on only MLP
                    # return MLPRegressionModel(input_dim, output_dim, hidden_dims, activation=activation)
                    return MLPRegressionModel(
                        input_dim, output_dim, hidden_dims,
                        batch_norm=batch_norm,
                        dropout=dropout,
                        activation=activation
                    )
        elif task_type == 'cox':
            self.metric = PartialLogLikelihood
            self.tensor_names = ('X', 'E', 'T')
            if feature_selection:
                # The model for Cox regression based on STG and MLP
                # return STGCoxModel(input_dim, output_dim, hidden_dims, device=self.device, activation=activation, sigma=sigma, lam=lam)
                return STGCoxModel(
                    input_dim, output_dim, hidden_dims,
                    device=self.device,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    activation=activation,
                    sigma=sigma,
                    lam=lam
                )
            else:
                # The model for Cox regression based on only MLP
                # return MLPCoxModel(input_dim, output_dim, hidden_dims, activation=activation)
                return MLPCoxModel(
                    input_dim, output_dim, hidden_dims,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    activation=activation
                )
        else:
            raise NotImplementedError()

    def train_step(self, feed_dict, meters=None):
        assert self._model.training

        loss, logits, monitors = self._model(feed_dict)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        #probe_infnan(logits, 'logits')
        if self.task_type=='cox':
            ci = calc_concordance_index(logits.detach().cpu().numpy(), 
                    feed_dict['E'].detach().cpu().numpy(), feed_dict['T'].detach().cpu().numpy())
        if self.extra_args=='l1-softthresh':
            self._model.mlp[0][0].weight.data = self._model.prox_op(self._model.mlp[0][0].weight)

        loss = as_float(loss)
        if meters is not None:
            meters.update(loss=loss)
            if self.task_type =='cox':
                meters.update(CI=ci)
            meters.update(monitors)

    # def get_dataloader(self, X, y, shuffle):
    #     if self.task_type in ('classification', 'multiclass'):
    #         data_loader = FastTensorDataLoader(torch.from_numpy(X).float().to(self.device), 
    #                     torch.from_numpy(y).long().to(self.device), tensor_names=self.tensor_names,
    #                     batch_size=self.batch_size, shuffle=shuffle)
    #     elif self.task_type == 'binary':
    #         data_loader = FastTensorDataLoader(torch.from_numpy(X).float().to(self.device), 
    #                     torch.from_numpy(y).float().reshape(-1, 1).to(self.device), tensor_names=self.tensor_names,
    #                     batch_size=self.batch_size, shuffle=shuffle)
    #     elif self.task_type == 'regression':
    #         data_loader = FastTensorDataLoader(torch.from_numpy(X).float().to(self.device), 
    #                     torch.from_numpy(y).float().to(self.device), tensor_names=self.tensor_names,
    #                     batch_size=self.batch_size, shuffle=shuffle)
    #     elif self.task_type == 'cox':
    #         assert isinstance(y, dict)
    #         data_loader = FastTensorDataLoader(torch.from_numpy(X).float().to(self.device), 
    #                     torch.from_numpy(y['E']).float().to(self.device),
    #                     torch.from_numpy(y['T']).float().to(self.device),
    #                     tensor_names=self.tensor_names,
    #                     batch_size=self.batch_size, shuffle=shuffle)
    #     else:
    #         raise NotImplementedError()

    #     return data_loader 
    
    def get_dataloader(self, X, y, shuffle, drop_last=False):
        if self.task_type in ('classification', 'multiclass'):
            x_tensor = _to_cpu_tensor_keep_dtype(X)
            y_tensor = _to_cpu_tensor_keep_dtype(y)
    
            data_loader = FastTensorDataLoader(
                x_tensor,
                y_tensor,
                tensor_names=self.tensor_names,
                batch_size=self.batch_size,
                shuffle=shuffle,
                device=self.device,
                input_dtype=torch.float32,
                label_dtypes={'label': torch.long},
                drop_last=drop_last
            )
    
        elif self.task_type == 'binary':
            x_tensor = _to_cpu_tensor_keep_dtype(X)
            y_tensor = _to_cpu_tensor_keep_dtype(y).reshape(-1, 1)
    
            data_loader = FastTensorDataLoader(
                x_tensor,
                y_tensor,
                tensor_names=self.tensor_names,
                batch_size=self.batch_size,
                shuffle=shuffle,
                device=self.device,
                input_dtype=torch.float32,
                label_dtypes={'label': torch.float32},
                drop_last=drop_last
            )
    
        elif self.task_type == 'regression':
            x_tensor = _to_cpu_tensor_keep_dtype(X)
            y_tensor = _to_cpu_tensor_keep_dtype(y)
    
            data_loader = FastTensorDataLoader(
                x_tensor,
                y_tensor,
                tensor_names=self.tensor_names,
                batch_size=self.batch_size,
                shuffle=shuffle,
                device=self.device,
                input_dtype=torch.float32,
                label_dtypes={'label': torch.float32},
                drop_last=drop_last
            )
    
        elif self.task_type == 'cox':
            assert isinstance(y, dict)
    
            x_tensor = _to_cpu_tensor_keep_dtype(X)
            e_tensor = _to_cpu_tensor_keep_dtype(y['E'])
            t_tensor = _to_cpu_tensor_keep_dtype(y['T'])
    
            data_loader = FastTensorDataLoader(
                x_tensor,
                e_tensor,
                t_tensor,
                tensor_names=self.tensor_names,
                batch_size=self.batch_size,
                shuffle=shuffle,
                device=self.device,
                input_dtype=torch.float32,
                label_dtypes={
                    'E': torch.float32,
                    'T': torch.float32
                },
                drop_last=drop_last
            )
    
        else:
            raise NotImplementedError()
    
        return data_loader


    def fit(self, X, y, nr_epochs, valid_X=None, valid_y=None, 
        verbose=True, meters=None, early_stop=None, print_interval=1, shuffle=False):
        data_loader = self.get_dataloader(X, y, shuffle, drop_last=True)

        if valid_X is not None:
            val_data_loader = self.get_dataloader(valid_X, valid_y, shuffle=False, drop_last=False)
        else:
            val_data_loader = None
        self.train(data_loader, nr_epochs, val_data_loader, verbose, meters, early_stop, print_interval)

    def evaluate(self, X, y):
        data_loader = self.get_dataloader(X, y, shuffle=False, drop_last=False)
        meters = GroupMeters()
        self.validate(data_loader, self.metric, meters, mode='test')
        print(meters.format_simple(''))

    # def predict(self, X, verbose=True):
    #     dataset = SimpleDataset(torch.from_numpy(X).float().to(self.device))
    #     data_loader = DataLoader(dataset, batch_size=X.shape[0], shuffle=False) 
    #     res = []
    #     self._model.eval()
    #     for feed_dict in data_loader:
    #         feed_dict_np = as_numpy(feed_dict)
    #         feed_dict = as_tensor(feed_dict)
    #         with torch.no_grad():
    #             output_dict = self._model(feed_dict)
    #         output_dict_np = as_numpy(output_dict)
    #         res.append(output_dict_np['pred'])
    #     return np.concatenate(res, axis=0)

    def predict(self, X, verbose=True):
        x_tensor = _to_cpu_tensor_keep_dtype(X)
    
        data_loader = FastTensorDataLoader(
            x_tensor,
            tensor_names=('input',),
            batch_size=self.batch_size,
            shuffle=False,
            device=self.device,
            input_dtype=torch.float32
        )
    
        res = []
        self._model.eval()
    
        with torch.no_grad():
            for feed_dict in data_loader:
                output_dict = self._model(feed_dict)
                output_dict_np = as_numpy(output_dict)
                res.append(output_dict_np['pred'])
    
        return np.concatenate(res, axis=0)

    def train_epoch(self, data_loader, meters=None):
        if meters is None:
            meters = GroupMeters()

        self._model.train()
        end = time.time()
        for feed_dict in data_loader:
            data_time = time.time() - end; end = time.time()
            self.train_step(feed_dict, meters=meters)
            step_time = time.time() - end; end = time.time()
            #if dev:
            #meters.update({'time/data': data_time, 'time/step': step_time})
        return meters

    def train(self, data_loader, nr_epochs, val_data_loader=None, verbose=True, 
        meters=None, early_stop=None, print_interval=1):
        if meters is None:
            meters = GroupMeters()

        for epoch in range(1, 1 + nr_epochs):
            meters.reset()
            if epoch == self.freeze_onward:
                self._model.freeze_weights()
            self.train_epoch(data_loader, meters=meters)

            if val_data_loader is not None:
                self.validate(val_data_loader, self.metric, meters)

            if verbose and epoch % print_interval == 0:
                caption = 'Epoch: {}:'.format(epoch)
                print(meters.format_simple(caption))

            if early_stop is not None and val_data_loader is not None:
                monitor_key = 'valid_loss'
                if monitor_key not in meters.avg:
                    raise ValueError("Early stopping requires validation data so that 'valid_loss' is available.")
                should_stop = early_stop.step(meters.avg[monitor_key], self._model)
                if should_stop:
                    if verbose:
                        print('Early stopping triggered at epoch {}.'.format(epoch))
                    early_stop.restore_best_weights(self._model)
                    break

    def validate_step(self, feed_dict, metric, meters=None, mode='valid'):
        with torch.no_grad():
            pred = self._model(feed_dict)
        if self.task_type in ('classification', 'multiclass'):
            result = metric(pred['logits'], self._model._get_label(feed_dict))
        elif self.task_type == 'binary':
            result = metric(pred['logits'], self._model._get_label(feed_dict).float())
        elif self.task_type == 'regression':
            result = metric(pred['pred'], self._model._get_label(feed_dict))
            
        elif self.task_type == 'cox':
            result = metric(pred['logits'], self._model._get_fail_indicator(feed_dict), 'noties') 
            val_CI = calc_concordance_index(pred['logits'].detach().cpu().numpy(), 
                    feed_dict['E'].detach().cpu().numpy(), feed_dict['T'].detach().cpu().numpy())
            result = as_float(result)
        else:
            raise NotImplementedError()

        if meters is not None:
            meters.update({mode+'_loss':result})
            if self.task_type=='cox':
                meters.update({mode+'_CI':val_CI})

    def validate(self, data_loader, metric, meters=None, mode='valid'):
        if meters is None:
            meters = GroupMeters()

        self._model.eval()
        end = time.time()
        for fd in data_loader:
            data_time = time.time() - end; end = time.time()
            self.validate_step(fd, metric, meters=meters, mode=mode)
            step_time = time.time() - end; end = time.time()

        return meters.avg

    def save_checkpoint(self, filename, extra=None):
        model = self._model

        state = {
            'model': state_dict(model, cpu=True),
            'optimizer': as_cpu(self._optimizer.state_dict()),
            'extra': extra
        }
        try:
            torch.save(state, filename)
            logger.info('Checkpoint saved: "{}".'.format(filename))
        except Exception:
            logger.exception('Error occurred when dump checkpoint "{}".'.format(filename))

    def load_checkpoint(self, filename):
        if osp.isfile(filename):
            model = self._model
            if isinstance(model, nn.DataParallel):
                model = model.module

            try:
                checkpoint = torch.load(filename)
                load_state_dict(model, checkpoint['model'])
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                logger.critical('Checkpoint loaded: {}.'.format(filename))
                return checkpoint['extra']
            except Exception:
                logger.exception('Error occurred when load checkpoint "{}".'.format(filename))
        else:
            logger.warning('No checkpoint found at specified position: "{}".'.format(filename))
        return None

    def get_gates(self, mode):
        return self._model.get_gates(mode)









################## Layers.py ###################   



# __all__ = [
#     'LinearLayer', 'MLPLayer', 'FeatureSelector',
# ]

class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size()) 
        self.sigma = sigma
        self.device = device
    
    def forward(self, prev_x):
        z = self.mu + self.sigma*self.noise.normal_()*self.training 
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x
    
    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self


class GatingLayer(nn.Module):
    '''To implement L1-based gating layer (so that we can compare L1 with L0(STG) in a fair way)
    '''
    def __init__(self, input_dim, device):
        super(GatingLayer, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.device = device
    
    def forward(self, prev_x):
        new_x = prev_x * self.mu 
        return new_x
    
    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return torch.sum(torch.abs(x))


class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation=None):
        if bias is None:
            bias = (batch_norm is None)

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if batch_norm is not None and batch_norm is not False:
            modules.append(get_batcnnorm(batch_norm, out_features, 1))
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, 1))
        super().__init__(*modules)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu', flatten=True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        nr_hiddens = len(hidden_dims)
        for i in range(nr_hiddens):
            layer = LinearLayer(dims[i], dims[i+1], batch_norm=batch_norm, dropout=dropout, activation=activation)
            modules.append(layer)
        layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)
        self.flatten = flatten

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input)
     



################## losses.py ###################



def PartialLogLikelihood(logits, fail_indicator, ties):
    '''
    fail_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model 
    ties: 'noties' or 'efron' or 'breslow'
    '''
    logL = 0
    # pre-calculate cumsum
    cumsum_y_pred = torch.cumsum(logits, 0)
    hazard_ratio = torch.exp(logits)
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)
    if ties == 'noties':
        log_risk = torch.log(cumsum_hazard_ratio)
        likelihood = logits - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * fail_indicator
        logL = -torch.sum(uncensored_likelihood)
    else:
        raise NotImplementedError()
    # negative average log-likelihood
    observations = torch.sum(fail_indicator, 0)
    return 1.0*logL / observations


def calc_concordance_index(logits, fail_indicator, fail_time):
    """
    Compute the concordance-index value.
    Parameters:
        label_true: dict, like {'e': event, 't': time}, Observation and Time in survival analyze.
        y_pred: np.array, predictive proportional risk of network.
    Returns:
        concordance index.
    """
    hr_pred = -logits 
    ci = concordance_index(fail_time,
                            hr_pred,
                            fail_indicator)
    return ci




################## matching.py ###################


# __all__ = ['NameMatcher', 'IENameMatcher']


class NameMatcher(object):
    def __init__(self, rules=None):
        if rules is None:
            self._rules = []
        elif isinstance(rules, dict):
            self._rules = list(rules.items())
        else:
            assert isinstance(rules, collections.Iterable)
            self._rules = list(rules)

        self._map = {}
        self._compiled_rules = []
        self._compiled = False

        self._matched = []
        self._unused = set()
        self._last_stat = None

    @property
    def rules(self):
        return self._rules

    def map(self):
        assert self._compiled
        return self._map

    def append_rule(self, rule):
        self._rules.append(tuple(rule))

    def insert_rule(self, index, rule):
        self._rules.insert(index, rule)

    def pop_rule(self, index=None):
        self._rules.pop(index)

    def begin(self, *, force_compile=False):
        if not self._compiled or force_compile:
            self.compile()
        self._matched = []
        self._unused = set(range(len(self._compiled_rules)))

    def end(self):
        return self._matched, {self._compiled_rules[i][0] for i in self._unused}

    def match(self, k):
        for i, (r, p, v) in enumerate(self._compiled_rules):
            if p.match(k):
                if i in self._unused:
                    self._unused.remove(i)
                self._matched.append((k, r, v))
                return v
        return None

    def compile(self):
        self._map = dict()
        self._compiled_rules = []

        for r, v in self._rules:
            self._map[r] = v
            p = fnmatch.translate(r)
            p = re.compile(p, flags=re.IGNORECASE)
            self._compiled_rules.append((r, p, v))
        self._compiled = True

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._last_stat = self.end()

    def get_last_stat(self):
        return self._last_stat



class IENameMatcher(object):
    def __init__(self, include=None, exclude=None):
        if include is None:
            self.include = None
        else:
            self.include = NameMatcher([(i, True) for i in include])

        if exclude is None:
            self.exclude = None
        else:
            self.exclude = NameMatcher([(e, True) for e in exclude])
        self._last_stat = None

    def begin(self):
        if self.include is not None:
            self.include.begin()
        if self.exclude is not None:
            self.exclude.begin()
        self._last_stat = (set(), set())

    def end(self):
        if self.include is not None:
            self.include.end()
        if self.exclude is not None:
            self.exclude.end()

        if len(self._last_stat[0]) < len(self._last_stat[1]):
            self._last_stat = ('included', self._last_stat[0])
        else:
            self._last_stat = ('excluded', self._last_stat[1])

    def match(self, k):
        if self.include is None:
            ret = True
        else:
            ret = bool(self.include.match(k))

        if self.exclude is not None:
            ret = ret and not bool(self.exclude.match(k))

        if ret:
            self._last_stat[0].add(k)
        else:
            self._last_stat[1].add(k)
        return ret

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def get_last_stat(self):
        return self._last_stat
    


################### meter.py ###################


def map_exec(func, *iterables):
    return list(map(func, *iterables))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    val = 0
    avg = 0
    sum = 0
    count = 0
    tot_count = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count


class GroupMeters(object):
    def __init__(self):
        self._meters = collections.defaultdict(AverageMeter)

    def reset(self):
        map_exec(AverageMeter.reset, self._meters.values())

    def update(self, updates=None, value=None, n=1, **kwargs):
        """
        Example:
            >>> meters.update(key, value)
            >>> meters.update({key1: value1, key2: value2})
            >>> meters.update(key1=value1, key2=value2)
        """
        if updates is None:
            updates = {}
        if updates is not None and value is not None:
            updates = {updates: value}
        updates.update(kwargs)
        for k, v in updates.items():
            self._meters[k].update(v, n=n)

    def __getitem__(self, name):
        return self._meters[name]

    def items(self):
        return self._meters.items()

    @property
    def sum(self):
        return {k: m.sum for k, m in self._meters.items() if m.count > 0}

    @property
    def avg(self):
        return {k: m.avg for k, m in self._meters.items() if m.count > 0}

    @property
    def val(self):
        return {k: m.val for k, m in self._meters.items() if m.count > 0}

    def format(self, caption, values, kv_format, glue):
        meters_kv = self._canonize_values(values)
        log_str = [caption]
        log_str.extend(itertools.starmap(kv_format.format, sorted(meters_kv.items())))
        return glue.join(log_str)

    def format_simple(self, caption, values='avg', compressed=True):
        if compressed:
            return self.format(caption, values, '{}={:4f}', ' ')
        else:
            return self.format(caption, values, '\t{} = {:4f}', '\n')

    def dump(self, filename, values='avg'):
        meters_kv = self._canonize_values(values)
        with open(filename, 'a') as f:
            #f.write(io.dumps_json(meters_kv, compressed=False))
            f.write(json.dumps(meters_kv, cls=JsonObjectEncoder, sort_keys=True, indent=4, separators=(',', ': ')))
            f.write('\n')

    def _canonize_values(self, values):
        if isinstance(values, six.string_types):
            assert values in ('avg', 'val', 'sum')
            meters_kv = getattr(self, values)
        else:
            meters_kv = values
        return meters_kv


class JsonObjectEncoder(json.JSONEncoder):
    """Adapted from https://stackoverflow.com/a/35483750"""

    def default(self, obj):
        if hasattr(obj, '__jsonify__'):
            json_object = obj.__jsonify__()
            if isinstance(json_object, six.string_types):
                return json_object
            return self.encode(json_object)
        else:
            raise TypeError("Object of type '%s' is not JSON serializable." % obj.__class__.__name__)

        if hasattr(obj, '__dict__'):
            d = dict(
                (key, value)
                for key, value in inspect.getmembers(obj)
                if not key.startswith("__")
                and not inspect.isabstract(value)
                and not inspect.isbuiltin(value)
                and not inspect.isfunction(value)
                and not inspect.isgenerator(value)
                and not inspect.isgeneratorfunction(value)
                and not inspect.ismethod(value)
                and not inspect.ismethoddescriptor(value)
                and not inspect.isroutine(value)
            )
            return self.default(d)

        return obj
################### models.py ###################


# __all__ = ['MLPModel', 'MLPRegressionModel', 'MLPClassificationModel', 'LinearRegressionModel', 'LinearClassificationModel']


class ModelIOKeysMixin(object):
    def _get_input(self, feed_dict):
        return feed_dict['input']

    def _get_label(self, feed_dict):
        return feed_dict['label']

    def _get_covariate(self, feed_dict):
        '''For cox'''
        return feed_dict['X']

    def _get_fail_indicator(self, feed_dict):
        '''For cox'''
        return feed_dict['E'].reshape(-1, 1)

    def _get_failure_time(self, feed_dict):
        '''For cox'''
        return feed_dict['T']

    def _compose_output(self, value):
        return dict(pred=value)


class MLPModel(MLPLayer):
    def freeze_weights(self):
        for name, p in self.named_parameters():
            if 'mu' not in name:
                p.requires_grad = False

    def get_gates(self, mode):
        if mode == 'raw':
            return self.mu.detach().cpu().numpy()
        elif mode == 'prob':
            return np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5)) 
        else:
            raise NotImplementedError()


class STGClassificationModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, nr_classes, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.FeatureSelector = FeatureSelector(input_dim, sigma, device)
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        self.reg = self.FeatureSelector.regularizer
        self.lam = lam 
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma
        
    def forward(self, feed_dict):
        x = self.FeatureSelector(self._get_input(feed_dict))
        logits = super().forward(x)
        if self.training:
            label = self._get_label(feed_dict)
            if logits.shape[-1] == 1:
                loss = nn.BCEWithLogitsLoss()(logits, label.float())
            else:
                loss = self.loss(logits, label)
            reg = torch.mean(self.reg((self.mu + 0.5)/self.sigma)) 
            total_loss = loss + self.lam * reg 
            return total_loss, dict(), dict() 
        else:
            return self._compose_output(logits)

    def _compose_output(self, logits):
        if logits.shape[-1] == 1:
            value = torch.sigmoid(logits)
            pred = (value >= 0.5).long().view(-1)
            return dict(prob=value, pred=pred, logits=logits)
        value = self.softmax(logits)
        _, pred = value.max(dim=1)
        return dict(prob=value, pred=pred, logits=logits)

class STGRegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.FeatureSelector = FeatureSelector(input_dim, sigma, device)
        self.loss = nn.MSELoss()
        self.reg = self.FeatureSelector.regularizer 
        self.lam = lam
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma

    def forward(self, feed_dict):
        x = self.FeatureSelector(self._get_input(feed_dict))
        pred = super().forward(x)
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            reg = torch.mean(self.reg((self.mu + 0.5)/self.sigma)) 
            total_loss = loss + self.lam * reg
            return total_loss, dict(), dict()
        else:
            return self._compose_output(pred)
 

class STGCoxModel(MLPModel, ModelIOKeysMixin):
    #TODO: Finish impl cox model.
    def __init__(self, input_dim, nr_classes, hidden_dims, device, lam, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.FeatureSelector = FeatureSelector(input_dim, sigma, device)
        self.loss = PartialLogLikelihood
        self.noties = 'noties'
        self.lam = lam
        self.reg = self.FeatureSelector.regularizer 
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma

    def forward(self, feed_dict):
        x = self.FeatureSelector(self._get_covariate(feed_dict))
        logits = super().forward(x)
        if self.training:
            loss = self.loss(logits, self._get_fail_indicator(feed_dict), self.noties)
            reg = torch.sum(self.reg((self.mu + 0.5)/self.sigma)) 
            total_loss = loss + reg 
            return total_loss, logits, dict()
        else:
            return self._compose_output(logits)

    def _compose_output(self, logits):
        return dict(logits=logits)



class L1RegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.loss = nn.MSELoss()
        self.lam = lam

    def forward(self, feed_dict):
        pred = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            reg = torch.mean(torch.abs(self.mlp[0][0].weight)) 
            total_loss = loss + self.lam * reg
            return total_loss, dict(), dict()
        else:
            return self._compose_output(pred)


class L1GateRegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.GateingLayer = GatingLayer(input_dim, device)
        self.reg = self.GateingLayer.regularizer
        self.mu = self.GateingLayer.mu
        self.loss = nn.MSELoss()
        self.lam = lam

    def forward(self, feed_dict):
        x = self.GateingLayer(self._get_input(feed_dict))
        pred = super().forward(x)
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            reg = torch.mean(self.reg(self.mu))
            total_loss = loss + self.lam * reg
            return total_loss, dict(), dict()
        else:
            return self._compose_output(pred)


class SoftThreshRegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.loss = nn.MSELoss()
        self.lam = lam

    def prox_plus(self, w):
        """Projection onto non-negative numbers
        """
        below = w < 0
        w[below] = 0
        return w

    def prox_op(self, w):
        return torch.sign(w) * self.prox_plus(torch.abs(w) - self.lam)

    def forward(self, feed_dict):
        pred = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            total_loss = loss 
            return total_loss, dict(), dict()
        else:
            return self._compose_output(pred)


   





class MLPCoxModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, nr_classes, hidden_dims, batch_norm=None, dropout=None, activation='relu'):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.loss = PartialLogLikelihood 
        self.noties = 'noties'

    def forward(self, feed_dict):
        logits = super().forward(self._get_covariate(feed_dict))
        if self.training:
            loss = self.loss(logits, self._get_fail_indicator(feed_dict), self.noties)
            return loss, logits, dict()
        else:
            return self._compose_output(logits)

    def _compose_output(self, logits):
        return dict(logits=logits)


class MLPRegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu'):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.loss = nn.MSELoss()

    def forward(self, feed_dict):
        pred = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            return loss, dict(), dict()
        else:
            return self._compose_output(pred)


class MLPClassificationModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, nr_classes, hidden_dims, batch_norm=None, dropout=None, activation='relu'):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feed_dict):
        logits = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(logits, self._get_label(feed_dict))
            return loss, dict(), dict()
        else:
            return self._compose_output(logits)

    def _compose_output(self, logits):
        value = self.softmax(logits)
        _, pred = value.max(dim=1)
        return dict(prob=value, pred=pred, logits=logits)


class LinearRegressionModel(MLPRegressionModel):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, [])


class LinearClassificationModel(MLPClassificationModel):
    def __init__(self, input_dim, nr_classes):
        super().__init__(input_dim, nr_classes, [])



    




###################saveio.py ###################



# __all__ = ['load_state_dict', 'load_weights']


def state_dict(model, include=None, exclude=None, cpu=True):
    if isinstance(model, nn.DataParallel):
        model = model.module

    state_dict = model.state_dict()

    matcher = IENameMatcher(include, exclude)
    with matcher:
        state_dict = {k: v for k, v in state_dict.items() if matcher.match(k)}
    stat = matcher.get_last_stat()
    if len(stat[1]) > 0:
        logger.critical('Weights {}: {}.'.format(stat[0], ', '.join(sorted(list(stat[1])))))

    if cpu:
        state_dict = as_cpu(state_dict)
    return state_dict


def load_state_dict(model, state_dict, include=None, exclude=None):
    if isinstance(model, nn.DataParallel):
        model = model.module

    matcher = IENameMatcher(include, exclude)
    with matcher:
        state_dict = {k: v for k, v in state_dict.items() if matcher.match(k)}
    stat = matcher.get_last_stat()
    if len(stat[1]) > 0:
        logger.critical('Weights {}: {}.'.format(stat[0], ', '.join(sorted(list(stat[1])))))

    # Build the tensors.
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            state_dict[k] = torch.from_numpy(v)

    error_msg = []
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                error_msg.append('While copying the parameter named {}, '
                                 'whose dimensions in the model are {} and '
                                 'whose dimensions in the checkpoint are {}.'
                                 .format(name, own_state[name].size(), param.size()))

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        error_msg.append('Missing keys in state_dict: "{}".'.format(missing))

    unexpected = set(state_dict.keys()) - set(own_state.keys())
    if len(unexpected) > 0:
        error_msg.append('Unexpected key "{}" in state_dict.'.format(unexpected))

    if len(error_msg):
        raise KeyError('\n'.join(error_msg))


def load_weights(model, filename, include=None, exclude=None, return_raw=True):
    if osp.isfile(filename):
        try:
            raw = weights = torch.load(filename)
            # Hack for checkpoint.
            if 'model' in weights and 'optimizer' in weights:
                weights = weights['model']

            try:
                load_state_dict(model, weights, include=include, exclude=exclude)
            except KeyError as e:
                logger.warning('Unexpected or missing weights found:\n' + e.args[0])
            logger.critical('Weights loaded: {}.'.format(filename))
            if return_raw:
                return raw
            return True
        except Exception:
            logger.exception('Error occurred when load weights {}.'.format(filename))
    else:
        logger.warning('No weights file found at specified position: {}.'.format(filename))
    return None






#################### utils.py ###################





class SimpleDataset(Dataset):
    '''
    Assuming X and y are numpy arrays and 
     with X.shape = (n_samples, n_features) 
          y.shape = (n_samples,)
    '''
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    
    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i]
        #data = np.array(data).astype(np.float32)
        if self.y is not None:
            return dict(input=data, label=self.y[i])
        else:
            return dict(input=data)


class FastTensorDataLoader:
    def __init__(
        self,
        *tensors,
        tensor_names,
        batch_size=32,
        shuffle=False,
        device=None,
        input_dtype=torch.float32,
        label_dtypes=None,
        drop_last=False
    ):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)

        self.tensors = tensors
        self.tensor_names = tensor_names
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.device = device
        self.input_dtype = input_dtype
        self.label_dtypes = label_dtypes or {}
        self.drop_last = drop_last

        
        if self.drop_last:
            self.n_batches = self.dataset_len // self.batch_size
        else:
            n_batches, remainder = divmod(self.dataset_len, self.batch_size)
            if remainder > 0:
                n_batches += 1
            self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.drop_last:
            if self.i + self.batch_size > self.dataset_len:
                raise StopIteration
        else:
            if self.i >= self.dataset_len:
                raise StopIteration

        batch = {}

        for k in range(len(self.tensor_names)):
            name = self.tensor_names[k]
            tensor = self.tensors[k][self.i:self.i + self.batch_size]

            # 只在取 batch 时转换
            if name in ("input", "X"):
                tensor = tensor.to(device=self.device, dtype=self.input_dtype)
            else:
                dtype = self.label_dtypes.get(name, None)
                if dtype is not None:
                    tensor = tensor.to(device=self.device, dtype=dtype)
                else:
                    tensor = tensor.to(device=self.device)

            batch[name] = tensor

        self.i += self.batch_size
        return batch
        

    def __len__(self):
        return self.n_batches





def probe_infnan(v, name, extras={}):
    nps = torch.isnan(v)
    s = nps.sum().item()
    if s > 0:
        print('>>> {} >>>'.format(name))
        print(name, s)
        print(v[nps])
        for k, val in extras.items():
            print(k, val, val.sum().item())
        quit()


class Identity(nn.Module):
    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args

def get_batcnnorm(bn, nr_features=None, nr_dims=1):
    if isinstance(bn, nn.Module):
        return bn

    assert 1 <= nr_dims <= 3

    if bn in (True, 'async'):
        clz_name = 'BatchNorm{}d'.format(nr_dims)
        return getattr(nn, clz_name)(nr_features)
    else:
        raise ValueError('Unknown type of batch normalization: {}.'.format(bn))


# def get_dropout(dropout, nr_dims=1):
#     if isinstance(dropout, nn.Module):
#         return dropout

#     if dropout is True:
#         dropout = 0.5
#     if nr_dims == 1:
#         return nn.Dropout(dropout, True)
#     else:
#         clz_name = 'Dropout{}d'.format(nr_dims)
#         return getattr(nn, clz_name)(dropout)
    
def get_dropout(dropout, nr_dims=1):
    if isinstance(dropout, nn.Module):
        return dropout

    if dropout is True:
        dropout = 0.5
    if nr_dims == 1:
        return nn.Dropout(dropout, inplace=False)
    else:
        clz_name = 'Dropout{}d'.format(nr_dims)
        return getattr(nn, clz_name)(dropout, inplace=False)

# def get_activation(act):
#     if isinstance(act, nn.Module):
#         return act

#     assert type(act) is str, 'Unknown type of activation: {}.'.format(act)
#     act_lower = act.lower()
#     if act_lower == 'identity':
#         return Identity()
#     elif act_lower == 'relu':
#         return nn.ReLU(True)
#     elif act_lower == 'selu':
#         return nn.SELU(True)
#     elif act_lower == 'sigmoid':
#         return nn.Sigmoid()
#     elif act_lower == 'tanh':
#         return nn.Tanh()
#     else:
#         try:
#             return getattr(nn, act)
#         except AttributeError:
#             raise ValueError('Unknown activation function: {}.'.format(act))

def get_activation(activation_name):
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leakyrelu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
    }
    key = activation_name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation: {activation_name}")
    return activations[key]


def get_optimizer(optimizer, model, *args, **kwargs):
    if isinstance(optimizer, (optim.Optimizer)):
        return optimizer

    if type(optimizer) is str:
        try:
            optimizer = getattr(optim, optimizer)
        except AttributeError:
            raise ValueError('Unknown optimizer type: {}.'.format(optimizer))
    return optimizer(filter(lambda p: p.requires_grad, model.parameters()), *args, **kwargs)
    

# def stmap(func, iterable):
#     if isinstance(iterable, six.string_types):
#         return func(iterable)
#     elif isinstance(iterable, (collections.Sequence, collections.UserList)):
#         return [stmap(func, v) for v in iterable]
#     elif isinstance(iterable, collections.Set):
#         return {stmap(func, v) for v in iterable}
#     elif isinstance(iterable, (collections.Mapping, collections.UserDict)):
#         return {k: stmap(func, v) for k, v in iterable.items()}
#     else:
#         return func(iterable)
def stmap(func, iterable):
    if isinstance(iterable, six.string_types):
        return func(iterable)
    elif isinstance(iterable, (collections.abc.Sequence, collections.UserList)):
        return [stmap(func, v) for v in iterable]
    elif isinstance(iterable, collections.abc.Set):
        return {stmap(func, v) for v in iterable}
    elif isinstance(iterable, (collections.abc.Mapping, collections.UserDict)):
        return {k: stmap(func, v) for k, v in iterable.items()}
    else:
        return func(iterable)

def _as_tensor(o):
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        return o
    if torch.is_tensor(o):
        return o
    return torch.from_numpy(np.array(o))


def as_tensor(obj):
    return stmap(_as_tensor, obj)


    
def _to_cpu_tensor_keep_dtype(x):
    """
    Convert numpy array to torch tensor but keep original dtype.
    Do not convert to float32 and do not move to GPU here.
    """
    if torch.is_tensor(x):
        return x.cpu()
    return torch.from_numpy(np.asarray(x))

    
def _as_numpy(o):
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        o = o
    if torch.is_tensor(o):
        return o.cpu().numpy()
    return np.array(o)


def as_numpy(obj):
    return stmap(_as_numpy, obj)


def _as_float(o):
    if isinstance(o, SKIP_TYPES):
        return o
    if torch.is_tensor(o):
        return o.item()
    arr = as_numpy(o)
    assert arr.size == 1
    return float(arr)


def as_float(obj):
    return stmap(_as_float, obj)


def _as_cpu(o):
    from torch.autograd import Variable
    if isinstance(o, Variable) or torch.is_tensor(o):
        return o.cpu()
    return o


def as_cpu(obj):
    return stmap(_as_cpu, obj)


## For synthetic dataset creation





    





# ============================================================
# utils
# ============================================================
def parse_list_arg(x, dtype=int):
    if isinstance(x, (list, tuple)):
        return [dtype(v) for v in x]
    return [dtype(v.strip()) for v in str(x).split(",") if v.strip()]


def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flatten_y_if_needed(y):
    y = _to_numpy(y)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.reshape(-1)
    return y


def _as_1d_cpu_numpy(y):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu()
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.reshape(-1)
        return y.numpy()
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.reshape(-1)
    return y


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _get_output_dim(task_type, y):
    if task_type == "binary":
        return 1
    if task_type == "multiclass":
        y = _flatten_y_if_needed(y)
        return int(len(np.unique(y)))
    return 1



def _build_hidden_dims(input_dim, digit, n_layers=2, shrink_ratio=2.0):
    """
    Backward-compatible helper.
    `digit` is treated as the start digit (old code may still call it `digit`).
    Hidden dimensions are generated progressively using `shrink_ratio`.
    """
    start_digit = max(1, int(digit))
    n_layers = max(2, int(n_layers))
    shrink_ratio = float(shrink_ratio)
    digits_list = [
        max(1, int(round(start_digit * (shrink_ratio ** i))))
        for i in range(n_layers)
    ]
    return [max(4, int(input_dim / max(1, int(d)))) for d in digits_list]


def _build_hidden_dims_from_digits_list(input_dim, digits_list):
    if digits_list is None or len(digits_list) == 0:
        raise ValueError("digits_list must not be empty")
    if len(digits_list) < 2:
        raise ValueError(f"digits_list must contain at least 2 layers, got {len(digits_list)}")
    return [max(4, int(input_dim / max(1, int(d)))) for d in digits_list]


def _safe_sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _prepare_binary_scores(pred):
    pred = np.asarray(pred)
    if pred.ndim > 1 and pred.shape[1] == 1:
        pred = pred.reshape(-1)

    # 如果已经是概率
    if np.all((pred >= 0.0) & (pred <= 1.0)):
        prob = pred
    else:
        prob = _safe_sigmoid(pred)

    y_hat = (prob >= 0.5).astype(int)
    return prob, y_hat


def _prepare_multiclass_scores(pred):
    pred = np.asarray(pred)

    if pred.ndim == 1:
        # 已经是类别标签
        y_hat = pred.astype(int)
        prob = None
    else:
        # 可能是 logits/prob
        row_sum = pred.sum(axis=1, keepdims=True)
        if np.all(pred >= 0) and np.allclose(row_sum, 1.0, atol=1e-4):
            prob = pred
        else:
            shifted = pred - np.max(pred, axis=1, keepdims=True)
            exp_x = np.exp(shifted)
            prob = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        y_hat = np.argmax(prob, axis=1)

    return prob, y_hat


def _compute_metrics(y_true, pred, task_type):
    y_true = _flatten_y_if_needed(y_true)
    pred = np.asarray(pred)

    if task_type == "regression":
        y_pred = pred.reshape(-1)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2),
            "loss": float(mse),
        }

    if task_type == "binary":
        prob, y_hat = _prepare_binary_scores(pred)
        acc = accuracy_score(y_true.astype(int), y_hat)
        eps = 1e-8
        bce = -np.mean(
            y_true * np.log(np.clip(prob, eps, 1 - eps)) +
            (1 - y_true) * np.log(np.clip(1 - prob, eps, 1 - eps))
        )
        return {
            "acc": float(acc),
            "loss": float(bce),
        }

    if task_type == "multiclass":
        prob, y_hat = _prepare_multiclass_scores(pred)
        acc = accuracy_score(y_true.astype(int), y_hat)

        if prob is not None:
            eps = 1e-8
            ce = -np.mean(np.log(np.clip(prob[np.arange(len(y_true)), y_true.astype(int)], eps, 1.0)))
            loss = float(ce)
        else:
            loss = float(1.0 - acc)

        return {
            "acc": float(acc),
            "loss": float(loss),
        }

    raise ValueError(f"Unsupported task_type: {task_type}")


def _score_from_metrics(metrics, eval_metric, task_type):
    if eval_metric in metrics:
        return float(metrics[eval_metric])
    if task_type in ("binary", "multiclass"):
        return float(metrics["acc"])
    return float(metrics["r2"])


def _build_stg_model(X_train, y_train, model_params, training_params, logger):
    task_type = model_params["task_type"]
    input_dim = X_train.shape[1]
    output_dim = _get_output_dim(task_type, y_train)

    model = STG(
        task_type=task_type,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=model_params.get("hidden_dims", [500, 50]),
        activation=model_params.get("activation", "tanh"),
        batch_norm=model_params.get("batch_norm", False),
        dropout=model_params.get("dropout_prob", 0.0),
        optimizer=model_params.get("optimizer", "Adam"),
        learning_rate=training_params["lr"],
        batch_size=training_params["batch_size"],
        feature_selection=True,
        sigma=training_params["sigma"],
        lam=training_params["lam"],
        random_state=training_params.get("seed", 42),
        device=training_params.get("device", "cpu"),
    )
    return model


def _fit_stg_model(model, X_train, y_train, X_val, y_val, training_params):
    early_stop = EarlyStopping(
        patience=training_params.get("patience", 50),
        min_delta=training_params.get("min_delta", 1e-4),
        mode="min"
    )

    nr_epochs = training_params.get("num_epochs", 3000)

    model.fit(
        X_train, y_train,
        nr_epochs=nr_epochs,
        valid_X=X_val,
        valid_y=y_val,
        print_interval=training_params.get("print_interval", 100),
        early_stop=early_stop
    )


# ============================================================
# single fold train for Optuna / CV
# ============================================================
def stg_single_fold_train(
    X_train, y_train, X_val, y_val,
    model_params, training_params,
    folder_path, logger, eval_metric
):
    os.makedirs(folder_path, exist_ok=True)


    task_type = model_params["task_type"]

    model = _build_stg_model(X_train, y_train, model_params, training_params, logger)

    start_time = time.time()
    _fit_stg_model(model, X_train, y_train, X_val, y_val, training_params)
    elapsed = time.time() - start_time

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_metrics = _compute_metrics(y_train, train_pred, task_type)
    val_metrics = _compute_metrics(y_val, val_pred, task_type)

    score = _score_from_metrics(val_metrics, eval_metric, task_type)

    prob_gates = np.asarray(model.get_gates(mode="prob")).reshape(-1)
    raw_gates = np.asarray(model.get_gates(mode="raw")).reshape(-1)

    np.savez(
        os.path.join(folder_path, "stg_fold_result.npz"),
        prob_gates=prob_gates,
        raw_gates=raw_gates,
        train_metrics=_to_jsonable(train_metrics),
        val_metrics=_to_jsonable(val_metrics),
    )

    return {
        "model": model,
        "train_metrics": [train_metrics],
        "val_metrics": [val_metrics],
        "train_losses": [train_metrics.get("loss", np.nan)],
        "val_losses": [val_metrics.get("loss", np.nan)],
        "score": float(score),
        "prob_gates": prob_gates,
        "raw_gates": raw_gates,
        "elapsed": elapsed,
    }


# ============================================================
# final training (single run, no optuna)
# ============================================================
def STG_model_train(data, label, model_params, training_params, folder_name, logger):
    total_start_time = time.time()
    os.makedirs(folder_name, exist_ok=True)

    X = data
    y = label

    task_type = model_params["task_type"]
    validation_split = training_params.get("validation_split", 0.2)

    if validation_split > 0:
        n_samples = X.shape[0]
        all_indices = np.arange(n_samples)
        stratify_y = _as_1d_cpu_numpy(y) if task_type in ("binary", "multiclass") else None
        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=validation_split,
            random_state=training_params.get("seed", 42),
            stratify=stratify_y
        )

        if isinstance(X, torch.Tensor):
            train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=X.device)
            val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=X.device)
            X_train, X_val = X[train_idx_t], X[val_idx_t]
        else:
            X_train, X_val = X[train_idx], X[val_idx]

        if isinstance(y, torch.Tensor):
            train_idx_t_y = torch.as_tensor(train_idx, dtype=torch.long, device=y.device)
            val_idx_t_y = torch.as_tensor(val_idx, dtype=torch.long, device=y.device)
            y_train, y_val = y[train_idx_t_y], y[val_idx_t_y]
        else:
            y_train, y_val = y[train_idx], y[val_idx]
    else:
        X_train, X_val, y_train, y_val = X, X, y, y

    logger.info(f"STG_model_train: X_train={X_train.shape}, X_val={X_val.shape}")

    model = _build_stg_model(X_train, y_train, model_params, training_params, logger)

    train_start_time = time.time()
    _fit_stg_model(model, X_train, y_train, X_val, y_val, training_params)
    train_time = time.time() - train_start_time

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_metrics = _compute_metrics(y_train, train_pred, task_type)
    val_metrics = _compute_metrics(y_val, val_pred, task_type)

    prob_gates = np.asarray(model.get_gates(mode="prob")).reshape(-1)
    raw_gates = np.asarray(model.get_gates(mode="raw")).reshape(-1)

    feature_indices = np.argsort(-prob_gates)
    feature_scores = prob_gates[feature_indices]

    np.savez(
        os.path.join(folder_name, "feature_selection_result.npz"),
        prob_gates=prob_gates,
        raw_gates=raw_gates,
        feature_indices=feature_indices,
        feature_scores=feature_scores
    )

    with open(os.path.join(folder_name, "feature_selection_result.json"), "w") as f:
        json.dump({
            "feature_indices": feature_indices.tolist(),
            "feature_scores": feature_scores.tolist(),
            "top_100_features": feature_indices[:100].tolist(),
            "top_100_scores": feature_scores[:100].tolist(),
            "train_metrics": _to_jsonable(train_metrics),
            "val_metrics": _to_jsonable(val_metrics),
            "sigma": float(training_params["sigma"]),
            "lam": float(training_params["lam"]),
        }, f, indent=2)

    total_time = time.time() - total_start_time
    with open(os.path.join(folder_name, "timing_info.txt"), "w") as f:
        f.write(f"STG Training Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Training time: {format_time(train_time)} ({train_time:.2f} seconds)\n")
        f.write(f"Total time:    {format_time(total_time)} ({total_time:.2f} seconds)\n")

    logger.info(f"STG training finished, training time: {format_time(train_time)}")
    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Val metrics: {val_metrics}")
    logger.info(f"Top 20 selected features: {feature_indices[:20].tolist()}")

    return {
        "model": model,
        "feature_indices": feature_indices,
        "feature_scores": feature_scores,
        "prob_gates": prob_gates,
        "raw_gates": raw_gates,
        "train_metrics": [train_metrics],
        "val_metrics": [val_metrics],
        "train_losses": [train_metrics.get("loss", np.nan)],
        "val_losses": [val_metrics.get("loss", np.nan)],
    }


# ============================================================
# optuna search
# ============================================================
def optuna_search_STG(data, label, model_params, training_params, folder_name, logger):
    import optuna

    total_start_time = time.time()

    X = data
    y = label

    eval_metric = training_params.get("eval_metric", None)
    use_cv = training_params.get("use_cv", True)

    if eval_metric is None:
        eval_metric = "acc" if model_params["task_type"] in ("binary", "multiclass") else "r2"

    minimize_metric = eval_metric in ["mse", "mae", "loss"]

    logger.info(f"Parameter search will use '{eval_metric}' as the evaluation metric.")
    logger.info(f"During optimization, the metric will be {'minimized' if minimize_metric else 'maximized'}.")
    logger.info(f"Hyperparameter search use_cv={use_cv}")

    weight_decay_min = training_params.get("weight_decay_min", 0.0)
    weight_decay_max = training_params.get("weight_decay_max", 1e-1)
    lr_min = training_params.get("lr_min", 1e-5)
    lr_max = training_params.get("lr_max", 1e-1)
    dropout_min = training_params.get("dropout_min", 0.1)
    dropout_max = training_params.get("dropout_max", 0.5)
    sigma_min = training_params.get("sigma_min", 0.1)
    sigma_max = training_params.get("sigma_max", 1.0)
    lam_min = training_params.get("lam_min", 1e-4)
    lam_max = training_params.get("lam_max", 1.0)

    start_digit_choices = parse_list_arg(training_params.get("digits_list", "50,100,200"), dtype=int)
    shrink_ratio_choices = parse_list_arg(training_params.get("shrink_ratio_list", "1.25,1.5,2.0"), dtype=float)
    batch_size_choices = parse_list_arg(training_params.get("batch_size_list", "16,32,64,128"), dtype=int)
    min_layers = max(2, int(training_params.get("min_layers", 2)))
    max_layers = max(min_layers, int(training_params.get("max_layers", 5)))

    optuna_folder = os.path.join(folder_name, "optuna_search")
    os.makedirs(optuna_folder, exist_ok=True)

    k_folds = training_params.get("n_splits", 3)
    val_ratio = training_params.get("validation_split", 0.2)

    if use_cv:
        if model_params["task_type"] in ("binary", "multiclass"):
            splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=training_params["seed"])
            logger.info(f"Using {k_folds}-fold stratified cross-validation during Optuna search.")
        else:
            splitter = KFold(n_splits=k_folds, shuffle=True, random_state=training_params["seed"])
            logger.info(f"Using {k_folds}-fold cross-validation during Optuna search.")
    else:
        splitter = None
        logger.info(
            f"use_cv=False, so Optuna search will use a single validation split "
            f"(validation_split={val_ratio}) instead of K-fold CV."
        )

    def objective(trial):
        trial_id = trial.number
        trial_start_time = time.time()

        weight_decay = trial.suggest_float(
            "weight_decay",
            max(weight_decay_min, 1e-8),
            max(weight_decay_max, max(weight_decay_min, 1e-8)),
            log=True
        )
        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
        dropout = trial.suggest_float("dropout", dropout_min, dropout_max)
        sigma = trial.suggest_float("sigma", sigma_min, sigma_max)
        lam = trial.suggest_float("lam", lam_min, lam_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", batch_size_choices)
        n_layers = trial.suggest_int("n_layers", min_layers, max_layers)
        start_digit = trial.suggest_categorical("start_digit", start_digit_choices)
        shrink_ratio = trial.suggest_categorical("shrink_ratio", shrink_ratio_choices)
        digits_list = [
            max(1, int(round(start_digit * (shrink_ratio ** i))))
            for i in range(n_layers)
        ]
        hidden_dims = _build_hidden_dims_from_digits_list(X.shape[1], digits_list)

        logger.info(
            f"Trial {trial_id}: weight_decay={weight_decay}, lr={lr}, dropout={dropout}, "
            f"sigma={sigma}, lam={lam}, batch_size={batch_size}, "
            f"n_layers={n_layers}, start_digit={start_digit}, shrink_ratio={shrink_ratio}, "
            f"generated_digits_list={digits_list}, generated_hidden_dims={hidden_dims}"
        )

        fold_scores = []
        fold_times = []

        if use_cv:
            if model_params["task_type"] in ("binary", "multiclass"):
                split_iter = splitter.split(np.arange(X.shape[0]), _as_1d_cpu_numpy(y))
            else:
                split_iter = splitter.split(np.arange(X.shape[0]))
        else:
            all_indices = np.arange(X.shape[0])
            label_np = _as_1d_cpu_numpy(y) if model_params["task_type"] in ("binary", "multiclass") else None
            train_idx, val_idx = train_test_split(
                all_indices,
                test_size=val_ratio,
                random_state=training_params["seed"],
                stratify=label_np if label_np is not None else None
            )
            split_iter = [(train_idx, val_idx)]

        n_eval_splits = k_folds if use_cv else 1

        for fold, (train_idx, val_idx) in enumerate(split_iter):
            fold_start_time = time.time()
            logger.info(f"Trial {trial_id}, Fold {fold+1}/{n_eval_splits}")

            if isinstance(X, torch.Tensor):
                train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=X.device)
                val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=X.device)
                X_train, X_val = X[train_idx_t], X[val_idx_t]
            else:
                X_train, X_val = X[train_idx], X[val_idx]

            if isinstance(y, torch.Tensor):
                train_idx_t_y = torch.as_tensor(train_idx, dtype=torch.long, device=y.device)
                val_idx_t_y = torch.as_tensor(val_idx, dtype=torch.long, device=y.device)
                y_train, y_val = y[train_idx_t_y], y[val_idx_t_y]
            else:
                y_train, y_val = y[train_idx], y[val_idx]

            current_model_params = copy.deepcopy(model_params)
            current_model_params["n_layers"] = n_layers
            current_model_params["start_digit"] = int(start_digit)
            current_model_params["shrink_ratio"] = float(shrink_ratio)
            current_model_params["digits_list"] = [int(d) for d in digits_list]
            current_model_params["hidden_dims"] = hidden_dims
            current_model_params["hidden_size"] = current_model_params["hidden_dims"][0]
            current_model_params["dropout_prob"] = dropout
            current_model_params["digits"] = int(start_digit)

            current_training_params = copy.deepcopy(training_params)
            current_training_params["weight_decay"] = weight_decay
            current_training_params["lr"] = lr
            current_training_params["batch_size"] = batch_size
            current_training_params["sigma"] = sigma
            current_training_params["lam"] = lam

            fold_result = stg_single_fold_train(
                X_train, y_train, X_val, y_val,
                current_model_params, current_training_params,
                os.path.join(optuna_folder, f"trial_{trial_id}_fold_{fold}"),
                logger,
                eval_metric=eval_metric
            )

            fold_score = float(fold_result["score"])
            fold_scores.append(fold_score)

            fold_time = time.time() - fold_start_time
            fold_times.append(fold_time)

            logger.info(
                f"Trial {trial_id}, Fold {fold+1}/{n_eval_splits} completed - "
                f"Time taken: {format_time(fold_time)}"
            )

        avg_score = float(np.mean(fold_scores))
        trial_time = time.time() - trial_start_time

        logger.info(
            f"Trial {trial_id} completed - average {eval_metric}: {avg_score:.6f}, "
            f"time taken: {format_time(trial_time)}"
        )

        return avg_score

    direction = "minimize" if minimize_metric else "maximize"
    db_file = os.path.join(optuna_folder, "optuna_STG.db")
    db_url = f"sqlite:///{db_file}"
    n_trials = training_params.get("n_trials", 20)
    n_jobs = training_params.get("n_jobs", 1)
    logger.info(
        f"Use {n_jobs} parallel workers to run {n_trials} Optuna optimization trials "
        f"(use_cv={use_cv})"
    )

    storage = RDBStorage(
        url=db_url,
        engine_kwargs={
            "poolclass": QueuePool,
            "pool_size": min(n_jobs + 1, 20),
            "max_overflow": 10,
            "pool_timeout": 30,
        }
    )

    sampler = TPESampler(seed=training_params["seed"])

    study = optuna.create_study(
        sampler=sampler,
        direction=direction,
        storage=storage,
        study_name="STG_shrink_ratio_v3",
        load_if_exists=True
    )


    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    best_params = study.best_params
    best_score = study.best_value

    with open(os.path.join(optuna_folder, "best_params.json"), "w") as f:
        json.dump(_to_jsonable(best_params), f, indent=2)

    search_time = time.time() - total_start_time


    
    logger.info(f"Best parameters found by Optuna: {best_params}")
    logger.info(f"Best {eval_metric}: {best_score}")

    final_model_params = copy.deepcopy(model_params)
    if "n_layers" in best_params:
        best_n_layers = max(2, int(best_params["n_layers"]))
        best_start_digit = int(best_params.get("start_digit", final_model_params.get("start_digit", final_model_params.get("digits", 100))))
        best_shrink_ratio = float(best_params.get("shrink_ratio", final_model_params.get("shrink_ratio", training_params.get("shrink_ratio", 2.0))))
        best_digits_list = [
            max(1, int(round(best_start_digit * (best_shrink_ratio ** i))))
            for i in range(best_n_layers)
        ]
        final_model_params["n_layers"] = best_n_layers
        final_model_params["start_digit"] = best_start_digit
        final_model_params["shrink_ratio"] = best_shrink_ratio
        final_model_params["digits_list"] = best_digits_list
        final_model_params["hidden_dims"] = _build_hidden_dims_from_digits_list(X.shape[1], best_digits_list)
        final_model_params["hidden_size"] = final_model_params["hidden_dims"][0]
        final_model_params["digits"] = int(best_start_digit)
    elif "digits" in best_params:
        fallback_n_layers = max(2, int(final_model_params.get("n_layers", 2)))
        fallback_start_digit = int(best_params["digits"])
        fallback_shrink_ratio = float(final_model_params.get("shrink_ratio", training_params.get("shrink_ratio", 2.0)))
        best_digits_list = [
            max(1, int(round(fallback_start_digit * (fallback_shrink_ratio ** i))))
            for i in range(fallback_n_layers)
        ]
        final_model_params["n_layers"] = fallback_n_layers
        final_model_params["start_digit"] = fallback_start_digit
        final_model_params["shrink_ratio"] = fallback_shrink_ratio
        final_model_params["digits_list"] = best_digits_list
        final_model_params["hidden_dims"] = _build_hidden_dims_from_digits_list(X.shape[1], best_digits_list)
        final_model_params["hidden_size"] = final_model_params["hidden_dims"][0]
        final_model_params["digits"] = fallback_start_digit
    else:
        final_model_params["hidden_dims"] = model_params.get("hidden_dims", [500, 50])
        final_model_params["hidden_size"] = final_model_params["hidden_dims"][0]
        final_model_params["digits"] = int(final_model_params.get("start_digit", final_model_params.get("digits", 100)))
    final_model_params["dropout_prob"] = best_params["dropout"]

    final_training_params = copy.deepcopy(training_params)
    final_training_params["weight_decay"] = best_params["weight_decay"]
    final_training_params["lr"] = best_params["lr"]
    final_training_params["batch_size"] = best_params["batch_size"]
    final_training_params["sigma"] = best_params["sigma"]
    final_training_params["lam"] = best_params["lam"]

    final_model_time_start = time.time()

    results = STG_model_train(
        X, y,
        final_model_params,
        final_training_params,
        folder_name,
        logger
    )
    final_model_time = time.time() - final_model_time_start

    total_time_for_txt = time.time() - total_start_time
    with open(os.path.join(optuna_folder, "best_params.txt"), "w") as f:
        f.write("Best hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write("\nBest architecture:\n")
        f.write(f"n_layers: {final_model_params.get('n_layers')}\n")
        f.write(f"start_digit: {final_model_params.get('start_digit')}\n")
        f.write(f"shrink_ratio: {final_model_params.get('shrink_ratio')}\n")
        f.write(f"digits_list: {final_model_params.get('digits_list')}\n")
        f.write(f"hidden_dims: {final_model_params.get('hidden_dims')}\n")
        f.write(f"\nBest value: {best_score}\n")
        f.write(f"Search time: {format_time(search_time)} ({search_time:.2f} seconds)\n")
        f.write(f"Final model training time: {format_time(final_model_time)} ({final_model_time:.2f} seconds)\n")
        f.write(f"Total time: {format_time(total_time_for_txt)} ({total_time_for_txt:.2f} seconds)\n")

    results["best_params"] = copy.deepcopy(best_params)
    results["best_value"] = float(best_score)
    results["final_model_params"] = copy.deepcopy(final_model_params)
    results["best_architecture"] = {
        "n_layers": final_model_params.get("n_layers"),
        "start_digit": final_model_params.get("start_digit"),
        "shrink_ratio": final_model_params.get("shrink_ratio"),
        "digits_list": final_model_params.get("digits_list"),
        "hidden_dims": final_model_params.get("hidden_dims"),
    }

    total_time = time.time() - total_start_time
    logger.info(f"Total processing time: {format_time(total_time)} ({total_time/60:.2f} minutes)")

    return results


# ============================================================
# public API: feature selection only
# ============================================================
def STG_fs(data, label, model_params, training_params, name, folder_name, logger):
    X = data["X_train"]
    y = label["y_train"]

    do_parameter_search = training_params.get("do_parameter_search", 0)

    if do_parameter_search == 1:
        logger.info("Start Optuna hyperparameter optimization for STG...")
        results = optuna_search_STG(X, y, model_params, training_params, folder_name, logger)
        if isinstance(results, dict) and results.get("best_architecture") is not None:
            best_architecture = results["best_architecture"]
            logger.info(
                f"best STG architecture from Optuna: n_layers={best_architecture.get('n_layers')}, "
                f"start_digit={best_architecture.get('start_digit')}, shrink_ratio={best_architecture.get('shrink_ratio')}, "
                f"digits_list={best_architecture.get('digits_list')}, hidden_dims={best_architecture.get('hidden_dims')}"
            )
    else:
        logger.info("Parameter optimization is disabled. Using single parameter values.")
        logger.info(f"Using parameters:")
        logger.info(f"  sigma: {training_params['sigma']}")
        logger.info(f"  lam: {training_params['lam']}")
        logger.info(f"  weight_decay: {training_params.get('weight_decay', 0.0)}")
        logger.info(f"  dropout: {model_params['dropout_prob']}")
        logger.info(f"  lr: {training_params['lr']}")
        logger.info(f"  batch_size: {training_params['batch_size']}")
        logger.info(f"  start_digit: {model_params.get('start_digit', model_params.get('digits', 100))}")
        logger.info(f"  shrink_ratio: {model_params.get('shrink_ratio', training_params.get('shrink_ratio', 2.0))}")
        logger.info(f"  n_layers: {model_params.get('n_layers', 2)}")
        logger.info(f"  generated_digits_list: {model_params.get('digits_list', None)}")
        logger.info(f"  hidden_dims: {model_params.get('hidden_dims', None)}")
        logger.info("  use_cv: False (no Optuna CV because do_parameter_search=0)")

        results = STG_model_train(X, y, model_params, training_params, folder_name, logger)
    np.save(os.path.join(folder_name, f'STG_{name}_prob_gates.npy'),
            results['prob_gates'])

    np.save(os.path.join(folder_name, f'STG_{name}_raw_gates.npy'),
            results['raw_gates'])

    np.save(os.path.join(folder_name, f'STG_{name}_idx.npy'),
            results['feature_indices'])

    np.save(os.path.join(folder_name, f'STG_{name}_scores.npy'),
            results['feature_scores'])
    return results


# ============================================================
# public API: feature selection + evaluation
# ============================================================
def STG_fs_with_evaluation(
    data, label,
    model_params, training_params,
    name, digits, folder_name, logger,
    feature_prediction,
    n_iters=20, feature_step=500, n_folds=3
):
    """
    Perform STG feature selection with evaluation across varying feature counts.

    Args:
        data: input X data
        label: target values
        model_params: model parameter dictionary
        training_params: training parameter dictionary
        name: dataset name
        digits: feature dimension divisor
        folder_name: result save directory
        logger: logger
        feature_prediction: upper limit on the number of features to evaluate
        n_iters: number of iterations
        feature_step: feature evaluation step size
        n_folds: number of cross-validation folds
    """

    total_start_time = time.time()
    logger.info(f" STG_fs_with_evaluation start {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    x_train = data['X_train']
    x_test = data['X_test']
    y_train = label['y_train']
    y_test = label['y_test']

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
    # seeds = np.random.choice(range(1000), n_iters, replace=False)
    seeds = [int(s) for s in np.random.choice(range(1000), n_iters, replace=False)]
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
            current_model_params['n_layers'] = int(current_model_params.get('n_layers', 2))
            current_model_params['start_digit'] = int(digits)
            current_model_params['shrink_ratio'] = float(
                current_model_params.get('shrink_ratio', current_training_params.get('shrink_ratio', 2.0))
            )
            current_model_params['digits_list'] = [
                max(1, int(round(current_model_params['start_digit'] * (current_model_params['shrink_ratio'] ** i))))
                for i in range(current_model_params['n_layers'])
            ]
            current_model_params['hidden_dims'] = _build_hidden_dims_from_digits_list(
                x_train.shape[1], current_model_params['digits_list']
            )
            current_model_params['hidden_size'] = current_model_params['hidden_dims'][0]
            current_model_params['digits'] = int(digits)

            current_training_params['device'] = device
            current_training_params['seed'] = seeds[iter]

            if current_training_params.get('do_parameter_search', False):
                logger.info("using Optuna for hyperparameter optimization and training...")
                logger.info(
                    f"current fixed STG architecture before Optuna: n_layers={current_model_params['n_layers']}, "
                    f"start_digit={current_model_params.get('start_digit')}, shrink_ratio={current_model_params.get('shrink_ratio')}, "
                    f"generated_digits_list={current_model_params.get('digits_list')}, hidden_dims={current_model_params.get('hidden_dims')}"
                )
                stg_results = optuna_search_STG(
                    x_train, y_train,
                    current_model_params,
                    current_training_params,
                    os.path.join(eval_folder, f'optuna_iter_{iter}'),
                    logger
                )

                best_params = stg_results.get('best_params', None) if isinstance(stg_results, dict) else None
                best_architecture = stg_results.get('best_architecture', None) if isinstance(stg_results, dict) else None
                if best_params is not None:
                    all_best_params.append(best_params)
                    logger.info(f"get best params: {best_params}")
                else:
                    best_params_file = os.path.join(eval_folder, f'optuna_iter_{iter}', 'optuna_search', 'best_params.json')
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
                        f"best STG architecture from Optuna: n_layers={best_architecture.get('n_layers')}, "
                        f"start_digit={best_architecture.get('start_digit')}, shrink_ratio={best_architecture.get('shrink_ratio')}, "
                        f"digits_list={best_architecture.get('digits_list')}, hidden_dims={best_architecture.get('hidden_dims')}"
                    )

            else:
                logger.info("using fixed parameters to train model...")
                stg_results = STG_model_train(
                    x_train, y_train,
                    current_model_params,
                    current_training_params,
                    os.path.join(eval_folder, f'model_iter_{iter}'),
                    logger
                )

            feature_ranking = np.asarray(stg_results['feature_indices'])
            feature_weights = np.asarray(stg_results['prob_gates'])

            results_weights.append(feature_weights)
            results_indices.append(feature_ranking)

            feature_ranking_file = os.path.join(iter_folder, f'feature_ranking_iter_{iter}.npz')
            np.savez(
                feature_ranking_file,
                feature_indices=feature_ranking,
                weights=feature_weights,
                raw_gates=np.asarray(stg_results['raw_gates']),
                iter=iter,
                seed=seeds[iter]
            )
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
                    logger.info(f"selected_features前几个: {selected_features[:10]}...")

                    if isinstance(x_train, torch.Tensor):
                        selected_tensor_train = torch.tensor(selected_features, device=x_train.device, dtype=torch.long)
                        selected_tensor_test = torch.tensor(selected_features, device=x_test.device, dtype=torch.long)
                        x_train_selected = x_train.index_select(1, selected_tensor_train)
                        x_test_selected = x_test.index_select(1, selected_tensor_test)
                    else:
                        x_train_selected = x_train[:, selected_features]
                        x_test_selected = x_test[:, selected_features]

                    final_score, best_dropout, best_l2, best_cv_val_loss, report_metrics = evaluate_feature_set(
                        x_train_selected, y_train,
                        x_test_selected, y_test,
                        num_features, current_model_params,
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
                    np.savez(
                        feature_eval_file,
                        num_features=num_features,
                        iter=iter,
                        selected_features=selected_features,
                        test_score=final_score,
                        best_cv_val_loss=best_cv_val_loss,
                        best_dropout=best_dropout,
                        best_l2=best_l2,
                        report_metrics=np.array(_to_jsonable(report_metrics), dtype=object)
                    )
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

            np.savez(
                iter_results_file,
                iter=iter,
                test_results=results[:, iter],
                cv_val_losses=cv_val_losses[:, iter],
                feature_indices=feature_ranking,
                feature_weights=feature_weights,
                feature_sequence=feature_sequence,
                seed=seeds[iter],
                timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                report_metrics_column=report_metrics_iter_column
            )
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
        'weights': np.array(results_weights, dtype=object),     # STG: prob_gates
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

    save_path = os.path.join(folder_name, f'STG_{name}_results.npz')
    np.savez(save_path, **final_results)

    logger.info("\nFinal results summary:")
    for step, num_features in enumerate(feature_sequence):
        if step >= len(mean_test_results):
            continue

        num_features = int(num_features)
        logger.info(f"\n{num_features} features results:")
        logger.info(f"Average val_loss: {mean_cv_val_losses[step]:.4f} ± {std_cv_val_losses[step]:.4f}")
        logger.info(f"Average test score: {mean_test_results[step]:.4f} ± {std_test_results[step]:.4f}")

    logger.info(f"\nResults saved to: {save_path}")

    return final_results