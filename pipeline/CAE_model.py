import os
import time
import copy
import json
import datetime
import joblib
import numpy as np
import torch
import math


from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error

from evaluation_utils import evaluate_feature_set


import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch.utils.data import TensorDataset, DataLoader
import optuna
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from sqlalchemy.pool import QueuePool

# Helper to set seeds and deterministic behavior
def set_torch_seed(seed, deterministic=True):
    if seed is None:
        return

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ConcreteSelect(nn.Module):

    def __init__(self, input_dim, output_dim, start_temp=10.0, min_temp=0.1, alpha=0.99999, **kwargs):
        super(ConcreteSelect, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.alpha = alpha

        # non-trainable temperature
        self.register_buffer("temp", torch.tensor(float(self.start_temp), dtype=torch.float32))

        # trainable logits: [output_dim, input_dim]
        self.logits = nn.Parameter(torch.empty(self.output_dim, self.input_dim))
        nn.init.xavier_normal_(self.logits)

        self.selections = None

    def forward(self, X):
        # training phase: soft selection with Gumbel-Softmax / Concrete
        if self.training:
            uniform = torch.rand_like(self.logits).clamp(min=1e-7, max=1.0 - 1e-7)
            gumbel = -torch.log(-torch.log(uniform))

            # update temperature
            new_temp = max(self.min_temp, float(self.temp.item()) * self.alpha)
            self.temp.fill_(new_temp)

            noisy_logits = (self.logits + gumbel) / self.temp
            samples = F.softmax(noisy_logits, dim=1)
            self.selections = samples
        else:
            # inference phase: hard discrete selection
            indices = torch.argmax(self.logits, dim=1)
            discrete_logits = F.one_hot(indices, num_classes=self.input_dim).float()
            self.selections = discrete_logits

        # X: [batch_size, input_dim]
        # selections: [output_dim, input_dim]
        # output: [batch_size, output_dim]
        Y = torch.matmul(X, self.selections.t())
        return Y


class StopperCallback:

    def __init__(self, mean_max_target=0.998, logger=None, verbose=0):
        self.mean_max_target = mean_max_target
        self.logger = logger
        self.verbose = verbose

    def _log(self, message):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def on_epoch_begin(self, model):
        if not self.verbose:
            return
        self._log(
            f"mean max of probabilities: {self.get_monitor_value(model):.6f} - temperature {float(model.concrete_select.temp.item()):.6f}"
        )

    def get_monitor_value(self, model):
        with torch.no_grad():
            probs = F.softmax(model.concrete_select.logits, dim=1)
            monitor_value = probs.max(dim=1).values.mean().item()
        return monitor_value


class _ConcreteAutoencoderModel(nn.Module):

    def __init__(self, input_dim, K, decoder_module, start_temp, min_temp, alpha):
        super(_ConcreteAutoencoderModel, self).__init__()
        self.concrete_select = ConcreteSelect(
            input_dim=input_dim,
            output_dim=K,
            start_temp=start_temp,
            min_temp=min_temp,
            alpha=alpha
        )
        self.decoder = decoder_module

    def forward(self, X):
        selected_features = self.concrete_select(X)
        outputs = self.decoder(selected_features)
        return outputs


class ConcreteAutoencoderFeatureSelector():

    def __init__(
        self,
        K,
        output_function,
        num_epochs=300,
        batch_size=None,
        learning_rate=0.001,
        weight_decay=0.0,
        start_temp=10.0,
        min_temp=0.1,
        tryout_limit=5,
        device=None,
        seed=None,
        deterministic=True,
        logger=None,
        selector_mode='unsupervised',
        task_type='regression',
        num_classes=None,
        val_patience=10,
        val_min_delta=0.0,
        verbose=0,
    ):
        self.K = K
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.probabilities = None
        self.indices = None
        self.seed = seed
        self.deterministic = deterministic
        self.logger = logger
        self.selector_mode = selector_mode
        self.task_type = task_type
        self.num_classes = num_classes
        self.val_patience = val_patience
        self.val_min_delta = val_min_delta
        self.verbose = verbose

    def _log(self, message):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def _build_criterion(self):
        if self.selector_mode == 'unsupervised':
            return nn.MSELoss()
        if self.task_type == 'binary':
            return nn.BCEWithLogitsLoss()
        if self.task_type == 'multiclass':
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _prepare_targets(self, y):
        if self.selector_mode == 'unsupervised':
            return y.float()
        if self.task_type == 'multiclass':
            return y.long().view(-1)
        if self.task_type == 'binary':
            y = y.float()
            if y.ndim == 1:
                y = y.view(-1, 1)
            return y
        y = y.float()
        if y.ndim == 1:
            y = y.view(-1, 1)
        return y

    def fit(self, X, Y=None, val_X=None, val_Y=None):
        if Y is None:
            Y = X

        assert len(X) == len(Y)
        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)

        set_torch_seed(self.seed, deterministic=self.deterministic)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y)
        Y = self._prepare_targets(Y)

        if validation_data is not None:
            val_X, val_Y = validation_data
            if not isinstance(val_X, torch.Tensor):
                val_X = torch.tensor(val_X, dtype=torch.float32)
            if not isinstance(val_Y, torch.Tensor):
                val_Y = torch.tensor(val_Y)
            val_Y = self._prepare_targets(val_Y)

        if self.batch_size is None:
            self.batch_size = max(len(X) // 256, 16)

        num_epochs = self.num_epochs
        steps_per_epoch = max((len(X) + self.batch_size - 1) // self.batch_size, 1)

        dataset = TensorDataset(X.float(), Y)
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator
        )

        input_dim = X.shape[1]

        for i in range(self.tryout_limit):
            if self.seed is not None:
                set_torch_seed(self.seed + i, deterministic=self.deterministic)

            alpha = math.exp(math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))

            decoder_module = self.output_function()
            self.model = _ConcreteAutoencoderModel(
                input_dim=input_dim,
                K=self.K,
                decoder_module=decoder_module,
                start_temp=self.start_temp,
                min_temp=self.min_temp,
                alpha=alpha
            ).to(self.device)

            optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            criterion = self._build_criterion()

            if self.verbose:
                self._log(str(self.model))

            stopper_callback = StopperCallback(logger=self.logger, verbose=self.verbose)
            best_state_dict = None
            best_monitor = -float("inf")
            best_val_loss = float("inf")
            no_improve_count = 0

            for epoch in range(num_epochs):
                self.model.train()
                stopper_callback.on_epoch_begin(self.model)

                running_loss = 0.0
                num_samples = 0

                for batch_X, batch_Y in loader:
                    batch_X = batch_X.to(self.device).float()
                    batch_Y = batch_Y.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    if self.selector_mode == 'supervised' and self.task_type == 'binary' and outputs.ndim == 1:
                        outputs = outputs.view(-1, 1)
                    loss = criterion(outputs, batch_Y)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * batch_X.size(0)
                    num_samples += batch_X.size(0)

                train_loss = running_loss / max(num_samples, 1)

                val_loss = None
                if validation_data is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(val_X.to(self.device).float())
                        if self.selector_mode == 'supervised' and self.task_type == 'binary' and val_out.ndim == 1:
                            val_out = val_out.view(-1, 1)
                        val_loss = criterion(val_out, val_Y.to(self.device)).item()
                    if self.verbose:
                        self._log(f"Epoch {epoch+1}/{num_epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")
                else:
                    if self.verbose:
                        self._log(f"Epoch {epoch+1}/{num_epochs} - loss: {train_loss:.6f}")

                monitor_value = stopper_callback.get_monitor_value(self.model)

                if validation_data is not None:
                    if val_loss + self.val_min_delta < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        best_state_dict = copy.deepcopy(self.model.state_dict())
                    else:
                        no_improve_count += 1
                    if no_improve_count >= self.val_patience:
                        self._log(
                            f"Early stop on validation loss: patience reached ({self.val_patience})"
                        )
                        break
                else:
                    if monitor_value > best_monitor:
                        best_monitor = monitor_value
                        best_state_dict = copy.deepcopy(self.model.state_dict())

                if monitor_value >= stopper_callback.mean_max_target:
                    self._log(
                        f"Stop training: monitor reached target ({monitor_value:.6f} >= {stopper_callback.mean_max_target})"
                    )
                    break

            if best_state_dict is not None:
                self.model.load_state_dict(best_state_dict)

            final_monitor = stopper_callback.get_monitor_value(self.model)
            if final_monitor >= stopper_callback.mean_max_target:
                break

            num_epochs *= 2

        self.model.eval()
        with torch.no_grad():
            probs = F.softmax(self.model.concrete_select.logits, dim=1)
            idx = torch.argmax(self.model.concrete_select.logits, dim=1)

        self.probabilities = probs.detach().cpu().numpy()
        self.indices = idx.detach().cpu().numpy()

        return self

    def get_indices(self):
        self.model.eval()
        with torch.no_grad():
            idx = torch.argmax(self.model.concrete_select.logits, dim=1)
        return idx.detach().cpu().numpy()

    def get_mask(self):
        self.model.eval()
        with torch.no_grad():
            idx = torch.argmax(self.model.concrete_select.logits, dim=1)
            mask = F.one_hot(idx, num_classes=self.model.concrete_select.logits.shape[1]).sum(dim=0)
        return mask.detach().cpu().numpy()

    def transform(self, X):
        indices = self.get_indices()
        if isinstance(X, torch.Tensor):
            return X[:, indices]
        else:
            X = np.asarray(X)
            return X[:, indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        return self.get_indices() if indices else self.get_mask()

    def get_params(self):
        return self.model

    def get_probabilities(self):
        if self.probabilities is None:
            raise ValueError("Model has not been fit yet.")
        return self.probabilities






# ============================================================
# utils
# ============================================================



def is_CAE_result_complete(result_file, expected_k=None, logger=None):
    """
     Check whether a CAE `feature_selection_result.npz` file is complete.

     Complete if:
     1. The file exists
     2. `np.load` can read it
     3. Required keys are present
     4. `feature_indices`, `feature_scores`, `selected_indices`, `selected_mask`,
         and `selection_probabilities` are non-empty
     5. If `expected_k` is not None, the raw length of `selected_indices`
         should equal `expected_k`

     Note:
     - `selected_indices` may contain duplicates, so the number of unique entries
        can be less than K.
     - This check validates the raw length of `selected_indices` against K.
    """
    
    if not os.path.exists(result_file):
        if logger is not None:
            logger.info(f"[CAE check] result file does not exist: {result_file}")
        return False

    required_keys = [
        "feature_indices",
        "feature_scores",
        "selected_indices",
        "selected_mask",
        "selection_probabilities",
    ]

    try:
        data = np.load(result_file, allow_pickle=True)

        missing_keys = [key for key in required_keys if key not in data.files]
        if len(missing_keys) > 0:
            if logger is not None:
                logger.warning(
                    f"[CAE check] incomplete result: missing keys {missing_keys} in {result_file}"
                )
            return False

        feature_indices = np.asarray(data["feature_indices"]).ravel()
        feature_scores = np.asarray(data["feature_scores"]).ravel()
        selected_indices = np.asarray(data["selected_indices"]).ravel()
        selected_mask = np.asarray(data["selected_mask"]).ravel()
        selection_probabilities = np.asarray(data["selection_probabilities"])

        if feature_indices.size == 0:
            if logger is not None:
                logger.warning(f"[CAE check] empty feature_indices in {result_file}")
            return False

        if feature_scores.size == 0:
            if logger is not None:
                logger.warning(f"[CAE check] empty feature_scores in {result_file}")
            return False

        if selected_indices.size == 0:
            if logger is not None:
                logger.warning(f"[CAE check] empty selected_indices in {result_file}")
            return False

        if selected_mask.size == 0:
            if logger is not None:
                logger.warning(f"[CAE check] empty selected_mask in {result_file}")
            return False

        if selection_probabilities.size == 0:
            if logger is not None:
                logger.warning(f"[CAE check] empty selection_probabilities in {result_file}")
            return False

        if expected_k is not None:
            expected_k = int(expected_k)

            if selected_indices.size != expected_k:
                if logger is not None:
                    logger.warning(
                        f"[CAE check] selected_indices length mismatch in {result_file}: "
                        f"expected K={expected_k}, got {selected_indices.size}"
                    )
                return False

        if logger is not None:
            logger.info(
                f"[CAE check] complete result found: {result_file}, "
                f"selected_indices length={selected_indices.size}, "
                f"unique selected={len(np.unique(selected_indices))}"
            )

        return True

    except Exception as e:
        if logger is not None:
            logger.warning(f"[CAE check] failed to load/check {result_file}: {e}")
        return False
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

# Selector mode normalization helper
def _normalize_selector_mode(value):
    mode = str(value).strip().lower()
    if mode not in ("unsupervised", "supervised"):
        raise ValueError(f"Unsupported selector_mode: {value}. Expected 'unsupervised' or 'supervised'.")
    return mode


def _safe_sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _prepare_binary_scores(pred):
    pred = np.asarray(pred)
    if pred.ndim > 1 and pred.shape[1] == 1:
        pred = pred.reshape(-1)

    if np.all((pred >= 0.0) & (pred <= 1.0)):
        prob = pred
    else:
        prob = _safe_sigmoid(pred)

    y_hat = (prob >= 0.5).astype(int)
    return prob, y_hat


def _prepare_multiclass_scores(pred):
    pred = np.asarray(pred)

    if pred.ndim == 1:
        y_hat = pred.astype(int)
        prob = None
    else:
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


# Helper to extract all selector outputs robustly
def _extract_selector_outputs(selector, reduction="max"):
    if hasattr(selector, "get_feature_scores") and hasattr(selector, "get_feature_ranking"):
        feature_scores = np.asarray(selector.get_feature_scores(reduction=reduction))
        feature_indices = np.asarray(selector.get_feature_ranking(reduction=reduction))
        selected_indices = np.asarray(selector.get_indices()) if hasattr(selector, "get_indices") else feature_indices[:0]
        if hasattr(selector, "get_mask"):
            selected_mask = np.asarray(selector.get_mask())
        elif hasattr(selector, "get_support"):
            selected_mask = np.asarray(selector.get_support())
        else:
            selected_mask = np.isin(np.arange(feature_scores.shape[0]), selected_indices)
        selection_probabilities = np.asarray(selector.get_probabilities()) if hasattr(selector, "get_probabilities") else feature_scores.copy()
        return feature_indices, feature_scores, selected_indices, selected_mask, selection_probabilities

    if not hasattr(selector, "get_probabilities"):
        raise AttributeError(
            "ConcreteAutoencoderFeatureSelector does not provide get_feature_scores/get_feature_ranking "
            "or get_probabilities, so feature outputs cannot be extracted."
        )

    probs = np.asarray(selector.get_probabilities())
    if probs.ndim == 1:
        feature_scores = probs.astype(float)
    elif probs.ndim == 2:
        if reduction == "mean":
            feature_scores = probs.mean(axis=0)
        else:
            feature_scores = probs.max(axis=0)
    else:
        raise ValueError(f"Unexpected probability shape from selector.get_probabilities(): {probs.shape}")

    feature_scores = np.asarray(feature_scores, dtype=float).reshape(-1)
    feature_indices = np.argsort(-feature_scores)
    selected_indices = np.asarray(selector.get_indices()) if hasattr(selector, "get_indices") else feature_indices[:0]
    if hasattr(selector, "get_support"):
        selected_mask = np.asarray(selector.get_support())
    else:
        selected_mask = np.isin(np.arange(feature_scores.shape[0]), selected_indices)
    selection_probabilities = probs
    return feature_indices, feature_scores, selected_indices, selected_mask, selection_probabilities


class CAEDecoder(torch.nn.Module):
    def __init__(self, input_k, hidden_dims, output_dim, dropout_rate=0.1):
        super().__init__()

        if hidden_dims is None or len(hidden_dims) == 0:
            raise ValueError("hidden_dims must not be empty")

        layers = []
        prev_dim = input_k
        for h in hidden_dims:
            h = max(4, int(h))
            layers.append(torch.nn.Linear(prev_dim, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = h

        layers.append(torch.nn.Linear(prev_dim, int(output_dim)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _make_decoder_builder(input_dim, K_select, hidden_dims, dropout_prob,
                          selector_mode="unsupervised", task_type="regression", num_classes=None):
    if hidden_dims is None or len(hidden_dims) == 0:
        raise ValueError("hidden_dims must not be empty")

    selector_mode = _normalize_selector_mode(selector_mode)
    normalized_hidden_dims = [max(4, int(h)) for h in hidden_dims]

    if selector_mode == "unsupervised":
        output_dim = int(input_dim)
    else:
        task_type = str(task_type).strip().lower()
        if task_type == "regression":
            output_dim = 1
        elif task_type == "binary":
            output_dim = 1
        elif task_type == "multiclass":
            if num_classes is None:
                raise ValueError("num_classes must be provided for supervised multiclass CAE")
            output_dim = int(num_classes)
        else:
            raise ValueError(f"Unsupported task_type for supervised CAE: {task_type}")

    def decoder():
        return CAEDecoder(
            input_k=K_select,
            hidden_dims=normalized_hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_prob
        )
    return decoder


def _build_selector(X_train, model_params, training_params):
    input_dim = X_train.shape[1]
    hidden_dims = model_params["hidden_dims"]
    K_select = training_params["K_select"]
    selector_mode = _normalize_selector_mode(
        training_params.get("selector_mode", model_params.get("selector_mode", "unsupervised"))
    )
    model_params["selector_mode"] = selector_mode
    training_params["selector_mode"] = selector_mode

    selector = ConcreteAutoencoderFeatureSelector(
        K=K_select,
        output_function=_make_decoder_builder(
            input_dim=input_dim,
            K_select=K_select,
            hidden_dims=hidden_dims,
            dropout_prob=model_params.get("dropout_prob", 0.0),
            selector_mode=selector_mode,
            task_type=model_params.get("task_type", "regression"),
            num_classes=model_params.get("num_classes", None)
        ),
        num_epochs=training_params["num_epochs"],
        batch_size=training_params["batch_size"],
        learning_rate=training_params["lr"],
        weight_decay=training_params.get("weight_decay", 0.0),
        start_temp=training_params["start_temp"],
        min_temp=training_params["min_temp"],
        tryout_limit=training_params["tryout_limit"],
        device=training_params.get("device", "cpu"),
        seed=training_params.get("seed", 42),
        deterministic=True,
        logger=training_params.get("logger", None),
        selector_mode=selector_mode,
        task_type=model_params.get("task_type", "regression"),
        num_classes=model_params.get("num_classes", None),
        val_patience=training_params.get("patience", 10),
        val_min_delta=training_params.get("min_delta", 0.0)
    )
    return selector


# ============================================================
# single fold train
# ============================================================
def CAE_single_fold_train(
    X_train, y_train, X_val, y_val,
    model_params, training_params,
    folder_path, logger, eval_metric
):

    X_train_np = _to_numpy(X_train).astype(np.float32)
    X_val_np = _to_numpy(X_val).astype(np.float32)
    y_train_np = _flatten_y_if_needed(y_train)
    y_val_np = _flatten_y_if_needed(y_val)

    selector_mode = _normalize_selector_mode(
        training_params.get("selector_mode", model_params.get("selector_mode", "unsupervised"))
    )
    training_params["logger"] = logger
    selector = _build_selector(X_train_np, model_params, training_params)

    start_time = time.time()
    if selector_mode == "unsupervised":
        selector.fit(X_train_np, X_train_np, X_val_np, X_val_np)
    else:
        selector.fit(X_train_np, y_train_np, X_val_np, y_val_np)
    elapsed = time.time() - start_time

    # Transform using the selected features to evaluate the reconstruction-based
    # post-training feature selection result
    selected_train = selector.transform(X_train_np)
    selected_val = selector.transform(X_val_np)

    # This is only a consistent evaluation entry for Optuna; actual downstream
    # evaluation is performed in `with_evaluation`. We therefore use the
    # reconstruction-style selector's ranking stability here.
    feature_indices, feature_scores, selected_indices, selected_mask, selection_probabilities = _extract_selector_outputs(
        selector,
        reduction=training_params.get("score_reduction", "max")
    )

    with torch.no_grad():
        model_device = selector.model.decoder.net[0].weight.device
        X_val_t = torch.tensor(X_val_np, dtype=torch.float32, device=model_device)
        pred_val = selector.model(X_val_t).detach().cpu().numpy()

    if selector_mode == "unsupervised":
        recon_mse = mean_squared_error(X_val_np.reshape(-1), pred_val.reshape(-1))
        val_metrics = {"loss": float(recon_mse), "recon_mse": float(recon_mse)}
        score = float(val_metrics["loss"])
    else:
        val_metrics = _compute_metrics(y_val_np, pred_val, model_params["task_type"])
        score = float(_score_from_metrics(val_metrics, eval_metric, model_params["task_type"]))

    # np.savez(
    #     os.path.join(folder_path, "CAE_fold_result.npz"),
    #     feature_indices=feature_indices,
    #     feature_scores=feature_scores,
    #     selected_indices=selected_indices,
    #     selected_mask=selected_mask,
    #     selection_probabilities=selection_probabilities,
    # )

    return {
        "model": selector,
        "feature_indices": feature_indices,
        "feature_scores": feature_scores,
        "selected_indices": selected_indices,
        "selected_mask": selected_mask,
        "selection_probabilities": selection_probabilities,
        "train_metrics": [],
        "val_metrics": [val_metrics],
        "train_losses": [],
        "val_losses": [val_metrics.get("loss", np.nan)],
        "score": float(score),
        "elapsed": elapsed,
        "X_train_selected": selected_train,
        "X_val_selected": selected_val,
    }


# ============================================================
# final train
# ============================================================
def CAE_model_train(
    data, label, name, model_params, training_params, folder_name, logger,
    save_outputs=True,
):
    total_start_time = time.time()
    if save_outputs:
        os.makedirs(folder_name, exist_ok=True)

    X = data
    y = label

    task_type = model_params["task_type"]
    validation_split = training_params.get("validation_split", 0.2)

    if validation_split > 0:
        n_samples = X.shape[0]
        all_indices = np.arange(n_samples)
        stratify_labels = None
        if model_params.get("task_type", "regression") in ("binary", "multiclass"):
            stratify_labels = _flatten_y_if_needed(y)
        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=validation_split,
            random_state=training_params.get("seed", 42),
            stratify=stratify_labels
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

    X_train_np = _to_numpy(X_train).astype(np.float32)
    X_val_np = _to_numpy(X_val).astype(np.float32)
    y_train_np = _flatten_y_if_needed(y_train)
    y_val_np = _flatten_y_if_needed(y_val)
    selector_mode = _normalize_selector_mode(
        training_params.get("selector_mode", model_params.get("selector_mode", "unsupervised"))
    )
    logger.info(f"CAE selector_mode: {selector_mode}")
    if selector_mode == "unsupervised":
        logger.info("CAE final training target: reconstruction X -> X")
    else:
        logger.info(f"CAE final training target: supervised X -> y ({model_params.get('task_type', 'regression')})")

    training_params["logger"] = logger
    logger.info(f"CAE_model_train: X_train={X_train_np.shape}, X_val={X_val_np.shape}")

    selector = _build_selector(X_train_np, model_params, training_params)

    train_start_time = time.time()
    if selector_mode == "unsupervised":
        selector.fit(X_train_np, X_train_np, X_val_np, X_val_np)
    else:
        selector.fit(X_train_np, y_train_np, X_val_np, y_val_np)
    train_time = time.time() - train_start_time

    feature_indices, feature_scores, selected_indices, selected_mask, selection_probabilities = _extract_selector_outputs(
        selector,
        reduction=training_params.get("score_reduction", "max")
    )

    if save_outputs:
        np.save(
            os.path.join(folder_name, f'CAE_{name}_results.npy'),
            selected_indices,
        )
        np.save(
            os.path.join(folder_name, f'CAE_{name}_weights.npy'),
            feature_scores,
        )

    return {
        "model": selector,
        "feature_indices": feature_indices,
        "feature_scores": feature_scores,
        "selected_indices": selected_indices,
        "selected_mask": selected_mask,
        "selection_probabilities": selection_probabilities,
        "train_metrics": [],
        "val_metrics": [],
        "train_losses": [],
        "val_losses": [],
    }


# ============================================================
# optuna search
# ============================================================
def optuna_search_CAE(
    data, label, name, model_params, training_params, folder_name, logger,
    save_final_outputs=True,
):


    total_start_time = time.time()

    X = data
    y = label
    selector_mode = _normalize_selector_mode(
        training_params.get("selector_mode", model_params.get("selector_mode", "unsupervised"))
    )
    logger.info(f"CAE selector_mode for Optuna search: {selector_mode}")

    if selector_mode == "unsupervised":
        logger.info("CAE Optuna search training target: reconstruction X -> X")
    else:
        logger.info(
            f"CAE Optuna search training target: supervised X -> y ({model_params.get('task_type', 'regression')})"
        )

    requested_eval_metric = training_params.get("eval_metric", None)
    use_cv = training_params.get("use_cv", True)

    if selector_mode == "unsupervised":
        if requested_eval_metric is not None and requested_eval_metric != "loss":
            logger.info(
                f"Requested eval_metric='{requested_eval_metric}' for CAE, "
                f"but unsupervised CAE always uses reconstruction loss for hyperparameter search. "
                f"Override eval_metric -> 'loss'."
            )
        eval_metric = "loss"
        minimize_metric = True
        logger.info(f"Parameter search will use '{eval_metric}' as the evaluation metric.")
        logger.info(f"During optimization, the metric will be minimized.")
    else:
        eval_metric = requested_eval_metric or "loss"
        minimize_metric = eval_metric == "loss"
        logger.info(f"Parameter search will use '{eval_metric}' as the evaluation metric.")
        logger.info(
            "During optimization, the metric will be minimized." if minimize_metric
            else "During optimization, the metric will be maximized."
        )
    logger.info(f"Hyperparameter search use_cv={use_cv}")

    weight_decay_min = training_params.get("weight_decay_min", 0.0)
    weight_decay_max = training_params.get("weight_decay_max", 1e-1)
    lr_min = training_params.get("lr_min", 1e-5)
    lr_max = training_params.get("lr_max", 1e-1)
    dropout_min = training_params.get("dropout_min", 0.1)
    dropout_max = training_params.get("dropout_max", 0.5)

    start_temp_min = training_params.get("start_temp_min", 5.0)
    start_temp_max = training_params.get("start_temp_max", 20.0)
    min_temp_min = training_params.get("min_temp_min", 0.01)
    min_temp_max = training_params.get("min_temp_max", 1.0)

    start_digit_choices = parse_list_arg(training_params.get("digits_list", "50,100,200"), dtype=int)
    shrink_ratio_choices = parse_list_arg(training_params.get("shrink_ratio_list", "1.25,1.5,2.0"), dtype=float)
    batch_size_choices = parse_list_arg(training_params.get("batch_size_list", "16,32,64,128"), dtype=int)
    min_layers = max(2, int(training_params.get("min_layers", 2)))
    max_layers = max(min_layers, int(training_params.get("max_layers", 5)))

    optuna_folder = os.path.join(folder_name, "optuna_search")
    os.makedirs(optuna_folder, exist_ok=True)

    n_trials = training_params.get("n_trials", 20)
    n_jobs = training_params.get("n_jobs", 1)

    db_file = os.path.join(optuna_folder, "optuna_CAE.db")
    db_url = f"sqlite:///{db_file}"

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

    direction = "minimize" if minimize_metric else "maximize"

    study = optuna.create_study(
        sampler=sampler,
        direction=direction,
        storage=storage,
        study_name="CAE_shrink_ratio_v3",
        load_if_exists=True
    )

    k_folds = training_params.get("n_splits", 3)
    val_ratio = training_params.get("validation_split", 0.2)

    if use_cv:
        if model_params.get("task_type", "regression") in ("binary", "multiclass"):
            splitter = StratifiedKFold(
                n_splits=k_folds,
                shuffle=True,
                random_state=training_params["seed"]
            )
            logger.info(f"Using {k_folds}-fold stratified cross-validation during Optuna search.")
        else:
            splitter = KFold(
                n_splits=k_folds,
                shuffle=True,
                random_state=training_params["seed"]
            )
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
        start_temp = trial.suggest_float("start_temp", start_temp_min, start_temp_max)
        min_temp = trial.suggest_float("min_temp", min_temp_min, min_temp_max, log=True)
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
            f"start_temp={start_temp}, min_temp={min_temp}, batch_size={batch_size}, "
            f"n_layers={n_layers}, start_digit={start_digit}, shrink_ratio={shrink_ratio}, "
            f"generated_digits_list={digits_list}, generated_hidden_dims={hidden_dims}"
        )

        fold_scores = []
        fold_times = []

        if use_cv:
            if model_params.get("task_type", "regression") in ("binary", "multiclass"):
                split_iter = splitter.split(np.arange(X.shape[0]), _flatten_y_if_needed(y))
            else:
                split_iter = splitter.split(np.arange(X.shape[0]))
        else:
            all_indices = np.arange(X.shape[0])
            stratify_labels = None
            if model_params.get("task_type", "regression") in ("binary", "multiclass"):
                stratify_labels = _flatten_y_if_needed(y)
            train_idx, val_idx = train_test_split(
                all_indices,
                test_size=val_ratio,
                random_state=training_params["seed"],
                stratify=stratify_labels
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
            current_model_params["dropout_prob"] = dropout
            current_model_params["digits"] = int(start_digit)

            current_training_params = copy.deepcopy(training_params)
            current_training_params["weight_decay"] = weight_decay
            current_training_params["lr"] = lr
            current_training_params["batch_size"] = batch_size
            current_training_params["start_temp"] = start_temp
            current_training_params["min_temp"] = min_temp

            fold_result = CAE_single_fold_train(
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

    logger.info(
        f"Use {n_jobs} parallel workers to run {n_trials} Optuna optimization trials "
        f"(use_cv={use_cv})"
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    joblib.dump(
        study,
        os.path.join(optuna_folder, "optuna_study.pkl"),
    )

    best_params = study.best_params
    best_score = study.best_value

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
        final_model_params["digits"] = fallback_start_digit
    else:
        final_model_params["hidden_dims"] = model_params.get("hidden_dims", _build_hidden_dims(X.shape[1], final_model_params.get("start_digit", final_model_params.get("digits", 100)), n_layers=final_model_params.get("n_layers", 2), shrink_ratio=final_model_params.get("shrink_ratio", training_params.get("shrink_ratio", 2.0))))
        final_model_params["digits"] = int(final_model_params.get("start_digit", final_model_params.get("digits", 100)))

    final_model_params["dropout_prob"] = best_params["dropout"]

    final_training_params = copy.deepcopy(training_params)
    final_training_params["weight_decay"] = best_params["weight_decay"]
    final_training_params["lr"] = best_params["lr"]
    final_training_params["batch_size"] = best_params["batch_size"]
    final_training_params["start_temp"] = best_params["start_temp"]
    final_training_params["min_temp"] = best_params["min_temp"]
    final_model_time_start = time.time()

    results = CAE_model_train(
        X, y,
        name,
        final_model_params,
        final_training_params,
        folder_name,
        logger,
        save_outputs=save_final_outputs,
    )
    final_model_time = time.time() - final_model_time_start
    search_time = time.time() - total_start_time - final_model_time

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

    total_time = time.time() - total_start_time
    logger.info(f"Total processing time: {format_time(total_time)} ({total_time/60:.2f} minutes)")

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
    return results


# ============================================================
# public API: feature selection only
# ============================================================
def CAE_fs(data, label, model_params, training_params, name, folder_name, logger, ):
    X = data["X_train"]
    y = label["y_train"]

    selector_mode = _normalize_selector_mode(
        training_params.get("selector_mode", model_params.get("selector_mode", "unsupervised"))
    )
    model_params["selector_mode"] = selector_mode
    training_params["selector_mode"] = selector_mode
    logger.info(f"CAE selector_mode: {selector_mode}")

    do_parameter_search = training_params.get("do_parameter_search", 0)

    if do_parameter_search == 1:
        logger.info("Start Optuna hyperparameter optimization for CAE...")
        if selector_mode == "unsupervised":
            logger.info("CAE hyperparameter search uses validation reconstruction loss (unsupervised selector).")
        else:
            logger.info(f"CAE hyperparameter search uses supervised validation metrics for task_type={model_params.get('task_type', 'regression')}.")
        results = optuna_search_CAE(X, y, name, model_params, training_params, folder_name, logger)
        if isinstance(results, dict) and results.get("best_architecture") is not None:
            best_architecture = results["best_architecture"]
            logger.info(
                f"best CAE architecture from Optuna: n_layers={best_architecture.get('n_layers')}, "
                f"start_digit={best_architecture.get('start_digit')}, shrink_ratio={best_architecture.get('shrink_ratio')}, "
                f"digits_list={best_architecture.get('digits_list')}, hidden_dims={best_architecture.get('hidden_dims')}"
            )
    else:
        logger.info("Parameter optimization is disabled. Using single parameter values.")
        logger.info("Using parameters:")
        logger.info(f"  start_temp: {training_params['start_temp']}")
        logger.info(f"  min_temp: {training_params['min_temp']}")
        logger.info(f"  tryout_limit: {training_params['tryout_limit']}")
        logger.info(f"  K_select: {training_params['K_select']}")
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

        results = CAE_model_train(X, y, name, model_params, training_params, folder_name, logger)

    return results


# ============================================================
# public API: feature selection + evaluation
# ============================================================
def CAE_fs_with_evaluation(
    data, label,
    model_params, training_params,
    name, digits, folder_name, logger,
    feature_prediction,
    n_iters=20, feature_step=500, n_folds=3
):
    total_start_time = time.time()
    logger.info(f" CAE_fs_with_evaluation start {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    x_train = data['X_train']
    x_test = data['X_test']
    y_train = label['y_train']
    y_test = label['y_test']

    device = x_train.device if isinstance(x_train, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")
    selector_mode = _normalize_selector_mode(
        training_params.get("selector_mode", model_params.get("selector_mode", "unsupervised"))
    )
    model_params["selector_mode"] = selector_mode
    training_params["selector_mode"] = selector_mode
    logger.info(f"CAE selector_mode: {selector_mode}")

    total_features = x_train.shape[1]
    fixed_k_select = int(training_params['K_select'])
    if fixed_k_select <= 0:
        raise ValueError(f"K_select must be positive, got {fixed_k_select}")
    if fixed_k_select > total_features:
        raise ValueError(f"K_select={fixed_k_select} exceeds total number of features={total_features}")
    max_features = fixed_k_select

    eval_folder = os.path.join(folder_name, 'evaluation')
    os.makedirs(eval_folder, exist_ok=True)
    checkpoint_folder = os.path.join(eval_folder, 'checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)
    iter_folder = os.path.join(eval_folder, 'iterations')
    os.makedirs(iter_folder, exist_ok=True)
    feature_folder = os.path.join(eval_folder, 'feature_evaluations')
    os.makedirs(feature_folder, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_folder, 'checkpoint.npz')

    feature_sequence = np.array([fixed_k_select], dtype=np.int64)
    logger.info(f"evaluate exactly K_select={fixed_k_select} hard-selected features per iteration (no progressive top-k evaluation)")
    n_steps = len(feature_sequence)

    start_iter = 0
    results = np.zeros((n_steps, n_iters))
    cv_val_losses = np.zeros((n_steps, n_iters))
    results_weights = []
    results_indices = []
    all_best_params = []
    all_best_architectures = []
    all_report_metrics = [[None for _ in range(n_iters)] for _ in range(n_steps)]
    seeds = [int(s) for s in np.random.choice(range(1000), n_iters, replace=False)]

    logger.info(f"total features: {total_features}")
    logger.info(f"evaluation settings:")
    logger.info(f"- K_select to evaluate directly: {fixed_k_select}")
    logger.info(f"- feature sequence: {feature_sequence}")
    logger.info(f"- number of evaluation points: {n_steps}")
    logger.info(f"- number of cross-validation folds: {n_folds}")

    dropout_probs = [0.1, 0.3, 0.7, 0.9]
    l2_lambdas = [0.0001, 0.001, 0.01, 0.1]
    logger.info(f"using device: {device}")

    for iter in range(start_iter, n_iters):
        iter_start_time = time.time()
        logger.info(f"start iteration {iter+1}/{n_iters}, seed {int(seeds[iter])}")

        try:
            current_model_params = model_params.copy()
            current_model_params['selector_mode'] = selector_mode
            current_training_params = training_params.copy()
            current_training_params['selector_mode'] = selector_mode

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
            current_model_params['digits'] = int(digits)

            current_training_params['device'] = device
            current_training_params['seed'] = int(seeds[iter])
            if selector_mode == "unsupervised":
                logger.info("CAE selector training remains unsupervised in evaluation mode; y is only used later for downstream evaluation.")
            else:
                logger.info(f"CAE selector training uses supervised targets in evaluation mode (task_type={current_model_params.get('task_type', 'regression')}).")

            if current_training_params.get('do_parameter_search', False):
                logger.info("using Optuna for hyperparameter optimization and training...")
                logger.info(
                    f"current fixed CAE architecture before Optuna: n_layers={current_model_params['n_layers']}, "
                    f"start_digit={current_model_params.get('start_digit')}, shrink_ratio={current_model_params.get('shrink_ratio')}, "
                    f"generated_digits_list={current_model_params.get('digits_list')}, hidden_dims={current_model_params.get('hidden_dims')}"
                )
                cat_results = optuna_search_CAE(
                    x_train, y_train, name,
                    current_model_params,
                    current_training_params,
                    os.path.join(eval_folder, f'optuna_iter_{iter}'),
                    logger,
                    save_final_outputs=False,
                )

                best_params = cat_results.get('best_params', None) if isinstance(cat_results, dict) else None
                best_architecture = cat_results.get('best_architecture', None) if isinstance(cat_results, dict) else None
                if best_params is not None:
                    all_best_params.append(best_params)
                    logger.info(f"get best params: {best_params}")
                    if best_architecture is not None:
                        all_best_architectures.append(best_architecture)
                        logger.info(
                            f"best CAE architecture from Optuna: n_layers={best_architecture.get('n_layers')}, "
                            f"start_digit={best_architecture.get('start_digit')}, shrink_ratio={best_architecture.get('shrink_ratio')}, "
                            f"digits_list={best_architecture.get('digits_list')}, hidden_dims={best_architecture.get('hidden_dims')}"
                        )
                    if 'dropout' in best_params:
                        current_model_params['dropout_prob'] = best_params['dropout']
                    if 'n_layers' in best_params:
                        best_n_layers = max(2, int(best_params['n_layers']))
                        best_start_digit = int(best_params.get('start_digit', current_model_params.get('start_digit', current_model_params.get('digits', 100))))
                        best_shrink_ratio = float(best_params.get('shrink_ratio', current_model_params.get('shrink_ratio', current_training_params.get('shrink_ratio', 2.0))))
                        best_digits_list = [
                            max(1, int(round(best_start_digit * (best_shrink_ratio ** i))))
                            for i in range(best_n_layers)
                        ]
                        current_model_params['n_layers'] = best_n_layers
                        current_model_params['start_digit'] = best_start_digit
                        current_model_params['shrink_ratio'] = best_shrink_ratio
                        current_model_params['digits_list'] = best_digits_list
                        current_model_params['hidden_dims'] = _build_hidden_dims_from_digits_list(
                            x_train.shape[1], best_digits_list
                        )
                        current_model_params['digits'] = int(best_start_digit)
                    elif 'digits' in best_params:
                        fallback_start_digit = int(best_params['digits'])
                        fallback_shrink_ratio = float(current_model_params.get('shrink_ratio', current_training_params.get('shrink_ratio', 2.0)))
                        fallback_n_layers = max(2, int(current_model_params.get('n_layers', 2)))
                        best_digits_list = [
                            max(1, int(round(fallback_start_digit * (fallback_shrink_ratio ** i))))
                            for i in range(fallback_n_layers)
                        ]
                        current_model_params['start_digit'] = fallback_start_digit
                        current_model_params['shrink_ratio'] = fallback_shrink_ratio
                        current_model_params['digits_list'] = best_digits_list
                        current_model_params['hidden_dims'] = _build_hidden_dims_from_digits_list(
                            x_train.shape[1], best_digits_list
                        )
                        current_model_params['digits'] = fallback_start_digit
                    if 'weight_decay' in best_params:
                        current_training_params['weight_decay'] = best_params['weight_decay']
                    if 'lr' in best_params:
                        current_training_params['lr'] = best_params['lr']
                    if 'batch_size' in best_params:
                        current_training_params['batch_size'] = best_params['batch_size']
                    if 'start_temp' in best_params:
                        current_training_params['start_temp'] = best_params['start_temp']
                    if 'min_temp' in best_params:
                        current_training_params['min_temp'] = best_params['min_temp']

            else:
                logger.info("using fixed parameters to train model...")
                logger.info(
                    f"current fixed CAE architecture: n_layers={current_model_params['n_layers']}, "
                    f"start_digit={current_model_params.get('start_digit')}, shrink_ratio={current_model_params.get('shrink_ratio')}, "
                    f"generated_digits_list={current_model_params.get('digits_list')}, hidden_dims={current_model_params.get('hidden_dims')}"
                )
                cat_results = CAE_model_train(
                    x_train, y_train, name,
                    current_model_params,
                    current_training_params,
                    os.path.join(eval_folder, f'model_iter_{iter}'),
                    logger,
                    save_outputs=False,
                )

            feature_ranking = np.asarray(cat_results['feature_indices'])
            feature_weights = np.asarray(cat_results['feature_scores'])
            raw_selected_indices = np.asarray(cat_results['selected_indices'], dtype=np.int64)
            selected_indices = np.asarray(list(dict.fromkeys(raw_selected_indices.tolist())), dtype=np.int64)
            selected_mask = np.asarray(cat_results['selected_mask'])
            selection_probabilities = np.asarray(cat_results['selection_probabilities'])

            if len(selected_indices) < len(raw_selected_indices):
                logger.info(
                    f"Deduplicated hard-selected indices from {len(raw_selected_indices)} slots to {len(selected_indices)} unique features"
                )

            results_weights.append(feature_weights)
            results_indices.append(selected_indices)

            feature_ranking_file = os.path.join(iter_folder, f'feature_ranking_iter_{iter}.npz')
            np.savez(
                feature_ranking_file,
                feature_indices=feature_ranking,
                feature_scores=feature_weights,
                selected_indices=selected_indices,
                raw_selected_indices=raw_selected_indices,
                selected_mask=selected_mask,
                selection_probabilities=selection_probabilities,
                iter=iter,
                seed=seeds[iter]
            )
            logger.info(f"feature ranking saved to: {feature_ranking_file}")

            feature_steps_range = range(n_steps)

            for step in feature_steps_range:
                num_features = int(feature_sequence[step])
                logger.info(f"\nevaluating exactly {num_features} hard-selected features")

                try:
                    selected_features = np.asarray(selected_indices, dtype=np.int64)
                    if len(selected_features) != num_features:
                        logger.warning(
                            f"unique selected_indices length ({len(selected_features)}) does not match K_select ({num_features}); using unique selected_indices length for evaluation"
                        )
                        num_features = len(selected_features)
                    logger.info(f"hard-selected unique feature indices (first few): {selected_features[:10]}...")

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
                    )

                    logger.info(f"The evaluation results for the number of features {num_features} have been saved to: {feature_eval_file}")

                    checkpoint_data = {
                        'last_feature_step': step,
                        'current_iter': iter,
                        'current_feature': num_features,
                        'results': results,
                        'cv_val_losses': cv_val_losses,
                        'results_weights': np.stack(results_weights, axis=0) if len(results_weights) > 0 else np.empty((0, total_features)),
                        'results_indices': np.array(results_indices, dtype=object) if len(results_indices) > 0 else np.empty((0,), dtype=object),
                        'seeds': np.array(seeds, dtype=np.int64),
                        'feature_sequence': feature_sequence,
                    }
                    np.savez(checkpoint_file, **checkpoint_data)


                    logger.info(f"Checkpoint updated, completed evaluation for the number of features {num_features} ({step+1}/{n_steps}), current iteration {iter+1}/{n_iters}")

                except Exception as e:
                    logger.error(f"Error occurred while evaluating the number of features {num_features}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

            iter_results_file = os.path.join(iter_folder, f'full_results_iter_{iter}.npz')
            np.savez(
                iter_results_file,
                iter=iter,
                test_results=results[:, iter],
                cv_val_losses=cv_val_losses[:, iter],
                feature_indices=feature_ranking,
                feature_scores=feature_weights,
                selected_indices=selected_indices,
                raw_selected_indices=raw_selected_indices,
                selected_mask=selected_mask,
                selection_probabilities=selection_probabilities,
                feature_sequence=feature_sequence,
                seed=seeds[iter],
            )

            logger.info(f"Iteration {iter+1} full results saved to: {iter_results_file}")

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

    weights_array = np.stack(results_weights, axis=0) if len(results_weights) > 0 else np.empty((0, total_features))
    indices_array = np.array(results_indices, dtype=object) if len(results_indices) > 0 else np.empty((0,), dtype=object)

    final_results = {
        'test_results': results,
        'cv_val_losses': cv_val_losses,
        'weights': weights_array,
        'indices': indices_array,
        'mean_test_results': mean_test_results,
        'std_test_results': std_test_results,
        'mean_cv_val_losses': mean_cv_val_losses,
        'std_cv_val_losses': std_cv_val_losses,
        'feature_sequence': feature_sequence,
        'total_features': np.array(total_features, dtype=np.int64),
        'max_features': np.array(max_features, dtype=np.int64),
        'n_steps': np.array(n_steps, dtype=np.int64),
        'seeds': np.array(seeds, dtype=np.int64),
    }

    save_path = os.path.join(folder_name, f'CAE_{name}_results.npz')
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

    return {
        'test_results': results,
        'cv_val_losses': cv_val_losses,
        'weights': weights_array,
        'indices': indices_array,
        'mean_test_results': mean_test_results,
        'std_test_results': std_test_results,
        'mean_cv_val_losses': mean_cv_val_losses,
        'std_cv_val_losses': std_cv_val_losses,
        'feature_sequence': feature_sequence,
        'total_features': total_features,
        'max_features': max_features,
        'n_steps': n_steps,
        'seeds': seeds,
    }