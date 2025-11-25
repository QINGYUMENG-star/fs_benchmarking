# evaluation_utils.py
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_recall_fscore_support,
    r2_score, mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr


# ------------------------------
# Model definition for evaluation
# ------------------------------
class EvaluationNet(nn.Module):
    """
    一个轻量的评估网络：两层隐藏层 + 可选多分类/二分类/回归的输出头。
    - task_type: 'binary' | 'multiclass' | 'regression'
    - num_classes: 多分类类别数（>=2）
    - dropout_prob: dropout 概率
    """
    def __init__(self, input_size, task_type, num_classes=None, dropout_prob=0.2):
        super().__init__()
        self.task_type = task_type
        hidden = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        if task_type == 'multiclass':
            assert num_classes is not None and num_classes >= 2, \
                "For 'multiclass', num_classes must be provided and >= 2."
            head = nn.Linear(32, num_classes)  # 输出 logits
        else:
            head = nn.Linear(32, 1)            # 二分类/回归输出 1 维

        self.network = nn.Sequential(hidden, head)

    def forward(self, x):
        return self.network(x)  # 直接输出 logits/数值，激活在外部按需求处理


def _get_criterion(task_type: str):
    """
    根据任务类型选择损失函数：
    - binary: BCEWithLogitsLoss
    - multiclass: CrossEntropyLoss
    - regression: MSELoss
    """
    if task_type == 'binary':
        return nn.BCEWithLogitsLoss()
    elif task_type == 'multiclass':
        return nn.CrossEntropyLoss()
    else:
        return nn.MSELoss()


# ------------------------------------------------
# Cross-validation fold training: return best val_loss
# ------------------------------------------------
def train_fold_return_best_valloss(
    X_train, y_train, X_val, y_val,
    max_features, model_params, dropout_prob, l2_lambda, device,
    max_epochs: int = 100, patience: int = 20, batch_size: int = 32, lr: float = 1e-3
) -> float:
    """
    仅用于调参/交叉验证阶段：训练并返回该参数组合下的“最小验证损失”(val_loss, 越小越好)。
    注意：该阶段**不**计算 AUC/F1/R 等报告指标。
    """
    task_type   = model_params['task_type']
    num_classes = model_params.get('num_classes', None)

    net = EvaluationNet(max_features, task_type, num_classes, dropout_prob).to(device)
    criterion = _get_criterion(task_type)
    optimz    = optim.Adam(net.parameters(), lr=0.001, weight_decay=l2_lambda)

    train_ds = TensorDataset(X_train, y_train)
    train_ld = DataLoader(train_ds, batch_size=8, shuffle=True)

    best_val_loss = float('inf')
    best_state = copy.deepcopy(net.state_dict())
    no_improve = 0

    for _ in range(max_epochs):
        net.train()
        for bx, by in train_ld:
            optimz.zero_grad()
            if task_type == 'multiclass':
                by = by.view(-1).long()
            else:
                by = by.view(-1, 1).float()
            logits = net(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimz.step()

        # 验证 loss
        net.eval()
        with torch.no_grad():
            val_logits = net(X_val)
            if task_type == 'multiclass':
                yv = y_val.view(-1).long()
            else:
                yv = y_val.view(-1, 1).float()
            val_loss = criterion(val_logits, yv).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(net.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # 返回该参数组合的最小 val_loss（用于调参比较）
    return best_val_loss


# -----------------------------------------
# Metrics for final reporting (test on final)
# -----------------------------------------

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import torch
from scipy.stats import pearsonr

def compute_report_metrics(
    task_type: str,
    y_true: np.ndarray,
    logits_or_preds: np.ndarray,
    num_classes: int = None,
    include_raw: bool = False,            # ⭐ 新增：是否把原始 y/预测存进 metrics
    proba_from: str = "logits",           # "logits" 或 "probs"，用于 multiclass/binary
):
    """
    最终模型在测试集上的报告指标。
    - binary：AUC、Accuracy、F1、Precision、Recall
    - multiclass：Accuracy、F1-macro、F1-weighted、每类 P/R/F1、每类 AUROC（含macro/weighted）
    - regression：MSE、MAE、R²、Pearson r

    参数
    ----
    task_type: 'binary' | 'multiclass' | 'regression'
    y_true:  一维 numpy 数组
    logits_or_preds: 
        - binary: (N,) 或 (N,1) 的 logits（若 proba_from='logits'）或概率（proba_from='probs'）
        - multiclass: (N,C) 的 logits（或概率，取决于 proba_from）
        - regression: (N,) 或 (N,1) 的预测值
    num_classes: 多分类类别数
    include_raw: 是否把原始 y/预测放进 metrics，便于你后续自算其他指标
    proba_from:  指示 logits_or_preds 是 logits 还是 probs（影响 sigmoid/softmax 的计算）
    """
    y_true = np.asarray(y_true).ravel().astype(np.int64 if task_type in ("binary","multiclass") else np.float32)
    metrics = {}

    if task_type == 'binary':
        arr = np.asarray(logits_or_preds).reshape(-1)
        if proba_from == "logits":
            prob = 1.0 / (1.0 + np.exp(-arr))
        else:
            prob = np.clip(arr, 1e-8, 1-1e-8)
        pred = (prob >= 0.5).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
        metrics.update({
            'auc': float(roc_auc_score(y_true, prob)),
            'accuracy': float(accuracy_score(y_true, pred)),
            'f1': float(f1),
            'precision': float(p),
            'recall': float(r),
        })
        if include_raw:
            metrics['raw'] = {
                'y_true': y_true.copy(),
                'probs': prob.copy(),
                'pred': pred.copy(),
                'logits': arr.copy() if proba_from == "logits" else None,
            }

    elif task_type == 'multiclass':
        assert num_classes is not None and num_classes >= 2
        logits_or_probs = np.asarray(logits_or_preds)
        if proba_from == "logits":
            # softmax
            with torch.no_grad():
                probs = torch.softmax(torch.from_numpy(logits_or_probs), dim=1).numpy()
            logits = logits_or_probs
        else:
            probs = np.clip(logits_or_probs, 1e-8, 1-1e-8)
            logits = None

        pred = probs.argmax(axis=1)

        metrics['accuracy']     = float(accuracy_score(y_true, pred))
        metrics['f1_macro']     = float(f1_score(y_true, pred, average='macro', zero_division=0))
        metrics['f1_weighted']  = float(f1_score(y_true, pred, average='weighted', zero_division=0))

        p_c, r_c, f1_c, _ = precision_recall_fscore_support(
            y_true, pred, average=None, labels=list(range(num_classes)), zero_division=0
        )
        per_class = {
            'precision': p_c.astype(float).tolist(),
            'recall':    r_c.astype(float).tolist(),
            'f1':        f1_c.astype(float).tolist(),
        }

        # per-class AUROC (one-vs-rest)
        try:
            y_true_ovr = np.eye(num_classes)[y_true.astype(int)]
            auc_per_class = []
            for c in range(num_classes):
                auc_c = roc_auc_score(y_true_ovr[:, c], probs[:, c])
                auc_per_class.append(float(auc_c))
            per_class['auc'] = auc_per_class
            metrics['auc_macro']    = float(np.mean(auc_per_class))
            supports = np.bincount(y_true.astype(int), minlength=num_classes)
            w = supports / (supports.sum() + 1e-8)
            metrics['auc_weighted'] = float((w * np.array(auc_per_class)).sum())
        except Exception:
            # 类别太偏或缺失时 AUROC 可能报错，忽略
            pass

        metrics['per_class'] = per_class

        if include_raw:
            metrics['raw'] = {
                'y_true': y_true.copy(),
                'probs': probs.copy(),
                'pred': pred.copy(),
                'logits': logits.copy() if logits is not None else None,
            }

    else:  # regression
        y_pred = np.asarray(logits_or_preds).reshape(-1)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        r   = pearsonr(y_true, y_pred)[0]
        metrics.update({
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'pearson_r': float(r if np.isfinite(r) else np.nan),
        })
        if include_raw:
            metrics['raw'] = {
                'y_true': y_true.copy(),
                'y_pred': y_pred.copy(),
            }

    return metrics

# ------------------------------------------------------------
# Main API: tune by val_loss, then train-final & report metrics
# ------------------------------------------------------------
def evaluate_feature_set(
    x_train, y_train, x_test, y_test, max_features,
    model_params, dropout_probs, l2_lambdas, n_folds, device
):
    """
    - 网格搜索阶段：仅使用 CV 的最小 val_loss 选择超参（越小越好）。
    - 选出最佳 (dropout, l2) 后：用全训练集重新训练一次（内部 10% 验证做早停），
      然后在测试集上一次性计算报告指标（AUC/Acc/F1/R 等），并返回。
    返回：
        final_score: 主要报告指标（binary=AUC；multiclass=F1-macro；regression=Pearson r）
        best_dropout, best_l2: 选出的最佳超参
        best_cv_val_loss: 调参阶段的最优平均验证损失
        report_metrics: 完整报告指标字典
    """
    # —— numpy 化 —— #
    if isinstance(x_train, torch.Tensor): x_train = x_train.cpu().numpy()
    if isinstance(x_test,  torch.Tensor): x_test  = x_test.cpu().numpy()
    if isinstance(y_train, torch.Tensor): y_train = y_train.cpu().numpy()
    if isinstance(y_test,  torch.Tensor): y_test  = y_test.cpu().numpy()

    task_type   = model_params['task_type']
    num_classes = model_params.get('num_classes', None)

    # —— 拆分器（分类分层） —— #
    if task_type in ('binary', 'multiclass'):
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True)
        y_for_split = y_train.ravel()
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True)
        y_for_split = y_train.ravel()

    # —— 全量张量 —— #
    X_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    X_test_t  = torch.as_tensor(x_test,  dtype=torch.float32, device=device)

    if task_type == 'multiclass':
        y_train_t = torch.as_tensor(y_train.ravel(), dtype=torch.long,   device=device)
        y_test_t  = torch.as_tensor(y_test.ravel(),  dtype=torch.long,   device=device)
    else:
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
        y_test_t  = torch.as_tensor(y_test,  dtype=torch.float32, device=device)
        if y_train_t.ndim == 1: y_train_t = y_train_t.view(-1, 1)
        if y_test_t.ndim  == 1: y_test_t  = y_test_t.view(-1, 1)

    # ---------- 调参：仅用最小 val_loss ---------- #
    best_cfg = None
    best_cv_val_loss = float('inf')

    for dp in dropout_probs:
        for l2 in l2_lambdas:
            fold_losses = []
            for tr_idx, va_idx in splitter.split(x_train, y_for_split):
                X_tr, X_va = X_train_t[tr_idx], X_train_t[va_idx]
                y_tr, y_va = y_train_t[tr_idx], y_train_t[va_idx]

                valloss = train_fold_return_best_valloss(
                    X_tr, y_tr, X_va, y_va,
                    max_features, model_params,
                    dp, l2, device
                )
                fold_losses.append(valloss)
            mean_valloss = float(np.mean(fold_losses))
            if mean_valloss < best_cv_val_loss:
                best_cv_val_loss = mean_valloss
                best_cfg = (dp, l2)

    best_dropout, best_l2 = best_cfg

    # ---------- 最终模型：内部 10% 验证做早停，训好后在测试集算报告指标 ---------- #
    net = EvaluationNet(max_features, task_type, num_classes, best_dropout).to(device)
    criterion = _get_criterion(task_type)
    optimz    = optim.Adam(net.parameters(), lr=1e-3, weight_decay=best_l2)

    # 内部 90/10 切分
    n = X_train_t.size(0)
    n_val = max(1, int(0.2 * n))
    perm = torch.randperm(n, device=device)
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    X_tr, y_tr = X_train_t[tr_idx], y_train_t[tr_idx]
    X_va, y_va = X_train_t[va_idx], y_train_t[va_idx]

    train_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=8, shuffle=True)
    best_state = copy.deepcopy(net.state_dict())
    no_improve, patience = 0, 20
    best_val_loss_final = float('inf')

    for _ in range(100):
        net.train()
        for bx, by in train_ld:
            optimz.zero_grad()
            if task_type == 'multiclass':
                by = by.view(-1).long()
            else:
                by = by.view(-1, 1).float()
            logits = net(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimz.step()

        # 内部验证
        net.eval()
        with torch.no_grad():
            logits_va = net(X_va)
            if task_type == 'multiclass':
                vy = y_va.view(-1).long()
            else:
                vy = y_va.view(-1, 1).float()
            val_loss = criterion(logits_va, vy).item()

        if val_loss < best_val_loss_final:
            best_val_loss_final = val_loss
            best_state = copy.deepcopy(net.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    net.load_state_dict(best_state)

    # 测试集一次性计算报告指标
    net.eval()
    with torch.no_grad():
        logits_test = net(X_test_t).detach().cpu().numpy()
    y_test_np = y_test_t.detach().cpu().numpy().ravel()

    report_metrics = compute_report_metrics(
        task_type=model_params['task_type'],
        y_true=y_test_np,
        logits_or_preds=logits_test,
        num_classes=model_params.get('num_classes'),
        include_raw=True,         # ⭐ 最终报告保存原始数组
        proba_from="logits"       # 如果你传的是 logits
    )

    # 选择一个“主指标”作为 final_score
    if task_type == 'binary':
        final_score = report_metrics.get('auc', None)
    elif task_type == 'multiclass':
        final_score = report_metrics.get('auc_macro', None)
    else:
        final_score = report_metrics.get('pearson_r', None)

    return final_score, best_dropout, best_l2, best_cv_val_loss, report_metrics