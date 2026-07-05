"""
train.py
--------
Training loop for fatigue prediction models.

Handles:
  - Weighted BCEWithLogitsLoss for class imbalance (pos_weight ~22.9x)
  - Early stopping on validation F1
  - Model checkpointing (best val F1)
  - Per-epoch metric logging (loss, F1, precision, recall, AUC-ROC)
  - 5-fold cross-validation mode
  - Attention weight export for interpretability (RQ1)
  - Baseline model training (BiLSTMBaseline, RandomForest, ARIMA, MovingAverage)

Usage examples
--------------
# Train attention model on full train/val split:
    python train.py --csv data/features_labeled.csv --model attention

# Run 5-fold CV:
    python train.py --csv data/features_labeled.csv --model attention --cv

# Train all models for comparison (RQ3):
    python train.py --csv data/features_labeled.csv --model all
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore", category=UserWarning)

# Local modules
from dataset import (
    FatigueDataset, build_dataloaders, kfold_player_splits,
    apply_normalization, compute_normalization_stats, collate_fn,
    player_wise_split,
    FEATURE_COLS, LABEL_COL, PLAYER_COL, MATCH_COL, WINDOW_COL,
)
from model import build_model, AttentionBiLSTM, BiLSTMBaseline

import pandas as pd


# ---------------------------------------------------------------------------
# Config dataclass (plain dict for simplicity + JSON serialisability)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Data
    "csv_path":          "data/features_labeled.csv",
    "batch_size":        32,
    "mode":              "sequence",
    "num_workers":       0,

    # Model
    "model_type":        "attention",   # "attention" | "bilstm" | "rf" | "arima" | "ma" | "all"
    "hidden_dim":        128,
    "num_layers":        2,
    "dropout":           0.3,

    # Training
    "epochs":            60,
    "lr":                1e-3,
    "weight_decay":      1e-4,
    "clip_grad_norm":    1.0,
    "scheduler":         "cosine",      # "cosine" | "plateau" | "none"
    "warmup_epochs":     3,

    # Early stopping
    "patience":          10,
    "min_delta":         1e-4,

    # Threshold (applied to sigmoid output for binary classification)
    "decision_threshold": 0.35,         # lower than 0.5 because class is rare

    # Output
    "output_dir":        "experiments",
    "save_attention":    True,          # export attention weights after training

    # CV
    "cv_folds":          5,
    "seed":              42,
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.35,
) -> Dict[str, float]:
    """
    Compute classification metrics from predicted probabilities.

    Args:
        y_true      : (N,) binary ground-truth labels
        y_pred_prob : (N,) predicted probabilities
        threshold   : decision boundary for binary classification

    Returns:
        dict of metric_name → float
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Guard against edge cases (all-zero predictions)
    n_pos_pred = y_pred.sum()
    n_pos_true = y_true.sum()

    metrics = {
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "accuracy":  accuracy_score(y_true, y_pred),
    }

    # AUC requires at least one positive example
    if n_pos_true > 0 and n_pos_true < len(y_true):
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_prob)
    else:
        metrics["auc_roc"] = float("nan")

    # Regression-style metrics on probabilities (for RQ3 comparison with ARIMA/MA)
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred_prob)))
    metrics["mae"]  = float(mean_absolute_error(y_true, y_pred_prob))

    # R² on probs (can be negative if model is worse than mean predictor)
    if np.var(y_true) > 0:
        metrics["r2"] = float(r2_score(y_true, y_pred_prob))
    else:
        metrics["r2"] = float("nan")

    metrics["n_pred_positive"] = int(n_pos_pred)
    metrics["n_true_positive"] = int(n_pos_true)
    metrics["n_total"]         = int(len(y_true))

    return metrics


# ---------------------------------------------------------------------------
# Core training / evaluation functions
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_norm: float = 1.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    One training epoch.

    Returns:
        mean_loss   : float
        all_probs   : (N,) predicted probabilities (valid positions only)
        all_labels  : (N,) ground-truth labels (valid positions only)
    """
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for batch in loader:
        features = batch["features"].to(device)   # (B, T, F)
        labels   = batch["labels"].to(device)     # (B, T) or (B,)
        lengths  = batch["lengths"].to(device)    # (B,)
        mask     = batch["mask"].to(device)       # (B, T)

        optimizer.zero_grad()
        logits, _ = model(features, lengths, mask)  # (B, T) or (B,)

        if logits.dim() == 2:
            # Sequence mode: flatten and mask out padding
            loss_raw = criterion(logits, labels.float())       # (B, T)
            valid_mask = mask & (labels != -100)
            loss = (loss_raw * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

            # Collect predictions for metrics
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            lbls  = labels.detach().cpu().numpy()
            for i in range(len(lengths)):
                L = lengths[i].item()
                all_probs.append(probs[i, :L])
                all_labels.append(lbls[i, :L])
        else:
            # Last mode: single label per sequence
            loss = criterion(logits, labels.float()).mean()
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item()

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    return total_loss / max(len(loader), 1), all_probs, all_labels


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluation pass — no gradient computation.

    Returns:
        mean_loss    : float
        all_probs    : (N,) predicted probabilities
        all_labels   : (N,) ground-truth labels
        all_attn     : (M,) flattened attention weights (for exportable inspection)
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_labels, all_attn = [], [], []

    for batch in loader:
        features = batch["features"].to(device)
        labels   = batch["labels"].to(device)
        lengths  = batch["lengths"].to(device)
        mask     = batch["mask"].to(device)

        logits, attn_weights = model(features, lengths, mask)

        if logits.dim() == 2:
            loss_raw = criterion(logits, labels.float())
            valid_mask = mask & (labels != -100)
            loss = (loss_raw * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

            probs = torch.sigmoid(logits).cpu().numpy()
            lbls  = labels.cpu().numpy()
            for i in range(len(lengths)):
                L = lengths[i].item()
                all_probs.append(probs[i, :L])
                all_labels.append(lbls[i, :L])
                all_attn.append(attn_weights[i, :L].cpu().numpy())
        else:
            loss = criterion(logits, labels.float()).mean()
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_attn.append(attn_weights.cpu().numpy())

        total_loss += loss.item()

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_attn   = np.concatenate(all_attn)

    return total_loss / max(len(loader), 1), all_probs, all_labels, all_attn


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitors a metric and signals early stopping when no improvement
    is seen for `patience` consecutive epochs.

    Saves the best model checkpoint automatically.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        checkpoint_path: str = "best_model.pt",
        mode: str = "max",   # "max" for F1, "min" for loss
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, value: float, model: nn.Module, epoch: int) -> bool:
        """
        Returns True if training should stop.
        Saves a checkpoint whenever a new best is found.
        """
        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1

        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Learning rate scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 3,
):
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6
        )
    else:
        return None


# ---------------------------------------------------------------------------
# Full training run for one deep learning model
# ---------------------------------------------------------------------------

def train_deep_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    run_dir: Path,
    device: torch.device,
    fold: Optional[int] = None,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Full training loop for AttentionBiLSTM or BiLSTMBaseline.

    Returns:
        best_metrics : dict of validation metrics at best checkpoint
        val_probs    : (N,) probabilities on validation set
        val_attn     : (N,) attention weights on validation set
    """
    tag = f"fold{fold}_" if fold is not None else ""
    checkpoint_path = str(run_dir / f"{tag}best_model.pt")

    # Loss: heavily weight the positive (fatigued) class
    pos_weight = torch.tensor(
        [train_loader.dataset.class_weights[1].item()]
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg["scheduler"], cfg["epochs"], cfg["warmup_epochs"])
    stopper   = EarlyStopping(
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        checkpoint_path=checkpoint_path,
        mode="max",
    )

    history = []
    print(f"\n{'='*58}")
    print(f"  Training {cfg['model_type'].upper()}"
          + (f"  [fold {fold}]" if fold is not None else ""))
    print(f"{'='*58}")
    print(f"  {'Epoch':>5}  {'TrLoss':>7}  {'VaLoss':>7}  "
          f"{'F1':>6}  {'Prec':>6}  {'Recall':>6}  {'AUC':>6}  {'LR':>8}")
    print(f"  {'-'*58}")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # Warmup: linearly scale LR for first warmup_epochs
        if epoch <= cfg["warmup_epochs"]:
            for g in optimizer.param_groups:
                g["lr"] = cfg["lr"] * epoch / cfg["warmup_epochs"]

        tr_loss, tr_probs, tr_labels = train_epoch(
            model, train_loader, optimizer, criterion, device, cfg["clip_grad_norm"]
        )
        va_loss, va_probs, va_labels, va_attn = eval_epoch(
            model, val_loader, criterion, device
        )

        metrics = compute_metrics(va_labels, va_probs, cfg["decision_threshold"])

        # Step scheduler
        if scheduler is not None and epoch > cfg["warmup_epochs"]:
            if cfg["scheduler"] == "plateau":
                scheduler.step(metrics["f1"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = {
            "epoch": epoch, "tr_loss": tr_loss, "va_loss": va_loss,
            "lr": current_lr, **metrics
        }
        history.append(row)

        # Print every epoch
        marker = " *" if stopper.counter == 0 else "  "
        print(f"{marker} {epoch:>5}  {tr_loss:>7.4f}  {va_loss:>7.4f}  "
              f"{metrics['f1']:>6.3f}  {metrics['precision']:>6.3f}  "
              f"{metrics['recall']:>6.3f}  {metrics['auc_roc']:>6.3f}  "
              f"{current_lr:>8.2e}")

        if stopper(metrics["f1"], model, epoch):
            print(f"\n  Early stopping at epoch {epoch}. "
                  f"Best F1={stopper.best_value:.4f} at epoch {stopper.best_epoch}.")
            break

    # Reload best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Final evaluation on validation set with best model
    _, va_probs, va_labels, va_attn = eval_epoch(model, val_loader, criterion, device)
    best_metrics = compute_metrics(va_labels, va_probs, cfg["decision_threshold"])
    best_metrics["best_epoch"] = stopper.best_epoch

    # Save training history
    hist_path = run_dir / f"{tag}history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Save attention weights if requested
    if cfg.get("save_attention") and cfg["model_type"] == "attention":
        attn_path = run_dir / f"{tag}val_attention.npy"
        np.save(attn_path, va_attn)
        print(f"  Attention weights saved → {attn_path}")

    print(f"\n  Best val metrics:")
    print(f"    F1={best_metrics['f1']:.4f}  "
          f"Prec={best_metrics['precision']:.4f}  "
          f"Recall={best_metrics['recall']:.4f}  "
          f"AUC={best_metrics['auc_roc']:.4f}")

    return best_metrics, va_probs, va_attn


# ---------------------------------------------------------------------------
# Baseline: Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    cfg: Dict,
    run_dir: Path,
) -> Dict:
    """
    Train a Random Forest on per-window features (no temporal structure).
    Features are the mean of each feature column across windows per match
    + all individual windows flattened.

    For direct comparison each window is treated independently.
    """
    from sklearn.utils.class_weight import compute_sample_weight

    feature_cols = cfg.get("feature_cols", FEATURE_COLS)

    X_train = df_train[feature_cols].values
    y_train = df_train[LABEL_COL].values
    X_val   = df_val[feature_cols].values
    y_val   = df_val[LABEL_COL].values

    sample_weights = compute_sample_weight("balanced", y_train)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=cfg["seed"],
        n_jobs=-1,
    )
    rf.fit(X_train, y_train, sample_weight=sample_weights)

    probs = rf.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, probs, cfg["decision_threshold"])

    # Feature importances
    importances = dict(zip(feature_cols, rf.feature_importances_.tolist()))
    metrics["feature_importances"] = importances

    import joblib
    joblib.dump(rf, run_dir / "rf_model.pkl")

    print(f"\n  Random Forest results:")
    print(f"    F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}  "
          f"Recall={metrics['recall']:.4f}  AUC={metrics['auc_roc']:.4f}")
    print(f"    Feature importances: {importances}")

    return metrics


# ---------------------------------------------------------------------------
# Baseline: ARIMA per-player APM time series
# ---------------------------------------------------------------------------

def train_arima(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    cfg: Dict,
) -> Dict:
    """
    ARIMA(1,1,1) baseline: forecast APM from historical windows.
    Fatigue proxy = when forecast APM drops > 1 std below player baseline.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("  statsmodels not installed — skipping ARIMA baseline.")
        print("  Install with: pip install statsmodels --break-system-packages")
        return {}

    apm_col = "apm"
    if apm_col not in df_train.columns:
        print("  'apm' column not found — skipping ARIMA.")
        return {}

    all_probs, all_labels = [], []

    for player_id, group in df_val.sort_values(WINDOW_COL).groupby(PLAYER_COL):
        apm_series = group[apm_col].values
        labels     = group[LABEL_COL].values

        if len(apm_series) < 5:
            continue

        try:
            model = ARIMA(apm_series, order=(1, 1, 1))
            fit   = model.fit()
            # One-step-ahead in-sample predictions as fatigue signal
            residuals = fit.resid
            # Normalise residuals to [0,1] as a fatigue probability proxy
            r_std = residuals.std()
            if r_std > 0:
                probs = np.clip(-residuals / (3 * r_std) + 0.5, 0, 1)
            else:
                probs = np.full(len(residuals), 0.5)

            all_probs.append(probs)
            all_labels.append(labels[:len(probs)])
        except Exception:
            continue

    if not all_probs:
        return {}

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    metrics    = compute_metrics(all_labels, all_probs, cfg["decision_threshold"])

    print(f"\n  ARIMA results:")
    print(f"    F1={metrics['f1']:.4f}  RMSE={metrics['rmse']:.4f}  "
          f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Baseline: Moving Average
# ---------------------------------------------------------------------------

def train_moving_average(
    df_val: pd.DataFrame,
    cfg: Dict,
    window: int = 3,
) -> Dict:
    """
    Moving average baseline: APM drop relative to rolling mean as fatigue proxy.
    Simple but interpretable reference point.
    """
    apm_col = "apm"
    if apm_col not in df_val.columns:
        print("  'apm' column not found — skipping Moving Average baseline.")
        return {}

    all_probs, all_labels = [], []

    for _, group in df_val.sort_values(WINDOW_COL).groupby(PLAYER_COL):
        apm    = group[apm_col].values
        labels = group[LABEL_COL].values

        if len(apm) < window + 1:
            continue

        rolling_mean = np.convolve(apm, np.ones(window) / window, mode="valid")
        # Align: compare each window to the rolling mean of preceding windows
        actual   = apm[window:]
        baseline = rolling_mean[:-1] if len(rolling_mean) > 1 else rolling_mean

        min_len = min(len(actual), len(baseline), len(labels) - window)
        if min_len <= 0:
            continue

        drop = baseline[:min_len] - actual[:min_len]
        drop_std = drop.std()
        if drop_std > 0:
            probs = np.clip(drop / (3 * drop_std) + 0.5, 0, 1)
        else:
            probs = np.full(min_len, 0.5)

        all_probs.append(probs)
        all_labels.append(labels[window : window + min_len])

    if not all_probs:
        return {}

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    metrics    = compute_metrics(all_labels, all_probs, cfg["decision_threshold"])

    print(f"\n  Moving Average (window={window}) results:")
    print(f"    F1={metrics['f1']:.4f}  RMSE={metrics['rmse']:.4f}  "
          f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

def run_cross_validation(
    df_trainval: pd.DataFrame,
    norm_stats: Dict,
    cfg: Dict,
    run_dir: Path,
    device: torch.device,
) -> Dict:
    """
    5-fold player-wise cross-validation for the deep models.
    Returns aggregated metrics across folds.
    """
    folds = kfold_player_splits(df_trainval, k=cfg["cv_folds"], seed=cfg["seed"])
    fold_results = []

    for fold_idx, (df_tr, df_vl) in enumerate(folds, start=1):
        print(f"\n{'─'*58}")
        print(f"  CV Fold {fold_idx}/{cfg['cv_folds']}  "
              f"({df_tr[PLAYER_COL].nunique()} train players / "
              f"{df_vl[PLAYER_COL].nunique()} val players)")

        # Recompute norm stats per fold to avoid leakage
        fold_stats = compute_normalization_stats(df_tr, cfg.get("feature_cols", FEATURE_COLS))
        df_tr_n = apply_normalization(df_tr, fold_stats)
        df_vl_n = apply_normalization(df_vl, fold_stats)

        ds_tr = FatigueDataset(df_tr_n, mode=cfg["mode"])
        ds_vl = FatigueDataset(df_vl_n, mode=cfg["mode"])

        tr_loader = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True,
                               collate_fn=collate_fn)
        vl_loader = DataLoader(ds_vl, batch_size=cfg["batch_size"], shuffle=False,
                               collate_fn=collate_fn)

        input_dim = ds_tr.sequences[0].shape[-1]
        model = build_model(
            cfg["model_type"], input_dim,
            cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"],
            cfg["mode"], device
        )

        fold_metrics, _, _ = train_deep_model(
            model, tr_loader, vl_loader, cfg, run_dir, device, fold=fold_idx
        )
        fold_results.append(fold_metrics)

    # Aggregate
    agg = {}
    keys = [k for k in fold_results[0] if isinstance(fold_results[0][k], float)]
    for k in keys:
        vals = [r[k] for r in fold_results if not np.isnan(r.get(k, float("nan")))]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals))

    print(f"\n{'='*58}")
    print(f"  CV Summary ({cfg['cv_folds']} folds):")
    print(f"    F1    = {agg.get('f1_mean', 0):.4f} ± {agg.get('f1_std', 0):.4f}")
    print(f"    Prec  = {agg.get('precision_mean', 0):.4f} ± {agg.get('precision_std', 0):.4f}")
    print(f"    Recall= {agg.get('recall_mean', 0):.4f} ± {agg.get('recall_std', 0):.4f}")
    print(f"    AUC   = {agg.get('auc_roc_mean', 0):.4f} ± {agg.get('auc_roc_std', 0):.4f}")

    return {"fold_results": fold_results, "aggregated": agg}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(cfg: Dict) -> None:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Output directory ----
    run_name = f"{cfg['model_type']}_{int(time.time())}"
    run_dir  = Path(cfg["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- Build data loaders ----
    train_loader, val_loader, test_loader, info = build_dataloaders(
        csv_path=cfg["csv_path"],
        batch_size=cfg["batch_size"],
        mode=cfg["mode"],
        num_workers=cfg["num_workers"],
        stats_save_path=str(run_dir / "norm_stats.json"),
    )
    cfg["feature_cols"] = info["feature_cols"]
    input_dim = info["n_features"]

    print(f"\nDataset  — train: {info['n_matches']['train']} matches, "
          f"val: {info['n_matches']['val']}, test: {info['n_matches']['test']}")
    print(f"Fatigue rates — train: {info['fatigue_rate']['train']*100:.2f}%, "
          f"val: {info['fatigue_rate']['val']*100:.2f}%, "
          f"test: {info['fatigue_rate']['test']*100:.2f}%")

    all_results = {}

    # ---- Deep models ----
    deep_types = []
    if cfg["model_type"] == "all":
        deep_types = ["attention", "bilstm"]
    elif cfg["model_type"] in ("attention", "bilstm"):
        deep_types = [cfg["model_type"]]

    for mtype in deep_types:
        model_cfg = {**cfg, "model_type": mtype}
        model = build_model(
            mtype, input_dim,
            cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"],
            cfg["mode"], device
        )

        if cfg.get("cv"):
            # Load raw df for CV
            df_full = pd.read_csv(cfg["csv_path"])
            all_players = df_full[PLAYER_COL].unique().tolist()
            tr_pl, vl_pl, te_pl = player_wise_split(all_players)
            df_trainval = df_full[df_full[PLAYER_COL].isin(tr_pl + vl_pl)].copy()
            norm_stats  = compute_normalization_stats(df_trainval)
            cv_results  = run_cross_validation(
                df_trainval, norm_stats, model_cfg, run_dir, device
            )
            all_results[f"{mtype}_cv"] = cv_results
        else:
            metrics, _, _ = train_deep_model(
                model, train_loader, val_loader, model_cfg, run_dir, device
            )
            all_results[mtype] = metrics

    # ---- Shallow baselines ----
    if cfg["model_type"] in ("rf", "all"):
        df_full  = pd.read_csv(cfg["csv_path"])
        all_players = df_full[PLAYER_COL].unique().tolist()
        tr_pl, vl_pl, _ = player_wise_split(all_players)
        df_tr = df_full[df_full[PLAYER_COL].isin(tr_pl)].copy()
        df_vl = df_full[df_full[PLAYER_COL].isin(vl_pl)].copy()
        stats  = compute_normalization_stats(df_tr)
        df_tr  = apply_normalization(df_tr, stats)
        df_vl  = apply_normalization(df_vl, stats)
        all_results["random_forest"] = train_random_forest(df_tr, df_vl, cfg, run_dir)

    if cfg["model_type"] in ("arima", "all"):
        df_full = pd.read_csv(cfg["csv_path"])
        all_players = df_full[PLAYER_COL].unique().tolist()
        tr_pl, vl_pl, _ = player_wise_split(all_players)
        df_tr = df_full[df_full[PLAYER_COL].isin(tr_pl)].copy()
        df_vl = df_full[df_full[PLAYER_COL].isin(vl_pl)].copy()
        all_results["arima"] = train_arima(df_tr, df_vl, cfg)
        all_results["moving_average"] = train_moving_average(df_vl, cfg)

    # ---- Save all results ----
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved → {results_path}")

    # ---- Comparison table ----
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  Model comparison summary")
        print(f"{'='*70}")
        print(f"  {'Model':<20}  {'F1':>6}  {'Prec':>6}  {'Recall':>6}  "
              f"{'AUC':>6}  {'RMSE':>7}")
        print(f"  {'-'*60}")
        for name, res in all_results.items():
            if isinstance(res, dict) and "f1" in res:
                print(f"  {name:<20}  {res.get('f1',0):>6.3f}  "
                      f"{res.get('precision',0):>6.3f}  "
                      f"{res.get('recall',0):>6.3f}  "
                      f"{res.get('auc_roc',0):>6.3f}  "
                      f"{res.get('rmse',0):>7.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fatigue prediction models")
    parser.add_argument("--csv",       default=DEFAULT_CONFIG["csv_path"])
    parser.add_argument("--model",     default="attention",
                        choices=["attention", "bilstm", "rf", "arima", "ma", "all"])
    parser.add_argument("--epochs",    type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch",     type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",        type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--hidden",    type=int,   default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--layers",    type=int,   default=DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--dropout",   type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--patience",  type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["decision_threshold"])
    parser.add_argument("--outdir",    default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--cv",        action="store_true", help="Run 5-fold cross-validation")
    parser.add_argument("--mode",      default="sequence", choices=["sequence", "last"])
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG}
    cfg.update({
        "csv_path":          args.csv,
        "model_type":        args.model,
        "epochs":            args.epochs,
        "batch_size":        args.batch,
        "lr":                args.lr,
        "hidden_dim":        args.hidden,
        "num_layers":        args.layers,
        "dropout":           args.dropout,
        "patience":          args.patience,
        "decision_threshold": args.threshold,
        "output_dir":        args.outdir,
        "cv":                args.cv,
        "mode":              args.mode,
    })

    main(cfg)