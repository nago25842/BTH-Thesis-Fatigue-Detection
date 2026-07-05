"""
evaluate.py
-----------
Final test-set evaluation for all trained models.

Produces:
  - Definitive RQ2 metrics  (F1, precision, recall, AUC) for AttentionBiLSTM
  - Definitive RQ3 comparison table across all models
  - Confusion matrix
  - Precision-Recall curve
  - ROC curve
  - Threshold sensitivity analysis
  - Per-player fatigue detection summary

All outputs saved to experiments/<run_dir>/evaluation/

Usage
-----
    python evaluate.py ^
        --csv   data/features_labeled.csv ^
        --ckpt  experiments/attention_<ts>/best_model.pt ^
        --ckpt2 experiments/all_<ts>/best_model.pt ^
        --norm  experiments/attention_<ts>/norm_stats.json ^
        --rf    experiments/all_<ts>/rf_model.pkl

    # Minimal (attention model only):
    python evaluate.py ^
        --csv  data/features_labeled.csv ^
        --ckpt experiments/attention_<ts>/best_model.pt ^
        --norm experiments/attention_<ts>/norm_stats.json
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score,
)

warnings.filterwarnings("ignore")

from dataset import (
    build_dataloaders, apply_normalization, FatigueDataset,
    player_wise_split, collate_fn,
    FEATURE_COLS, LABEL_COL, PLAYER_COL, MATCH_COL, WINDOW_COL,
)
from model import build_model

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

PALETTE = {
    "purple": "#7F77DD", "teal":  "#1D9E75",
    "amber":  "#EF9F27", "coral": "#D85A30",
    "gray":   "#888780", "blue":  "#378ADD",
    "green":  "#639922", "red":   "#E24B4A",
}
MODEL_COLORS = {
    "Attention-BiLSTM": PALETTE["purple"],
    "BiLSTM":           PALETTE["teal"],
    "Random Forest":    PALETTE["amber"],
    "ARIMA":            PALETTE["coral"],
    "Moving Average":   PALETTE["gray"],
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top":  False,   "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13,        "axes.labelsize": 11,
})


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Run model on a DataLoader.

    Returns:
        probs      : (N,) predicted probabilities
        labels     : (N,) ground-truth labels
        attn       : (N,) attention weights
        player_ids : (N,) player id per window
    """
    model.eval()
    all_probs, all_labels, all_attn, all_players = [], [], [], []

    for batch in loader:
        feats      = batch["features"].to(device)
        lengths    = batch["lengths"].to(device)
        mask       = batch["mask"].to(device)
        labels     = batch["labels"].numpy()
        player_ids = batch["player_ids"]

        logits, attn = model(feats, lengths, mask)
        probs = torch.sigmoid(logits).cpu().numpy()

        for i in range(feats.shape[0]):
            L = lengths[i].item()
            all_probs.append(probs[i, :L])
            all_labels.append(labels[i, :L])
            all_attn.append(attn[i, :L].cpu().numpy())
            all_players.extend([player_ids[i]] * L)

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_attn),
        all_players,
    )


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.35,
    model_name: str = "",
) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "model":      model_name,
        "threshold":  threshold,
        "f1":         f1_score(y_true, y_pred, zero_division=0),
        "precision":  precision_score(y_true, y_pred, zero_division=0),
        "recall":     recall_score(y_true, y_pred, zero_division=0),
        "accuracy":   accuracy_score(y_true, y_pred),
        "auc_roc":    roc_auc_score(y_true, y_prob) if y_true.sum() > 0 else float("nan"),
        "avg_precision": average_precision_score(y_true, y_prob) if y_true.sum() > 0 else float("nan"),
        "rmse":       float(np.sqrt(mean_squared_error(y_true, y_prob))),
        "mae":        float(mean_absolute_error(y_true, y_prob)),
        "r2":         float(r2_score(y_true, y_prob)) if np.var(y_true) > 0 else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_total":    int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_predicted_positive": int(y_pred.sum()),
    }
    return metrics


# ---------------------------------------------------------------------------
# Plot: confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    out_dir: Path,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Purples")

    labels = ["Normal", "Fatigued"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, rotation=90, va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {model_name}", pad=10)

    thresh_color = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh_color else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_").replace("-", "")
    fig.savefig(out_dir / f"confusion_matrix_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: Precision-Recall curve
# ---------------------------------------------------------------------------

def plot_pr_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> None:
    """results = {model_name: (y_true, y_prob)}"""
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, (y_true, y_prob) in results.items():
        if y_true.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        color = MODEL_COLORS.get(name, PALETTE["gray"])
        ax.plot(rec, prec, label=f"{name}  (AP={ap:.3f})",
                color=color, linewidth=1.8)

    # Baseline: random classifier
    pos_rate = list(results.values())[0][0].mean()
    ax.axhline(pos_rate, color=PALETTE["gray"], linestyle="--",
               linewidth=1, label=f"Random (AP={pos_rate:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))

    for name, (y_true, y_prob) in results.items():
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        color = MODEL_COLORS.get(name, PALETTE["gray"])
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})",
                color=color, linewidth=1.8)

    ax.plot([0, 1], [0, 1], color=PALETTE["gray"], linestyle="--",
            linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: Threshold sensitivity
# ---------------------------------------------------------------------------

def plot_threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    out_dir: Path,
) -> float:
    """
    Plot F1, precision, recall across decision thresholds.
    Returns the threshold that maximises F1 on the test set.
    """
    thresholds = np.linspace(0.05, 0.90, 80)
    f1s, precs, recs = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))

    best_idx = int(np.argmax(f1s))
    best_t   = float(thresholds[best_idx])

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(thresholds, f1s,   color=PALETTE["purple"], linewidth=2,   label="F1")
    ax.plot(thresholds, precs, color=PALETTE["teal"],   linewidth=1.5, label="Precision",
            linestyle="--")
    ax.plot(thresholds, recs,  color=PALETTE["coral"],  linewidth=1.5, label="Recall",
            linestyle=":")
    ax.axvline(best_t, color=PALETTE["amber"], linewidth=1.2, linestyle="-.",
               label=f"Best threshold = {best_t:.2f}  (F1={f1s[best_idx]:.3f})")
    ax.axvline(0.35,   color=PALETTE["gray"],  linewidth=0.8, linestyle="--",
               label="Training threshold = 0.35")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold sensitivity — {model_name}", pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim([0.05, 0.90]); ax.set_ylim([0, 1])
    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_").replace("-", "")
    fig.savefig(out_dir / f"threshold_sensitivity_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Best threshold on test set: {best_t:.2f}  →  "
          f"F1={f1s[best_idx]:.4f}, "
          f"Prec={precs[best_idx]:.4f}, "
          f"Recall={recs[best_idx]:.4f}")
    return best_t


# ---------------------------------------------------------------------------
# Plot: model comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bars(
    all_metrics: List[Dict],
    out_dir: Path,
) -> None:
    metrics_to_plot = ["f1", "precision", "recall", "auc_roc"]
    labels = ["F1", "Precision", "Recall", "AUC-ROC"]
    models = [m["model"] for m in all_metrics]
    n_m, n_met = len(models), len(metrics_to_plot)
    x = np.arange(n_met)
    width = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(all_metrics):
        vals = [m.get(k, 0) for k in metrics_to_plot]
        color = MODEL_COLORS.get(m["model"], PALETTE["gray"])
        offset = (i - n_m / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=m["model"],
                      color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
                        color=color)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.12])
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — test set", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right",
              ncol=2 if n_m > 3 else 1)
    plt.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: per-player fatigue detection rate
# ---------------------------------------------------------------------------

def plot_player_fatigue_rate(
    probs: np.ndarray,
    labels: np.ndarray,
    player_ids: List[str],
    threshold: float,
    out_dir: Path,
    top_n: int = 30,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "player_id": player_ids,
        "prob":      probs,
        "label":     labels,
        "pred":      (probs >= threshold).astype(int),
    })

    per_player = df.groupby("player_id").agg(
        n_windows       =("label",  "count"),
        n_true_fatigue  =("label",  "sum"),
        n_pred_fatigue  =("pred",   "sum"),
        mean_prob       =("prob",   "mean"),
        true_rate       =("label",  "mean"),
        pred_rate       =("pred",   "mean"),
    ).reset_index()

    # Show top players by predicted fatigue rate
    top = per_player.nlargest(top_n, "pred_rate")

    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.32)))
    y = np.arange(len(top))
    ax.barh(y, top["true_rate"], height=0.4, align="center",
            color=PALETTE["coral"], alpha=0.8, label="True fatigue rate")
    ax.barh(y + 0.4, top["pred_rate"], height=0.4, align="center",
            color=PALETTE["purple"], alpha=0.8, label="Predicted fatigue rate")
    ax.set_yticks(y + 0.2)
    ax.set_yticklabels(top["player_id"], fontsize=8)
    ax.set_xlabel("Fatigue rate (fraction of windows)")
    ax.set_title(f"Per-player fatigue rate — top {top_n} most fatigued (predicted)", pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_dir / "per_player_fatigue.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    per_player.to_csv(out_dir / "per_player_stats.csv", index=False)
    return per_player


# ---------------------------------------------------------------------------
# Final results table (thesis Table)
# ---------------------------------------------------------------------------

def print_results_table(all_metrics: List[Dict]) -> None:
    print(f"\n{'='*80}")
    print(f"  FINAL TEST SET RESULTS  —  thesis Tables (RQ2 + RQ3)")
    print(f"{'='*80}")
    print(f"  {'Model':<22} {'F1':>6} {'Prec':>6} {'Recall':>6} "
          f"{'AUC':>6} {'AP':>6} {'RMSE':>7} {'R²':>7}")
    print(f"  {'-'*76}")
    for m in all_metrics:
        print(f"  {m['model']:<22} "
              f"{m.get('f1', 0):>6.3f} "
              f"{m.get('precision', 0):>6.3f} "
              f"{m.get('recall', 0):>6.3f} "
              f"{m.get('auc_roc', float('nan')):>6.3f} "
              f"{m.get('avg_precision', float('nan')):>6.3f} "
              f"{m.get('rmse', 0):>7.4f} "
              f"{m.get('r2', float('nan')):>7.3f}")
    print(f"{'='*80}")

    # Highlight best per metric
    print(f"\n  Best per metric:")
    for metric in ["f1", "auc_roc", "precision", "recall"]:
        vals = [(m["model"], m.get(metric, 0)) for m in all_metrics
                if not np.isnan(m.get(metric, float("nan")))]
        if vals:
            best_model, best_val = max(vals, key=lambda x: x[1])
            print(f"    {metric:<15} → {best_model:<22} ({best_val:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {out_dir}\n")

    # ---- Load norm stats + build test loader ----
    with open(args.norm) as f:
        norm_stats = json.load(f)
    feature_cols = list(norm_stats.keys())
    print(f"Features: {feature_cols}")

    df_full = pd.read_csv(args.csv)
    all_players = df_full[PLAYER_COL].unique().tolist()
    _, _, te_pl = player_wise_split(all_players)
    df_test = df_full[df_full[PLAYER_COL].isin(te_pl)].copy()
    df_test = apply_normalization(df_test, norm_stats, feature_cols)
    ds_test = FatigueDataset(df_test, feature_cols=feature_cols, mode="sequence")

    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        ds_test, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    total_windows = sum(s.shape[0] for s in ds_test.sequences)
    fatigue_rate  = ds_test.fatigue_rate()
    print(f"\nTest set: {len(ds_test)} matches, {total_windows} windows, "
          f"{fatigue_rate*100:.2f}% fatigue rate\n")

    all_metrics = []
    curve_data  = {}     # for PR + ROC plots

    # ----------------------------------------------------------------
    # 1. Attention-BiLSTM (primary model — RQ2)
    # ----------------------------------------------------------------
    print("=" * 55)
    print("Evaluating: Attention-BiLSTM")
    print("=" * 55)
    attn_model = build_model("attention", len(feature_cols), device=device)
    attn_model.load_state_dict(torch.load(args.ckpt, map_location=device))

    probs_a, labels_a, attn_a, players_a = run_inference(attn_model, test_loader, device)
    metrics_a = compute_all_metrics(labels_a, probs_a, args.threshold, "Attention-BiLSTM")
    all_metrics.append(metrics_a)
    curve_data["Attention-BiLSTM"] = (labels_a, probs_a)

    plot_confusion_matrix(labels_a, probs_a, args.threshold, "Attention-BiLSTM", out_dir)
    best_t = plot_threshold_sensitivity(labels_a, probs_a, "Attention-BiLSTM", out_dir)

    # Also evaluate at best threshold
    metrics_best = compute_all_metrics(labels_a, probs_a, best_t, "Attention-BiLSTM (best-t)")
    print(f"\n  At training threshold ({args.threshold:.2f}): "
          f"F1={metrics_a['f1']:.4f}, Prec={metrics_a['precision']:.4f}, "
          f"Recall={metrics_a['recall']:.4f}, AUC={metrics_a['auc_roc']:.4f}")
    print(f"  At best threshold    ({best_t:.2f}): "
          f"F1={metrics_best['f1']:.4f}, Prec={metrics_best['precision']:.4f}, "
          f"Recall={metrics_best['recall']:.4f}")

    plot_player_fatigue_rate(probs_a, labels_a, players_a, args.threshold, out_dir)

    # ----------------------------------------------------------------
    # 2. BiLSTM baseline (RQ3)
    # ----------------------------------------------------------------
    if args.ckpt2:
        print("\n" + "=" * 55)
        print("Evaluating: BiLSTM baseline")
        print("=" * 55)
        bilstm_model = build_model("bilstm", len(feature_cols), device=device)
        bilstm_model.load_state_dict(torch.load(args.ckpt2, map_location=device))
        probs_b, labels_b, _, _ = run_inference(bilstm_model, test_loader, device)
        metrics_b = compute_all_metrics(labels_b, probs_b, args.threshold, "BiLSTM")
        all_metrics.append(metrics_b)
        curve_data["BiLSTM"] = (labels_b, probs_b)
        plot_confusion_matrix(labels_b, probs_b, args.threshold, "BiLSTM", out_dir)
        print(f"  F1={metrics_b['f1']:.4f}, AUC={metrics_b['auc_roc']:.4f}")

    # ----------------------------------------------------------------
    # 3. Random Forest baseline (RQ3)
    # ----------------------------------------------------------------
    if args.rf:
        print("\n" + "=" * 55)
        print("Evaluating: Random Forest")
        print("=" * 55)
        try:
            import joblib
            rf = joblib.load(args.rf)
            X_test = df_test[feature_cols].values
            y_test = df_test[LABEL_COL].values
            probs_rf = rf.predict_proba(X_test)[:, 1]
            metrics_rf = compute_all_metrics(y_test, probs_rf, args.threshold, "Random Forest")
            all_metrics.append(metrics_rf)
            curve_data["Random Forest"] = (y_test, probs_rf)
            plot_confusion_matrix(y_test, probs_rf, args.threshold, "Random Forest", out_dir)
            print(f"  F1={metrics_rf['f1']:.4f}, AUC={metrics_rf['auc_roc']:.4f}")
        except Exception as e:
            print(f"  Could not load RF model: {e}")

    # ----------------------------------------------------------------
    # 4. ARIMA + Moving Average (RQ3)
    # ----------------------------------------------------------------
    try:
        from statsmodels.tsa.arima.model import ARIMA
        print("\n" + "=" * 55)
        print("Evaluating: ARIMA baseline")
        print("=" * 55)

        df_test_raw = df_full[df_full[PLAYER_COL].isin(te_pl)].copy()
        all_probs_ar, all_labels_ar = [], []
        all_probs_ma, all_labels_ma = [], []

        for _, group in df_test_raw.sort_values(WINDOW_COL).groupby(PLAYER_COL):
            if "apm" not in group.columns or len(group) < 5:
                continue
            apm    = group["apm"].values
            labels = group[LABEL_COL].values

            # ARIMA
            try:
                fit = ARIMA(apm, order=(1, 1, 1)).fit()
                res = fit.resid
                r_std = res.std()
                if r_std > 0:
                    p = np.clip(-res / (3 * r_std) + 0.5, 0, 1)
                else:
                    p = np.full(len(res), 0.5)
                all_probs_ar.append(p)
                all_labels_ar.append(labels[:len(p)])
            except Exception:
                pass

            # Moving Average
            w = 3
            if len(apm) >= w + 1:
                rolling = np.convolve(apm, np.ones(w) / w, mode="valid")
                actual  = apm[w:]
                base    = rolling[:-1] if len(rolling) > 1 else rolling
                mn      = min(len(actual), len(base), len(labels) - w)
                if mn > 0:
                    drop  = base[:mn] - actual[:mn]
                    d_std = drop.std()
                    p = np.clip(drop / (3 * d_std + 1e-8) + 0.5, 0, 1)
                    all_probs_ma.append(p)
                    all_labels_ma.append(labels[w:w + mn])

        if all_probs_ar:
            p_ar = np.concatenate(all_probs_ar)
            l_ar = np.concatenate(all_labels_ar)
            m_ar = compute_all_metrics(l_ar, p_ar, args.threshold, "ARIMA")
            all_metrics.append(m_ar)
            curve_data["ARIMA"] = (l_ar, p_ar)
            print(f"  ARIMA   F1={m_ar['f1']:.4f}, AUC={m_ar['auc_roc']:.4f}")

        if all_probs_ma:
            p_ma = np.concatenate(all_probs_ma)
            l_ma = np.concatenate(all_labels_ma)
            m_ma = compute_all_metrics(l_ma, p_ma, args.threshold, "Moving Average")
            all_metrics.append(m_ma)
            curve_data["Moving Average"] = (l_ma, p_ma)
            print(f"  MA      F1={m_ma['f1']:.4f}, AUC={m_ma['auc_roc']:.4f}")

    except ImportError:
        print("  statsmodels not available — skipping ARIMA/MA evaluation.")

    # ----------------------------------------------------------------
    # Plots that need all models
    # ----------------------------------------------------------------
    if len(curve_data) >= 2:
        plot_pr_curves(curve_data, out_dir)
        plot_roc_curves(curve_data, out_dir)

    plot_comparison_bars(all_metrics, out_dir)
    print_results_table(all_metrics)

    # ----------------------------------------------------------------
    # Save everything
    # ----------------------------------------------------------------
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(out_dir / "test_results.csv", index=False)

    with open(out_dir / "test_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n{'='*55}")
    print(f"All evaluation outputs saved to: {out_dir}")
    print(f"  ├── confusion_matrix_*.png")
    print(f"  ├── threshold_sensitivity_*.png")
    print(f"  ├── pr_curves.png")
    print(f"  ├── roc_curves.png")
    print(f"  ├── model_comparison.png")
    print(f"  ├── per_player_fatigue.png")
    print(f"  ├── per_player_stats.csv")
    print(f"  ├── test_results.csv        ← thesis Table")
    print(f"  └── test_results.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final test set evaluation")
    parser.add_argument("--csv",       required=True,
                        help="Path to labelled windows CSV")
    parser.add_argument("--ckpt",      required=True,
                        help="Attention-BiLSTM best_model.pt")
    parser.add_argument("--norm",      required=True,
                        help="norm_stats.json from training run")
    parser.add_argument("--ckpt2",     default=None,
                        help="BiLSTM baseline best_model.pt (optional)")
    parser.add_argument("--rf",        default=None,
                        help="rf_model.pkl path (optional)")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Decision threshold (default 0.35)")
    parser.add_argument("--out",       default=None,
                        help="Output dir (default: next to --ckpt)")
    args = parser.parse_args()

    if args.out is None:
        args.out = str(Path(args.ckpt).parent / "evaluation")

    main(args)