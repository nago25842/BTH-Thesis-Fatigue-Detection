"""
interpret.py
------------
Interpretability analysis for RQ1:
  "Which behavioral features most indicate performance degradation?"

Three complementary methods:
  1. SHAP values       – per-feature contribution to each prediction
  2. Permutation importance – model-agnostic feature ranking
  3. Attention weights – temporal patterns the model focuses on

All outputs are saved to experiments/<run_dir>/interpretability/
including publication-ready plots and a CSV summary table.

Usage
-----
# Point at your best trained attention model checkpoint:
    python interpret.py \\
        --csv  data/features_labeled.csv \\
        --ckpt experiments/attention_<timestamp>/best_model.pt \\
        --norm experiments/attention_<timestamp>/norm_stats.json

# With explicit output dir:
    python interpret.py \\
        --csv  data/features_labeled.csv \\
        --ckpt experiments/attention_<timestamp>/best_model.pt \\
        --norm experiments/attention_<timestamp>/norm_stats.json \\
        --out  experiments/attention_<timestamp>/interpretability
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

from dataset import (
    build_dataloaders, apply_normalization, compute_normalization_stats,
    player_wise_split, collate_fn, FatigueDataset,
    FEATURE_COLS, LABEL_COL, PLAYER_COL, MATCH_COL, WINDOW_COL,
)
from model import build_model, AttentionBiLSTM

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

PALETTE = {
    "purple":  "#7F77DD",
    "teal":    "#1D9E75",
    "amber":   "#EF9F27",
    "coral":   "#D85A30",
    "gray":    "#888780",
    "red":     "#E24B4A",
    "blue":    "#378ADD",
}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

FEATURE_LABELS = {
    "apm":             "APM",
    "apm_variance":    "APM variance",
    "action_gap_mean": "Action gap (mean)",
    "action_gap_std":  "Action gap (std)",
    "error_rate":      "Error rate",
    "resource_eff":    "Resource efficiency",
    "pause_freq":      "Pause frequency",
}


def pretty(col: str) -> str:
    return FEATURE_LABELS.get(col, col)


# ---------------------------------------------------------------------------
# 1. SHAP — DeepLIFT/GradientSHAP via captum
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: nn.Module,
    val_loader,
    feature_cols: List[str],
    device: torch.device,
    n_background: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute GradientSHAP values for each feature at each time step.

    Falls back to vanilla gradient * input (Integrated Gradients style)
    if captum is not installed.

    Returns:
        shap_vals : (N, F)   mean |SHAP| per window per feature
        X_flat    : (N, F)   raw feature values
        y_flat    : (N,)     labels
    """
    try:
        from captum.attr import GradientShap
        use_captum = True
    except ImportError:
        use_captum = False
        print("  captum not installed — using gradient×input as SHAP proxy.")
        print("  Install: pip install captum --break-system-packages")

    model.eval()

    # Collect all validation windows (flattened — one row per window)
    all_features, all_labels = [], []
    for batch in val_loader:
        feats   = batch["features"]   # (B, T, F)
        labels  = batch["labels"]     # (B, T)
        lengths = batch["lengths"]
        for i in range(feats.shape[0]):
            L = lengths[i].item()
            all_features.append(feats[i, :L].numpy())   # (L, F)
            all_labels.append(labels[i, :L].numpy())    # (L,)

    X_flat = np.concatenate(all_features, axis=0)   # (N, F)
    y_flat = np.concatenate(all_labels,   axis=0)   # (N,)

    # For SHAP we treat each window independently (no temporal context)
    # by wrapping the model to accept (1, 1, F) inputs
    class WindowWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            # x: (B, F) → unsqueeze T dim → (B, 1, F)
            x3 = x.unsqueeze(1)
            lengths = torch.ones(x.shape[0], dtype=torch.long, device=x.device)
            mask    = torch.ones(x.shape[0], 1, dtype=torch.bool, device=x.device)
            logits, _ = self.base(x3, lengths, mask)
            # logits: (B, 1) or (B,) — return (B, 1) for captum
            return logits.view(-1, 1)

    wrapper = WindowWrapper(model).to(device)
    wrapper.eval()

    X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(device)
    background = X_tensor[
        np.random.choice(len(X_tensor), min(n_background, len(X_tensor)), replace=False)
    ]

    if use_captum:
        from captum.attr import GradientShap
        gs = GradientShap(wrapper)
        # Process in chunks to avoid OOM
        chunk = 256
        shap_chunks = []
        for start in range(0, len(X_tensor), chunk):
            end = min(start + chunk, len(X_tensor))
            attrs = gs.attribute(
                X_tensor[start:end],
                baselines=background,
                n_samples=10,
                stdevs=0.1,
            )
            shap_chunks.append(attrs.detach().cpu().numpy())
        shap_vals = np.concatenate(shap_chunks, axis=0)   # (N, F)
    else:
        # Gradient × input fallback
        X_tensor.requires_grad_(True)
        out = wrapper(X_tensor)
        out.sum().backward()
        shap_vals = (X_tensor.grad * X_tensor).detach().cpu().numpy()

    return shap_vals, X_flat, y_flat


def plot_shap_summary(
    shap_vals: np.ndarray,
    X_flat: np.ndarray,
    y_flat: np.ndarray,
    feature_cols: List[str],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Beeswarm-style SHAP summary plot (mean |SHAP| bar chart as fallback).
    Returns a DataFrame of mean |SHAP| per feature (for the CSV table).
    """
    F = len(feature_cols)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)   # (F,)
    order = np.argsort(mean_abs_shap)[::-1]

    df_shap = pd.DataFrame({
        "feature":       [feature_cols[i] for i in order],
        "feature_label": [pretty(feature_cols[i]) for i in order],
        "mean_abs_shap": mean_abs_shap[order],
        "rank":          range(1, F + 1),
    })

    # ---- Bar chart ----
    fig, ax = plt.subplots(figsize=(7, max(3, F * 0.55)))
    colors = [PALETTE["purple"] if i == 0 else PALETTE["gray"] for i in range(F)]
    bars = ax.barh(
        df_shap["feature_label"][::-1],
        df_shap["mean_abs_shap"][::-1],
        color=colors[::-1],
        height=0.6,
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature importance (SHAP)", pad=12)
    ax.axvline(0, color="black", linewidth=0.5)
    for bar, val in zip(bars, df_shap["mean_abs_shap"][::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=PALETTE["gray"])
    plt.tight_layout()
    fig.savefig(out_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Dot plot: SHAP value distribution coloured by feature value ----
    fig, axes = plt.subplots(1, F, figsize=(max(8, F * 1.6), 4), sharey=False)
    if F == 1:
        axes = [axes]
    cmap = LinearSegmentedColormap.from_list("rv", ["#378ADD", "#E24B4A"])
    for ax, feat_idx in zip(axes, order):
        feat_name   = feature_cols[feat_idx]
        sv          = shap_vals[:, feat_idx]
        fv          = X_flat[:, feat_idx]
        fv_norm = (fv - fv.min()) / (np.ptp(fv) + 1e-8)
        ax.scatter(sv, np.zeros_like(sv) + np.random.uniform(-0.2, 0.2, len(sv)),
                   c=fv_norm, cmap=cmap, alpha=0.4, s=8, linewidths=0)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel(f"{pretty(feat_name)}", fontsize=9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Feature value\n(low → high)", fontsize=9)
    fig.suptitle("SHAP distribution per feature", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  SHAP plots saved → {out_dir}")
    return df_shap


# ---------------------------------------------------------------------------
# 2. Permutation importance
# ---------------------------------------------------------------------------

def permutation_importance(
    model: nn.Module,
    val_loader,
    feature_cols: List[str],
    device: torch.device,
    n_repeats: int = 10,
    threshold: float = 0.35,
) -> pd.DataFrame:
    """
    Permutation importance: for each feature, randomly shuffle its values
    across all validation windows and measure the drop in F1.

    A large drop = the feature is important.
    A small or negative drop = the feature adds little signal.

    Returns a DataFrame sorted by importance (descending).
    """
    from sklearn.metrics import f1_score

    model.eval()

    def get_f1(loader_override=None) -> float:
        all_probs, all_labels = [], []
        src = loader_override if loader_override is not None else val_loader
        with torch.no_grad():
            for batch in src:
                feats   = batch["features"].to(device)
                lengths = batch["lengths"].to(device)
                mask    = batch["mask"].to(device)
                labels  = batch["labels"]
                logits, _ = model(feats, lengths, mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                lbls  = labels.numpy()
                for i in range(len(batch["lengths"])):
                    L = batch["lengths"][i].item()
                    all_probs.append(probs[i, :L])
                    all_labels.append(lbls[i, :L])
        p = np.concatenate(all_probs)
        l = np.concatenate(all_labels)
        return f1_score(l, (p >= threshold).astype(int), zero_division=0)

    baseline_f1 = get_f1()
    print(f"  Baseline F1: {baseline_f1:.4f}")

    results = []
    for feat_idx, feat_name in enumerate(feature_cols):
        drops = []
        for _ in range(n_repeats):
            # Shuffle this feature across all batches in-memory
            # Collect all batches, shuffle, re-evaluate
            all_feats, all_lengths, all_masks, all_labels_list = [], [], [], []
            for batch in val_loader:
                all_feats.append(batch["features"].clone())
                all_lengths.append(batch["lengths"])
                all_masks.append(batch["mask"])
                all_labels_list.append(batch["labels"])

            # Shuffle the feature across ALL windows globally
            global_feat_vals = []
            for f in all_feats:
                for b in range(f.shape[0]):
                    global_feat_vals.append(f[b, :, feat_idx].numpy())
            flat = np.concatenate(global_feat_vals)
            np.random.shuffle(flat)

            # Write shuffled values back
            ptr = 0
            for f in all_feats:
                for b in range(f.shape[0]):
                    T = f.shape[1]
                    f[b, :, feat_idx] = torch.tensor(flat[ptr:ptr + T], dtype=torch.float32)
                    ptr += T

            # Evaluate with shuffled feature
            all_probs, all_labels_eval = [], []
            with torch.no_grad():
                for feats, lengths, mask, labels in zip(
                    all_feats, all_lengths, all_masks, all_labels_list
                ):
                    feats   = feats.to(device)
                    lengths = lengths.to(device)
                    mask    = mask.to(device)
                    logits, _ = model(feats, lengths, mask)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    lbls  = labels.numpy()
                    for i in range(len(lengths)):
                        L = lengths[i].item()
                        all_probs.append(probs[i, :L])
                        all_labels_eval.append(lbls[i, :L])

            p = np.concatenate(all_probs)
            l = np.concatenate(all_labels_eval)
            perm_f1 = f1_score(l, (p >= threshold).astype(int), zero_division=0)
            drops.append(baseline_f1 - perm_f1)

        mean_drop = float(np.mean(drops))
        std_drop  = float(np.std(drops))
        results.append({
            "feature":       feat_name,
            "feature_label": pretty(feat_name),
            "importance":    mean_drop,
            "std":           std_drop,
            "baseline_f1":   baseline_f1,
        })
        print(f"    {pretty(feat_name):<25}  ΔF1 = {mean_drop:+.4f} ± {std_drop:.4f}")

    df_perm = pd.DataFrame(results).sort_values("importance", ascending=False).reset_index(drop=True)
    df_perm["rank"] = range(1, len(df_perm) + 1)
    return df_perm


def plot_permutation_importance(df_perm: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, max(3, len(df_perm) * 0.55)))
    colors = [
        PALETTE["coral"] if v > 0 else PALETTE["blue"]
        for v in df_perm["importance"]
    ]
    ax.barh(
        df_perm["feature_label"][::-1],
        df_perm["importance"][::-1],
        xerr=df_perm["std"][::-1],
        color=colors[::-1],
        height=0.6,
        capsize=3,
        error_kw={"linewidth": 1, "color": PALETTE["gray"]},
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean F1 drop when feature is shuffled")
    ax.set_title("Permutation feature importance", pad=12)
    legend_patches = [
        mpatches.Patch(color=PALETTE["coral"], label="Positive impact"),
        mpatches.Patch(color=PALETTE["blue"],  label="Negative / no impact"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, frameon=False)
    plt.tight_layout()
    fig.savefig(out_dir / "permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Permutation importance plot saved → {out_dir}")


# ---------------------------------------------------------------------------
# 3. Attention weight analysis
# ---------------------------------------------------------------------------

def analyze_attention_weights(
    model: nn.Module,
    val_loader,
    feature_cols: List[str],
    device: torch.device,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Analyse the attention weights learned by the model:
      - Average attention weight over time (does attention peak at certain windows?)
      - Attention vs fatigue label (do fatigued windows get higher attention?)
      - Per-match attention heatmaps for a sample of matches

    Returns a DataFrame with per-window attention stats.
    """
    model.eval()

    records = []
    sample_matches = []   # for heatmap: store (attn, labels, features, match_id)

    with torch.no_grad():
        for batch in val_loader:
            feats      = batch["features"].to(device)
            lengths    = batch["lengths"].to(device)
            mask       = batch["mask"].to(device)
            labels     = batch["labels"].numpy()
            match_ids  = batch["match_ids"]
            player_ids = batch["player_ids"]

            _, attn = model(feats, lengths, mask)   # attn: (B, T)
            attn_np = attn.cpu().numpy()

            for i in range(feats.shape[0]):
                L     = lengths[i].item()
                a     = attn_np[i, :L]
                lbl   = labels[i, :L]
                feat  = feats[i, :L].cpu().numpy()

                for t in range(L):
                    records.append({
                        "match_id":    match_ids[i],
                        "player_id":   player_ids[i],
                        "window_idx":  t,
                        "attn_weight": float(a[t]),
                        "label":       int(lbl[t]),
                        "rel_position": t / max(L - 1, 1),   # 0=start, 1=end
                        **{f: float(feat[t, fi]) for fi, f in enumerate(feature_cols)},
                    })

                # Keep a few matches for heatmap visualisation
                if len(sample_matches) < 12:
                    sample_matches.append((a, lbl, feat, match_ids[i], player_ids[i]))

    df_attn = pd.DataFrame(records)

    # ---- Plot 1: Mean attention by relative position ----
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.linspace(0, 1, 11)
    df_attn["pos_bin"] = pd.cut(df_attn["rel_position"], bins=bins, labels=False)
    mean_by_pos = df_attn.groupby("pos_bin")["attn_weight"].mean()
    ax.bar(
        mean_by_pos.index / 10,
        mean_by_pos.values,
        width=0.09,
        color=PALETTE["purple"],
        alpha=0.8,
        align="edge",
    )
    ax.set_xlabel("Relative position in match (0 = start, 1 = end)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Temporal attention distribution across match duration", pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / "attention_by_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: Attention weight — fatigued vs normal windows ----
    fatigue_attn = df_attn[df_attn["label"] == 1]["attn_weight"].values
    normal_attn  = df_attn[df_attn["label"] == 0]["attn_weight"].values

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins_h = np.linspace(0, df_attn["attn_weight"].max(), 30)
    ax.hist(normal_attn,  bins=bins_h, alpha=0.6, color=PALETTE["blue"],
            label=f"Normal (n={len(normal_attn):,})", density=True)
    ax.hist(fatigue_attn, bins=bins_h, alpha=0.7, color=PALETTE["coral"],
            label=f"Fatigued (n={len(fatigue_attn):,})", density=True)
    ax.set_xlabel("Attention weight")
    ax.set_ylabel("Density")
    ax.set_title("Attention weights: fatigued vs normal windows", pad=12)
    ax.legend(frameon=False)

    mean_f = fatigue_attn.mean() if len(fatigue_attn) > 0 else 0
    mean_n = normal_attn.mean()
    ax.axvline(mean_f, color=PALETTE["coral"], linestyle="--", linewidth=1.2,
               label=f"Mean fatigued = {mean_f:.4f}")
    ax.axvline(mean_n, color=PALETTE["blue"],  linestyle="--", linewidth=1.2,
               label=f"Mean normal   = {mean_n:.4f}")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "attention_fatigue_vs_normal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 3: Sample attention heatmaps ----
    n_show = min(6, len(sample_matches))
    if n_show > 0:
        fig, axes = plt.subplots(n_show, 1, figsize=(9, n_show * 1.4))
        if n_show == 1:
            axes = [axes]
        cmap_attn = LinearSegmentedColormap.from_list("attn", ["#F1EFE8", "#534AB7"])
        for ax, (a, lbl, feat, mid, pid) in zip(axes, sample_matches[:n_show]):
            T = len(a)
            im = ax.imshow(
                a.reshape(1, -1), aspect="auto", cmap=cmap_attn,
                vmin=0, vmax=a.max() + 1e-8,
            )
            # Mark fatigued windows with a red tick
            for t in range(T):
                if lbl[t] == 1:
                    ax.add_patch(plt.Rectangle(
                        (t - 0.5, -0.5), 1, 1,
                        linewidth=1.5, edgecolor=PALETTE["coral"],
                        facecolor="none",
                    ))
            ax.set_yticks([])
            ax.set_xticks(range(T))
            ax.set_xticklabels(range(T), fontsize=7)
            ax.set_xlabel(f"Window index  |  player {pid}", fontsize=8)
        fig.colorbar(im, ax=axes, orientation="vertical",
                     fraction=0.01, pad=0.01, label="Attention weight")
        fig.suptitle(
            "Attention heatmaps (orange border = fatigued window)", fontsize=11, y=1.01
        )
        plt.tight_layout()
        fig.savefig(out_dir / "attention_heatmaps.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Summary stats ----
    print(f"\n  Attention analysis:")
    print(f"    Mean attention — normal  : {mean_n:.5f}")
    print(f"    Mean attention — fatigued: {mean_f:.5f}")
    if len(fatigue_attn) > 0:
        ratio = mean_f / (mean_n + 1e-10)
        print(f"    Ratio (fatigued/normal)  : {ratio:.3f}x")

    df_attn.to_csv(out_dir / "attention_per_window.csv", index=False)
    print(f"  Attention data saved → {out_dir / 'attention_per_window.csv'}")
    return df_attn


# ---------------------------------------------------------------------------
# 4. Combined feature ranking table
# ---------------------------------------------------------------------------

def build_ranking_table(
    df_shap: pd.DataFrame,
    df_perm: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Merge SHAP and permutation rankings into one table.
    Compute a combined rank score = mean of both ranks.
    This is the table that goes directly into your thesis.
    """
    df = df_shap[["feature", "feature_label", "mean_abs_shap", "rank"]].rename(
        columns={"rank": "shap_rank", "mean_abs_shap": "shap_score"}
    )
    df_p = df_perm[["feature", "importance", "std", "rank"]].rename(
        columns={"rank": "perm_rank", "importance": "perm_importance", "std": "perm_std"}
    )
    df = df.merge(df_p, on="feature", how="outer")
    df["combined_rank"] = (df["shap_rank"].fillna(len(feature_cols)) +
                           df["perm_rank"].fillna(len(feature_cols))) / 2
    df = df.sort_values("combined_rank").reset_index(drop=True)
    df["final_rank"] = range(1, len(df) + 1)

    print(f"\n  {'='*60}")
    print(f"  Combined feature ranking (RQ1 answer)")
    print(f"  {'='*60}")
    print(f"  {'Rank':<5} {'Feature':<25} {'SHAP':>8} {'ΔF1':>8} {'Perm±':>8}")
    print(f"  {'-'*58}")
    for _, row in df.iterrows():
        print(f"  {int(row['final_rank']):<5} "
              f"{row['feature_label']:<25} "
              f"{row['shap_score']:>8.4f} "
              f"{row['perm_importance']:>8.4f} "
              f"±{row['perm_std']:>6.4f}")

    # ---- Dual-axis bar chart ----
    fig, ax1 = plt.subplots(figsize=(8, max(3.5, len(df) * 0.65)))
    y_pos = np.arange(len(df))
    ax2   = ax1.twiny()

    # SHAP bars (bottom axis)
    shap_norm = df["shap_score"] / (df["shap_score"].max() + 1e-8)
    ax1.barh(y_pos - 0.2, shap_norm, height=0.35,
             color=PALETTE["purple"], alpha=0.85, label="SHAP (normalised)")

    # Permutation bars (top axis)
    perm_norm = df["perm_importance"] / (df["perm_importance"].abs().max() + 1e-8)
    ax2.barh(y_pos + 0.2, perm_norm, height=0.35,
             color=PALETTE["teal"],   alpha=0.85, label="Permutation ΔF1 (normalised)")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df["feature_label"])
    ax1.invert_yaxis()
    ax1.set_xlabel("SHAP importance (normalised)", color=PALETTE["purple"])
    ax2.set_xlabel("Permutation importance (normalised)", color=PALETTE["teal"])
    ax1.set_title("Feature importance — SHAP vs permutation", pad=18)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
               frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "combined_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    df.to_csv(out_dir / "feature_ranking.csv", index=False)
    print(f"\n  Feature ranking table → {out_dir / 'feature_ranking.csv'}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {out_dir}\n")

    # ---- Load normalisation stats ----
    with open(args.norm) as f:
        norm_stats = json.load(f)

    feature_cols = list(norm_stats.keys())
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # ---- Build val loader ----
    train_loader, val_loader, test_loader, info = build_dataloaders(
        csv_path=args.csv,
        batch_size=32,
        mode="sequence",
        num_workers=0,
        stats_save_path=None,
    )
    # Override normalisation with saved stats (avoid re-fitting)
    # Re-build val dataset with saved norm stats
    df_full = pd.read_csv(args.csv)
    all_players = df_full[PLAYER_COL].unique().tolist()
    _, vl_pl, _ = player_wise_split(all_players)
    df_val = df_full[df_full[PLAYER_COL].isin(vl_pl)].copy()
    df_val = apply_normalization(df_val, norm_stats, feature_cols)
    ds_val = FatigueDataset(df_val, feature_cols=feature_cols, mode="sequence")
    from torch.utils.data import DataLoader
    val_loader = DataLoader(ds_val, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # ---- Load model ----
    input_dim = len(feature_cols)
    model = build_model("attention", input_dim=input_dim, device=device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {args.ckpt}\n")

    # ---- 1. SHAP ----
    print("=" * 55)
    print("1. SHAP values")
    print("=" * 55)
    shap_vals, X_flat, y_flat = compute_shap_values(
        model, val_loader, feature_cols, device
    )
    df_shap = plot_shap_summary(shap_vals, X_flat, y_flat, feature_cols, out_dir)

    # ---- 2. Permutation importance ----
    print("\n" + "=" * 55)
    print("2. Permutation importance")
    print("=" * 55)
    df_perm = permutation_importance(
        model, val_loader, feature_cols, device, n_repeats=10
    )
    plot_permutation_importance(df_perm, out_dir)

    # ---- 3. Attention weights ----
    print("\n" + "=" * 55)
    print("3. Attention weight analysis")
    print("=" * 55)
    df_attn = analyze_attention_weights(
        model, val_loader, feature_cols, device, out_dir
    )

    # ---- 4. Combined ranking ----
    print("\n" + "=" * 55)
    print("4. Combined feature ranking (RQ1)")
    print("=" * 55)
    df_rank = build_ranking_table(df_shap, df_perm, feature_cols, out_dir)

    # ---- Save summary JSON ----
    summary = {
        "shap_ranking":        df_shap[["feature", "mean_abs_shap", "rank"]].to_dict("records"),
        "perm_ranking":        df_perm[["feature", "importance", "std", "rank"]].to_dict("records"),
        "combined_ranking":    df_rank[["feature", "final_rank"]].to_dict("records"),
        "attention_stats": {
            "mean_normal":   float(df_attn[df_attn["label"] == 0]["attn_weight"].mean()),
            "mean_fatigued": float(df_attn[df_attn["label"] == 1]["attn_weight"].mean())
            if (df_attn["label"] == 1).any() else None,
        },
    }
    with open(out_dir / "interpretability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print(f"All interpretability outputs saved to:")
    print(f"  {out_dir}")
    print(f"  ├── shap_importance.png")
    print(f"  ├── shap_distribution.png")
    print(f"  ├── permutation_importance.png")
    print(f"  ├── attention_by_position.png")
    print(f"  ├── attention_fatigue_vs_normal.png")
    print(f"  ├── attention_heatmaps.png")
    print(f"  ├── attention_per_window.csv")
    print(f"  ├── feature_ranking.csv")
    print(f"  └── interpretability_summary.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpretability analysis for fatigue model")
    parser.add_argument("--csv",  required=True, help="Path to labelled windows CSV")
    parser.add_argument("--ckpt", required=True, help="Path to best_model.pt checkpoint")
    parser.add_argument("--norm", required=True, help="Path to norm_stats.json")
    parser.add_argument("--out",  default=None,  help="Output directory (default: next to ckpt)")
    args = parser.parse_args()

    if args.out is None:
        args.out = str(Path(args.ckpt).parent / "interpretability")

    main(args)