"""
dataset.py
----------
FatigueDataset and data loading utilities for:
  "Predicting Player Fatigue in Esports Using Attention-Based
   Sequence Models and Behavioral Telemetry"

Expected CSV columns (one row = one 2-minute window):
    player_id       : str   – unique player identifier
    match_id        : str   – replay identifier
    window_idx      : int   – sequential index within match (0-based)
    apm             : float – actions per minute (mean over window)
    apm_variance    : float – variance of per-second APM
    action_gap_mean : float – mean gap between actions (seconds)
    action_gap_std  : float – std of action gaps
    error_rate      : float – proxy error metric (misclicks / total actions)
    resource_eff    : float – resource efficiency score
    pause_freq      : float – pause events per minute
    fatigue_label   : int   – 0 = normal, 1 = fatigued (automated proxy)
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "apm",
    "apm_variance",
    "action_gap_mean",
    #"action_gap_std",
    "error_rate",
    #"resource_eff",
    #"pause_freq",
]

LABEL_COL = "fatigue_binary"
PLAYER_COL = "player_id"
MATCH_COL = "replay_id"
WINDOW_COL = "window_idx"

# Minimum windows a match must have to be included
MIN_WINDOWS_PER_MATCH = 3

# Random seed for reproducibility
SEED = 42


# ---------------------------------------------------------------------------
# Helper: player-wise split
# ---------------------------------------------------------------------------

def player_wise_split(
    player_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split unique player IDs into train / val / test sets.

    Player-wise splitting ensures no player appears in more than one
    partition, preventing data leakage from the same player's style
    being seen in both training and evaluation.

    Args:
        player_ids  : list of unique player identifier strings
        train_ratio : fraction of players for training (default 0.70)
        val_ratio   : fraction of players for validation (default 0.15)
        seed        : random seed for reproducibility

    Returns:
        (train_players, val_players, test_players)
    """
    players = sorted(set(player_ids))  # sort for determinism before shuffle
    rng = random.Random(seed)
    rng.shuffle(players)

    n = len(players)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_players = players[:n_train]
    val_players = players[n_train : n_train + n_val]
    test_players = players[n_train + n_val :]

    return train_players, val_players, test_players


# ---------------------------------------------------------------------------
# Helper: per-feature normalisation statistics
# ---------------------------------------------------------------------------

def compute_normalization_stats(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std for each feature column.
    Call this on the TRAINING split only, then apply to all splits
    to prevent leakage.

    Returns:
        dict  {feature_name: {"mean": float, "std": float}}
    """
    stats = {}
    for col in feature_cols:
        mean = float(df[col].mean())
        std = float(df[col].std())
        if std < 1e-8:
            std = 1.0  # avoid division by zero for constant features
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_normalization(
    df: pd.DataFrame,
    stats: Dict[str, Dict[str, float]],
    feature_cols: List[str] = FEATURE_COLS,
) -> pd.DataFrame:
    """
    Z-score normalise feature columns using pre-computed stats.
    Returns a copy of df with normalised features.
    """
    df = df.copy()
    for col in feature_cols:
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        df[col] = (df[col] - mean) / std
    return df


# ---------------------------------------------------------------------------
# Core Dataset class
# ---------------------------------------------------------------------------

class FatigueDataset(Dataset):
    """
    PyTorch Dataset for temporal fatigue prediction.

    Each sample is one match (sequence of 2-minute windows).
    The model receives the full sequence; loss is computed at every
    time step (sequence labelling), or at the final step (last-step
    classification) depending on `mode`.

    Args:
        df          : DataFrame filtered to the players in this split
        feature_cols: list of column names to use as input features
        label_col   : column name for the binary fatigue label
        mode        : "sequence" (label per window) or "last" (final
                      window only — useful for binary match-level output)
        min_len     : drop matches with fewer than this many windows

    Attributes:
        sequences   : list of (T, F) float32 tensors  [T=time, F=features]
        labels      : list of (T,) int64 tensors  (or scalar for "last")
        match_ids   : list of match_id strings for traceability
        player_ids  : list of player_id strings for traceability
        class_weights: (2,) tensor for weighted loss / sampler
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = FEATURE_COLS,
        label_col: str = LABEL_COL,
        mode: str = "sequence",
        min_len: int = MIN_WINDOWS_PER_MATCH,
    ):
        assert mode in ("sequence", "last"), \
            f"mode must be 'sequence' or 'last', got '{mode}'"

        self.feature_cols = feature_cols
        self.label_col = label_col
        self.mode = mode

        self.sequences: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.match_ids: List[str] = []
        self.player_ids: List[str] = []

        self._build(df, min_len)
        self.class_weights = self._compute_class_weights()

    # ------------------------------------------------------------------
    def _build(self, df: pd.DataFrame, min_len: int) -> None:
        """Group by match, sort by window index, build tensors."""
        grouped = df.sort_values(WINDOW_COL).groupby(MATCH_COL)

        for match_id, group in grouped:
            if len(group) < min_len:
                continue

            features = group[self.feature_cols].values.astype(np.float32)
            labels = group[self.label_col].values.astype(np.int64)
            player = group[PLAYER_COL].iloc[0]

            feat_tensor = torch.tensor(features)   # (T, F)
            if self.mode == "sequence":
                label_tensor = torch.tensor(labels)       # (T,)
            else:
                label_tensor = torch.tensor(labels[-1])   # scalar

            self.sequences.append(feat_tensor)
            self.labels.append(label_tensor)
            self.match_ids.append(str(match_id))
            self.player_ids.append(str(player))

    # ------------------------------------------------------------------
    def _compute_class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights for the two classes.
        Used for WeightedRandomSampler and BCEWithLogitsLoss.
        """
        if self.mode == "sequence":
            all_labels = torch.cat(self.labels)
        else:
            all_labels = torch.stack(self.labels)

        n_total = len(all_labels)
        n_pos = int(all_labels.sum().item())
        n_neg = n_total - n_pos

        if n_pos == 0 or n_neg == 0:
            return torch.tensor([1.0, 1.0])

        w_neg = n_total / (2.0 * n_neg)
        w_pos = n_total / (2.0 * n_pos)
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "features": self.sequences[idx],    # (T, F)
            "labels":   self.labels[idx],        # (T,) or scalar
            "match_id": self.match_ids[idx],
            "player_id": self.player_ids[idx],
            "length":   self.sequences[idx].shape[0],  # T (for masking)
        }

    # ------------------------------------------------------------------
    def fatigue_rate(self) -> float:
        """Fraction of windows labelled as fatigued (for reporting)."""
        if self.mode == "sequence":
            all_labels = torch.cat(self.labels)
        else:
            all_labels = torch.stack(self.labels)
        return float(all_labels.float().mean().item())

    def summary(self) -> str:
        lengths = [s.shape[0] for s in self.sequences]
        return (
            f"  Matches     : {len(self.sequences)}\n"
            f"  Windows     : {sum(lengths)}\n"
            f"  Seq len     : min={min(lengths)}, "
            f"max={max(lengths)}, mean={np.mean(lengths):.1f}\n"
            f"  Fatigue rate: {self.fatigue_rate()*100:.2f}%\n"
            f"  Class weights [neg, pos]: "
            f"[{self.class_weights[0]:.3f}, {self.class_weights[1]:.3f}]"
        )


# ---------------------------------------------------------------------------
# Collate function for variable-length sequences
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Pads sequences in a batch to the same length.

    Returns a dict with:
        features  : (B, T_max, F) padded float tensor
        labels    : (B, T_max) padded int tensor  (or (B,) for 'last' mode)
        lengths   : (B,) int tensor of original sequence lengths
        mask      : (B, T_max) bool tensor — True where data exists
        match_ids : list of str
        player_ids: list of str
    """
    features = [item["features"] for item in batch]    # list of (T_i, F)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    # Pad sequences along T dimension
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    # (B, T_max, F)

    # Build padding mask: True = real data, False = pad
    B, T_max, _ = features_padded.shape
    mask = torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)
    # (B, T_max)

    # Labels: handle both sequence and scalar modes
    first_label = batch[0]["labels"]
    if first_label.dim() == 0:
        # "last" mode — scalar labels
        labels = torch.stack([item["labels"] for item in batch])
    else:
        # "sequence" mode — pad label sequences too
        labels_padded = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,  # -100 is ignored by CrossEntropyLoss
        )
        labels = labels_padded

    return {
        "features":   features_padded,
        "labels":     labels,
        "lengths":    lengths,
        "mask":       mask,
        "match_ids":  [item["match_id"]  for item in batch],
        "player_ids": [item["player_id"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Top-level factory: load CSV → splits → DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    mode: str = "sequence",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    feature_cols: List[str] = FEATURE_COLS,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    stats_save_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Full pipeline: read CSV → player split → normalise → datasets →
    DataLoaders.

    Args:
        csv_path             : path to the labelled windows CSV
        batch_size           : samples per batch
        mode                 : "sequence" or "last"
        train_ratio          : fraction of players in train set
        val_ratio            : fraction of players in val set
        feature_cols         : list of feature column names
        num_workers          : DataLoader workers (0 = main process)
        use_weighted_sampler : oversample minority class in training
        stats_save_path      : if given, save normalisation stats as JSON

    Returns:
        (train_loader, val_loader, test_loader, info_dict)

        info_dict contains split sizes, class weights, norm stats, etc.
    """
    # ---- Load ----
    df = pd.read_csv(csv_path)
    _validate_columns(df, feature_cols)

    # ---- Drop short matches ----
    match_lengths = df.groupby(MATCH_COL)[WINDOW_COL].count()
    valid_matches = match_lengths[match_lengths >= MIN_WINDOWS_PER_MATCH].index
    df = df[df[MATCH_COL].isin(valid_matches)].copy()

    # ---- Player-wise split ----
    all_players = df[PLAYER_COL].unique().tolist()
    train_players, val_players, test_players = player_wise_split(
        all_players, train_ratio=train_ratio, val_ratio=val_ratio
    )

    df_train = df[df[PLAYER_COL].isin(train_players)].copy()
    df_val   = df[df[PLAYER_COL].isin(val_players)].copy()
    df_test  = df[df[PLAYER_COL].isin(test_players)].copy()

    # ---- Normalise (fit on train only) ----
    norm_stats = compute_normalization_stats(df_train, feature_cols)
    df_train = apply_normalization(df_train, norm_stats, feature_cols)
    df_val   = apply_normalization(df_val,   norm_stats, feature_cols)
    df_test  = apply_normalization(df_test,  norm_stats, feature_cols)

    if stats_save_path:
        Path(stats_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stats_save_path, "w") as f:
            json.dump(norm_stats, f, indent=2)

    # ---- Build datasets ----
    ds_train = FatigueDataset(df_train, feature_cols, mode=mode)
    ds_val   = FatigueDataset(df_val,   feature_cols, mode=mode)
    ds_test  = FatigueDataset(df_test,  feature_cols, mode=mode)

    # ---- Samplers ----
    train_sampler = None
    if use_weighted_sampler and mode == "sequence":
        # Weight each sequence by the proportion of fatigued windows
        sample_weights = _sequence_sample_weights(ds_train)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    # ---- DataLoaders ----
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    info = {
        "n_players":       {"train": len(train_players), "val": len(val_players), "test": len(test_players)},
        "n_matches":       {"train": len(ds_train), "val": len(ds_val), "test": len(ds_test)},
        "n_features":      len(feature_cols),
        "feature_cols":    feature_cols,
        "class_weights":   ds_train.class_weights.tolist(),
        "norm_stats":      norm_stats,
        "fatigue_rate":    {
            "train": ds_train.fatigue_rate(),
            "val":   ds_val.fatigue_rate(),
            "test":  ds_test.fatigue_rate(),
        },
    }

    return train_loader, val_loader, test_loader, info


# ---------------------------------------------------------------------------
# 5-fold cross-validation split generator
# ---------------------------------------------------------------------------

def kfold_player_splits(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = SEED,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate k player-wise cross-validation folds.

    Each fold yields (train_df, val_df).  Test set remains separate and
    is NOT touched during CV — call build_dataloaders for final eval.

    Usage:
        folds = kfold_player_splits(df_trainval, k=5)
        for fold_idx, (df_tr, df_vl) in enumerate(folds):
            ds_tr = FatigueDataset(df_tr)
            ds_vl = FatigueDataset(df_vl)
            ...
    """
    players = sorted(df[PLAYER_COL].unique())
    rng = random.Random(seed)
    rng.shuffle(players)

    fold_size = len(players) // k
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else len(players)
        val_players = players[val_start:val_end]
        train_players = [p for p in players if p not in set(val_players)]

        df_tr = df[df[PLAYER_COL].isin(train_players)].copy()
        df_vl = df[df[PLAYER_COL].isin(val_players)].copy()
        folds.append((df_tr, df_vl))

    return folds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, feature_cols: List[str]) -> None:
    required = set(feature_cols) | {LABEL_COL, PLAYER_COL, MATCH_COL, WINDOW_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}\n"
            f"Found: {sorted(df.columns.tolist())}"
        )


def _sequence_sample_weights(ds: FatigueDataset) -> torch.Tensor:
    """
    Per-sequence weight = fraction of fatigued windows in that sequence.
    Sequences with no fatigue get a small base weight so they're still
    sampled occasionally.
    """
    weights = []
    for label_seq in ds.labels:
        frac = float(label_seq.float().mean().item())
        weights.append(max(frac, 0.05))  # floor at 5% so normal seqs appear
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Quick smoke-test  (run: python dataset.py --csv your_file.csv)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the data loader")
    parser.add_argument("--csv", required=True, help="Path to labelled windows CSV")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mode", default="sequence", choices=["sequence", "last"])
    args = parser.parse_args()

    print("=" * 60)
    print("FatigueDataset smoke test")
    print("=" * 60)

    train_loader, val_loader, test_loader, info = build_dataloaders(
        csv_path=args.csv,
        batch_size=args.batch_size,
        mode=args.mode,
        stats_save_path="experiments/norm_stats.json",
    )

    print(f"\nPlayers  — train: {info['n_players']['train']}, "
          f"val: {info['n_players']['val']}, test: {info['n_players']['test']}")
    print(f"Matches  — train: {info['n_matches']['train']}, "
          f"val: {info['n_matches']['val']}, test: {info['n_matches']['test']}")
    print(f"Features : {info['n_features']}")
    print(f"Fatigue rate — train: {info['fatigue_rate']['train']*100:.2f}%, "
          f"val: {info['fatigue_rate']['val']*100:.2f}%, "
          f"test: {info['fatigue_rate']['test']*100:.2f}%")
    print(f"Class weights [neg, pos]: {info['class_weights']}")

    print("\n--- First training batch ---")
    batch = next(iter(train_loader))
    print(f"  features shape : {batch['features'].shape}")
    print(f"  labels shape   : {batch['labels'].shape}")
    print(f"  lengths        : {batch['lengths'].tolist()}")
    print(f"  mask shape     : {batch['mask'].shape}")
    print(f"  sample player  : {batch['player_ids'][0]}")

    print("\nDataset summaries:")
    train_ds = train_loader.dataset
    print("TRAIN:\n" + train_ds.summary())
    val_ds = val_loader.dataset
    print("VAL:\n"   + val_ds.summary())
    test_ds = test_loader.dataset
    print("TEST:\n"  + test_ds.summary())

    print("\n✓ All checks passed.")