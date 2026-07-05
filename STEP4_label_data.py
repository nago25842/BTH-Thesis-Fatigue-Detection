"""
STEP 4 — Automated Fatigue Label Generation
=============================================
Identifies fatigue by comparing late-game windows to early-game baselines.
"""

import pandas as pd
import numpy as np
import os

INPUT_CSV  = "./data/features_raw.csv"
OUTPUT_CSV = "./data/features_labeled.csv"

def compute_player_labels(player_df):
    player_df = player_df.sort_values("window_idx").copy()
    baseline = player_df.head(2) # First 4 minutes
    
    b_apm = max(baseline["apm"].mean(), 1.0)
    b_err = max(baseline["error_rate"].mean(), 0.001)
    b_gap = max(baseline["action_gap_mean"].mean(), 0.001)

    scores = []
    for _, row in player_df.iterrows():
        apm_drop = max(0, (b_apm - row["apm"]) / b_apm)
        err_rise = min(max(0, (row["error_rate"] - b_err) / b_err), 1.0)
        gap_rise = min(max(0, (row["action_gap_mean"] - b_gap) / b_gap), 1.0)

        # Weighted composite score
        score = (0.5 * apm_drop) + (0.3 * err_rise) + (0.2 * gap_rise)
        scores.append(round(min(score, 1.0), 4))

    player_df["fatigue_score"] = scores
    player_df["fatigue_binary"] = (player_df["fatigue_score"] >= 0.3).astype(int)
    return player_df

def main():
    if not os.path.exists(INPUT_CSV):
        print("❌ Run STEP 3 first.")
        return

    df = pd.read_csv(INPUT_CSV)
    labeled_parts = [compute_player_labels(g) for _, g in df.groupby("player_id")]
    labeled_df = pd.concat(labeled_parts, ignore_index=True)
    
    labeled_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Labeled dataset saved. Fatigued windows: {labeled_df['fatigue_binary'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()