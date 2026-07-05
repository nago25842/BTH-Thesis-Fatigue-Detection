"""
STEP 3 — Feature Extraction from SC2EGSet JSON Dataset
======================================================
Updated to process JSON telemetry instead of raw .SC2Replay files.
"""

import pandas as pd
import numpy as np
import os
import json
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
REPLAY_DIR   = "C:/Thesis/data/replays/"       # Updated to your local path
OUTPUT_CSV   = "./data/features_raw.csv"
OUTPUT_META  = "./data/replay_metadata.json"
WINDOW_SEC   = 120                      
MIN_DURATION = 15 * 60                  
MAX_DURATION = 30 * 60                  
MAX_REPLAYS  = 5000                     

def get_replay_files(directory):
    """Collect all .json match file paths recursively."""
    # This looks through ALL subfolders, no matter how deep
    files = glob.glob(os.path.join(directory, "**/*.json"), recursive=True)
    
    # Filter out summary, mapping, and log files so we only get raw match data
    clean_files = [
        f for f in files 
        if "_summary" not in f 
        and "_mapping" not in f 
        and "_failed" not in f
    ]
    
    return clean_files[:MAX_REPLAYS]

def extract_features_from_json(file_path, replay_id):
    """Parses Raw Event JSON and calculates behavioral features."""
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    # 1. Timing Configuration
    loops_per_sec = 22.4  # Standard for 'Faster' game speed
    total_loops = data["header"]["elapsedGameLoops"]
    duration_sec = total_loops / loops_per_sec

    # Thesis Filter: 15-30 minutes
    if not (MIN_DURATION <= duration_sec <= MAX_DURATION):
        return None

    # 2. Get all events
    events = data.get("gameEvents", [])
    if not events:
        return None

    # 3. Identify the two main players
    # In professional JSONs, observers often have high userIDs. 
    # We take the two userIDs with the most events.
    from collections import Counter
    user_counts = Counter([e.get("userid", {}).get("userId") for e in events if e.get("userid")])
    top_players = [user_id for user_id, count in user_counts.most_common(2)]

    rows = []
    for p_id in top_players:
        # Filter events for this player
        p_events = [e for e in events if e.get("userid", {}).get("userId") == p_id]
        p_id_str = f"{replay_id}_p{p_id}"

        # Calculate Baseline APM (First 3 minutes)
        baseline_limit = 180 * loops_per_sec
        b_events = [e for e in p_events if e["loop"] < baseline_limit]
        # APM = (Actions / Seconds) * 60
        baseline_apm = (len(b_events) / 180.0) * 60.0 if b_events else 100.0

        t = 0
        window_idx = 0
        while t + WINDOW_SEC <= duration_sec:
            start_loop = t * loops_per_sec
            end_loop = (t + WINDOW_SEC) * loops_per_sec
            
            # Get events in this specific 2-minute window
            win_events = [e for e in p_events if start_loop <= e["loop"] < end_loop]
            
            if len(win_events) > 5: # Skip empty windows
                # --- FEATURE 1: APM ---
                apm = (len(win_events) / WINDOW_SEC) * 60.0
                
                # --- FEATURE 2: APM VARIANCE ---
                # Divide window into four 30s segments to see if they are getting "jittery"
                segments = []
                for s in range(0, WINDOW_SEC, 30):
                    s_l = (t + s) * loops_per_sec
                    e_l = (t + s + 30) * loops_per_sec
                    segments.append(len([e for e in win_events if s_l <= e["loop"] < e_l]))
                apm_variance = np.var(segments)

                # --- FEATURE 3: ERROR RATE (Camera Jumps) ---
                # Frequent, erratic camera updates often signal loss of focus/fatigue
                cams = [e for e in win_events if e["evtTypeName"] == "CameraUpdate"]
                camera_rate = len(cams) / len(win_events)

                # --- FEATURE 4: ACTION GAP ---
                # Average time (seconds) between events
                loops = sorted([e["loop"] for e in win_events])
                gaps = np.diff(loops) / loops_per_sec if len(loops) > 1 else [0]
                action_gap_mean = np.mean(gaps)

                rows.append({
                    "apm": round(apm, 2),
                    "apm_variance": round(apm_variance, 4),
                    "error_rate": round(camera_rate, 4),
                    "action_gap_mean": round(action_gap_mean, 4),
                    "apm_vs_baseline": round((apm - baseline_apm) / max(baseline_apm, 1), 4),
                    "replay_id": replay_id,
                    "player_id": p_id_str,
                    "window_idx": window_idx,
                    "match_pct": round(t / duration_sec, 3)
                })

            t += WINDOW_SEC
            window_idx += 1

    return rows, {"file": os.path.basename(file_path), "duration": duration_sec}

def main():
    print("=" * 55)
    print("  Phase 2 — JSON Feature Extraction")
    print("=" * 55)

    if not os.path.exists(REPLAY_DIR):
        print(f"❌ Path not found: {REPLAY_DIR}")
        return

    files = get_replay_files(REPLAY_DIR)
    print(f"📂 Found {len(files)} JSON replays.")

    all_rows = []
    meta = []
    
    for i, fpath in enumerate(tqdm(files, desc="Extracting")):
        res = extract_features_from_json(fpath, f"rep_{i:05d}")
        if res:
            data_rows, match_meta = res
            all_rows.extend(data_rows)
            meta.append(match_meta)

    os.makedirs("./data", exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"✅ Saved {len(all_rows)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()