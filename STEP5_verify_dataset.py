"""
STEP 5 — Dataset Verification
==============================
Final check to ensure the dataset is ready for the Attention-LSTM model.
"""

import pandas as pd
import os

INPUT_CSV = "./data/features_labeled.csv"

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Missing: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"📂 Verification Report for {len(df)} windows")
    print(f"── Class Balance: {df['fatigue_binary'].value_counts(normalize=True).to_dict()}")
    print(f"── Feature Check: APM mean = {df['apm'].mean():.2f}")
    
    if len(df) >= 1000:
        print("✅ Sample size sufficient for Bachelor's Thesis.")
    else:
        print("⚠️ Warning: Low sample size.")

if __name__ == "__main__":
    main()