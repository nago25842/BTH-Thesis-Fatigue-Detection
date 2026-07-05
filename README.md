# Detecting Cognitive Fatigue in Esports via Behavioral Telemetry

This repository contains the source code, feature engineering pipeline, and model architecture for the Bachelor's Thesis: *"Detecting Cognitive Fatigue in Esports via Behavioral Telemetry: A Deep Learning Approach using StarCraft II Replay Data."*

## Project Overview
This project explores the automated detection of cognitive fatigue in professional *StarCraft II* players using non-invasive behavioral telemetry. The system utilizes an Attention-BiLSTM architecture to identify temporal patterns of performance degradation in 2-minute sliding windows extracted from match replay files.

## Repository Structure
- `dataset.py`: Logic for data loading, preprocessing, and sequence windowing.
- `model.py`: Neural network architecture definitions, including the Attention-BiLSTM and baseline models.
- `train.py`: Training routines, class-weighted binary cross-entropy loss, and validation loops.
- `evaluate.py`: Evaluation scripts for test-set performance, including AUC-ROC and PR-curve generation.
- `interpret.py`: Implementation of SHAP attribution and permutation importance for auditing decision logic.
- `STEP3_extract_features.py`: Data engineering pipeline to parse raw `.SC2Replay` files into tabular behavioral features.
- `STEP4_label_data.py`: Automated proxy-labeling module for performance degradation.
- `STEP5_verify_dataset.py`: Verification scripts to ensure integrity of processed sequences.
- `features_labeled.csv`: Sample of the processed, labeled feature dataset.
- `replay_metadata.json`: Metadata mapping for the provided sample replays.

## Requirements
To run the analysis, ensure the following dependencies are installed:
- `python >= 3.8`
- `torch`
- `scikit-learn`
- `pandas`
- `numpy`
- `shap`
- `sc2reader`

Install via:
```bash
pip install torch scikit-learn pandas numpy shap sc2reader
