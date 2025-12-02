#!/usr/bin/env python3
"""Quick script to verify GNN AUC calculation."""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Load labels
with open('data/splits/ucsd_ped2_labels.json', 'r') as f:
    labels_dict = json.load(f)

# Load all GNN scores
all_scores = []
all_labels = []

gnn_dir = Path('data/processed/gnn_scores')
for score_file in sorted(gnn_dir.glob('Test*_gnn_scores.npy')):
    seq_name = score_file.stem.replace('_gnn_scores', '')
    
    if seq_name in labels_dict:
        scores = np.load(score_file)
        labs = np.array(labels_dict[seq_name])
        
        print(f"{seq_name}: {len(scores)} scores, {len(labs)} labels, "
              f"score range [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Check if sequence has both normal and anomaly frames
        if len(np.unique(labs)) > 1:
            fpr, tpr, _ = roc_curve(labs, scores)
            seq_auc = auc(fpr, tpr)
            print(f"  → Per-sequence AUC: {seq_auc:.4f}")
        else:
            print(f"  → Skipped (all frames are {('normal' if labs[0] == 0 else 'anomaly')})")
        
        all_scores.extend(scores)
        all_labels.extend(labs)

# Compute overall AUC
all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

print(f"\n{'='*70}")
print(f"Total frames: {len(all_labels)}")
print(f"Normal frames: {(all_labels == 0).sum()}")
print(f"Anomaly frames: {(all_labels == 1).sum()}")

fpr, tpr, _ = roc_curve(all_labels, all_scores)
overall_auc = auc(fpr, tpr)

print(f"\n🎯 Overall GNN AUC: {overall_auc:.4f} ({overall_auc*100:.2f}%)")
print(f"{'='*70}")
