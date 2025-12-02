#!/usr/bin/env python3
"""
Evaluate GNN Anomaly Detection Performance
Purpose: Compute ROC/AUC metrics for GNN-based anomaly scores
Compare against baseline L2 method
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def load_labels(labels_path='data/splits/ucsd_ped2_labels.json'):
    """Load frame-level anomaly labels."""
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return labels


def load_gnn_scores(score_dir='data/processed/gnn_scores'):
    """Load GNN anomaly scores for all test sequences."""
    score_dir = Path(score_dir)
    score_files = sorted(score_dir.glob('Test*_gnn_scores.npy'))
    
    scores = {}
    for score_file in score_files:
        seq_name = score_file.stem.replace('_gnn_scores', '')
        scores[seq_name] = np.load(score_file)
    
    return scores


def load_baseline_scores(score_dir='data/processed/baseline_scores'):
    """Load baseline L2 scores for comparison."""
    score_dir = Path(score_dir)
    score_files = sorted(score_dir.glob('Test*_scores.npy'))
    
    scores = {}
    for score_file in score_files:
        seq_name = score_file.stem.replace('_scores', '')
        scores[seq_name] = np.load(score_file)
    
    return scores


def evaluate_method(scores, labels, method_name='GNN'):
    """
    Evaluate anomaly detection performance.
    
    Args:
        scores (dict): Anomaly scores per sequence
        labels (dict): Ground truth labels per sequence
        method_name (str): Name of method for display
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n📊 Evaluating {method_name} Performance")
    print("-" * 70)
    
    results = {}
    all_scores = []
    all_labels = []
    
    # Process each sequence
    for seq_name in sorted(scores.keys()):
        if seq_name not in labels:
            print(f"  ⚠️  {seq_name}: No labels found")
            continue
        
        seq_scores = scores[seq_name]
        seq_labels = np.array(labels[seq_name])
        
        # Check alignment
        if len(seq_scores) != len(seq_labels):
            print(f"  ⚠️  {seq_name}: Score/label mismatch ({len(seq_scores)} vs {len(seq_labels)})")
            continue
        
        # Compute ROC/AUC for this sequence
        fpr, tpr, thresholds = roc_curve(seq_labels, seq_scores)
        roc_auc = auc(fpr, tpr)
        
        results[seq_name] = {
            'auc': roc_auc,
            'num_frames': len(seq_scores),
            'num_anomalies': int(seq_labels.sum())
        }
        
        print(f"  {seq_name}: AUC = {roc_auc:.4f} ({results[seq_name]['num_anomalies']} anomalies)")
        
        # Accumulate for overall metrics
        all_scores.extend(seq_scores)
        all_labels.extend(seq_labels)
    
    # Compute overall AUC
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    fpr_overall, tpr_overall, _ = roc_curve(all_labels, all_scores)
    auc_overall = auc(fpr_overall, tpr_overall)
    
    print("-" * 70)
    print(f"  📈 Overall AUC: {auc_overall:.4f}")
    
    # Compute summary statistics
    per_seq_aucs = [v['auc'] for v in results.values()]
    
    summary = {
        'method': method_name,
        'overall_auc': float(auc_overall),
        'mean_auc': float(np.mean(per_seq_aucs)),
        'std_auc': float(np.std(per_seq_aucs)),
        'min_auc': float(np.min(per_seq_aucs)),
        'max_auc': float(np.max(per_seq_aucs)),
        'num_sequences': len(results),
        'total_frames': len(all_labels),
        'total_anomalies': int(all_labels.sum()),
        'per_sequence': results,
        'fpr': fpr_overall.tolist(),
        'tpr': tpr_overall.tolist()
    }
    
    return summary


def plot_comparison(gnn_summary, baseline_summary, output_path='reports/gnn_vs_baseline_roc.png'):
    """
    Plot ROC curves comparing GNN vs Baseline.
    
    Args:
        gnn_summary (dict): GNN evaluation results
        baseline_summary (dict): Baseline evaluation results
        output_path (str): Where to save plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot GNN ROC
    plt.plot(gnn_summary['fpr'], gnn_summary['tpr'], 
             label=f"GNN (AUC = {gnn_summary['overall_auc']:.4f})",
             linewidth=2, color='blue')
    
    # Plot Baseline ROC
    plt.plot(baseline_summary['fpr'], baseline_summary['tpr'],
             label=f"Baseline L2 (AUC = {baseline_summary['overall_auc']:.4f})",
             linewidth=2, color='red', linestyle='--')
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5000)')
    
    # Formatting
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: GNN vs Baseline L2', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 ROC plot saved: {output_path}")
    
    plt.close()


def main():
    """Main evaluation function."""
    print("\n" + "=" * 70)
    print("📊 GNN ANOMALY DETECTION EVALUATION")
    print("=" * 70)
    
    # Load labels
    labels = load_labels()
    print(f"\n✓ Loaded labels for {len(labels)} test sequences")
    
    # Load GNN scores
    try:
        gnn_scores = load_gnn_scores()
        print(f"✓ Loaded GNN scores for {len(gnn_scores)} sequences")
    except Exception as e:
        print(f"❌ Failed to load GNN scores: {e}")
        return
    
    # Load baseline scores (for comparison)
    try:
        baseline_scores = load_baseline_scores()
        print(f"✓ Loaded baseline scores for {len(baseline_scores)} sequences")
        has_baseline = True
    except Exception as e:
        print(f"⚠️  Baseline scores not found (skipping comparison)")
        has_baseline = False
    
    # Evaluate GNN
    gnn_summary = evaluate_method(gnn_scores, labels, method_name='GNN')
    
    # Evaluate baseline (if available)
    if has_baseline:
        baseline_summary = evaluate_method(baseline_scores, labels, method_name='Baseline L2')
        
        # Compare
        print("\n" + "=" * 70)
        print("📈 PERFORMANCE COMPARISON")
        print("=" * 70)
        print(f"   GNN:      AUC = {gnn_summary['overall_auc']:.4f}")
        print(f"   Baseline: AUC = {baseline_summary['overall_auc']:.4f}")
        improvement = (gnn_summary['overall_auc'] - baseline_summary['overall_auc']) * 100
        print(f"   Improvement: {improvement:+.2f} percentage points")
        
        if gnn_summary['overall_auc'] > baseline_summary['overall_auc']:
            print(f"\n🎉 GNN OUTPERFORMS BASELINE by {improvement:.2f}%!")
        elif gnn_summary['overall_auc'] < baseline_summary['overall_auc']:
            print(f"\n⚠️  GNN underperforms baseline by {-improvement:.2f}%")
        else:
            print(f"\n   GNN and baseline perform equally")
        
        # Plot comparison
        plot_comparison(gnn_summary, baseline_summary)
    else:
        baseline_summary = None
    
    # Save results
    output_dir = Path('data/processed/evaluation_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'gnn': gnn_summary,
        'baseline': baseline_summary
    }
    
    output_path = output_dir / 'gnn_evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   • GNN Overall AUC: {gnn_summary['overall_auc']:.4f}")
    print(f"   • GNN Mean AUC: {gnn_summary['mean_auc']:.4f} ± {gnn_summary['std_auc']:.4f}")
    print(f"   • Best sequence: {gnn_summary['max_auc']:.4f}")
    print(f"   • Worst sequence: {gnn_summary['min_auc']:.4f}")
    
    if has_baseline and gnn_summary['overall_auc'] > 0.80:
        print(f"\n🏆 EXCELLENT! GNN achieved >80% AUC!")
    elif gnn_summary['overall_auc'] > 0.70:
        print(f"\n👍 GOOD! GNN achieved >70% AUC")
    elif gnn_summary['overall_auc'] > 0.60:
        print(f"\n   GNN achieved reasonable performance (>60% AUC)")
    else:
        print(f"\n⚠️  GNN performance below expectations (<60% AUC)")
        print(f"   Consider: longer training, hyperparameter tuning, or architecture changes")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
