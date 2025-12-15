#!/usr/bin/env python3
"""
ShanghaiTech Anomaly Detection Evaluation
Purpose: Evaluate GNN-based anomaly detection on ShanghaiTech dataset
Metrics: Frame-level AUC, precision, recall, F1-score

Evaluation Strategy:
1. Load trained GNN model (from UCSD Ped2 or ShanghaiTech-specific)
2. Compute reconstruction errors on ShanghaiTech test sequences
3. Generate anomaly scores and evaluate against ground truth
4. Compare cross-dataset transfer performance

Usage: python scripts/evaluate_shanghaitech_scores.py
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

class ShanghaiTechEvaluator:
    """Evaluate anomaly detection performance on ShanghaiTech"""
    
    def __init__(self):
        self.labels_file = Path('data/splits/shanghaitech_labels.json')
        self.scores_dir = Path('data/processed/shanghaitech/anomaly_scores')
        self.results_dir = Path('data/processed/shanghaitech/evaluation_results')
        
        # Create output directories
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_labels(self) -> Dict[str, List[int]]:
        """Load ground truth anomaly labels"""
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
        
        with open(self.labels_file, 'r') as f:
            return json.load(f)
    
    def load_anomaly_scores(self, method: str = 'gnn') -> Dict[str, np.ndarray]:
        """
        Load precomputed anomaly scores
        
        Args:
            method: Scoring method ('gnn', 'baseline', 'ensemble')
        """
        scores_file = self.scores_dir / f'{method}_anomaly_scores.npy'
        
        if not scores_file.exists():
            raise FileNotFoundError(
                f"Anomaly scores not found: {scores_file}\n"
                f"Please run the appropriate scoring script first."
            )
        
        scores = np.load(scores_file, allow_pickle=True).item()
        return scores
    
    def compute_metrics(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """
        Compute evaluation metrics
        
        Args:
            y_true: Ground truth labels (0/1)
            y_scores: Anomaly scores (continuous)
        
        Returns:
            Dictionary of metrics
        """
        # ROC AUC
        auc_score = roc_auc_score(y_true, y_scores)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        
        # Find best F1 threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_f1_idx]
        best_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else pr_thresholds[-1]
        
        # Metrics at best threshold
        y_pred = (y_scores >= best_threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        best_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        best_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        metrics = {
            'auc': float(auc_score),
            'best_f1': float(best_f1),
            'best_precision': float(best_precision),
            'best_recall': float(best_recall),
            'best_threshold': float(best_threshold),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        return metrics
    
    def evaluate_method(self, method: str = 'gnn') -> Dict:
        """
        Evaluate a specific anomaly detection method
        
        Args:
            method: Method name ('gnn', 'baseline', 'ensemble')
        """
        print(f"\n{'='*60}")
        print(f"Evaluating ShanghaiTech - Method: {method.upper()}")
        print(f"{'='*60}")
        
        # Load labels and scores
        print("\n📋 Loading data...")
        labels_dict = self.load_labels()
        
        try:
            scores_dict = self.load_anomaly_scores(method)
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
            return None
        
        # Aggregate all sequences
        all_labels = []
        all_scores = []
        sequence_results = {}
        
        print(f"\n🔍 Processing {len(labels_dict)} sequences...")
        
        for seq_name in sorted(labels_dict.keys()):
            if seq_name not in scores_dict:
                print(f"   ⚠️ Missing scores for {seq_name}")
                continue
            
            seq_labels = np.array(labels_dict[seq_name])
            seq_scores = scores_dict[seq_name]
            
            # Ensure matching lengths
            if len(seq_labels) != len(seq_scores):
                print(f"   ⚠️ Length mismatch for {seq_name}: "
                      f"labels={len(seq_labels)}, scores={len(seq_scores)}")
                continue
            
            all_labels.extend(seq_labels)
            all_scores.extend(seq_scores)
            
            # Per-sequence metrics
            if len(np.unique(seq_labels)) > 1:  # Only if contains both classes
                seq_metrics = self.compute_metrics(seq_labels, seq_scores)
                sequence_results[seq_name] = seq_metrics
        
        # Overall metrics
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total frames: {len(all_labels):,}")
        print(f"   Normal frames: {np.sum(all_labels == 0):,}")
        print(f"   Anomalous frames: {np.sum(all_labels == 1):,}")
        print(f"   Anomaly ratio: {np.mean(all_labels)*100:.2f}%")
        
        overall_metrics = self.compute_metrics(all_labels, all_scores)
        
        # Display results
        print(f"\n🎯 Overall Performance:")
        print(f"   AUC: {overall_metrics['auc']*100:.2f}%")
        print(f"   Best F1: {overall_metrics['best_f1']:.4f}")
        print(f"   Best Precision: {overall_metrics['best_precision']:.4f}")
        print(f"   Best Recall: {overall_metrics['best_recall']:.4f}")
        print(f"   Best Threshold: {overall_metrics['best_threshold']:.4f}")
        
        # Confusion matrix
        print(f"\n📈 Confusion Matrix:")
        print(f"   True Positives: {overall_metrics['true_positives']:,}")
        print(f"   False Positives: {overall_metrics['false_positives']:,}")
        print(f"   True Negatives: {overall_metrics['true_negatives']:,}")
        print(f"   False Negatives: {overall_metrics['false_negatives']:,}")
        
        # Save results
        results = {
            'method': method,
            'dataset': 'ShanghaiTech',
            'overall_metrics': overall_metrics,
            'sequence_metrics': sequence_results,
            'total_sequences': len(sequence_results),
            'total_frames': len(all_labels)
        }
        
        output_file = self.results_dir / f'{method}_evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return results
    
    def plot_roc_curve(self, method: str = 'gnn'):
        """Plot ROC curve for method"""
        labels_dict = self.load_labels()
        scores_dict = self.load_anomaly_scores(method)
        
        all_labels = []
        all_scores = []
        
        for seq_name in sorted(labels_dict.keys()):
            if seq_name in scores_dict:
                all_labels.extend(labels_dict[seq_name])
                all_scores.extend(scores_dict[seq_name])
        
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        auc = roc_auc_score(all_labels, all_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{method.upper()} (AUC={auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - ShanghaiTech Dataset')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = self.results_dir / f'{method}_roc_curve.png'
        plt.savefig(plot_file, dpi=300)
        print(f"   Plot saved: {plot_file}")
        plt.close()

def main():
    """Main execution function"""
    evaluator = ShanghaiTechEvaluator()
    
    # Check for available scoring methods
    methods_to_evaluate = []
    
    for method in ['gnn', 'baseline', 'ensemble']:
        scores_file = evaluator.scores_dir / f'{method}_anomaly_scores.npy'
        if scores_file.exists():
            methods_to_evaluate.append(method)
    
    if not methods_to_evaluate:
        print("="*60)
        print("⚠️ No anomaly scores found!")
        print("="*60)
        print("\nPlease generate anomaly scores first using:")
        print("  - GNN scoring script (for 'gnn' method)")
        print("  - Baseline scoring script (for 'baseline' method)")
        print("  - Ensemble scoring script (for 'ensemble' method)")
        return
    
    print("="*60)
    print("ShanghaiTech Anomaly Detection Evaluation")
    print("="*60)
    print(f"\nFound methods: {', '.join(methods_to_evaluate)}")
    
    # Evaluate each method
    for method in methods_to_evaluate:
        results = evaluator.evaluate_method(method)
        if results:
            evaluator.plot_roc_curve(method)
    
    print("\n" + "="*60)
    print("✅ ShanghaiTech evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
