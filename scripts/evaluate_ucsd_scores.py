#!/usr/bin/env python3
"""
evaluate_ucsd_scores.py - Modular Evaluation Script
Evaluates anomaly scores for UCSD Ped2 dataset with ROC/AUC metrics.
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    f1_score, precision_score, recall_score
)

class UCSDScoreEvaluator:
    """Modular evaluator for UCSD Ped2 baseline anomaly scores"""
    
    def __init__(self, scores_dir: str = 'data/processed/baseline_scores'):
        self.scores_dir = Path(scores_dir)
        self.labels_file = Path('data/splits/ucsd_ped2_labels.json')
        self.results_dir = Path('data/processed/evaluation_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluation_results = {}
    
    def load_labels(self) -> Dict[str, List[int]]:
        """Load ground truth labels"""
        print("Loading ground truth labels...")
        
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
        
        with open(self.labels_file) as f:
            labels_data = json.load(f)
        
        print(f"  ✓ Loaded labels for {len(labels_data)} sequences")
        return labels_data
    
    def load_scores(self) -> Dict[str, np.ndarray]:
        """Load anomaly scores"""
        print("Loading anomaly scores...")
        
        if not self.scores_dir.exists():
            raise FileNotFoundError(f"Scores directory not found: {self.scores_dir}")
        
        score_files = [f for f in self.scores_dir.iterdir() 
                      if f.suffix == '.npy' and f.stem.endswith('_scores')]
        
        scores = {}
        for score_file in sorted(score_files):
            seq_name = score_file.stem.replace('_scores', '')
            scores[seq_name] = np.load(score_file)
            print(f"  ✓ Loaded {seq_name}: {len(scores[seq_name])} scores")
        
        return scores
    
    def compute_sequence_metrics(self, scores: np.ndarray, labels: List[int]) -> Dict:
        """Compute metrics for a single sequence"""
        try:
            # Convert labels to numpy array
            labels_arr = np.array(labels)
            
            # Check if we have any positive samples
            if not np.any(labels_arr):
                print(f"    No anomalies found in sequence")
                return {
                    'auc': None,  # Use None instead of 0.0 for no anomalies
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'num_frames': len(labels),
                    'num_anomalies': 0,
                    'has_anomalies': False
                }
            
            # Check if all samples are positive (should not happen but handle gracefully)
            if np.all(labels_arr):
                print(f"    All frames are anomalous")
                return {
                    'auc': None,  # Use None for all-anomaly case
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1': 1.0,
                    'num_frames': len(labels),
                    'num_anomalies': int(np.sum(labels_arr)),
                    'has_anomalies': False  # Mark as invalid for AUC calculation
                }
            
            # Compute ROC AUC
            auc = roc_auc_score(labels_arr, scores)
            
            # Handle NaN AUC
            if np.isnan(auc):
                print(f"    AUC is NaN")
                return {
                    'auc': None,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'num_frames': len(labels),
                    'num_anomalies': int(np.sum(labels_arr)),
                    'has_anomalies': False,
                    'error': 'AUC is NaN'
                }
            
            # Find optimal threshold using Youden's index
            fpr, tpr, thresholds = roc_curve(labels_arr, scores)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Compute classification metrics at optimal threshold
            predictions = (scores >= optimal_threshold).astype(int)
            precision = precision_score(labels_arr, predictions, zero_division=0)
            recall = recall_score(labels_arr, predictions, zero_division=0)
            f1 = f1_score(labels_arr, predictions, zero_division=0)
            
            return {
                'auc': float(auc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'optimal_threshold': float(optimal_threshold),
                'num_frames': len(labels),
                'num_anomalies': int(np.sum(labels_arr)),
                'has_anomalies': True
            }
            
        except Exception as e:
            print(f"    ✗ Error computing metrics: {e}")
            return {
                'auc': None,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'num_frames': len(labels),
                'num_anomalies': int(np.sum(labels)) if len(labels) > 0 else 0,
                'has_anomalies': False,
                'error': str(e)
            }
    
    def evaluate_all_sequences(self, scores: Dict[str, np.ndarray], 
                             labels: Dict[str, List[int]]) -> Dict:
        """Evaluate all sequences and compute aggregate metrics"""
        print("Evaluating sequences...")
        
        sequence_results = {}
        valid_aucs = []
        total_frames = 0
        total_anomalies = 0
        
        for seq_name in sorted(scores.keys()):
            if seq_name not in labels:
                print(f"  {seq_name}: No labels found, skipping")
                continue
            
            print(f"  Evaluating {seq_name}...")
            
            # Get scores and labels for this sequence
            seq_scores = scores[seq_name]
            seq_labels = labels[seq_name]
            
            # Check length alignment
            if len(seq_scores) != len(seq_labels):
                print(f"    Length mismatch: {len(seq_scores)} scores vs {len(seq_labels)} labels")
                continue
            
            # Compute metrics
            metrics = self.compute_sequence_metrics(seq_scores, seq_labels)
            sequence_results[seq_name] = metrics
            
            # Track valid results
            if metrics['has_anomalies'] and metrics['auc'] is not None:
                valid_aucs.append(metrics['auc'])
                print(f"    ✓ AUC: {metrics['auc']:.4f}")
            elif metrics['auc'] is None:
                print(f"    ➖ AUC: N/A (no anomalies or invalid)")
            else:
                print(f"    AUC: {metrics.get('auc', 'N/A')}")
            
            total_frames += metrics['num_frames']
            total_anomalies += metrics['num_anomalies']
        
        # Compute aggregate statistics
        aggregate_stats = {
            'num_sequences_evaluated': len(sequence_results),
            'num_sequences_with_anomalies': len(valid_aucs),
            'total_frames': total_frames,
            'total_anomalies': total_anomalies,
            'anomaly_rate': total_anomalies / total_frames if total_frames > 0 else 0.0
        }
        
        if valid_aucs:
            aggregate_stats.update({
                'mean_auc': float(np.mean(valid_aucs)),
                'std_auc': float(np.std(valid_aucs)),
                'min_auc': float(np.min(valid_aucs)),
                'max_auc': float(np.max(valid_aucs))
            })
        else:
            aggregate_stats.update({
                'mean_auc': 0.0,
                'std_auc': 0.0,
                'min_auc': 0.0,
                'max_auc': 0.0
            })
        
        return {
            'sequence_results': sequence_results,
            'aggregate_stats': aggregate_stats
        }
    
    def save_results(self, results: Dict, output_dir: Optional[str] = None) -> None:
        """Save evaluation results"""
        if output_dir:
            results_dir = Path(output_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
        else:
            results_dir = self.results_dir
        
        print("Saving evaluation results...")
        
        # Save detailed results as JSON
        results_file = results_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Saved results: {results_file}")
        
        # Save summary text report
        summary_file = results_dir / 'evaluation_summary.txt'
        with open(summary_file, 'w') as f:
            stats = results['aggregate_stats']
            f.write("UCSD Ped2 Anomaly Detection Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset Overview:\n")
            f.write(f"  - Total sequences evaluated: {stats['num_sequences_evaluated']}\n")
            f.write(f"  - Sequences with anomalies: {stats['num_sequences_with_anomalies']}\n")
            f.write(f"  - Total frames: {stats['total_frames']}\n")
            f.write(f"  - Total anomalies: {stats['total_anomalies']}\n")
            f.write(f"  - Anomaly rate: {stats['anomaly_rate']:.1%}\n\n")
            
            f.write(f"Performance Metrics:\n")
            f.write(f"  - Mean AUC: {stats['mean_auc']:.4f} (±{stats['std_auc']:.4f})\n")
            f.write(f"  - AUC Range: {stats['min_auc']:.4f} - {stats['max_auc']:.4f}\n\n")
            
            f.write("Per-Sequence Results:\n")
            for seq_name, metrics in results['sequence_results'].items():
                if metrics['has_anomalies']:
                    f.write(f"  - {seq_name}: AUC={metrics['auc']:.4f} "
                          f"({metrics['num_anomalies']}/{metrics['num_frames']} anomalies)\n")
                else:
                    f.write(f"  - {seq_name}: No anomalies\n")
        
        print(f"  ✓ Saved summary: {summary_file}")
    
    def run_evaluation(self, output_dir: Optional[str] = None) -> Dict:
        """Run complete evaluation pipeline"""
        print("=" * 60)
        print("UCSD ANOMALY DETECTION EVALUATION")
        print("=" * 60)
        
        # Load data
        labels = self.load_labels()
        scores = self.load_scores()
        
        # Run evaluation
        results = self.evaluate_all_sequences(scores, labels)
        
        # Save results
        self.save_results(results, output_dir)
        
        # Print summary
        stats = results['aggregate_stats']
        print()
        print("=" * 60)
        print("EVALUATION COMPLETE")
        print(f"Mean AUC: {stats['mean_auc']:.4f} (±{stats['std_auc']:.4f})")
        print(f"{stats['num_sequences_with_anomalies']}/{stats['num_sequences_evaluated']} sequences evaluated")
        print("=" * 60)
        
        return results

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Evaluate UCSD anomaly detection scores')
    parser.add_argument('--scores_dir', default='data/processed/baseline_scores',
                       help='Directory containing anomaly scores')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for results (default: data/processed/evaluation_results)')
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not Path('data/splits/ucsd_ped2_labels.json').exists():
        print("Error: ucsd_ped2_labels.json not found. Run make_ucsd_label_masks.py first.")
        return
    
    if not Path(args.scores_dir).exists():
        print(f"Error: Scores directory not found: {args.scores_dir}")
        print("Run score_ucsd_baseline.py first to generate scores.")
        return
    
    # Run evaluator
    evaluator = UCSDScoreEvaluator(scores_dir=args.scores_dir)
    results = evaluator.run_evaluation(output_dir=args.output_dir)
    
    # Print key metrics
    stats = results['aggregate_stats']
    print(f"\nKey Results:")
    print(f"   Mean AUC: {stats['mean_auc']:.4f}")
    print(f"   Evaluated: {stats['num_sequences_with_anomalies']} sequences")
    print(f"   Total frames: {stats['total_frames']} ({stats['total_anomalies']} anomalies)")

if __name__ == "__main__":
    main()