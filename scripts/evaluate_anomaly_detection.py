#!/usr/bin/env python3
"""
Frame-Level Evaluation/Metrics Module
Evaluates anomaly detection performance using ROC/AUC metrics

This module:
1. Loads true labels from validated JSON files
2. Computes ROC/AUC for each test sequence 
3. Aggregates results with per-sequence and overall performance
4. Optionally computes precision/recall at optimal F1 threshold

Usage: python scripts/evaluate_anomaly_detection.py
"""

import os
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    f1_score, precision_score, recall_score
)

class AnomalyDetectionEvaluator:
    def __init__(self, scores_dir: str = 'data/processed/anomaly_scores'):
        self.scores_dir = Path(scores_dir)
        self.labels_file = Path('data/splits/ucsd_ped2_labels.json')
        self.results_dir = Path('data/processed/evaluation_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluation_results = {}
        
    def load_ground_truth_labels(self) -> Dict[str, List[int]]:
        """
        Step 2a: Load true labels from validated JSON file
        """
        print("🏷️ Loading ground truth labels...")
        
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
        
        with open(self.labels_file) as f:
            labels_data = json.load(f)
        
        print(f"  ✓ Loaded labels for {len(labels_data)} sequences")
        
        # Print label statistics
        total_frames = sum(len(labels) for labels in labels_data.values())
        total_anomalous = sum(sum(labels) for labels in labels_data.values())
        anomaly_rate = total_anomalous / total_frames if total_frames > 0 else 0
        
        print(f"  Dataset statistics:")
        print(f"     Total frames: {total_frames}")
        print(f"     Anomalous frames: {total_anomalous} ({anomaly_rate:.1%})")
        print(f"     Normal frames: {total_frames - total_anomalous} ({1-anomaly_rate:.1%})")
        
        return labels_data
    
    def load_anomaly_scores(self, method: str = 'baseline_l2') -> Dict[str, np.ndarray]:
        """
        Load precomputed anomaly scores from scoring module
        """
        print(f"\nLoading anomaly scores (method: {method})...")
        
        scores_dir = self.scores_dir / f'{method}_scores'
        
        if not scores_dir.exists():
            raise FileNotFoundError(f"Scores directory not found: {scores_dir}")
        
        anomaly_scores = {}
        
        for score_file in sorted(scores_dir.glob('*_anomaly_scores.npy')):
            # Extract sequence name (e.g., Test001_anomaly_scores.npy -> Test001)
            seq_name = score_file.stem.replace('_anomaly_scores', '')
            
            try:
                scores = np.load(score_file)
                anomaly_scores[seq_name] = scores
                print(f"  ✓ {seq_name}: {len(scores)} scores loaded")
                
            except Exception as e:
                print(f"  ✗ Error loading {score_file}: {e}")
                continue
        
        if not anomaly_scores:
            raise ValueError(f"No anomaly scores found in {scores_dir}")
        
        print(f"  Loaded scores for {len(anomaly_scores)} sequences")
        return anomaly_scores
    
    def evaluate_sequence(self, seq_name: str, y_true: List[int], y_scores: np.ndarray) -> Dict:
        """
        Step 2b: Compute ROC/AUC for a single sequence
        """
        y_true = np.array(y_true)
        
        # Check if sequence has any anomalies (need both classes for AUC)
        if len(np.unique(y_true)) < 2:
            anomaly_count = np.sum(y_true)
            if anomaly_count == 0:
                status = "all_normal"
                auc_score = np.nan  # Cannot compute AUC with only normal frames
            else:
                status = "all_anomalous"
                auc_score = np.nan  # Cannot compute AUC with only anomalous frames
            
            return {
                'sequence': seq_name,
                'status': status,
                'num_frames': len(y_true),
                'num_anomalies': int(anomaly_count),
                'auc': auc_score,
                'optimal_threshold': np.nan,
                'optimal_f1': np.nan,
                'precision_at_optimal': np.nan,
                'recall_at_optimal': np.nan
            }
        
        # Compute ROC AUC
        try:
            auc_score = roc_auc_score(y_true, y_scores)
        except Exception as e:
            print(f"  ⚠ Warning: Could not compute AUC for {seq_name}: {e}")
            auc_score = np.nan
        
        # Find optimal threshold using F1 score
        optimal_threshold = np.nan
        optimal_f1 = np.nan
        precision_at_optimal = np.nan
        recall_at_optimal = np.nan
        
        try:
            # Try multiple thresholds to find best F1
            thresholds = np.linspace(y_scores.min(), y_scores.max(), 100)
            best_f1 = 0
            
            for threshold in thresholds:
                y_pred = (y_scores >= threshold).astype(int)
                f1 = f1_score(y_true, y_pred)
                
                if f1 > best_f1:
                    best_f1 = f1
                    optimal_threshold = threshold
                    optimal_f1 = f1
                    precision_at_optimal = precision_score(y_true, y_pred, zero_division=0)
                    recall_at_optimal = recall_score(y_true, y_pred, zero_division=0)
                    
        except Exception as e:
            print(f"  ⚠ Warning: Could not find optimal threshold for {seq_name}: {e}")
        
        return {
            'sequence': seq_name,
            'status': 'evaluated',
            'num_frames': len(y_true),
            'num_anomalies': int(np.sum(y_true)),
            'auc': float(auc_score) if not np.isnan(auc_score) else None,
            'optimal_threshold': float(optimal_threshold) if not np.isnan(optimal_threshold) else None,
            'optimal_f1': float(optimal_f1) if not np.isnan(optimal_f1) else None,
            'precision_at_optimal': float(precision_at_optimal) if not np.isnan(precision_at_optimal) else None,
            'recall_at_optimal': float(recall_at_optimal) if not np.isnan(recall_at_optimal) else None
        }
    
    def evaluate_all_sequences(self, labels_data: Dict[str, List[int]], 
                             anomaly_scores: Dict[str, np.ndarray]) -> Dict:
        """
        Step 2b-c: Evaluate all sequences and aggregate results
        """
        print("\nEvaluating anomaly detection performance...")
        
        sequence_results = []
        valid_aucs = []
        
        for seq_name in sorted(labels_data.keys()):
            if seq_name not in anomaly_scores:
                print(f"  ⚠ Warning: No scores found for {seq_name}, skipping...")
                continue
            
            y_true = labels_data[seq_name]
            y_scores = anomaly_scores[seq_name]
            
            # Verify alignment
            if len(y_true) != len(y_scores):
                print(f"  ✗ {seq_name}: Length mismatch - {len(y_true)} labels vs {len(y_scores)} scores")
                continue
            
            # Evaluate sequence
            result = self.evaluate_sequence(seq_name, y_true, y_scores)
            sequence_results.append(result)
            
            # Print sequence result
            if result['auc'] is not None and not np.isnan(result['auc']):
                valid_aucs.append(result['auc'])
                print(f"  ✓ {seq_name}: AUC={result['auc']:.3f}, "
                      f"F1={result['optimal_f1']:.3f} (anomalies: {result['num_anomalies']}/{result['num_frames']})")
            else:
                print(f"  - {seq_name}: {result['status']} "
                      f"(anomalies: {result['num_anomalies']}/{result['num_frames']})")
        
        # Aggregate results
        if valid_aucs:
            mean_auc = np.mean(valid_aucs)
            std_auc = np.std(valid_aucs)
            median_auc = np.median(valid_aucs)
        else:
            mean_auc = std_auc = median_auc = np.nan
        
        # Compute overall metrics (concatenate all sequences)
        all_true = np.concatenate([
            labels_data[r['sequence']] for r in sequence_results 
            if r['sequence'] in anomaly_scores and r['status'] == 'evaluated'
        ])
        all_scores = np.concatenate([
            anomaly_scores[r['sequence']] for r in sequence_results
            if r['sequence'] in anomaly_scores and r['status'] == 'evaluated'  
        ])
        
        overall_auc = roc_auc_score(all_true, all_scores) if len(np.unique(all_true)) > 1 else np.nan
        
        aggregated_results = {
            'method': 'baseline_l2_distance',
            'total_sequences': len(sequence_results),
            'sequences_with_auc': len(valid_aucs),
            'mean_auc': float(mean_auc) if not np.isnan(mean_auc) else None,
            'std_auc': float(std_auc) if not np.isnan(std_auc) else None,
            'median_auc': float(median_auc) if not np.isnan(median_auc) else None,
            'overall_auc': float(overall_auc) if not np.isnan(overall_auc) else None,
            'total_frames': len(all_true),
            'total_anomalies': int(np.sum(all_true)),
            'sequence_results': sequence_results
        }
        
        return aggregated_results
    
    def plot_roc_curves(self, labels_data: Dict[str, List[int]], 
                       anomaly_scores: Dict[str, np.ndarray], 
                       save_plots: bool = True) -> None:
        """
        Plot ROC curves for visualization
        """
        print("\nGenerating ROC curve plots...")
        
        plt.figure(figsize=(12, 8))
        
        # Plot individual sequence ROC curves
        plt.subplot(2, 2, 1)
        for seq_name in sorted(labels_data.keys()):
            if seq_name not in anomaly_scores:
                continue
                
            y_true = np.array(labels_data[seq_name])
            y_scores = anomaly_scores[seq_name]
            
            if len(np.unique(y_true)) < 2:
                continue  # Skip sequences with only one class
            
            try:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc = roc_auc_score(y_true, y_scores)
                plt.plot(fpr, tpr, alpha=0.7, label=f'{seq_name} (AUC={auc:.3f})')
            except:
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by Sequence')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot overall ROC curve
        plt.subplot(2, 2, 2)
        all_true = np.concatenate([labels_data[seq] for seq in labels_data.keys() if seq in anomaly_scores])
        all_scores = np.concatenate([anomaly_scores[seq] for seq in labels_data.keys() if seq in anomaly_scores])
        
        if len(np.unique(all_true)) > 1:
            fpr, tpr, _ = roc_curve(all_true, all_scores)
            auc = roc_auc_score(all_true, all_scores)
            plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Overall AUC = {auc:.3f}')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Overall ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot score distributions
        plt.subplot(2, 2, 3)
        normal_scores = all_scores[all_true == 0]
        anomaly_scores_subset = all_scores[all_true == 1]
        
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
        plt.hist(anomaly_scores_subset, bins=50, alpha=0.7, label='Anomalous', density=True)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot AUC by sequence
        plt.subplot(2, 2, 4)
        seq_names = []
        seq_aucs = []
        for seq_name in sorted(labels_data.keys()):
            if seq_name not in anomaly_scores:
                continue
            y_true = np.array(labels_data[seq_name])
            if len(np.unique(y_true)) < 2:
                continue
            try:
                auc = roc_auc_score(y_true, anomaly_scores[seq_name])
                seq_names.append(seq_name)
                seq_aucs.append(auc)
            except:
                continue
        
        if seq_aucs:
            plt.bar(range(len(seq_aucs)), seq_aucs)
            plt.xticks(range(len(seq_names)), seq_names, rotation=45)
            plt.ylabel('AUC')
            plt.title('AUC by Sequence')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = self.results_dir / 'roc_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved ROC plots: {plot_file}")
        
        plt.close()
    
    def save_evaluation_results(self, results: Dict) -> None:
        """
        Save evaluation results to JSON file
        """
        print("\nSaving evaluation results...")
        
        results_file = self.results_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✓ Saved detailed results: {results_file}")
        
        # Also save a summary report
        summary_file = self.results_dir / 'evaluation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("ANOMALY DETECTION EVALUATION SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Method: {results['method']}\n")
            f.write(f"Total Sequences: {results['total_sequences']}\n")
            f.write(f"Sequences with AUC: {results['sequences_with_auc']}\n\n")
            
            if results['mean_auc'] is not None:
                f.write(f"Mean AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}\n")
                f.write(f"Median AUC: {results['median_auc']:.3f}\n")
            
            if results['overall_auc'] is not None:
                f.write(f"Overall AUC: {results['overall_auc']:.3f}\n\n")
            
            f.write(f"Total Frames: {results['total_frames']}\n")
            f.write(f"Total Anomalies: {results['total_anomalies']}\n")
            f.write(f"Anomaly Rate: {results['total_anomalies']/results['total_frames']:.1%}\n\n")
            
            f.write("PER-SEQUENCE RESULTS:\n")
            f.write("-" * 50 + "\n")
            for seq_result in results['sequence_results']:
                seq_name = seq_result['sequence']
                if seq_result['auc'] is not None:
                    f.write(f"{seq_name}: AUC={seq_result['auc']:.3f}, "
                           f"F1={seq_result['optimal_f1']:.3f}, "
                           f"Anomalies={seq_result['num_anomalies']}/{seq_result['num_frames']}\n")
                else:
                    f.write(f"{seq_name}: {seq_result['status']} "
                           f"(Anomalies={seq_result['num_anomalies']}/{seq_result['num_frames']})\n")
        
        print(f"  ✓ Saved summary report: {summary_file}")
    
    def run_evaluation(self, method: str = 'baseline_l2', plot_results: bool = True) -> Dict:
        """
        Complete evaluation pipeline
        """
        print("=" * 60)
        print("ANOMALY DETECTION EVALUATION PIPELINE")
        print(f"   Method: {method}")
        print("=" * 60)
        
        # Step 2a: Load ground truth labels
        labels_data = self.load_ground_truth_labels()
        
        # Load anomaly scores
        anomaly_scores = self.load_anomaly_scores(method)
        
        # Step 2b-c: Evaluate all sequences
        results = self.evaluate_all_sequences(labels_data, anomaly_scores)
        
        # Generate plots
        if plot_results:
            self.plot_roc_curves(labels_data, anomaly_scores)
        
        # Save results
        self.save_evaluation_results(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        if results['mean_auc'] is not None:
            print(f"Mean AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
            print(f"Overall AUC: {results['overall_auc']:.3f}")
        else:
            print("No valid AUC scores computed")
        print(f"Evaluated {results['sequences_with_auc']}/{results['total_sequences']} sequences")
        print("=" * 60)
        
        return results

def main():
    """Main execution function"""
    # Check if required files exist
    if not Path('data/splits/ucsd_ped2_labels.json').exists():
        print("Error: Labels file not found. Run validate_ucsd_dataset.py first.")
        return
    
    if not Path('data/processed/anomaly_scores').exists():
        print("Error: Anomaly scores not found. Run baseline_anomaly_scoring.py first.")
        return
    
    # Initialize and run evaluator
    evaluator = AnomalyDetectionEvaluator()
    results = evaluator.run_evaluation()
    
    return results

if __name__ == "__main__":
    main()