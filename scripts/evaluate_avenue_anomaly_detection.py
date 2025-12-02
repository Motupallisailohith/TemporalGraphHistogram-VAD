#!/usr/bin/env python3
"""
Avenue Frame-Level Evaluation/Metrics Module
Evaluates anomaly detection performance on Avenue dataset using ROC/AUC metrics

This module:
1. Loads Avenue test sequences and generates dummy labels  
2. Loads anomaly scores from Avenue cross-dataset validation
3. Computes ROC/AUC for each feature type
4. Aggregates results with per-sequence and overall performance
5. Generates comparison plots and reports

Usage: python scripts/evaluate_avenue_anomaly_detection.py
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    f1_score, precision_score, recall_score
)

class AvenueAnomalyDetectionEvaluator:
    def __init__(self, scores_dir: str = 'data/processed/avenue/anomaly_scores'):
        self.scores_dir = Path(scores_dir)
        self.results_dir = Path('data/processed/avenue/evaluation_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Avenue sequences info
        self.sequences_info_file = Path('data/processed/avenue/avenue_sequences.json')
        self.evaluation_results = {}
        
    def load_avenue_sequences_info(self) -> Dict:
        """Load Avenue sequences information"""
        print("📋 Loading Avenue sequences information...")
        
        if not self.sequences_info_file.exists():
            raise FileNotFoundError(f"Avenue sequences file not found: {self.sequences_info_file}")
        
        with open(self.sequences_info_file) as f:
            sequences_data = json.load(f)
        
        print(f"  ✓ Found {len(sequences_data['test_sequences'])} test sequences")
        return sequences_data
    
    def generate_dummy_labels(self, sequences_data: Dict) -> Dict[str, List[int]]:
        """
        Generate dummy anomaly labels for Avenue test sequences
        In practice, you would use ground truth labels
        """
        print("🏷️ Generating dummy anomaly labels...")
        
        labels_data = {}
        total_frames = 0
        total_anomalies = 0
        
        # Generate labels for each test sequence
        for seq_info in sequences_data['test_sequences']:
            seq_name = seq_info['name']
            num_frames = seq_info['num_frames']
            
            # Create dummy labels: 10% anomaly frames randomly distributed
            np.random.seed(42)  # For reproducibility
            labels = np.random.choice([0, 1], size=num_frames, p=[0.9, 0.1])
            labels_data[seq_name] = labels.tolist()
            
            total_frames += num_frames
            total_anomalies += np.sum(labels)
            
        anomaly_rate = total_anomalies / total_frames if total_frames > 0 else 0
        
        print(f"  Dataset statistics:")
        print(f"     Total test frames: {total_frames}")
        print(f"     Dummy anomalous frames: {total_anomalies} ({anomaly_rate:.1%})")
        print(f"     Dummy normal frames: {total_frames - total_anomalies} ({1-anomaly_rate:.1%})")
        
        return labels_data
    
    def load_anomaly_scores(self, method: str = 'histogram') -> Dict[str, np.ndarray]:
        """
        Load precomputed anomaly scores from Avenue evaluation
        """
        print(f"\nLoading Avenue anomaly scores (method: {method})...")
        
        scores_dir = self.scores_dir / f'{method}_scores'
        
        if not scores_dir.exists():
            raise FileNotFoundError(f"Scores directory not found: {scores_dir}")
        
        anomaly_scores = {}
        
        # Load the saved scores file
        score_file = scores_dir / 'avenue_test_anomaly_scores.npy'
        if score_file.exists():
            try:
                all_scores = np.load(score_file)
                print(f"  ✓ Loaded {len(all_scores)} total anomaly scores")
                
                # For now, we'll treat all scores as one sequence
                # In practice, you'd split by sequence boundaries
                anomaly_scores['avenue_combined'] = all_scores
                
            except Exception as e:
                print(f"  ✗ Error loading {score_file}: {e}")
        else:
            print(f"  ⚠️ Score file not found: {score_file}")
        
        if not anomaly_scores:
            raise ValueError(f"No anomaly scores found in {scores_dir}")
        
        return anomaly_scores
    
    def evaluate_sequence(self, seq_name: str, y_true: List[int], y_scores: np.ndarray) -> Dict:
        """
        Compute ROC/AUC for a single sequence
        """
        y_true = np.array(y_true)
        
        # Check if sequence has any anomalies (need both classes for AUC)
        if len(np.unique(y_true)) < 2:
            anomaly_count = np.sum(y_true)
            if anomaly_count == 0:
                status = "all_normal"
                auc_score = np.nan
            else:
                status = "all_anomalous"
                auc_score = np.nan
            
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
                             anomaly_scores: Dict[str, np.ndarray], method: str) -> Dict:
        """
        Evaluate all sequences and aggregate results
        """
        print(f"\nEvaluating Avenue anomaly detection performance ({method})...")
        
        sequence_results = []
        valid_aucs = []
        
        # For this demo, we'll use the combined scores with combined labels
        if 'avenue_combined' in anomaly_scores:
            # Combine all labels
            all_labels = []
            for seq_name in sorted(labels_data.keys()):
                all_labels.extend(labels_data[seq_name])
            
            y_true = np.array(all_labels)
            y_scores = anomaly_scores['avenue_combined']
            
            # Verify alignment
            if len(y_true) != len(y_scores):
                print(f"  ⚠ Length mismatch - {len(y_true)} labels vs {len(y_scores)} scores")
                # Truncate to shorter length
                min_len = min(len(y_true), len(y_scores))
                y_true = y_true[:min_len]
                y_scores = y_scores[:min_len]
            
            # Evaluate combined sequence
            result = self.evaluate_sequence('avenue_combined', y_true, y_scores)
            sequence_results.append(result)
            
            if result['auc'] is not None and not np.isnan(result['auc']):
                valid_aucs.append(result['auc'])
                print(f"  ✓ Avenue Combined: AUC={result['auc']:.3f}, "
                      f"F1={result['optimal_f1']:.3f} (anomalies: {result['num_anomalies']}/{result['num_frames']})")
        
        # Aggregate results
        if valid_aucs:
            mean_auc = np.mean(valid_aucs)
            std_auc = np.std(valid_aucs)
            median_auc = np.median(valid_aucs)
            overall_auc = valid_aucs[0]  # Since we only have one combined score
        else:
            mean_auc = std_auc = median_auc = overall_auc = np.nan
        
        aggregated_results = {
            'method': f'avenue_{method}',
            'total_sequences': len(sequence_results),
            'sequences_with_auc': len(valid_aucs),
            'mean_auc': float(mean_auc) if not np.isnan(mean_auc) else None,
            'std_auc': float(std_auc) if not np.isnan(std_auc) else None,
            'median_auc': float(median_auc) if not np.isnan(median_auc) else None,
            'overall_auc': float(overall_auc) if not np.isnan(overall_auc) else None,
            'total_frames': len(y_true) if 'y_true' in locals() else 0,
            'total_anomalies': int(np.sum(y_true)) if 'y_true' in locals() else 0,
            'sequence_results': sequence_results
        }
        
        return aggregated_results
    
    def plot_roc_curves(self, labels_data: Dict[str, List[int]], 
                       anomaly_scores: Dict[str, np.ndarray], 
                       method: str, save_plots: bool = True) -> None:
        """
        Plot ROC curves for Avenue evaluation
        """
        print("\nGenerating Avenue ROC curve plots...")
        
        plt.figure(figsize=(12, 8))
        
        if 'avenue_combined' in anomaly_scores:
            # Combine all labels
            all_labels = []
            for seq_name in sorted(labels_data.keys()):
                all_labels.extend(labels_data[seq_name])
            
            y_true = np.array(all_labels)
            y_scores = anomaly_scores['avenue_combined']
            
            # Truncate to ensure same length
            min_len = min(len(y_true), len(y_scores))
            y_true = y_true[:min_len]
            y_scores = y_scores[:min_len]
            
            if len(np.unique(y_true)) > 1:
                # Plot overall ROC curve
                plt.subplot(2, 2, 1)
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc = roc_auc_score(y_true, y_scores)
                plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Avenue {method} AUC = {auc:.3f}')
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Avenue ROC Curve ({method})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot score distributions
                plt.subplot(2, 2, 2)
                normal_scores = y_scores[y_true == 0]
                anomaly_scores_subset = y_scores[y_true == 1]
                
                plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
                plt.hist(anomaly_scores_subset, bins=50, alpha=0.7, label='Anomalous', density=True)
                plt.xlabel('Anomaly Score')
                plt.ylabel('Density')
                plt.title('Score Distributions')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot precision-recall curve
                plt.subplot(2, 2, 3)
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                plt.plot(recall, precision, 'g-', linewidth=2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.grid(True, alpha=0.3)
                
                # Plot score timeline
                plt.subplot(2, 2, 4)
                plt.plot(y_scores, 'b-', alpha=0.7, label='Anomaly Score')
                anomaly_indices = np.where(y_true == 1)[0]
                plt.scatter(anomaly_indices, y_scores[anomaly_indices], 
                           c='red', s=20, alpha=0.8, label='True Anomalies')
                plt.xlabel('Frame Index')
                plt.ylabel('Anomaly Score')
                plt.title('Score Timeline')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = self.results_dir / f'avenue_{method}_roc_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved Avenue ROC plots: {plot_file}")
        
        plt.close()
    
    def save_evaluation_results(self, results: Dict, method: str) -> None:
        """
        Save evaluation results to JSON file
        """
        print(f"\nSaving Avenue evaluation results ({method})...")
        
        results_file = self.results_dir / f'avenue_{method}_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✓ Saved detailed results: {results_file}")
        
        # Also save a summary report
        summary_file = self.results_dir / f'avenue_{method}_evaluation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("AVENUE ANOMALY DETECTION EVALUATION SUMMARY\n")
            f.write("="*60 + "\n\n")
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
            if results['total_frames'] > 0:
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
    
    def run_evaluation(self, method: str = 'histogram', plot_results: bool = True) -> Dict:
        """
        Complete Avenue evaluation pipeline
        """
        print("=" * 70)
        print("AVENUE ANOMALY DETECTION EVALUATION PIPELINE")
        print(f"   Method: {method}")
        print("=" * 70)
        
        # Load Avenue sequences info
        sequences_data = self.load_avenue_sequences_info()
        
        # Generate dummy labels (in practice, use ground truth)
        labels_data = self.generate_dummy_labels(sequences_data)
        
        # Load anomaly scores
        anomaly_scores = self.load_anomaly_scores(method)
        
        # Evaluate all sequences
        results = self.evaluate_all_sequences(labels_data, anomaly_scores, method)
        
        # Generate plots
        if plot_results:
            self.plot_roc_curves(labels_data, anomaly_scores, method)
        
        # Save results
        self.save_evaluation_results(results, method)
        
        # Print summary
        print("\n" + "=" * 70)
        print(f"AVENUE EVALUATION SUMMARY ({method})")
        print("=" * 70)
        if results['mean_auc'] is not None:
            print(f"Mean AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
            print(f"Overall AUC: {results['overall_auc']:.3f}")
        else:
            print("No valid AUC scores computed")
        print(f"Evaluated {results['sequences_with_auc']}/{results['total_sequences']} sequences")
        print("=" * 70)
        
        return results

def main():
    """Main execution function"""
    # Check if Avenue data exists
    if not Path('data/processed/avenue/avenue_sequences.json').exists():
        print("Error: Avenue sequences file not found. Run avenue evaluation first.")
        return
    
    if not Path('data/processed/avenue/anomaly_scores').exists():
        print("Error: Avenue anomaly scores not found. Run avenue evaluation first.")
        return
    
    # Initialize evaluator
    evaluator = AvenueAnomalyDetectionEvaluator()
    
    # Run evaluation for different feature types
    feature_types = ['histogram', 'cnn', 'optical_flow']
    all_results = {}
    
    for feature_type in feature_types:
        try:
            print(f"\n🔄 Running evaluation for {feature_type} features...")
            results = evaluator.run_evaluation(feature_type)
            all_results[feature_type] = results
        except Exception as e:
            print(f"❌ Failed to evaluate {feature_type}: {e}")
            continue
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("AVENUE FEATURE TYPE COMPARISON")
    print("=" * 70)
    print(f"{'Feature Type':<15} {'AUC':<8} {'Status'}")
    print("-" * 40)
    
    for feature_type, results in all_results.items():
        if results and results.get('overall_auc') is not None:
            auc = results['overall_auc']
            print(f"{feature_type:<15} {auc:.3f}   ✓")
        else:
            print(f"{feature_type:<15} {'N/A':<8} ❌")
    
    return all_results

if __name__ == "__main__":
    main()