"""
STEP 1: Diagnose Ensemble Issues
Purpose: Analyze why weighted averaging underperforms
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


class EnsembleDiagnostics:
    """
    Diagnose problems in current ensemble approach.
    
    Logical Flow:
    1. Load baseline and GNN scores
    2. Analyze score distributions
    3. Check for correlation issues
    4. Identify normalization problems
    """
    
    def __init__(self, 
                 baseline_dir='data/processed/baseline_scores',
                 gnn_dir='data/processed/gnn_scores',
                 labels_file='data/splits/ucsd_ped2_labels.json'):
        """
        Initialize diagnostics.
        
        What this does:
        - Store paths to score directories
        - Load ground truth labels
        """
        self.baseline_dir = baseline_dir
        self.gnn_dir = gnn_dir
        self.labels = self.load_labels(labels_file)
        
        # Create reports directory
        os.makedirs('reports', exist_ok=True)
    
    def load_labels(self, labels_file):
        """
        Load ground truth labels.
        
        Input: JSON file with frame-level labels
        Output: Dictionary {seq_name: [0,1,0,1,...]}
        """
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        return labels
    
    def load_scores(self, seq_name):
        """
        Load both baseline and GNN scores for a sequence.
        
        Logical Steps:
        1. Load baseline scores from .npy file
        2. Load GNN scores from .npy file
        3. Verify shapes match
        4. Return both score arrays
        
        Returns:
            tuple: (baseline_scores, gnn_scores)
        """
        # Handle different naming conventions for baseline
        baseline_candidates = [
            os.path.join(self.baseline_dir, f'{seq_name}_scores.npy'),
            os.path.join(self.baseline_dir, f'{seq_name}_baseline_scores.npy')
        ]
        
        baseline_path = None
        for path in baseline_candidates:
            if os.path.exists(path):
                baseline_path = path
                break
        
        if baseline_path is None:
            raise FileNotFoundError(f"Baseline scores not found for {seq_name}")
        
        # Handle different naming conventions for GNN
        gnn_candidates = [
            os.path.join(self.gnn_dir, f'{seq_name}_gnn_scores.npy'),
            os.path.join(self.gnn_dir, f'{seq_name}_scores.npy')
        ]
        
        gnn_path = None
        for path in gnn_candidates:
            if os.path.exists(path):
                gnn_path = path
                break
                
        if gnn_path is None:
            raise FileNotFoundError(f"GNN scores not found for {seq_name}")
        
        baseline_scores = np.load(baseline_path)
        gnn_scores = np.load(gnn_path)
        
        assert len(baseline_scores) == len(gnn_scores), \
            f"Score length mismatch for {seq_name}: baseline={len(baseline_scores)}, gnn={len(gnn_scores)}"
        
        return baseline_scores, gnn_scores
    
    def analyze_score_distributions(self):
        """
        Analyze score distributions to identify normalization issues.
        
        Why this matters:
        - Baseline and GNN scores may have different scales
        - Simple averaging assumes comparable scales
        - Mismatched scales cause dominant model to overwhelm
        
        Logical Steps:
        1. Load all scores
        2. Compute statistics (min, max, mean, std)
        3. Plot distributions
        4. Check for scale differences
        """
        print("\n" + "="*70)
        print("SCORE DISTRIBUTION ANALYSIS")
        print("="*70)
        
        all_baseline = []
        all_gnn = []
        
        for seq_name in self.labels.keys():
            try:
                baseline_scores, gnn_scores = self.load_scores(seq_name)
                all_baseline.extend(baseline_scores)
                all_gnn.extend(gnn_scores)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        all_baseline = np.array(all_baseline)
        all_gnn = np.array(all_gnn)
        
        # Statistics
        print("\nBaseline Scores:")
        print(f"   Range: [{all_baseline.min():.4f}, {all_baseline.max():.4f}]")
        print(f"   Mean: {all_baseline.mean():.4f}, Std: {all_baseline.std():.4f}")
        
        print("\nGNN Scores:")
        print(f"   Range: [{all_gnn.min():.4f}, {all_gnn.max():.4f}]")
        print(f"   Mean: {all_gnn.mean():.4f}, Std: {all_gnn.std():.4f}")
        
        # Check scale ratio
        baseline_range = all_baseline.max() - all_baseline.min()
        gnn_range = all_gnn.max() - all_gnn.min()
        
        if baseline_range > 0 and gnn_range > 0:
            scale_ratio = gnn_range / baseline_range
            print(f"\nScale Ratio (GNN/Baseline): {scale_ratio:.2f}")
            
            if scale_ratio > 2.0 or scale_ratio < 0.5:
                print("   WARNING: Significant scale mismatch detected!")
                print("   -> This explains why simple averaging fails")
            else:
                print("   OK: Scales are reasonably matched")
        else:
            scale_ratio = 1.0
            print("\nWARNING: One of the score ranges is zero!")
        
        # Plot distributions
        self.plot_distributions(all_baseline, all_gnn)
        
        return {
            'baseline_range': (float(all_baseline.min()), float(all_baseline.max())),
            'gnn_range': (float(all_gnn.min()), float(all_gnn.max())),
            'baseline_mean': float(all_baseline.mean()),
            'gnn_mean': float(all_gnn.mean()),
            'scale_ratio': float(scale_ratio)
        }
    
    def plot_distributions(self, baseline_scores, gnn_scores):
        """
        Visualize score distributions.
        
        What this shows:
        - Histogram of baseline scores
        - Histogram of GNN scores
        - Helps identify scale/distribution mismatches
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Baseline
        axes[0].hist(baseline_scores, bins=50, alpha=0.7, color='blue')
        axes[0].set_title('Baseline Score Distribution')
        axes[0].set_xlabel('Anomaly Score')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(baseline_scores.mean(), color='red', linestyle='--', 
                       label=f'Mean: {baseline_scores.mean():.3f}')
        axes[0].legend()
        
        # GNN
        axes[1].hist(gnn_scores, bins=50, alpha=0.7, color='green')
        axes[1].set_title('GNN Score Distribution')
        axes[1].set_xlabel('Anomaly Score')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(gnn_scores.mean(), color='red', linestyle='--',
                       label=f'Mean: {gnn_scores.mean():.3f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('reports/score_distributions.png', dpi=150)
        print("\nDistribution plot saved: reports/score_distributions.png")
    
    def analyze_correlation(self):
        """
        Measure correlation between baseline and GNN scores.
        
        Why this matters:
        - High correlation means models make similar predictions
        - Low correlation means complementary strengths
        - Negative correlation is unusual but informative
        
        Logical Steps:
        1. Compute Pearson correlation
        2. Compute Spearman rank correlation
        3. Visualize scatter plot
        """
        print("\n" + "="*70)
        print("SCORE CORRELATION ANALYSIS")
        print("="*70)
        
        all_baseline = []
        all_gnn = []
        
        for seq_name in self.labels.keys():
            try:
                baseline_scores, gnn_scores = self.load_scores(seq_name)
                all_baseline.extend(baseline_scores)
                all_gnn.extend(gnn_scores)
            except FileNotFoundError:
                continue
        
        all_baseline = np.array(all_baseline)
        all_gnn = np.array(all_gnn)
        
        # Pearson correlation (linear)
        pearson_corr = np.corrcoef(all_baseline, all_gnn)[0, 1]
        
        # Spearman correlation (rank-based)
        spearman_corr, _ = spearmanr(all_baseline, all_gnn)
        
        print(f"\nPearson Correlation: {pearson_corr:.4f}")
        print(f"Spearman Correlation: {spearman_corr:.4f}")
        
        if abs(pearson_corr) > 0.7:
            print("\n   WARNING: High correlation detected!")
            print("   -> Models make similar predictions")
            print("   -> Ensemble may not add much value")
        elif abs(pearson_corr) < 0.3:
            print("\n   GOOD: Low correlation detected!")
            print("   -> Models have complementary strengths")
            print("   -> Ensemble should improve performance")
        else:
            print("\n   MODERATE: Moderate correlation detected")
            print("   -> Ensemble may provide some benefit")
        
        # Scatter plot
        self.plot_correlation(all_baseline, all_gnn, pearson_corr)
        
        return {
            'pearson': float(pearson_corr),
            'spearman': float(spearman_corr)
        }
    
    def plot_correlation(self, baseline_scores, gnn_scores, correlation):
        """
        Visualize correlation between methods.
        """
        plt.figure(figsize=(8, 6))
        
        # Use scatter with alpha for density visualization
        plt.scatter(baseline_scores, gnn_scores, alpha=0.3, s=1)
        plt.xlabel('Baseline Scores')
        plt.ylabel('GNN Scores')
        plt.title(f'Score Correlation (r={correlation:.3f})')
        plt.grid(True, alpha=0.3)
        
        # Add trend line if correlation is significant
        if abs(correlation) > 0.1:
            z = np.polyfit(baseline_scores, gnn_scores, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(baseline_scores.min(), baseline_scores.max(), 100)
            plt.plot(x_trend, p(x_trend), 'r--', alpha=0.8, label=f'Trend (r={correlation:.3f})')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('reports/score_correlation.png', dpi=150)
        print("Correlation plot saved: reports/score_correlation.png")
    
    def analyze_per_method_performance(self):
        """
        Analyze which method performs better on which sequences.
        
        Why this matters:
        - Maybe baseline excels on certain anomaly types
        - Maybe GNN excels on others
        - Can inform selective ensemble strategies
        
        Logical Steps:
        1. Compute AUC for each method per sequence
        2. Identify where each excels
        3. Look for patterns
        """
        print("\n" + "="*70)
        print("PER-SEQUENCE PERFORMANCE ANALYSIS")
        print("="*70)
        
        results = []
        
        for seq_name in sorted(self.labels.keys()):
            try:
                baseline_scores, gnn_scores = self.load_scores(seq_name)
                labels = np.array(self.labels[seq_name])
                
                # Skip sequences with all same label
                if len(np.unique(labels)) < 2:
                    print(f"\n{seq_name}: Skipped (all labels same)")
                    continue
                
                # Compute AUC for each
                try:
                    baseline_auc = roc_auc_score(labels, baseline_scores)
                except:
                    baseline_auc = 0.5
                
                try:
                    gnn_auc = roc_auc_score(labels, gnn_scores)
                except:
                    gnn_auc = 0.5
                
                winner = 'Baseline' if baseline_auc > gnn_auc else 'GNN'
                margin = abs(baseline_auc - gnn_auc)
                
                results.append({
                    'sequence': seq_name,
                    'baseline_auc': baseline_auc,
                    'gnn_auc': gnn_auc,
                    'winner': winner,
                    'margin': margin
                })
                
                print(f"\n{seq_name}:")
                print(f"   Baseline: {baseline_auc:.4f}")
                print(f"   GNN:      {gnn_auc:.4f}")
                print(f"   Winner:   {winner} (margin: {margin:.4f})")
                
            except FileNotFoundError:
                print(f"\n{seq_name}: Skipped (missing scores)")
                continue
        
        # Summary
        if results:
            baseline_wins = sum(1 for r in results if r['winner'] == 'Baseline')
            gnn_wins = sum(1 for r in results if r['winner'] == 'GNN')
            
            print("\n" + "="*70)
            print("Summary:")
            print(f"   Baseline wins: {baseline_wins} sequences")
            print(f"   GNN wins:      {gnn_wins} sequences")
            
            # Overall AUC
            all_baseline_auc = np.mean([r['baseline_auc'] for r in results])
            all_gnn_auc = np.mean([r['gnn_auc'] for r in results])
            print(f"   Average Baseline AUC: {all_baseline_auc:.4f}")
            print(f"   Average GNN AUC: {all_gnn_auc:.4f}")
        
        return results
    
    def run_full_diagnosis(self):
        """
        Run complete diagnostic suite.
        
        What this does:
        1. Analyze score distributions
        2. Check correlation
        3. Compare per-sequence performance
        4. Generate diagnostic report
        """
        print("\n" + "="*70)
        print("ENSEMBLE DIAGNOSTIC SUITE")
        print("="*70)
        
        # Run all diagnostics
        distribution_stats = self.analyze_score_distributions()
        correlation_stats = self.analyze_correlation()
        per_seq_results = self.analyze_per_method_performance()
        
        # Compile report
        report = {
            'distribution_stats': distribution_stats,
            'correlation_stats': correlation_stats,
            'per_sequence_results': per_seq_results,
            'summary': {
                'scale_mismatch': distribution_stats['scale_ratio'] > 2.0 or distribution_stats['scale_ratio'] < 0.5,
                'low_correlation': abs(correlation_stats['pearson']) < 0.3,
                'ensemble_potential': abs(correlation_stats['pearson']) < 0.7
            }
        }
        
        # Save report
        with open('reports/ensemble_diagnostics.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nDiagnostic report saved: reports/ensemble_diagnostics.json")
        
        # Summary recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        if report['summary']['scale_mismatch']:
            print("1. CRITICAL: Apply score normalization before ensemble")
            print("   -> Use MinMax, RobustScaler, or rank-based normalization")
        
        if report['summary']['low_correlation']:
            print("2. GOOD: Low correlation suggests ensemble will help")
            print("   -> Try weighted average, rank fusion, or stacking")
        
        if not report['summary']['ensemble_potential']:
            print("3. WARNING: High correlation may limit ensemble gains")
            print("   -> Focus on improving individual methods instead")
        
        return report


def main():
    """
    Run ensemble diagnostics.
    """
    diagnostics = EnsembleDiagnostics()
    report = diagnostics.run_full_diagnosis()
    
    print("\nDiagnostics complete!")
    print("   Review reports/score_distributions.png")
    print("   Review reports/score_correlation.png")
    print("   Review reports/ensemble_diagnostics.json")


if __name__ == "__main__":
    main()