"""
STEP 2: Score Normalization
Purpose: Normalize scores to comparable ranges
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import os


class ScoreNormalizer:
    """
    Normalize anomaly scores to [0,1] range.
    
    Supports multiple normalization strategies:
    1. Min-Max: Scale to [0,1] based on min/max
    2. Z-Score + Sigmoid: Standardize then squash to [0,1]
    3. Robust: Use median/IQR (robust to outliers)
    4. Rank: Convert to rank percentiles
    """
    
    def __init__(self, method='minmax'):
        """
        Initialize normalizer.
        
        Args:
            method (str): 'minmax', 'zscore', 'robust', or 'rank'
        """
        self.method = method
    
    def normalize_minmax(self, scores):
        """
        Min-Max normalization.
        
        Formula: (x - min) / (max - min)
        
        Logical Steps:
        1. Find min and max values
        2. Scale to [0,1]
        
        Pros: Simple, preserves relative distances
        Cons: Sensitive to outliers
        """
        min_val = scores.min()
        max_val = scores.max()
        
        if max_val - min_val < 1e-8:
            return np.ones_like(scores) * 0.5
        
        normalized = (scores - min_val) / (max_val - min_val)
        return normalized
    
    def normalize_zscore(self, scores):
        """
        Z-score normalization + sigmoid.
        
        Formula: sigmoid((x - mean) / std)
        
        Logical Steps:
        1. Standardize (z-score)
        2. Apply sigmoid to map to [0,1]
        
        Pros: Handles outliers better, smooth mapping
        Cons: May compress dynamic range
        """
        mean = scores.mean()
        std = scores.std()
        
        if std < 1e-8:
            return np.ones_like(scores) * 0.5
        
        z_scores = (scores - mean) / std
        normalized = 1 / (1 + np.exp(-z_scores))  # Sigmoid
        return normalized
    
    def normalize_robust(self, scores):
        """
        Robust normalization using median and IQR.
        
        Formula: sigmoid((x - median) / IQR)
        
        Logical Steps:
        1. Compute median and IQR
        2. Robust z-score
        3. Sigmoid to [0,1]
        
        Pros: Very robust to outliers
        Cons: More computation
        """
        median = np.median(scores)
        q75, q25 = np.percentile(scores, [75, 25])
        iqr = q75 - q25
        
        if iqr < 1e-8:
            return np.ones_like(scores) * 0.5
        
        robust_z = (scores - median) / iqr
        normalized = 1 / (1 + np.exp(-robust_z))  # Sigmoid
        return normalized
    
    def normalize_rank(self, scores):
        """
        Rank-based normalization.
        
        Formula: (rank - 1) / (N - 1)
        
        Logical Steps:
        1. Convert scores to ranks
        2. Normalize ranks to [0,1]
        
        Pros: Order-preserving, very robust
        Cons: Loses magnitude information
        """
        ranks = rankdata(scores)
        normalized = (ranks - 1) / (len(ranks) - 1)
        return normalized
    
    def normalize(self, scores):
        """
        Apply selected normalization method.
        
        Args:
            scores (np.ndarray): Raw anomaly scores
        
        Returns:
            np.ndarray: Normalized scores in [0,1]
        """
        if self.method == 'minmax':
            return self.normalize_minmax(scores)
        elif self.method == 'zscore':
            return self.normalize_zscore(scores)
        elif self.method == 'robust':
            return self.normalize_robust(scores)
        elif self.method == 'rank':
            return self.normalize_rank(scores)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def normalize_all_sequences(self, scores_dict):
        """
        Normalize scores for all sequences.
        
        Args:
            scores_dict (dict): {seq_name: scores_array}
        
        Returns:
            dict: {seq_name: normalized_scores}
        """
        normalized_dict = {}
        
        for seq_name, scores in scores_dict.items():
            normalized_dict[seq_name] = self.normalize(scores)
        
        return normalized_dict


def test_normalization_methods():
    """
    Test and compare different normalization methods.
    
    What this does:
    1. Load baseline and GNN scores
    2. Apply each normalization method
    3. Check resulting distributions
    4. Visualize differences
    """
    print("\n" + "="*70)
    print("TESTING NORMALIZATION METHODS")
    print("="*70)
    
    # Example scores (simulating your actual data)
    np.random.seed(42)
    baseline_scores = np.random.uniform(0.04, 0.07, 1000)  # Small range
    gnn_scores = np.random.uniform(0.45, 1.92, 1000)       # Large range
    
    methods = ['minmax', 'zscore', 'robust', 'rank']
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, method in enumerate(methods):
        print(f"\nTesting {method} normalization...")
        
        normalizer = ScoreNormalizer(method=method)
        
        baseline_norm = normalizer.normalize(baseline_scores)
        gnn_norm = normalizer.normalize(gnn_scores)
        
        print(f"   Baseline: [{baseline_norm.min():.4f}, {baseline_norm.max():.4f}], "
              f"mean={baseline_norm.mean():.4f}")
        print(f"   GNN:      [{gnn_norm.min():.4f}, {gnn_norm.max():.4f}], "
              f"mean={gnn_norm.mean():.4f}")
        
        # Plot distributions
        axes[0, i].hist(baseline_norm, bins=30, alpha=0.7, color='blue', label='Baseline')
        axes[0, i].set_title(f'{method.title()} - Baseline')
        axes[0, i].set_xlim([0, 1])
        axes[0, i].set_ylabel('Frequency')
        
        axes[1, i].hist(gnn_norm, bins=30, alpha=0.7, color='green', label='GNN')
        axes[1, i].set_title(f'{method.title()} - GNN')
        axes[1, i].set_xlim([0, 1])
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].set_xlabel('Normalized Score')
    
    plt.tight_layout()
    plt.savefig('reports/normalization_comparison.png', dpi=150)
    print("\nNormalization comparison saved: reports/normalization_comparison.png")


def compare_normalization_on_real_data():
    """
    Compare normalization methods on real score data.
    """
    try:
        from phase5_4a_diagnose_ensemble import EnsembleDiagnostics
        
        print("\n" + "="*70)
        print("COMPARING NORMALIZATION ON REAL DATA")
        print("="*70)
        
        # Load real data
        diagnostics = EnsembleDiagnostics()
        
        # Get first sequence for testing
        seq_names = list(diagnostics.labels.keys())
        if not seq_names:
            print("No sequences found in labels!")
            return
            
        seq_name = seq_names[0]
        try:
            baseline_scores, gnn_scores = diagnostics.load_scores(seq_name)
            
            print(f"\nTesting on {seq_name}:")
            print(f"   Baseline range: [{baseline_scores.min():.4f}, {baseline_scores.max():.4f}]")
            print(f"   GNN range:      [{gnn_scores.min():.4f}, {gnn_scores.max():.4f}]")
            
            methods = ['minmax', 'zscore', 'robust', 'rank']
            
            for method in methods:
                normalizer = ScoreNormalizer(method=method)
                
                baseline_norm = normalizer.normalize(baseline_scores)
                gnn_norm = normalizer.normalize(gnn_scores)
                
                print(f"\n{method.upper()} Normalization:")
                print(f"   Baseline: [{baseline_norm.min():.4f}, {baseline_norm.max():.4f}], mean={baseline_norm.mean():.4f}")
                print(f"   GNN:      [{gnn_norm.min():.4f}, {gnn_norm.max():.4f}], mean={gnn_norm.mean():.4f}")
                
                # Check if ranges are now comparable
                baseline_range = baseline_norm.max() - baseline_norm.min()
                gnn_range = gnn_norm.max() - gnn_norm.min()
                
                if baseline_range > 0:
                    scale_ratio = gnn_range / baseline_range
                    print(f"   Scale ratio: {scale_ratio:.2f}")
                
        except Exception as e:
            print(f"Error loading real data: {e}")
            print("Real data test failed, showing simulated results only.")
            
    except ImportError:
        print("Cannot import diagnostics module. Showing simulated results only.")


def demonstrate_scale_fix():
    """
    Demonstrate how normalization fixes scale mismatch.
    """
    print("\n" + "="*70)
    print("DEMONSTRATING SCALE MISMATCH FIX")
    print("="*70)
    
    # Simulate your actual score ranges
    baseline_raw = np.random.uniform(0.04, 0.07, 1000)  # Small range like yours
    gnn_raw = np.random.uniform(0.45, 1.92, 1000)       # Large range like yours
    
    print(f"\nRAW SCORES:")
    print(f"   Baseline: [{baseline_raw.min():.4f}, {baseline_raw.max():.4f}]")
    print(f"   GNN:      [{gnn_raw.min():.4f}, {gnn_raw.max():.4f}]")
    
    # Simple average (what you were doing before)
    simple_avg = (baseline_raw + gnn_raw) / 2
    print(f"\nSIMPLE AVERAGE (NO NORMALIZATION):")
    print(f"   Result dominated by GNN: [{simple_avg.min():.4f}, {simple_avg.max():.4f}]")
    print(f"   Mean: {simple_avg.mean():.4f} (close to GNN mean: {gnn_raw.mean():.4f})")
    
    # Normalized average
    normalizer = ScoreNormalizer(method='minmax')
    baseline_norm = normalizer.normalize(baseline_raw)
    gnn_norm = normalizer.normalize(gnn_raw)
    normalized_avg = (baseline_norm + gnn_norm) / 2
    
    print(f"\nNORMALIZED AVERAGE:")
    print(f"   Baseline normalized: [{baseline_norm.min():.4f}, {baseline_norm.max():.4f}]")
    print(f"   GNN normalized:      [{gnn_norm.min():.4f}, {gnn_norm.max():.4f}]")
    print(f"   Result balanced:     [{normalized_avg.min():.4f}, {normalized_avg.max():.4f}]")
    print(f"   Mean: {normalized_avg.mean():.4f} (balanced between both)")


def main():
    """
    Run normalization testing.
    """
    test_normalization_methods()
    compare_normalization_on_real_data()
    demonstrate_scale_fix()
    
    print("\nNormalization testing complete!")
    print("   Review reports/normalization_comparison.png")


if __name__ == "__main__":
    main()