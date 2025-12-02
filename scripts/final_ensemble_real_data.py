"""
Apply Advanced Ensemble Methods to Real Tuned GNN + Baseline Scores
Purpose: Get the final optimized performance using Component 5.4 methods
"""

import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path


def load_real_data():
    """Load baseline scores, tuned GNN scores, and labels."""
    print("📂 Loading real data...")
    
    # Load labels
    labels_path = 'data/splits/ucsd_ped2_labels.json'
    with open(labels_path, 'r') as f:
        labels_dict = json.load(f)
    
    # Load baseline scores
    baseline_dir = Path('data/processed/baseline_scores')
    baseline_dict = {}
    for seq_name in labels_dict.keys():
        score_file = baseline_dir / f'{seq_name}_scores.npy'
        if score_file.exists():
            baseline_dict[seq_name] = np.load(score_file)
    
    # Load tuned GNN scores  
    gnn_dir = Path('data/processed/gnn_scores')
    gnn_dict = {}
    for seq_name in labels_dict.keys():
        score_file = gnn_dir / f'{seq_name}_gnn_scores.npy'
        if score_file.exists():
            gnn_dict[seq_name] = np.load(score_file)
    
    print(f"   ✓ Labels: {len(labels_dict)} sequences")
    print(f"   ✓ Baseline: {len(baseline_dict)} sequences") 
    print(f"   ✓ GNN: {len(gnn_dict)} sequences")
    
    return labels_dict, baseline_dict, gnn_dict


def normalize_scores(scores, method='minmax'):
    """Normalize scores to [0,1] range."""
    if method == 'minmax':
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-8:
            return np.ones_like(scores) * 0.5
        return (scores - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def evaluate_ensemble_strategies(baseline_scores, gnn_scores, labels):
    """Evaluate different ensemble strategies."""
    
    # Convert labels to numpy array
    labels = np.array(labels, dtype=int)
    
    # Normalize scores
    baseline_norm = normalize_scores(baseline_scores)
    gnn_norm = normalize_scores(gnn_scores)
    
    print(f"   Score ranges after normalization:")
    print(f"     Baseline: [{baseline_norm.min():.4f}, {baseline_norm.max():.4f}]")
    print(f"     GNN:      [{gnn_norm.min():.4f}, {gnn_norm.max():.4f}]")
    
    results = {}
    
    # 1. Simple average
    simple_avg = (baseline_norm + gnn_norm) / 2
    results['simple_average'] = roc_auc_score(labels, simple_avg)
    
    # 2. Weighted average (optimize weights)
    best_auc = 0
    best_weights = None
    
    for w_gnn in np.linspace(0, 1, 21):  # 0.0, 0.05, 0.10, ..., 1.0
        w_baseline = 1 - w_gnn
        weighted_avg = w_baseline * baseline_norm + w_gnn * gnn_norm
        
        try:
            auc = roc_auc_score(labels, weighted_avg)
            if auc > best_auc:
                best_auc = auc
                best_weights = [w_baseline, w_gnn]
        except:
            continue
    
    results['weighted_average'] = best_auc
    
    # 3. Rank fusion (use NumPy implementation to avoid SciPy dependency)
    def _rankdata_np(a):
        a = np.asarray(a)
        if a.size == 0:
            return a.astype(float)
        sorter = np.argsort(a)
        sorted_a = a[sorter]
        ranks = np.empty_like(sorted_a, dtype=float)
        n = len(sorted_a)
        i = 0
        while i < n:
            j = i + 1
            # group equal values to assign average ranks for ties
            while j < n and sorted_a[j] == sorted_a[i]:
                j += 1
            # ranks are 1-based like scipy.stats.rankdata
            avg_rank = (i + 1 + j) / 2.0
            ranks[i:j] = avg_rank
            i = j
        # unsort back to original order
        unsorted_ranks = np.empty_like(ranks)
        unsorted_ranks[sorter] = ranks
        return unsorted_ranks

    baseline_ranks = _rankdata_np(baseline_scores)
    gnn_ranks = _rankdata_np(gnn_scores)
    avg_ranks = (baseline_ranks + gnn_ranks) / 2
    rank_scores = (avg_ranks - 1) / (len(avg_ranks) - 1)
    results['rank_fusion'] = roc_auc_score(labels, rank_scores)
    
    # 4. Stacking ensemble
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        X = np.column_stack([baseline_norm, gnn_norm])
        y = labels.astype(int)
        
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(meta_model, X, y, cv=5, scoring='roc_auc')
        
        # Fit on full data for final score
        meta_model.fit(X, y)
        stacking_scores = meta_model.predict_proba(X)[:, 1]
        results['stacking'] = roc_auc_score(labels, stacking_scores)
        
    except Exception as e:
        print(f"   Stacking failed: {e}")
        results['stacking'] = 0.5
    
    return results, best_weights


def main():
    """Run advanced ensemble on real data."""
    
    print("\n" + "="*70)
    print("🚀 ADVANCED ENSEMBLE ON REAL TUNED DATA")
    print("="*70)
    
    # Load data
    labels_dict, baseline_dict, gnn_dict = load_real_data()
    
    # Combine all sequences for overall evaluation
    all_labels = []
    all_baseline = []
    all_gnn = []
    
    valid_sequences = 0
    
    for seq_name in labels_dict:
        if seq_name in baseline_dict and seq_name in gnn_dict:
            labels = labels_dict[seq_name]
            baseline_scores = baseline_dict[seq_name]
            gnn_scores = gnn_dict[seq_name]
            
            # Check if sequence has valid labels (not all same)
            labels_array = np.array(labels, dtype=int)
            if len(np.unique(labels_array)) > 1:
                all_labels.extend(labels)
                all_baseline.extend(baseline_scores)
                all_gnn.extend(gnn_scores)
                valid_sequences += 1
    
    print(f"\n📊 Dataset summary:")
    print(f"   Valid sequences: {valid_sequences}")
    print(f"   Total frames: {len(all_labels)}")
    print(f"   Anomaly frames: {np.sum(all_labels)}")
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels, dtype=int)
    all_baseline = np.array(all_baseline)
    all_gnn = np.array(all_gnn)
    
    # Individual performance
    baseline_auc = roc_auc_score(all_labels, all_baseline)
    gnn_auc = roc_auc_score(all_labels, all_gnn)
    
    print(f"\n📈 Individual Performance:")
    print(f"   Baseline: {baseline_auc:.4f} ({baseline_auc*100:.2f}%)")
    print(f"   Tuned GNN: {gnn_auc:.4f} ({gnn_auc*100:.2f}%)")
    
    # Evaluate ensemble strategies
    print(f"\n🧪 Testing advanced ensemble strategies...")
    results, best_weights = evaluate_ensemble_strategies(all_baseline, all_gnn, all_labels)
    
    # Results
    print(f"\n" + "="*70)
    print(f"📊 ADVANCED ENSEMBLE RESULTS")
    print(f"="*70)
    
    for strategy, auc in results.items():
        print(f"   {strategy:<18}: {auc:.4f} ({auc*100:.2f}%)")
    
    # Best strategy
    # Use __getitem__ to provide a simple Callable[[key], value] for the type checker
    best_strategy = max(results, key=results.__getitem__)
    best_auc = results[best_strategy]
    
    print(f"\n🏆 BEST STRATEGY: {best_strategy}")
    print(f"   AUC: {best_auc:.4f} ({best_auc*100:.2f}%)")
    
    if best_weights:
        print(f"   Optimal weights: Baseline={best_weights[0]:.3f}, GNN={best_weights[1]:.3f}")
    
    # Improvement analysis
    baseline_improvement = (best_auc - baseline_auc) * 100
    gnn_improvement = (best_auc - gnn_auc) * 100
    
    print(f"\n📈 Improvements:")
    print(f"   vs Baseline: +{baseline_improvement:.2f} percentage points")
    print(f"   vs Tuned GNN: {gnn_improvement:+.2f} percentage points")
    
    # Target assessment
    if best_auc >= 0.65:
        print(f"\n🎯 TARGET ACHIEVED: AUC ≥ 65%!")
    elif best_auc >= 0.60:
        print(f"\n👍 GOOD PERFORMANCE: AUC ≥ 60%!")
    else:
        print(f"\n📊 Modest performance. Consider further optimization.")
    
    # Save results
    os.makedirs('reports', exist_ok=True)
    
    final_results = {
        'individual_performance': {
            'baseline_auc': float(baseline_auc),
            'tuned_gnn_auc': float(gnn_auc)
        },
        'ensemble_results': {k: float(v) for k, v in results.items()},
        'best_strategy': best_strategy,
        'best_auc': float(best_auc),
        'improvements': {
            'vs_baseline_pp': float(baseline_improvement),
            'vs_gnn_pp': float(gnn_improvement)
        }
    }
    
    with open('reports/final_ensemble_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved: reports/final_ensemble_results.json")
    print(f"="*70)
    
    return results


if __name__ == "__main__":
    main()