#!/usr/bin/env python3
"""
Complete Phase 5 Ablation Study Summary
Combines feature and graph structure ablation results for comprehensive analysis.
"""

import json
import os
from pathlib import Path

def load_ablation_results():
    """Load all ablation results."""
    
    # Load feature ablation results
    feature_results_path = "reports/ablations/feature_ablation_results.json"
    with open(feature_results_path, 'r') as f:
        feature_results = json.load(f)
    
    # Load graph structure ablation results
    graph_results_path = "reports/ablations/graph_structure_ablation.json"
    with open(graph_results_path, 'r') as f:
        graph_results = json.load(f)
    
    return feature_results, graph_results

def analyze_complete_ablation():
    """Analyze complete ablation study results."""
    
    feature_results, graph_results = load_ablation_results()
    
    print("=" * 80)
    print("🎯 COMPLETE PHASE 5 ABLATION STUDY SUMMARY")
    print("=" * 80)
    
    # Feature ablation analysis
    print("\n📊 COMPONENT 5.1c: FEATURE ABLATION RESULTS")
    print("-" * 50)
    feature_aucs = feature_results  # Direct access since it's already the results dict
    for feature_type, auc in sorted(feature_aucs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature_type:<15}: {auc:.4f}")
    
    best_feature = max(feature_aucs.items(), key=lambda x: x[1])
    worst_feature = min(feature_aucs.items(), key=lambda x: x[1])
    
    print(f"\n🏆 Best single feature: {best_feature[0]} (AUC: {best_feature[1]:.4f})")
    print(f"❌ Worst single feature: {worst_feature[0]} (AUC: {worst_feature[1]:.4f})")
    
    # Graph structure ablation analysis
    print("\n📊 COMPONENT 5.2: GRAPH STRUCTURE ABLATION RESULTS")
    print("-" * 50)
    
    print("Window size effects:")
    window_aucs = graph_results['window_sizes']
    for window, auc in sorted(window_aucs.items(), key=lambda x: int(x[0].split('=')[1])):
        interpretation = ""
        if window == "k=1":
            interpretation = " (too local)"
        elif window == "k=2":
            interpretation = " (baseline)"
        elif window == "k=3":
            interpretation = " (moderate)"
        elif window == "k=5":
            interpretation = " (global)"
        print(f"  {window:<8}: {auc:.4f}{interpretation}")
    
    print("\nEdge weighting schemes:")
    edge_aucs = graph_results['edge_weighting']
    for scheme, auc in sorted(edge_aucs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scheme:<12}: {auc:.4f}")
    
    # Best configurations
    best_window = max(window_aucs.items(), key=lambda x: x[1])
    best_edge = max(edge_aucs.items(), key=lambda x: x[1])
    
    print(f"\n🏆 Best window size: {best_window[0]} (AUC: {best_window[1]:.4f})")
    print(f"🏆 Best edge scheme: {best_edge[0]} (AUC: {best_edge[1]:.4f})")
    
    # Comparative analysis
    print("\n🔍 COMPARATIVE ANALYSIS")
    print("-" * 50)
    
    print("Feature importance ranking:")
    ranked_features = sorted(feature_aucs.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, auc) in enumerate(ranked_features, 1):
        print(f"  {i}. {feature:<15}: {auc:.4f}")
    
    print("\nGraph structure insights:")
    print(f"  • Window size impact: {max(window_aucs.values()) - min(window_aucs.values()):.4f} AUC range")
    print(f"  • Edge weighting impact: {max(edge_aucs.values()) - min(edge_aucs.values()):.4f} AUC range")
    
    # Performance gaps analysis
    print("\n📈 PERFORMANCE GAPS")
    print("-" * 50)
    
    # Compare with expected ensemble performance (62.90%)
    expected_ensemble = 0.6290
    best_single_auc = best_feature[1]
    best_graph_auc = max(max(window_aucs.values()), max(edge_aucs.values()))
    
    print(f"Expected ensemble AUC: {expected_ensemble:.4f}")
    print(f"Best single feature AUC: {best_single_auc:.4f}")
    print(f"Best graph structure AUC: {best_graph_auc:.4f}")
    print(f"Ensemble improvement potential: {expected_ensemble - best_single_auc:.4f}")
    
    # Recommendations
    print("\n🎯 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 50)
    
    print("1. Feature combination strategy:")
    print(f"   • Primary: {best_feature[0]} (strongest single performance)")
    print(f"   • Secondary: Consider ensemble with complementary features")
    
    print("\n2. Graph structure optimization:")
    print(f"   • Use window size k={best_window[0].split('=')[1]} for temporal connectivity")
    print(f"   • Apply {best_edge[0]} edge weighting for optimal performance")
    
    print("\n3. Next steps:")
    print("   • Implement optimized graph structure in main pipeline")
    print("   • Test feature ensemble combinations")
    print("   • Validate on full test set")
    
    # Save comprehensive summary
    summary_data = {
        'feature_ablation': feature_results,
        'graph_structure_ablation': graph_results,
        'analysis': {
            'best_feature': {
                'type': best_feature[0],
                'auc': best_feature[1]
            },
            'best_window': {
                'size': best_window[0],
                'auc': best_window[1]
            },
            'best_edge_scheme': {
                'type': best_edge[0],
                'auc': best_edge[1]
            },
            'performance_gaps': {
                'expected_ensemble': expected_ensemble,
                'best_single_feature': best_single_auc,
                'best_graph_structure': best_graph_auc,
                'improvement_potential': expected_ensemble - best_single_auc
            }
        },
        'recommendations': {
            'primary_feature': best_feature[0],
            'optimal_window_k': int(best_window[0].split('=')[1]),
            'optimal_edge_weighting': best_edge[0],
            'next_steps': [
                "Implement optimized graph structure",
                "Test feature ensemble combinations", 
                "Validate on full test set"
            ]
        }
    }
    
    # Ensure output directory exists
    os.makedirs("reports/ablations", exist_ok=True)
    
    # Save comprehensive summary
    summary_path = "reports/ablations/complete_ablation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n✅ Complete ablation summary saved to: {summary_path}")
    
    return summary_data

if __name__ == "__main__":
    summary_data = analyze_complete_ablation()