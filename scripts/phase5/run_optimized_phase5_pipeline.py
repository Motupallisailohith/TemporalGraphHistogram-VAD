#!/usr/bin/env python3
"""
Updated Phase 5 Pipeline with Optimized Configurations
Incorporates findings from feature and graph structure ablation studies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

def run_optimized_phase5_pipeline():
    """
    Execute Phase 5 pipeline with optimized configurations from ablation studies.
    """
    
    print("=" * 80)
    print("🚀 OPTIMIZED PHASE 5 PIPELINE")
    print("=" * 80)
    print("Incorporating ablation study findings:")
    
    # Load ablation study results
    ablation_summary = None
    try:
        with open("reports/ablations/complete_ablation_summary.json", 'r') as f:
            ablation_summary = json.load(f)
        
        recommendations = ablation_summary['recommendations']
        print(f"✓ Primary feature: {recommendations['primary_feature']}")
        print(f"✓ Optimal window k: {recommendations['optimal_window_k']}")
        print(f"✓ Optimal edge weighting: {recommendations['optimal_edge_weighting']}")
        print()
        
    except FileNotFoundError:
        print("⚠️  Ablation summary not found. Using default configurations.")
        recommendations = {
            'primary_feature': 'optical_flow',
            'optimal_window_k': 5,
            'optimal_edge_weighting': 'binary'
        }
    
    # Component 5.1c: Feature Ablation (Already completed)
    print("📊 Component 5.1c: Feature Ablation Study")
    print("-" * 50)
    
    feature_results_path = "reports/ablations/feature_ablation_results.json"
    if Path(feature_results_path).exists():
        print("✅ Feature ablation already completed!")
        with open(feature_results_path, 'r') as f:
            feature_results = json.load(f)
        
        print("Feature performance ranking:")
        for i, (feature, auc) in enumerate(sorted(feature_results.items(), 
                                                 key=lambda x: x[1], reverse=True), 1):
            status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"  {status} {feature:<15}: {auc:.4f}")
    else:
        print("❌ Feature ablation not found. Run 5.1c_train_ablation_models_clean.py first")
        return False
    
    print()
    
    # Component 5.2: Graph Structure Ablation (Already completed)
    print("📊 Component 5.2: Graph Structure Ablation Study")
    print("-" * 50)
    
    graph_results_path = "reports/ablations/graph_structure_ablation.json"
    if Path(graph_results_path).exists():
        print("✅ Graph structure ablation already completed!")
        with open(graph_results_path, 'r') as f:
            graph_results = json.load(f)
        
        print("Window size performance:")
        window_results = graph_results['window_sizes']
        best_window_auc = max(window_results.values())
        for window, auc in sorted(window_results.items(), key=lambda x: int(x[0].split('=')[1])):
            status = "🏆" if auc == best_window_auc else "  "
            print(f"  {status} {window:<8}: {auc:.4f}")
        
        print("\nEdge weighting performance:")
        edge_results = graph_results['edge_weighting']
        best_edge_auc = max(edge_results.values())
        for scheme, auc in sorted(edge_results.items(), key=lambda x: x[1], reverse=True):
            status = "🏆" if auc == best_edge_auc else "  "
            print(f"  {status} {scheme:<12}: {auc:.4f}")
    else:
        print("❌ Graph structure ablation not found. Run 5.2_graph_structure_ablation.py first")
        return False
    
    print()
    
    # Summary and Next Steps
    print("🎯 OPTIMIZATION INSIGHTS")
    print("-" * 50)
    
    # Initialize analysis variable to prevent unbound errors
    analysis = None
    
    if ablation_summary is not None:
        analysis = ablation_summary['analysis']
        
        print(f"Best feature combination potential:")
        print(f"  • Primary: {analysis['best_feature']['type']} (AUC: {analysis['best_feature']['auc']:.4f})")
        print(f"  • Ensemble target: {analysis['performance_gaps']['expected_ensemble']:.4f}")
        print(f"  • Improvement gap: {analysis['performance_gaps']['improvement_potential']:.4f}")
        
        print(f"\nOptimal graph structure:")
        print(f"  • Window size: k={recommendations['optimal_window_k']}")
        print(f"  • Edge weighting: {recommendations['optimal_edge_weighting']}")
        print(f"  • Performance: {analysis['best_edge_scheme']['auc']:.4f} AUC")
    else:
        print("Ablation summary not available - using default configurations")
    
    print(f"\n📋 NEXT STEPS")
    print("-" * 50)
    print("1. ✅ Complete feature ablation study")
    print("2. ✅ Complete graph structure ablation study") 
    print("3. 🔄 Implement optimized configurations in main pipeline")
    print("4. 🔄 Test ensemble combinations with optimal features")
    print("5. 🔄 Validate performance improvements on full test set")
    
    # Implementation recommendations
    print(f"\n💡 IMPLEMENTATION RECOMMENDATIONS")
    print("-" * 50)
    print("To apply these optimizations:")
    print("1. Update FeatureGraphBuilder parameters:")
    print(f"   - window_k = {recommendations['optimal_window_k']}")
    print(f"   - similarity_weighted = {recommendations['optimal_edge_weighting'] == 'similarity'}")
    print()
    print("2. Focus ensemble development on:")
    print(f"   - {recommendations['primary_feature']} as primary feature")
    print("   - Complementary feature combinations")
    print()
    print("3. Expected performance improvements:")
    if analysis is not None:
        improvement = analysis['performance_gaps']['improvement_potential']
        print(f"   - Target AUC gain: +{improvement:.4f}")
        current_best = analysis['best_feature']['auc'] 
        target_auc = analysis['performance_gaps']['expected_ensemble']
        print(f"   - From {current_best:.4f} → {target_auc:.4f}")
    else:
        print("   - Analysis data not available")
    
    print("\n✅ Phase 5 ablation studies complete!")
    print("🎯 Ready for optimization implementation phase")
    
    return True

if __name__ == "__main__":
    success = run_optimized_phase5_pipeline()
    if success:
        print(f"\n🚀 Phase 5 pipeline executed successfully!")
    else:
        print(f"\n❌ Phase 5 pipeline encountered issues. Check previous components.")