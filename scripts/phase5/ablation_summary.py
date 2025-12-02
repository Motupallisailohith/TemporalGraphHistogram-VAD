#!/usr/bin/env python3
"""
Component 5.1 Summary: Feature Ablation Study Results

Generate comprehensive summary of feature ablation results.
"""

import json
import os
from pathlib import Path

def generate_summary():
    """Generate comprehensive ablation study summary."""
    
    print("=" * 80)
    print("🔬 FEATURE ABLATION STUDY - COMPREHENSIVE SUMMARY")
    print("=" * 80)
    
    # Load results
    results_file = Path('reports/ablations/feature_ablation_results.json')
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        print("❌ Results file not found")
        return
    
    print("\n📊 ABLATION RESULTS:")
    print("-" * 40)
    
    # Sort by AUC performance
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (feature, auc) in enumerate(sorted_results, 1):
        performance = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"
        print(f"{performance} {rank}. {feature:20s}: AUC = {auc:.4f}")
    
    print("\n🔍 FEATURE ANALYSIS:")
    print("-" * 40)
    
    # Analyze each feature
    for feature, auc in sorted_results:
        if feature == 'histogram':
            interpretation = analyze_histogram(auc)
        elif feature == 'optical_flow':
            interpretation = analyze_optical_flow(auc)
        elif feature == 'cnn':
            interpretation = analyze_cnn(auc)
        else:
            interpretation = "Unknown feature type"
            
        print(f"\n📈 {feature.upper()} Features:")
        print(f"   AUC: {auc:.4f}")
        print(f"   {interpretation}")
    
    print("\n🎯 KEY INSIGHTS:")
    print("-" * 40)
    
    best_feature = sorted_results[0][0]
    best_auc = sorted_results[0][1]
    
    if best_auc > 0.6:
        overall_assessment = "Strong performance"
    elif best_auc > 0.5:
        overall_assessment = "Moderate performance"
    elif best_auc == 0.5:
        overall_assessment = "Random performance (no learning)"
    else:
        overall_assessment = "Below-random performance (potential issue)"
    
    print(f"• Best performing feature: {best_feature}")
    print(f"• Best AUC achieved: {best_auc:.4f}")
    print(f"• Overall assessment: {overall_assessment}")
    
    # Compare with baseline
    baseline_comparison()
    
    print("\n📋 RECOMMENDATIONS:")
    print("-" * 40)
    generate_recommendations(results)
    
    print("\n" + "=" * 80)

def analyze_histogram(auc):
    """Analyze histogram feature performance."""
    if auc > 0.5:
        return f"Histogram features show promise with {auc:.1%} AUC. The 256-bin grayscale histograms capture appearance patterns."
    elif auc == 0.5:
        return "Histogram features show random performance. May need better temporal modeling."
    else:
        return "Histogram features underperform. Potential issues with feature extraction or model training."

def analyze_optical_flow(auc):
    """Analyze optical flow feature performance.""" 
    if auc == 0.5:
        return "Optical flow shows random performance. Single-frame aggregation may lose motion information."
    elif auc > 0.5:
        return f"Optical flow features work well ({auc:.1%} AUC). Motion patterns help detect anomalies."
    else:
        return "Optical flow underperforms. May need better motion feature extraction."

def analyze_cnn(auc):
    """Analyze CNN feature performance."""
    if auc == 0.5:
        return "CNN features show random performance. ResNet50 features may need fine-tuning for anomaly detection."
    elif auc > 0.5:
        return f"CNN features perform well ({auc:.1%} AUC). Deep semantic features capture complex patterns."
    else:
        return "CNN features underperform. Consider different pre-trained models or feature layers."

def baseline_comparison():
    """Compare with baseline performance."""
    print("\n📐 BASELINE COMPARISON:")
    print("-" * 40)
    
    # Reference to existing baseline results if available
    baseline_file = Path('data/processed/baseline_scores/baseline_auc_results.json')
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        baseline_auc = baseline.get('overall_auc', 0.5)
        print(f"• Previous baseline AUC: {baseline_auc:.4f}")
    else:
        print("• No baseline results found for comparison")
    
    # Reference to Phase 4 results
    phase4_file = Path('reports/final_ensemble_results.json')
    if phase4_file.exists():
        with open(phase4_file, 'r') as f:
            phase4 = json.load(f)
        phase4_auc = phase4.get('test_auc', 0.629)  # Known Phase 4 result
        print(f"• Phase 4 ensemble AUC: {phase4_auc:.4f}")
        print(f"• Individual features vs ensemble: Feature ablation helps understand ensemble components")

def generate_recommendations(results):
    """Generate actionable recommendations."""
    best_auc = max(results.values())
    
    if best_auc <= 0.5:
        print("🔧 IMMEDIATE ACTIONS NEEDED:")
        print("   • Fix feature extraction pipeline")
        print("   • Check optical flow and CNN feature extraction")
        print("   • Verify temporal graph construction")
        print("   • Consider different GNN architectures")
    else:
        print("🚀 OPTIMIZATION OPPORTUNITIES:")
        
        # Histogram specific recommendations
        if 'histogram' in results and results['histogram'] > 0.4:
            print("   • Histogram features show potential - consider:")
            print("     - Different bin sizes (128, 512)")
            print("     - Color histograms instead of grayscale")
            print("     - Spatial histogram regions")
        
        # Feature combination recommendations
        print("   • Consider feature fusion strategies:")
        print("     - Early fusion (concatenate features)")
        print("     - Late fusion (ensemble predictions)")
        print("     - Attention-based fusion")
        
        print("   • Improve temporal modeling:")
        print("     - Longer temporal windows")
        print("     - LSTM/Transformer encoders")
        print("     - Dynamic graph structures")

if __name__ == "__main__":
    generate_summary()