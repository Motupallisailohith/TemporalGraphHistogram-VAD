#!/usr/bin/env python3
"""
UCSD Test001 Multi-Method Comparison
Compare baseline L2, GNN, ensemble, and baseline scores on the same sequence
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def compare_test001_scores():
    """Compare all available Test001 anomaly scores"""
    print("🎯 UCSD Test001 Multi-Method Comparison")
    print("=" * 60)
    
    # Define score files to compare
    score_files = {
        'baseline_l2': "data/processed/anomaly_scores/baseline_l2_scores/Test001_anomaly_scores.npy",
        'gnn': "data/processed/gnn_scores/Test001_gnn_scores.npy", 
        'ensemble': "data/processed/ensemble_scores/Test001_ensemble_scores.npy",
        'baseline': "data/processed/baseline_scores/Test001_scores.npy"
    }
    
    # Load all available scores
    scores_data = {}
    for method, file_path in score_files.items():
        if Path(file_path).exists():
            try:
                scores = np.load(file_path)
                scores_data[method] = scores
                print(f"✓ {method}: {len(scores)} scores")
                print(f"  Range: [{scores.min():.6f}, {scores.max():.6f}]")
                print(f"  Mean ± Std: {scores.mean():.6f} ± {scores.std():.6f}")
            except Exception as e:
                print(f"❌ Error loading {method}: {e}")
        else:
            print(f"⚠️  Missing {method}: {file_path}")
    
    if not scores_data:
        print("❌ No score files found!")
        return
    
    print(f"\n📊 Loaded {len(scores_data)} methods for Test001")
    
    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(18, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Raw score timelines
    plt.subplot(3, 3, 1)
    for i, (method, scores) in enumerate(scores_data.items()):
        color = colors[i % len(colors)]
        plt.plot(scores, label=f'{method}', linewidth=2, color=color, alpha=0.8)
    plt.xlabel('Frame Index')
    plt.ylabel('Raw Anomaly Score')
    plt.title('Test001 Raw Score Timelines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Normalized score timelines
    plt.subplot(3, 3, 2)
    normalized_scores = {}
    for i, (method, scores) in enumerate(scores_data.items()):
        # Min-max normalization
        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        normalized_scores[method] = normalized
        color = colors[i % len(colors)]
        plt.plot(normalized, label=f'{method}', linewidth=2, color=color, alpha=0.8)
    plt.xlabel('Frame Index')
    plt.ylabel('Normalized Anomaly Score [0,1]')
    plt.title('Test001 Normalized Score Timelines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Score distributions
    plt.subplot(3, 3, 3)
    for i, (method, scores) in enumerate(scores_data.items()):
        color = colors[i % len(colors)]
        plt.hist(scores, bins=30, alpha=0.6, label=f'{method}', density=True, color=color)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Score Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Box plots
    plt.subplot(3, 3, 4)
    box_data = list(scores_data.values())
    box_labels = list(scores_data.keys())
    bp = plt.boxplot(box_data, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.xticks(range(1, len(box_labels) + 1), box_labels, rotation=45)
    plt.ylabel('Anomaly Score')
    plt.title('Score Distribution Comparison')
    plt.grid(True, alpha=0.3)
    
    # 5. Cumulative distributions
    plt.subplot(3, 3, 5)
    for i, (method, scores) in enumerate(scores_data.items()):
        sorted_scores = np.sort(scores)
        y_vals = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        color = colors[i % len(colors)]
        plt.plot(sorted_scores, y_vals, label=f'{method}', linewidth=2, color=color)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Method correlation matrix
    if len(scores_data) >= 2:
        plt.subplot(3, 3, 6)
        methods = list(scores_data.keys())
        n_methods = len(methods)
        correlation_matrix = np.ones((n_methods, n_methods))
        
        # Compute correlations (handle different lengths)
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    scores1 = scores_data[method1]
                    scores2 = scores_data[method2]
                    min_len = min(len(scores1), len(scores2))
                    if min_len > 1:
                        corr = np.corrcoef(scores1[:min_len], scores2[:min_len])[0, 1]
                        correlation_matrix[i, j] = corr
        
        im = plt.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(n_methods), methods, rotation=45)
        plt.yticks(range(n_methods), methods)
        plt.title('Cross-Method Correlation Matrix')
        
        # Add correlation values
        for i in range(n_methods):
            for j in range(n_methods):
                plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                        ha='center', va='center', fontweight='bold')
    
    # 7. Score ranges comparison
    plt.subplot(3, 3, 7)
    methods = list(scores_data.keys())
    ranges = [scores_data[m].max() - scores_data[m].min() for m in methods]
    bars = plt.bar(methods, ranges, color=colors[:len(methods)], alpha=0.7)
    plt.ylabel('Score Range (Max - Min)')
    plt.title('Dynamic Range Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, range_val in zip(bars, ranges):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + range_val*0.01,
                f'{range_val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Percentile analysis
    plt.subplot(3, 3, 8)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for i, (method, scores) in enumerate(scores_data.items()):
        perc_vals = [np.percentile(scores, p) for p in percentiles]
        color = colors[i % len(colors)]
        plt.plot(percentiles, perc_vals, 'o-', label=method, linewidth=2, 
                markersize=6, color=color)
    plt.xlabel('Percentile')
    plt.ylabel('Anomaly Score')
    plt.title('Percentile Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Statistical summary table (as text)
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Create summary statistics
    summary_text = "STATISTICAL SUMMARY\n" + "="*25 + "\n\n"
    for method, scores in scores_data.items():
        summary_text += f"{method.upper()}:\n"
        summary_text += f"  Count: {len(scores)}\n"
        summary_text += f"  Mean: {scores.mean():.6f}\n"
        summary_text += f"  Std: {scores.std():.6f}\n"
        summary_text += f"  Min: {scores.min():.6f}\n"
        summary_text += f"  Max: {scores.max():.6f}\n"
        summary_text += f"  CV: {scores.std()/scores.mean():.1%}\n\n"
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("data/processed/anomaly_scores/comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / "test001_multi_method_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {plot_file}")
    
    plt.show()
    
    # Generate detailed analysis report
    generate_test001_report(scores_data, normalized_scores, output_dir)
    
    return scores_data

def generate_test001_report(scores_data, normalized_scores, output_dir):
    """Generate detailed text report for Test001 comparison"""
    
    report_file = output_dir / "test001_comparison_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("UCSD TEST001 MULTI-METHOD COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METHODS ANALYZED:\n")
        f.write("-" * 30 + "\n")
        for method in scores_data.keys():
            f.write(f"• {method.upper()}\n")
        f.write("\n")
        
        f.write("DETAILED STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Method':<15} {'Count':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Range':<12}\n")
        f.write("-" * 85 + "\n")
        
        for method, scores in scores_data.items():
            f.write(f"{method:<15} "
                   f"{len(scores):<8} "
                   f"{scores.mean():<12.6f} "
                   f"{scores.std():<12.6f} "
                   f"{scores.min():<12.6f} "
                   f"{scores.max():<12.6f} "
                   f"{scores.max()-scores.min():<12.6f}\n")
        
        f.write(f"\nPERFORMANCE RANKING (by mean score):\n")
        f.write("-" * 40 + "\n")
        sorted_methods = sorted(scores_data.keys(), 
                              key=lambda x: scores_data[x].mean(), reverse=True)
        for i, method in enumerate(sorted_methods, 1):
            mean_score = scores_data[method].mean()
            f.write(f"{i}. {method.upper()}: {mean_score:.6f}\n")
        
        f.write(f"\nSTABILITY ANALYSIS (by coefficient of variation):\n")
        f.write("-" * 50 + "\n")
        cv_data = {}
        for method, scores in scores_data.items():
            cv = scores.std() / scores.mean() if scores.mean() != 0 else float('inf')
            cv_data[method] = cv
        
        sorted_cv = sorted(cv_data.keys(), key=lambda x: cv_data[x])
        for i, method in enumerate(sorted_cv, 1):
            cv = cv_data[method]
            stability = "Highly Stable" if cv < 0.1 else "Stable" if cv < 0.2 else "Moderate" if cv < 0.5 else "Unstable"
            f.write(f"{i}. {method.upper()}: CV={cv:.1%} ({stability})\n")
        
        # Correlation analysis
        if len(scores_data) >= 2:
            f.write(f"\nCORRELATION ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            methods = list(scores_data.keys())
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i < j:  # Only upper triangle
                        scores1 = scores_data[method1]
                        scores2 = scores_data[method2]
                        min_len = min(len(scores1), len(scores2))
                        if min_len > 1:
                            corr = np.corrcoef(scores1[:min_len], scores2[:min_len])[0, 1]
                            f.write(f"{method1} vs {method2}: r={corr:.3f}\n")
        
        f.write(f"\nRECOMMENDATIONS:\n")
        f.write("-" * 20 + "\n")
        
        best_method = sorted_methods[0]
        most_stable = sorted_cv[0]
        
        f.write(f"• HIGHEST DETECTION SENSITIVITY: {best_method.upper()}\n")
        f.write(f"• MOST STABLE PERFORMANCE: {most_stable.upper()}\n")
        
        if 'ensemble' in scores_data:
            f.write(f"• ENSEMBLE DETECTED: Shows combined method performance\n")
        
        if best_method == most_stable:
            f.write(f"• RECOMMENDED: {best_method.upper()} (optimal balance)\n")
        else:
            f.write(f"• FOR DETECTION: Use {best_method.upper()}\n")
            f.write(f"• FOR STABILITY: Use {most_stable.upper()}\n")
    
    print(f"✅ Detailed report saved: {report_file}")

def main():
    """Main execution"""
    scores_data = compare_test001_scores()
    
    if scores_data:
        print(f"\n🎯 QUICK SUMMARY:")
        print("-" * 30)
        for method, scores in scores_data.items():
            print(f"{method.upper()}:")
            print(f"  Length: {len(scores)}")
            print(f"  Mean: {scores.mean():.6f}")
            print(f"  Range: [{scores.min():.6f}, {scores.max():.6f}]")
            print()

if __name__ == "__main__":
    main()