#!/usr/bin/env python3
"""
Avenue Anomaly Scores Comparison Tool
Compare different model outputs (baseline, CNN, GNN, histogram, optical flow) 
for Avenue cross-dataset validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import stats

class AvenueScoresComparator:
    """Compare anomaly scores from different models/methods"""
    
    def __init__(self, scores_dir="data/processed/avenue/anomaly_scores"):
        self.scores_dir = Path(scores_dir)
        self.scores_data = {}
        self.results_dir = Path("data/processed/avenue/comparison_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_all_scores(self):
        """Load all available anomaly score files"""
        print("📊 Loading Avenue anomaly scores...")
        
        # Score types to look for
        score_types = {
            'histogram': 'histogram_scores/avenue_test_anomaly_scores.npy',
            'cnn': 'cnn_scores/avenue_test_anomaly_scores.npy', 
            'optical_flow': 'optical_flow_scores/avenue_test_anomaly_scores.npy',
            'baseline': 'baseline_scores/avenue_test_anomaly_scores.npy'  # if exists
        }
        
        # Also load UCSD baseline L2 scores for comparison
        ucsd_baseline_dir = Path("data/processed/anomaly_scores/baseline_l2_scores")
        if ucsd_baseline_dir.exists():
            print("  📋 Loading UCSD L2 baseline scores for comparison...")
            ucsd_baseline_scores = []
            for test_file in sorted(ucsd_baseline_dir.glob("Test*_anomaly_scores.npy")):
                try:
                    scores = np.load(test_file)
                    ucsd_baseline_scores.extend(scores)
                    print(f"    ✓ {test_file.name}: {len(scores)} scores")
                except Exception as e:
                    print(f"    ❌ Error loading {test_file}: {e}")
            
            if ucsd_baseline_scores:
                self.scores_data['ucsd_baseline_l2'] = np.array(ucsd_baseline_scores)
                print(f"    📊 Total UCSD baseline: {len(ucsd_baseline_scores)} scores")
        
        for method, score_file in score_types.items():
            score_path = self.scores_dir / score_file
            if score_path.exists():
                try:
                    scores = np.load(score_path)
                    self.scores_data[method] = scores
                    print(f"  ✓ {method}: {len(scores)} scores")
                    print(f"    Range: [{scores.min():.4f}, {scores.max():.4f}]")
                    print(f"    Mean ± Std: {scores.mean():.4f} ± {scores.std():.4f}")
                except Exception as e:
                    print(f"  ❌ Error loading {method}: {e}")
            else:
                print(f"  ⚠️  Missing {method}: {score_path}")
        
        if not self.scores_data:
            print("❌ No anomaly score files found!")
            return False
            
        print(f"\n✅ Loaded {len(self.scores_data)} score types")
        return True
    
    def compute_statistics(self):
        """Compute detailed statistics for each score type"""
        print("\n📈 Computing detailed statistics...")
        
        stats_data = {}
        
        for method, scores in self.scores_data.items():
            stats_data[method] = {
                'count': len(scores),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'mean': float(scores.mean()),
                'median': float(np.median(scores)),
                'std': float(scores.std()),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75)),
                'iqr': float(np.percentile(scores, 75) - np.percentile(scores, 25)),
                'skewness': float(stats.skew(scores)),
                'kurtosis': float(stats.kurtosis(scores)),
                'zero_count': int(np.sum(scores == 0)),
                'nan_count': int(np.sum(np.isnan(scores))),
                'unique_values': int(len(np.unique(scores)))
            }
        
        # Save statistics
        stats_file = self.results_dir / "avenue_scores_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"  ✓ Statistics saved to {stats_file}")
        return stats_data
    
    def create_comparison_plots(self):
        """Create comprehensive comparison visualizations"""
        print("\n📊 Creating comparison plots...")
        
        if len(self.scores_data) < 2:
            print("  ⚠️  Need at least 2 score types for comparison")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Score distributions (histograms)
        plt.subplot(3, 3, 1)
        for i, (method, scores) in enumerate(self.scores_data.items()):
            color = colors[i % len(colors)]
            plt.hist(scores, bins=50, alpha=0.7, label=f'{method}', density=True, color=color)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Box plots for comparison
        plt.subplot(3, 3, 2)
        box_data = [scores for scores in self.scores_data.values()]
        box_labels = list(self.scores_data.keys())
        bp = plt.boxplot(box_data)
        plt.xticks(range(1, len(box_labels) + 1), box_labels)
        plt.ylabel('Anomaly Score')
        plt.title('Score Distribution Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Score timelines
        plt.subplot(3, 3, 3)
        for i, (method, scores) in enumerate(self.scores_data.items()):
            # Show first 1000 points for clarity
            display_scores = scores[:1000] if len(scores) > 1000 else scores
            color = colors[i % len(colors)]
            plt.plot(display_scores, alpha=0.8, label=f'{method}', linewidth=1, color=color)
        plt.xlabel('Frame Index')
        plt.ylabel('Anomaly Score')
        plt.title('Score Timeline (first 1000 frames)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Cumulative distributions
        plt.subplot(3, 3, 4)
        for method, scores in self.scores_data.items():
            sorted_scores = np.sort(scores)
            y_vals = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
            plt.plot(sorted_scores, y_vals, label=f'{method}', linewidth=2)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Q-Q plots (if we have exactly 2 methods)
        if len(self.scores_data) == 2:
            plt.subplot(3, 3, 5)
            methods = list(self.scores_data.keys())
            scores1 = self.scores_data[methods[0]]
            scores2 = self.scores_data[methods[1]]
            
            # Sample same number of points for Q-Q plot
            min_len = min(len(scores1), len(scores2))
            q1 = np.percentile(scores1, np.linspace(0, 100, min_len))
            q2 = np.percentile(scores2, np.linspace(0, 100, min_len))
            
            plt.scatter(q1, q2, alpha=0.6, s=20)
            plt.plot([min(q1.min(), q2.min()), max(q1.max(), q2.max())], 
                    [min(q1.min(), q2.min()), max(q1.max(), q2.max())], 
                    'r--', linewidth=2, label='Perfect correlation')
            plt.xlabel(f'{methods[0]} Quantiles')
            plt.ylabel(f'{methods[1]} Quantiles')
            plt.title('Q-Q Plot Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. Score statistics radar chart
        plt.subplot(3, 3, 6)
        if len(self.scores_data) <= 4:  # Only if manageable number of methods
            stats_to_plot = ['mean', 'std', 'min', 'max']
            angles = np.linspace(0, 2*np.pi, len(stats_to_plot), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for method, scores in self.scores_data.items():
                values = [scores.mean(), scores.std(), scores.min(), scores.max()]
                # Normalize values to 0-1 scale for visualization
                max_vals = [max([self.scores_data[m].mean() for m in self.scores_data]),
                           max([self.scores_data[m].std() for m in self.scores_data]),
                           max([self.scores_data[m].min() for m in self.scores_data]),
                           max([self.scores_data[m].max() for m in self.scores_data])]
                normalized = [v/mv if mv > 0 else 0 for v, mv in zip(values, max_vals)]
                normalized += normalized[:1]  # Complete the circle
                
                plt.polar(angles, normalized, 'o-', linewidth=2, label=method)
            
            plt.xticks(angles[:-1], stats_to_plot)
            plt.title('Statistics Comparison')
            plt.legend()
        
        # 7. Pairwise correlations (if multiple methods)
        plt.subplot(3, 3, 7)
        if len(self.scores_data) >= 2:
            methods = list(self.scores_data.keys())
            correlation_matrix = np.zeros((len(methods), len(methods)))
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        # Compute correlation on common length
                        scores1 = self.scores_data[method1]
                        scores2 = self.scores_data[method2]
                        min_len = min(len(scores1), len(scores2))
                        corr = np.corrcoef(scores1[:min_len], scores2[:min_len])[0, 1]
                        correlation_matrix[i, j] = corr
            
            im = plt.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im)
            plt.xticks(range(len(methods)), methods, rotation=45)
            plt.yticks(range(len(methods)), methods)
            plt.title('Cross-Method Correlation Matrix')
            
            # Add correlation values as text
            for i in range(len(methods)):
                for j in range(len(methods)):
                    plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                            ha='center', va='center', fontweight='bold')
        
        # 8. Score range comparison
        plt.subplot(3, 3, 8)
        methods = list(self.scores_data.keys())
        min_vals = [self.scores_data[m].min() for m in methods]
        max_vals = [self.scores_data[m].max() for m in methods]
        ranges = [max_vals[i] - min_vals[i] for i in range(len(methods))]
        
        x_pos = np.arange(len(methods))
        plt.bar(x_pos, ranges, alpha=0.7)
        plt.xlabel('Method')
        plt.ylabel('Score Range (Max - Min)')
        plt.title('Anomaly Score Ranges')
        plt.xticks(x_pos, methods, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 9. Percentile comparison
        plt.subplot(3, 3, 9)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for method, scores in self.scores_data.items():
            perc_vals = [np.percentile(scores, p) for p in percentiles]
            plt.plot(percentiles, perc_vals, 'o-', label=method, linewidth=2, markersize=6)
        plt.xlabel('Percentile')
        plt.ylabel('Anomaly Score')
        plt.title('Percentile Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plot_file = self.results_dir / "avenue_scores_comprehensive_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Comprehensive comparison saved: {plot_file}")
        
        plt.show()
    
    def create_detailed_report(self, stats_data):
        """Create a detailed text report of the comparison"""
        print("\n📝 Creating detailed comparison report...")
        
        report_file = self.results_dir / "avenue_scores_comparison_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("AVENUE ANOMALY SCORES COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("METHODS ANALYZED:\n")
            f.write("-" * 30 + "\n")
            for method in self.scores_data.keys():
                f.write(f"• {method.upper()}\n")
            
            f.write("\nDETAILED STATISTICS:\n")
            f.write("-" * 30 + "\n")
            
            # Create a formatted table
            headers = ["Method", "Count", "Mean", "Std", "Min", "Max", "Median", "IQR"]
            f.write(f"{headers[0]:<15} {headers[1]:<8} {headers[2]:<8} {headers[3]:<8} {headers[4]:<8} {headers[5]:<8} {headers[6]:<8} {headers[7]:<8}\n")
            f.write("-" * 80 + "\n")
            
            for method, stats in stats_data.items():
                f.write(f"{method:<15} "
                       f"{stats['count']:<8} "
                       f"{stats['mean']:<8.4f} "
                       f"{stats['std']:<8.4f} "
                       f"{stats['min']:<8.4f} "
                       f"{stats['max']:<8.4f} "
                       f"{stats['median']:<8.4f} "
                       f"{stats['iqr']:<8.4f}\n")
            
            # Performance ranking
            f.write("\nPERFORMACE RANKING (by mean score):\n")
            f.write("-" * 40 + "\n")
            sorted_methods = sorted(stats_data.keys(), 
                                  key=lambda x: stats_data[x]['mean'], reverse=True)
            for i, method in enumerate(sorted_methods, 1):
                mean_score = stats_data[method]['mean']
                f.write(f"{i}. {method.upper()}: {mean_score:.6f}\n")
            
            # Stability analysis
            f.write("\nSTABILITY ANALYSIS (by coefficient of variation):\n")
            f.write("-" * 50 + "\n")
            cv_data = {}
            for method, stats in stats_data.items():
                cv = stats['std'] / stats['mean'] if stats['mean'] != 0 else float('inf')
                cv_data[method] = cv
            
            sorted_cv = sorted(cv_data.keys(), key=lambda x: cv_data[x])
            for i, method in enumerate(sorted_cv, 1):
                cv = cv_data[method]
                stability = "Highly Stable" if cv < 0.1 else "Stable" if cv < 0.2 else "Moderate" if cv < 0.5 else "Unstable"
                f.write(f"{i}. {method.upper()}: CV={cv:.4f} ({stability})\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            
            best_mean = sorted_methods[0]
            most_stable = sorted_cv[0]
            
            # Check if we have baseline comparison
            baseline_methods = [m for m in stats_data.keys() if 'baseline' in m.lower()]
            gnn_methods = [m for m in stats_data.keys() if 'baseline' not in m.lower()]
            
            f.write(f"• HIGHEST MEAN ANOMALY DETECTION: {best_mean.upper()}\n")
            f.write(f"• MOST STABLE METHOD: {most_stable.upper()}\n")
            
            if baseline_methods and gnn_methods:
                # Compare GNN vs Baseline performance
                best_baseline = max(baseline_methods, key=lambda x: stats_data[x]['mean'])
                best_gnn = max(gnn_methods, key=lambda x: stats_data[x]['mean'])
                
                baseline_score = stats_data[best_baseline]['mean']
                gnn_score = stats_data[best_gnn]['mean']
                improvement = ((gnn_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
                
                f.write(f"• BASELINE vs GNN COMPARISON:\n")
                f.write(f"  - Best Baseline ({best_baseline}): {baseline_score:.6f}\n")
                f.write(f"  - Best GNN ({best_gnn}): {gnn_score:.6f}\n")
                if improvement > 0:
                    f.write(f"  - GNN Improvement: +{improvement:.1f}% over baseline\n")
                else:
                    f.write(f"  - Performance Change: {improvement:.1f}%\n")
            
            if best_mean == most_stable:
                f.write(f"• RECOMMENDED METHOD: {best_mean.upper()} (combines high performance and stability)\n")
            else:
                f.write(f"• FOR MAXIMUM DETECTION: Use {best_mean.upper()}\n")
                f.write(f"• FOR CONSISTENCY: Use {most_stable.upper()}\n")
        
        print(f"  ✓ Detailed report saved: {report_file}")
    
    def run_full_comparison(self):
        """Run complete comparison analysis"""
        print("🔍 AVENUE ANOMALY SCORES COMPARISON ANALYSIS")
        print("=" * 60)
        
        # Load all available scores
        if not self.load_all_scores():
            return
        
        # Compute statistics
        stats_data = self.compute_statistics()
        
        # Create visualizations
        self.create_comparison_plots()
        
        # Create detailed report
        self.create_detailed_report(stats_data)
        
        print("\n✅ Comparison analysis complete!")
        print(f"📁 Results saved in: {self.results_dir}")
        
        return {
            'scores': self.scores_data,
            'statistics': stats_data
        }

def main():
    """Main execution function"""
    comparator = AvenueScoresComparator()
    results = comparator.run_full_comparison()
    
    if results:
        print("\n🎯 QUICK SUMMARY:")
        print("-" * 30)
        for method, scores in results['scores'].items():
            stats = results['statistics'][method]
            print(f"{method.upper()}:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std:  {stats['std']:.6f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print()

if __name__ == "__main__":
    main()