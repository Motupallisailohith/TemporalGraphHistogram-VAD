#!/usr/bin/env python3
"""
Component 5.1d: Evaluate Ablations
Purpose: Evaluate trained models on different features to determine contribution
"""

import os
import json
import torch  # type: ignore
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from tqdm import tqdm
import time
from torch_geometric.data import Data

# Import model architecture
import sys
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'scripts' / 'phase5'))

from ablation_models import AblationGNNAutoencoder


class AblationEvaluator:
    """
    Evaluate ablation models to determine feature contributions.
    """
    
    def __init__(self, config):
        """
        Initialize evaluator.
        
        Args:
            config: dict with evaluation parameters
                - model_dir: path to trained ablation models
                - output_dir: where to save evaluation results
                - device: 'cuda' or 'cpu'
                - feature_types: list of feature types to evaluate
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        print(f"🔧 Initializing AblationEvaluator")
        print(f"   Device: {self.device}")
        print(f"   Model dir: {config['model_dir']}")
        print(f"   Output dir: {config['output_dir']}")
        print(f"   Feature types: {config.get('feature_types', ['histogram'])}")
        
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Load test labels
        self.test_labels = self.load_test_labels()
    
    def load_test_labels(self):
        """Load test sequence labels."""
        labels_file = 'data/splits/ucsd_ped2_labels.json'
        
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                all_labels = json.load(f)
            
            # Filter for test sequences
            test_labels = {k: v for k, v in all_labels.items() if k.startswith('Test')}
            print(f"   ✓ Loaded labels for {len(test_labels)} test sequences")
            return test_labels
        else:
            print(f"   ❌ Labels file not found: {labels_file}")
            return {}
    
    def load_trained_model(self, feature_type):
        """
        Load trained ablation model.
        
        Args:
            feature_type: str, feature type
        
        Returns:
            model: loaded PyTorch model
        """
        model_dir = Path(self.config['model_dir'])
        model_path = model_dir / f'gnn_ablation_{feature_type}.pth'
        history_path = model_dir / f'training_history_{feature_type}.json'
        
        if not model_path.exists():
            print(f"   ❌ Model not found: {model_path}")
            return None
        
        # Load training history to get architecture
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            model_params = history['model_params']
            input_dim = history['input_dim']
        else:
            print(f"   ⚠️ History not found, using default params")
            model_params = {'hidden_dim': 512, 'latent_dim': 128, 'dropout': 0.1}
            input_dim = self.get_input_dimension(feature_type)
        
        # Initialize model
        model = AblationGNNAutoencoder(
            input_dim=input_dim,
            hidden_dim=model_params['hidden_dim'],
            latent_dim=model_params['latent_dim'],
            dropout=model_params['dropout']
        ).to(self.device)
        
        # Load weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print(f"   ✓ Loaded {feature_type} model ({input_dim}D)")
            return model
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            return None
    
    def get_input_dimension(self, feature_type):
        """Get input dimension for feature type."""
        dims = {
            'histogram': 256,
            'optical_flow': 64,
            'cnn': 2048,
            'combined': 256 + 64 + 2048
        }
        return dims.get(feature_type, 256)
    
    def load_test_graph(self, seq_name, feature_type):
        """
        Load test graph for evaluation.
        
        Args:
            seq_name: str, test sequence name
            feature_type: str, feature type
        
        Returns:
            graph: PyTorch Geometric Data object
        """
        # Get the correct graph directory for this feature type
        if feature_type == 'combined':
            # For combined, we'll use histogram graphs as the base
            graph_dir = Path('data/processed/temporal_graphs_histogram')
        else:
            graph_dir = Path(f'data/processed/temporal_graphs_{feature_type}')
            
        graph_file = graph_dir / f'{seq_name}_graph.npz'
        
        if not graph_file.exists():
            return None
        
        try:
            graph_data = np.load(graph_file)
            
            # Use correct key names from the saved graph files
            features = torch.from_numpy(graph_data['node_features']).float()
            edge_index = torch.from_numpy(graph_data['edge_index']).long()
            
            # Compute edge weights from adjacency matrix
            adj_matrix = graph_data['adjacency_matrix']
            edge_weights = []
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                weight = adj_matrix[src, dst]
                edge_weights.append(weight)
            edge_weight = torch.tensor(edge_weights).float()
            
            graph = Data(
                x=features,
                edge_index=edge_index,
                edge_weight=edge_weight
            )
            
            return graph.to(self.device)
            
        except Exception as e:
            print(f"     Warning: Failed to load {graph_file}: {e}")
            return None
    
    def compute_anomaly_scores(self, model, graph):
        """
        Compute anomaly scores using reconstruction error.
        
        Args:
            model: trained GNN autoencoder
            graph: test graph
        
        Returns:
            scores: np.ndarray of anomaly scores
        """
        with torch.no_grad():
            # Forward pass
            x_recon, z = model(graph.x, graph.edge_index, graph.edge_weight)
            
            # Compute reconstruction error per node (frame)
            reconstruction_error = torch.nn.functional.mse_loss(
                x_recon, graph.x, reduction='none'
            )
            
            # Average error across features for each frame
            scores = reconstruction_error.mean(dim=1).cpu().numpy()
        
        return scores
    
    def evaluate_single_model(self, feature_type):
        """
        Evaluate a single ablation model.
        
        Args:
            feature_type: str, feature type to evaluate
        
        Returns:
            dict: evaluation results
        """
        print(f"\n🔍 Evaluating {feature_type} model...")
        
        # Load model
        model = self.load_trained_model(feature_type)
        if model is None:
            return None
        
        # Get test sequences
        splits_file = 'data/splits/ucsd_ped2_splits.json'
        if os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            
            # Handle different split file formats
            if 'test_sequences' in splits:
                test_sequences = splits['test_sequences']
            elif 'test' in splits:
                # Extract sequence names from paths
                test_paths = splits['test']
                test_sequences = []
                for path in test_paths:
                    seq_name = os.path.basename(path)
                    test_sequences.append(seq_name)
            else:
                raise ValueError("Unknown splits file format")
        else:
            test_sequences = [f'Test{i:03d}' for i in range(1, 13)]
        
        all_scores = []
        all_labels = []
        sequence_results = {}
        
        valid_sequences = 0
        
        for seq_name in test_sequences:
            # Load test graph
            graph = self.load_test_graph(seq_name, feature_type)
            if graph is None:
                print(f"     Skipping {seq_name} (no graph)")
                continue
            
            # Get labels
            if seq_name not in self.test_labels:
                print(f"     Skipping {seq_name} (no labels)")
                continue
            
            labels = np.array(self.test_labels[seq_name], dtype=int)
            
            # Check if labels have variation (needed for AUC)
            if len(np.unique(labels)) < 2:
                print(f"     Skipping {seq_name} (no label variation)")
                continue
            
            # Compute anomaly scores
            scores = self.compute_anomaly_scores(model, graph)
            
            # Align scores and labels
            min_len = min(len(scores), len(labels))
            scores = scores[:min_len]
            labels = labels[:min_len]
            
            # Compute sequence AUC
            try:
                seq_auc = roc_auc_score(labels, scores)
                sequence_results[seq_name] = {
                    'auc': seq_auc,
                    'num_frames': len(scores),
                    'num_anomalies': labels.sum(),
                    'score_range': [float(scores.min()), float(scores.max())]
                }
                
                # Accumulate for overall AUC
                all_scores.extend(scores)
                all_labels.extend(labels)
                
                valid_sequences += 1
                print(f"     {seq_name}: AUC = {seq_auc:.4f}")
                
            except Exception as e:
                print(f"     Error with {seq_name}: {e}")
        
        if not all_scores:
            print(f"   ❌ No valid sequences for {feature_type}")
            return None
        
        # Overall AUC
        overall_auc = roc_auc_score(all_labels, all_scores)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        
        results = {
            'feature_type': feature_type,
            'overall_auc': overall_auc,
            'num_sequences': valid_sequences,
            'total_frames': len(all_scores),
            'total_anomalies': sum(all_labels),
            'sequence_results': sequence_results,
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        }
        
        print(f"   ✓ Overall AUC: {overall_auc:.4f} ({valid_sequences} sequences)")
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate all available ablation models."""
        print(f"\n📊 Evaluating all ablation models...")
        
        # Find available models
        model_dir = Path(self.config['model_dir'])
        model_files = list(model_dir.glob('gnn_ablation_*.pth'))
        
        feature_types = []
        for model_file in model_files:
            # Extract feature type from filename
            feature_type = model_file.stem.replace('gnn_ablation_', '')
            feature_types.append(feature_type)
        
        if not feature_types:
            print(f"   ❌ No ablation models found in {model_dir}")
            return {}
        
        print(f"   Found models for: {feature_types}")
        
        all_results = {}
        
        for feature_type in feature_types:
            try:
                result = self.evaluate_single_model(feature_type)
                if result is not None:
                    all_results[feature_type] = result
            except Exception as e:
                print(f"   ❌ Failed to evaluate {feature_type}: {e}")
        
        return all_results
    
    def generate_ablation_report(self, results):
        """
        Generate comprehensive ablation study report.
        
        Args:
            results: dict of evaluation results by feature type
        """
        print(f"\n📋 Generating ablation report...")
        
        if not results:
            print(f"   ❌ No results to report")
            return
        
        # Create summary table
        summary = {
            'ablation_results': {},
            'performance_ranking': [],
            'feature_contributions': {},
            'analysis': {}
        }
        
        # Collect AUC scores
        auc_scores = {}
        for feature_type, result in results.items():
            auc_scores[feature_type] = result['overall_auc']
            
            summary['ablation_results'][feature_type] = {
                'auc': result['overall_auc'],
                'num_sequences': result['num_sequences'],
                'total_frames': result['total_frames'],
                'anomaly_rate': result['total_anomalies'] / result['total_frames']
            }
        
        # Rank by performance
        ranked_features = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
        summary['performance_ranking'] = ranked_features
        
        # Analyze feature contributions
        if 'combined' in auc_scores:
            combined_auc = auc_scores['combined']
            
            for feature_type, auc in auc_scores.items():
                if feature_type != 'combined':
                    contribution = (auc / combined_auc) * 100
                    summary['feature_contributions'][feature_type] = {
                        'individual_auc': auc,
                        'contribution_percent': contribution,
                        'vs_combined': auc - combined_auc
                    }
        
        # Analysis insights
        best_feature = ranked_features[0][0] if ranked_features else None
        worst_feature = ranked_features[-1][0] if ranked_features else None
        
        summary['analysis'] = {
            'best_feature': best_feature,
            'worst_feature': worst_feature,
            'performance_gap': ranked_features[0][1] - ranked_features[-1][1] if len(ranked_features) > 1 else 0,
            'baseline_comparison': {
                'baseline_auc': 0.4816,  # From your previous results
                'improvements': {ft: auc - 0.4816 for ft, auc in auc_scores.items()}
            }
        }
        
        # Save results
        results_file = os.path.join(self.config['output_dir'], 'ablation_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'evaluation_results': results,
                'summary': summary
            }, f, indent=2)
        
        # Generate visualization
        self.plot_ablation_results(auc_scores, summary)
        
        print(f"   ✓ Report saved: {results_file}")
        
        return summary
    
    def plot_ablation_results(self, auc_scores, summary):
        """Create visualization of ablation results."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Subplot 1: AUC comparison
            plt.subplot(2, 2, 1)
            features = list(auc_scores.keys())
            aucs = list(auc_scores.values())
            
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(features)]
            bars = plt.bar(features, aucs, color=colors)
            
            plt.title('Feature Ablation Results')
            plt.ylabel('AUC Score')
            plt.ylim([0, 1])
            
            # Add value labels on bars
            for bar, auc in zip(bars, aucs):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{auc:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            
            # Subplot 2: Performance ranking
            plt.subplot(2, 2, 2)
            ranked = summary['performance_ranking']
            rank_features = [item[0] for item in ranked]
            rank_aucs = [item[1] for item in ranked]
            
            plt.barh(rank_features, rank_aucs, color='lightblue')
            plt.title('Performance Ranking')
            plt.xlabel('AUC Score')
            
            # Subplot 3: Feature contributions (if combined available)
            if 'feature_contributions' in summary and summary['feature_contributions']:
                plt.subplot(2, 2, 3)
                contrib_data = summary['feature_contributions']
                contrib_features = list(contrib_data.keys())
                contrib_values = [contrib_data[f]['contribution_percent'] for f in contrib_features]
                
                plt.pie(contrib_values, labels=contrib_features, autopct='%1.1f%%')
                plt.title('Feature Contributions')
            
            # Subplot 4: vs Baseline comparison
            plt.subplot(2, 2, 4)
            baseline_auc = summary['analysis']['baseline_comparison']['baseline_auc']
            improvements = summary['analysis']['baseline_comparison']['improvements']
            
            imp_features = list(improvements.keys())
            imp_values = list(improvements.values())
            
            colors = ['green' if x > 0 else 'red' for x in imp_values]
            plt.bar(imp_features, imp_values, color=colors, alpha=0.7)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title('Improvement vs Baseline')
            plt.ylabel('AUC Improvement')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            plot_file = os.path.join(self.config['output_dir'], 'ablation_analysis.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ Visualization saved: {plot_file}")
            
        except Exception as e:
            print(f"   Warning: Failed to create plots: {e}")


def main():
    """
    Evaluate all ablation models.
    """
    print("\n" + "="*70)
    print("🚀 COMPONENT 5.1d: EVALUATE ABLATIONS")
    print("="*70)
    print("Evaluating feature contribution through ablation studies")
    
    config = {
        'model_dir': 'models/ablation_models',
        'output_dir': 'reports/ablation_study',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'feature_types': ['histogram']  # Only histogram model was trained successfully
    }
    
    # Check prerequisites
    model_dir = Path(config['model_dir'])
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        print("   Run Component 5.1c first to train ablation models")
        return
    
    # Check if histogram graphs exist
    histogram_dir = Path('data/processed/temporal_graphs_histogram')
    if not histogram_dir.exists():
        print(f"❌ Histogram graph directory not found: {histogram_dir}")
        print("   Run Component 5.1b first to build feature graphs")
        return
    
    # Initialize evaluator
    evaluator = AblationEvaluator(config)
    
    # Evaluate all models
    results = evaluator.evaluate_all_models()
    
    if not results:
        print(f"\n❌ No models evaluated successfully")
        return
    
    # Generate report
    summary = evaluator.generate_ablation_report(results)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 ABLATION STUDY SUMMARY")
    print("="*70)
    
    if summary and summary.get('performance_ranking'):
        print(f"🏆 Performance Ranking:")
        for i, (feature_type, auc) in enumerate(summary['performance_ranking'], 1):
            print(f"   {i}. {feature_type:<15}: {auc:.4f} AUC")
        
        if summary.get('analysis'):
            best_feature = summary['analysis'].get('best_feature', 'Unknown')
            performance_gap = summary['analysis'].get('performance_gap', 0.0)
            
            print(f"\n🎯 Key Findings:")
            print(f"   Best feature: {best_feature}")
            print(f"   Performance gap: {performance_gap:.4f}")
        
        if summary.get('feature_contributions'):
            print(f"\n🧩 Feature Contributions:")
            for feature_type, contrib in summary['feature_contributions'].items():
                contribution_pct = contrib.get('contribution_percent', 0.0) if isinstance(contrib, dict) else 0.0
                print(f"   {feature_type}: {contribution_pct:.1f}% of combined performance")
        
        if summary.get('analysis', {}).get('baseline_comparison', {}).get('improvements'):
            baseline_improvements = summary['analysis']['baseline_comparison']['improvements']
            print(f"\n📈 vs Baseline (48.16% AUC):")
            for feature_type, improvement in baseline_improvements.items():
                improvement_pp = improvement * 100
                print(f"   {feature_type}: +{improvement_pp:.2f} percentage points")
    else:
        print("❌ No valid summary generated or no performance data available")
    
    print(f"\n📂 Detailed results: {config['output_dir']}")
    print(f"✅ Ablation study complete!")


if __name__ == "__main__":
    main()