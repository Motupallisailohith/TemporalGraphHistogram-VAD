#!/usr/bin/env python3
"""
PHASE 3 - COMPONENT 4B: GNN Anomaly Scoring
Purpose: Score test sequences using trained GNN autoencoder
Input: 
  - Trained model (models/gnn_autoencoder.pth)
  - Test temporal graphs (Test001-Test012_temporal_graph.npz)
Output: Anomaly scores per frame (Test001-Test012_gnn_scores.npy)

Scoring Logic:
  1. Load trained GNN autoencoder
  2. For each test sequence:
     - Forward pass through model
     - Compute reconstruction error per frame
     - High error = anomaly (unfamiliar pattern)
  3. Save anomaly scores
"""

import os
import json
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# Import model architecture from training script
import sys
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'scripts'))

from phase3_4a_train_gnn import GNNAutoencoder


class GNNScorer:
    """
    Score test sequences using trained GNN autoencoder.
    
    High reconstruction error = anomaly
    Low reconstruction error = normal
    """
    
    def __init__(self,
                 model_path='models/gnn_autoencoder.pth',
                 graph_dir='data/processed/temporal_graphs',
                 output_dir='data/processed/gnn_scores',
                 input_dim=2048,
                 hidden_dim=512,
                 latent_dim=128,
                 device=None):
        """
        Initialize GNN scorer.
        
        Args:
            model_path (str): Path to trained model
            graph_dir (str): Directory with temporal graphs
            output_dir (str): Where to save anomaly scores
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            latent_dim (int): Latent space dimension
            device (str): 'cuda' or 'cpu'
        """
        self.model_path = Path(model_path)
        self.graph_dir = Path(graph_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Initializing GNN Scorer")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.model_path}")
        
        # Load trained model
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Load trained GNN autoencoder.
        
        Returns:
            GNNAutoencoder: Loaded model in eval mode
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Initialize model architecture
        model = GNNAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        ).to(self.device)
        
        # Load trained weights
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        # Set to evaluation mode
        model.eval()
        
        print(f"   ✓ Loaded trained model from {self.model_path}")
        
        return model
    
    def load_temporal_graph(self, graph_path):
        """
        Load temporal graph from .npz file.
        
        Args:
            graph_path (Path): Path to .npz file
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        try:
            # Load .npz file
            data = np.load(graph_path)
            
            # Extract components
            node_features = data['node_features']  # (num_nodes, 2048)
            edge_index = data['edge_index']        # (2, num_edges)
            
            # Convert to PyTorch tensors
            x = torch.tensor(node_features, dtype=torch.float32)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            
            # Create PyTorch Geometric Data object
            graph = Data(x=x, edge_index=edge_index)
            
            return graph
            
        except Exception as e:
            print(f"  ✗ Error loading {graph_path.name}: {e}")
            return None
    
    def compute_anomaly_scores(self, graph):
        """
        Compute per-frame anomaly scores using reconstruction error.
        
        Args:
            graph (Data): PyTorch Geometric Data object
            
        Returns:
            np.ndarray: Anomaly scores per frame (num_frames,)
        """
        # Move graph to device
        graph = graph.to(self.device)
        
        # Forward pass (no gradients needed)
        with torch.no_grad():
            x_recon = self.model(graph.x, graph.edge_index)
        
        # Compute reconstruction error per frame
        # MSE per frame: mean((original - reconstructed)^2) across feature dimensions
        errors = torch.mean((graph.x - x_recon) ** 2, dim=1)  # (num_frames,)
        
        # Convert to numpy
        scores = errors.cpu().numpy()
        
        return scores
    
    def score_sequence(self, seq_name):
        """
        Score one test sequence.
        
        Args:
            seq_name (str): Sequence name (e.g., 'Test001')
            
        Returns:
            dict: Scoring results or None if error
        """
        # Load temporal graph
        graph_path = self.graph_dir / f'{seq_name}_temporal_graph.npz'
        graph = self.load_temporal_graph(graph_path)
        
        if graph is None:
            return None
        
        # Compute anomaly scores
        scores = self.compute_anomaly_scores(graph)
        
        # Save scores
        output_path = self.output_dir / f'{seq_name}_gnn_scores.npy'
        np.save(output_path, scores)
        
        # Return results
        return {
            'status': 'success',
            'seq_name': seq_name,
            'num_frames': len(scores),
            'score_shape': scores.shape,
            'score_range': (float(scores.min()), float(scores.max())),
            'score_mean': float(scores.mean()),
            'score_std': float(scores.std()),
            'output_path': str(output_path)
        }
    
    def score_all_sequences(self):
        """
        Score all test sequences.
        
        Returns:
            dict: Scoring summary
        """
        # Find all test graph files
        test_files = sorted(self.graph_dir.glob('Test*_temporal_graph.npz'))
        
        if not test_files:
            print(f"\n❌ ERROR: No test graphs found in {self.graph_dir}")
            return {}
        
        # Extract sequence names
        test_sequences = [f.stem.replace('_temporal_graph', '') for f in test_files]
        
        print(f"\n🎯 SCORING TEST SEQUENCES WITH GNN")
        print("=" * 70)
        print(f"   Found {len(test_sequences)} test sequences")
        print("-" * 70)
        
        # Initialize tracking
        summary = {}
        total_frames = 0
        
        # Score each sequence
        for seq_name in tqdm(test_sequences, desc="Scoring sequences"):
            result = self.score_sequence(seq_name)
            
            if result is not None:
                summary[seq_name] = result
                total_frames += result['num_frames']
                
                print(f"  ✓ {seq_name}: {result['num_frames']} frames, "
                      f"score range [{result['score_range'][0]:.6f}, {result['score_range'][1]:.6f}], "
                      f"mean {result['score_mean']:.6f}")
            else:
                summary[seq_name] = {'status': 'failed'}
                print(f"  ✗ {seq_name}: Failed")
        
        # Compute overall statistics
        successful = sum(1 for v in summary.values() if v['status'] == 'success')
        
        summary_data = {
            'scoring_summary': summary,
            'total_sequences': len(test_sequences),
            'successful_sequences': successful,
            'total_frames': total_frames,
            'model_path': str(self.model_path),
            'device': str(self.device)
        }
        
        # Save summary
        summary_path = self.output_dir / 'gnn_scores_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n" + "=" * 70)
        print(f"✅ GNN SCORING COMPLETE")
        print(f"   📊 Processed: {successful}/{len(test_sequences)} sequences")
        print(f"   🎞️ Total frames: {total_frames}")
        print(f"   💾 Output directory: {self.output_dir}")
        print(f"   📋 Summary: {summary_path}")
        print("=" * 70)
        
        return summary_data


def main():
    """
    Main execution function for Component 4B.
    """
    print("\n" + "=" * 70)
    print("🚀 PHASE 3 - COMPONENT 4B: GNN ANOMALY SCORING")
    print("=" * 70)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("\n⚠️ No GPU detected. Running on CPU.")
        device = 'cpu'
    
    # Check prerequisites
    model_path = Path('models/gnn_autoencoder.pth')
    if not model_path.exists():
        print(f"\n❌ ERROR: Trained model not found!")
        print(f"   Expected: {model_path}")
        print(f"\n💡 Solution: Train GNN model first:")
        print(f"      python scripts/phase3_4a_train_gnn.py")
        return
    
    graph_dir = Path('data/processed/temporal_graphs')
    test_graphs = list(graph_dir.glob('Test*_temporal_graph.npz'))
    
    if not test_graphs:
        print(f"\n❌ ERROR: No test graphs found!")
        print(f"   Expected: Test001-Test012_temporal_graph.npz in {graph_dir}")
        return
    
    print(f"\n✓ Found trained model: {model_path}")
    print(f"✓ Found {len(test_graphs)} test graphs")
    
    # Initialize scorer
    scorer = GNNScorer(
        model_path='models/gnn_autoencoder.pth',
        graph_dir='data/processed/temporal_graphs',
        output_dir='data/processed/gnn_scores',
        input_dim=2048,
        hidden_dim=512,
        latent_dim=128,
        device=device
    )
    
    # Score all test sequences
    summary = scorer.score_all_sequences()
    
    if summary.get('successful_sequences') == summary.get('total_sequences'):
        print(f"\n🎉 ALL SEQUENCES SCORED SUCCESSFULLY!")
        print(f"\n📂 Generated files:")
        for seq_name in sorted(summary['scoring_summary'].keys()):
            if summary['scoring_summary'][seq_name]['status'] == 'success':
                shape = summary['scoring_summary'][seq_name]['score_shape']
                print(f"   {seq_name}_gnn_scores.npy: ({shape[0]},)")
        
        print(f"\n🎯 NEXT STEP: Evaluate GNN performance")
        print(f"   Expected: GNN ~85% AUC vs Baseline ~51% AUC")
        print(f"   Command: python scripts/evaluate_gnn_scores.py")
    else:
        failed = summary.get('total_sequences', 0) - summary.get('successful_sequences', 0)
        print(f"\n⚠️ {failed} sequences failed. Check logs above for details.")
    

if __name__ == '__main__':
    main()
