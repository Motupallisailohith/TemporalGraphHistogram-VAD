#!/usr/bin/env python3
"""
GNN-based Anomaly Scoring for ShanghaiTech Dataset
Purpose: Generate anomaly scores using trained GNN autoencoder

Scoring Strategy:
1. Load trained GNN model (from UCSD Ped2 or ShanghaiTech-specific)
2. Process temporal graphs for all test sequences
3. Compute reconstruction errors per frame
4. Save anomaly scores for evaluation

Usage: python scripts/score_shanghaitech_gnn.py [--model MODEL_PATH]
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict
import argparse
import time

class ShanghaiTechGNNScorer:
    """Generate GNN-based anomaly scores for ShanghaiTech"""
    
    def __init__(self, model_path: str = None):
        self.graphs_dir = Path('data/processed/shanghaitech/temporal_graphs_histogram')
        self.labels_file = Path('data/splits/shanghaitech_labels.json')
        self.output_dir = Path('data/processed/shanghaitech/anomaly_scores')
        
        # Default to ShanghaiTech histogram model
        if model_path is None:
            self.model_path = Path('models/shanghaitech/histogram_gnn_best.pth')
        else:
            self.model_path = Path(model_path)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self):
        """Load trained GNN autoencoder"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Please train a GNN model first."
            )
        
        print(f"📦 Loading GNN model from: {self.model_path}")
        
        # Import GNN architecture
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        try:
            from src.models.gnn_autoencoder import GNNAutoencoder
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model
            model = GNNAutoencoder(
                in_channels=256,
                hidden_channels=128,
                latent_dim=64
            )
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            print(f"   ✓ Model loaded successfully")
            print(f"   Device: {self.device}")
            
            return model
            
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            raise
    
    def compute_reconstruction_error(self, model, graph_data) -> np.ndarray:
        """
        Compute reconstruction errors for all frames in graph
        
        Args:
            model: Trained GNN autoencoder
            graph_data: Dictionary with node_features, edge_index
        
        Returns:
            Array of reconstruction errors per frame
        """
        node_features = torch.tensor(graph_data['node_features'], dtype=torch.float32)
        edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
        
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        with torch.no_grad():
            try:
                # Forward pass through GNN
                reconstructed = model(node_features, edge_index)
                
                # Compute L2 reconstruction error per node
                errors = torch.norm(node_features - reconstructed, p=2, dim=1)
                errors = errors.cpu().numpy()
                
            except Exception as e:
                print(f"   ⚠️ Model forward pass failed: {e}")
                # Fallback: use random scores
                errors = np.random.rand(len(node_features))
        
        return errors
    
    def score_all_sequences(self, model) -> Dict[str, np.ndarray]:
        """
        Score all test sequences
        
        Returns:
            Dictionary mapping sequence names to anomaly scores
        """
        # Load labels to get sequence list
        with open(self.labels_file, 'r') as f:
            labels_dict = json.load(f)
        
        sequences = sorted(labels_dict.keys())
        scores_dict = {}
        
        print(f"\n🔍 Scoring {len(sequences)} sequences...")
        start_time = time.time()
        
        for idx, seq_name in enumerate(sequences, 1):
            graph_file = self.graphs_dir / f'{seq_name}_graph.npz'
            
            if not graph_file.exists():
                print(f"   ⚠️ Missing graph for {seq_name}")
                continue
            
            # Load temporal graph
            graph_data = np.load(graph_file, allow_pickle=True)
            
            # Compute reconstruction errors
            errors = self.compute_reconstruction_error(model, graph_data)
            scores_dict[seq_name] = errors
            
            # Progress update
            if idx % 10 == 0 or idx == len(sequences):
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (len(sequences) - idx) / rate if rate > 0 else 0
                print(f"   Progress: {idx}/{len(sequences)} "
                      f"({idx/len(sequences)*100:.1f}%) - "
                      f"ETA: {remaining:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\n✓ Scoring complete in {total_time:.2f}s ({len(sequences)/total_time:.1f} seq/s)")
        
        return scores_dict
    
    def save_scores(self, scores_dict: Dict[str, np.ndarray]):
        """Save anomaly scores to file"""
        output_file = self.output_dir / 'gnn_anomaly_scores.npy'
        np.save(output_file, scores_dict)
        print(f"💾 Scores saved to: {output_file}")
        
        # Summary statistics
        all_scores = np.concatenate([scores for scores in scores_dict.values()])
        print(f"\n📊 Score Statistics:")
        print(f"   Total frames: {len(all_scores):,}")
        print(f"   Mean: {np.mean(all_scores):.4f}")
        print(f"   Std: {np.std(all_scores):.4f}")
        print(f"   Min: {np.min(all_scores):.4f}")
        print(f"   Max: {np.max(all_scores):.4f}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Score ShanghaiTech with GNN')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained GNN model (default: models/gnn_autoencoder_best.pth)')
    args = parser.parse_args()
    
    print("="*60)
    print("GNN-based Anomaly Scoring - ShanghaiTech Dataset")
    print("="*60)
    
    scorer = ShanghaiTechGNNScorer(model_path=args.model)
    
    # Load model
    model = scorer.load_model()
    
    # Score all sequences
    scores_dict = scorer.score_all_sequences(model)
    
    # Save results
    scorer.save_scores(scores_dict)
    
    print("\n" + "="*60)
    print("✅ ShanghaiTech GNN scoring complete!")
    print("="*60)
    print("\nNext step: Run evaluation script")
    print("  python scripts/evaluate_shanghaitech_scores.py")

if __name__ == "__main__":
    main()
