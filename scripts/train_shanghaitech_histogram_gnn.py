#!/usr/bin/env python3
"""
Train GNN Autoencoder on ShanghaiTech Histogram Features
Purpose: Train detection-free anomaly detection model using temporal graphs

Training Strategy:
1. Load temporal graphs (built from 256-dim histograms)
2. Train GNN autoencoder to reconstruct normal patterns
3. Save best model for anomaly scoring

Architecture: 256 → 128 → 64 → 128 → 256

Usage: python scripts/train_shanghaitech_histogram_gnn.py
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import time
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.gnn_autoencoder import GNNAutoencoder

class ShanghaiTechGNNTrainer:
    """Train GNN autoencoder on ShanghaiTech histogram features"""
    
    def __init__(self):
        self.graphs_dir = Path('data/processed/shanghaitech/temporal_graphs_histogram')
        self.labels_file = Path('data/splits/shanghaitech_labels.json')
        self.output_dir = Path('models/shanghaitech')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_training_data(self):
        """Load temporal graphs for training (normal frames only)"""
        with open(self.labels_file, 'r') as f:
            labels_dict = json.load(f)
        
        print(f"📂 Loading training data (normal frames only)...")
        
        train_graphs = []
        total_frames = 0
        normal_frames = 0
        
        for seq_name in sorted(labels_dict.keys()):
            graph_file = self.graphs_dir / f'{seq_name}_graph.npz'
            
            if not graph_file.exists():
                continue
            
            # Load graph
            graph_data = np.load(graph_file, allow_pickle=True)
            node_features = graph_data['node_features']
            edge_index = graph_data['edge_index']
            
            # Load labels for this sequence
            seq_labels = np.array(labels_dict[seq_name])
            
            # Filter to normal frames only
            normal_mask = (seq_labels == 0)
            
            if np.sum(normal_mask) > 0:
                # Extract subgraph with normal frames
                normal_features = node_features[normal_mask]
                
                # Rebuild edge index for normal frames only
                # For simplicity, create a new k-NN graph on normal frames
                # Or use all normal frames with their features
                
                train_graphs.append({
                    'node_features': normal_features,
                    'edge_index': self._build_temporal_edges(len(normal_features), k=5)
                })
                
                total_frames += len(seq_labels)
                normal_frames += len(normal_features)
        
        print(f"   ✓ Loaded {len(train_graphs)} graphs")
        print(f"   Total frames: {total_frames:,}")
        print(f"   Normal frames: {normal_frames:,} ({normal_frames/total_frames*100:.1f}%)")
        print(f"   Anomalous frames: {total_frames-normal_frames:,} (excluded from training)")
        
        return train_graphs
    
    def _build_temporal_edges(self, num_nodes, k=5):
        """Build k-nearest temporal neighbor edges"""
        edges = []
        
        for i in range(num_nodes):
            # Forward neighbors
            for j in range(1, min(k+1, num_nodes-i)):
                edges.append([i, i+j])
            
            # Backward neighbors
            for j in range(1, min(k+1, i+1)):
                edges.append([i, i-j])
        
        if edges:
            return np.array(edges).T
        else:
            return np.array([[],[]])
    
    def train_model(self, train_graphs, epochs=50, learning_rate=0.001):
        """
        Train GNN autoencoder
        
        Args:
            train_graphs: List of graph dictionaries
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        print(f"\n🚀 Training GNN Autoencoder...")
        print(f"   Architecture: 256 → 128 → 64 → 128 → 256")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {learning_rate}")
        
        # Initialize model
        model = GNNAutoencoder(
            in_channels=256,
            hidden_channels=128,
            latent_dim=64
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        best_loss = float('inf')
        history = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            
            for graph in train_graphs:
                node_features = torch.tensor(graph['node_features'], dtype=torch.float32).to(self.device)
                edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).to(self.device)
                
                # Forward pass
                reconstructed = model(node_features, edge_index)
                
                # Compute loss
                loss = model.reconstruction_loss(node_features, reconstructed)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            history.append(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), self.output_dir / 'histogram_gnn_best.pth')
            
            # Progress update
            if (epoch + 1) % 5 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Best: {best_loss:.6f} - Time: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Training complete!")
        print(f"   Total time: {total_time:.2f}s ({total_time/epochs:.2f}s/epoch)")
        print(f"   Best loss: {best_loss:.6f}")
        print(f"   Model saved: {self.output_dir / 'histogram_gnn_best.pth'}")
        
        # Save training history
        history_data = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'best_loss': float(best_loss),
            'final_loss': float(avg_loss),
            'training_time': total_time,
            'history': [float(x) for x in history]
        }
        
        with open(self.output_dir / 'histogram_gnn_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)
        
        return model

def main():
    """Main execution function"""
    print("="*60)
    print("Train GNN Autoencoder - ShanghaiTech Histograms")
    print("="*60)
    
    trainer = ShanghaiTechGNNTrainer()
    
    # Load training data (normal frames only)
    train_graphs = trainer.load_training_data()
    
    if not train_graphs:
        print("❌ No training data found!")
        return
    
    # Train model
    model = trainer.train_model(train_graphs, epochs=50)
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Score test sequences: python scripts/score_shanghaitech_gnn.py --model models/shanghaitech/histogram_gnn_best.pth --features histogram")
    print("  2. Evaluate results: python scripts/evaluate_shanghaitech_scores.py")

if __name__ == "__main__":
    main()
