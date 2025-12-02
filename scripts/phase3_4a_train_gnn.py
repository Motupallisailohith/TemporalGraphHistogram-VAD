#!/usr/bin/env python3
"""
PHASE 3 - COMPONENT 4A: Train GNN Autoencoder
Purpose: Train a Graph Neural Network autoencoder on normal training sequences
Input: Training temporal graphs (Train001-Train016_temporal_graph.npz)
Output: Trained GNN model (models/gnn_autoencoder.pth)

Architecture: 2048 → 512 → 128 → 512 → 2048 (GCN layers)
Loss: MSE reconstruction error on node features
Training: 50 epochs on all normal sequences

This is the CORE INNOVATION of the thesis!
"""

import os
import json
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import wandb
from datetime import datetime


class GNNAutoencoder(nn.Module):
    """
    Graph Neural Network Autoencoder for temporal video anomaly detection.
    
    Architecture:
      Encoder: 2048 → 512 → 128 (compress to latent space)
      Decoder: 128 → 512 → 2048 (reconstruct original features)
      
    Key idea:
      - Learns to reconstruct NORMAL patterns
      - High reconstruction error = anomaly (unfamiliar pattern)
    """
    
    def __init__(self, input_dim=2048, hidden_dim=512, latent_dim=128):
        """
        Initialize GNN autoencoder.
        
        Args:
            input_dim (int): Input node feature dimension (2048 from ResNet50)
            hidden_dim (int): Hidden layer dimension (512)
            latent_dim (int): Latent space dimension (128)
        """
        super(GNNAutoencoder, self).__init__()
        
        # Store dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # ENCODER: Compress node features
        self.encoder_conv1 = GCNConv(input_dim, hidden_dim)  # 2048 → 512
        self.encoder_conv2 = GCNConv(hidden_dim, latent_dim)  # 512 → 128
        
        # DECODER: Reconstruct node features
        self.decoder_conv1 = GCNConv(latent_dim, hidden_dim)  # 128 → 512
        self.decoder_conv2 = GCNConv(hidden_dim, input_dim)   # 512 → 2048
    
    def encode(self, x, edge_index):
        """
        Encode node features to latent space.
        
        Args:
            x (Tensor): Node features (num_nodes, 2048)
            edge_index (Tensor): Edge connectivity (2, num_edges)
            
        Returns:
            Tensor: Latent representation (num_nodes, 128)
        """
        # Layer 1: 2048 → 512
        x = self.encoder_conv1(x, edge_index)
        x = F.relu(x)
        
        # Layer 2: 512 → 128
        x = self.encoder_conv2(x, edge_index)
        x = F.relu(x)
        
        return x
    
    def decode(self, z, edge_index):
        """
        Decode latent representation back to original space.
        
        Args:
            z (Tensor): Latent features (num_nodes, 128)
            edge_index (Tensor): Edge connectivity (2, num_edges)
            
        Returns:
            Tensor: Reconstructed features (num_nodes, 2048)
        """
        # Layer 1: 128 → 512
        z = self.decoder_conv1(z, edge_index)
        z = F.relu(z)
        
        # Layer 2: 512 → 2048
        z = self.decoder_conv2(z, edge_index)
        
        return z
    
    def forward(self, x, edge_index):
        """
        Complete forward pass (encode → decode).
        
        Args:
            x (Tensor): Node features (num_nodes, 2048)
            edge_index (Tensor): Edge connectivity (2, num_edges)
            
        Returns:
            Tensor: Reconstructed features (num_nodes, 2048)
        """
        # Encode
        z = self.encode(x, edge_index)
        
        # Decode
        x_recon = self.decode(z, edge_index)
        
        return x_recon


class GNNTrainer:
    """
    Trainer for GNN autoencoder on normal temporal graphs.
    """
    
    def __init__(self,
                 graph_dir='data/processed/temporal_graphs',
                 model_dir='models',
                 input_dim=2048,
                 hidden_dim=512,
                 latent_dim=128,
                 num_epochs=50,
                 learning_rate=0.001,
                 device=None,
                 enable_wandb=True):
        """
        Initialize GNN trainer.
        
        Args:
            graph_dir (str): Directory with temporal graphs
            model_dir (str): Where to save trained model
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            latent_dim (int): Latent space dimension
            num_epochs (int): Training epochs
            learning_rate (float): Learning rate
            device (str): Training device
            enable_wandb (bool): Enable W&B tracking
        """
        self.enable_wandb = enable_wandb
        
        if self.enable_wandb:
            wandb.init(  # type: ignore
                project="temporalgraph-vad-complete",
                name=f"phase4_gnn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["phase4", "gnn_training", "autoencoder"],
                config={
                    "phase": "4_gnn_training",
                    "architecture": "gnn_autoencoder",
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "latent_dim": latent_dim,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "optimizer": "Adam",
                    "loss_function": "mse"
                }
            )
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            device (str): 'cuda' or 'cpu'
        """
        self.graph_dir = Path(graph_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Initializing GNN Trainer")
        print(f"   Device: {self.device}")
        print(f"   Architecture: {input_dim} → {hidden_dim} → {latent_dim} → {hidden_dim} → {input_dim}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {learning_rate}")
    
    def load_temporal_graph(self, graph_path):
        """
        Load temporal graph from .npz file and convert to PyTorch Geometric Data.
        
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
    
    def load_training_graphs(self):
        """
        Load all training temporal graphs.
        
        Returns:
            list: List of PyTorch Geometric Data objects
        """
        # Find all training graph files
        train_files = sorted(self.graph_dir.glob('Train*_temporal_graph.npz'))
        
        if not train_files:
            print(f"\n❌ ERROR: No training graphs found in {self.graph_dir}")
            print(f"   Expected: Train001-Train016_temporal_graph.npz")
            return []
        
        print(f"\n📊 Loading training graphs...")
        print(f"   Found {len(train_files)} training sequences")
        
        # Load all graphs
        graphs = []
        for graph_path in tqdm(train_files, desc="Loading graphs"):
            graph = self.load_temporal_graph(graph_path)
            if graph is not None:
                graphs.append(graph)
        
        print(f"   ✓ Loaded {len(graphs)} training graphs")
        
        return graphs
    
    def train_model(self, training_graphs):
        """
        Train GNN autoencoder on normal training sequences.
        
        Args:
            training_graphs (list): List of PyTorch Geometric Data objects
            
        Returns:
            dict: Training history
        """
        print(f"\n🧠 TRAINING GNN AUTOENCODER")
        print("=" * 70)
        
        # Initialize model
        model = GNNAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Initialize loss function
        criterion = nn.MSELoss()
        
        # Training history
        history = {
            'epoch_losses': [],
            'graph_losses': [],
            'best_loss': float('inf'),
            'best_epoch': 0
        }
        
        # Model save paths
        best_model_path = self.model_dir / 'gnn_autoencoder_best.pth'
        final_model_path = self.model_dir / 'gnn_autoencoder.pth'
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            model.train()
            epoch_loss = 0.0
            
            # Process each training graph
            for graph in training_graphs:
                # Move graph to device
                graph = graph.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                x_recon = model(graph.x, graph.edge_index)
                
                # Compute reconstruction loss
                loss = criterion(x_recon, graph.x)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
            
            # Average loss for epoch
            avg_loss = epoch_loss / len(training_graphs)
            history['epoch_losses'].append(avg_loss)
            
            # Log to W&B
            if self.enable_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "best_loss": history['best_loss']
                })
            
            # Track best model
            if avg_loss < history['best_loss']:
                history['best_loss'] = avg_loss
                history['best_epoch'] = epoch + 1
                
                # Save best model
                torch.save(model.state_dict(), best_model_path)
                
                # Log best model update
                if self.enable_wandb:
                    wandb.log({"best_model_epoch": epoch + 1, "new_best_loss": avg_loss})  # type: ignore
            
            # Print progress (every 5 epochs)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{self.num_epochs} | Loss: {avg_loss:.6f} | Best: {history['best_loss']:.6f} (epoch {history['best_epoch']})")
        
        training_time = time.time() - start_time
        
        # Save final model
        torch.save(model.state_dict(), final_model_path)
        
        # Log final training statistics
        if self.enable_wandb:
            wandb.log({
                "training_complete": True,
                "final_loss": history['epoch_losses'][-1],
                "best_final_loss": history['best_loss'],
                "best_epoch_final": history['best_epoch'],
                "training_time_seconds": training_time,
                "total_epochs_completed": self.num_epochs,
                "phase4_gnn_training_complete": True
            })
        
        print(f"\n" + "=" * 70)
        print(f"✅ TRAINING COMPLETE")
        print(f"   ⏱️ Training time: {training_time:.2f} seconds")
        print(f"   📉 Final loss: {history['epoch_losses'][-1]:.6f}")
        print(f"   🏆 Best loss: {history['best_loss']:.6f} (epoch {history['best_epoch']})")
        print(f"   💾 Saved models:")
        print(f"      {final_model_path} (final)")
        print(f"      {best_model_path} (best)")
        print("=" * 70)
        
        # Save training history
        history['training_time_seconds'] = training_time
        history['num_graphs'] = len(training_graphs)
        history['num_epochs'] = self.num_epochs
        history['learning_rate'] = self.learning_rate
        history['device'] = str(self.device)
        
        history_path = self.model_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n📋 Training history saved: {history_path}")
        
        return history


def main():
    """
    Main execution function for Component 4A.
    """
    print("\n" + "=" * 70)
    print("🚀 PHASE 3 - COMPONENT 4A: TRAIN GNN AUTOENCODER")
    print("=" * 70)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   PyTorch version: {torch.__version__}")
        device = 'cuda'
    else:
        print("\n⚠️ No GPU detected. Training on CPU (will be slower).")
        device = 'cpu'
    
    # Check prerequisites
    graph_dir = Path('data/processed/temporal_graphs')
    train_graphs = list(graph_dir.glob('Train*_temporal_graph.npz'))
    
    if not train_graphs:
        print(f"\n❌ ERROR: No training graphs found!")
        print(f"   Expected: Train001-Train016_temporal_graph.npz in {graph_dir}")
        print(f"\n💡 Solution:")
        print(f"   1. Extract CNN features for training sequences:")
        print(f"      python scripts/extract_train_cnn_features.py")
        print(f"   2. Build training temporal graphs:")
        print(f"      python scripts/generate_training_graphs.py")
        return
    
    print(f"\n✓ Found {len(train_graphs)} training graphs")
    
    # Initialize trainer
    trainer = GNNTrainer(
        graph_dir='data/processed/temporal_graphs',
        model_dir='models',
        input_dim=2048,      # ResNet50 features
        hidden_dim=512,      # Compression layer
        latent_dim=128,      # Latent space
        num_epochs=50,       # Full training
        learning_rate=0.001,
        device=device
    )
    
    # Load training graphs
    training_graphs = trainer.load_training_graphs()
    
    if not training_graphs:
        print("\n❌ Failed to load training graphs")
        return
    
    # Train model
    history = trainer.train_model(training_graphs)
    
    print(f"\n🎉 COMPONENT 4A COMPLETE!")
    print(f"   📦 Trained model ready: models/gnn_autoencoder.pth")
    print(f"   🎯 Next step: Run Component 4B (GNN Scoring)")
    print(f"      python scripts/phase3_4b_score_gnn.py")
    
    # Finish W&B
    if trainer.enable_wandb:
        wandb.finish()  # type: ignore
    

if __name__ == '__main__':
    main()
