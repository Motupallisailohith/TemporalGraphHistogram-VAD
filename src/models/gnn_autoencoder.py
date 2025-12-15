"""
GNN Autoencoder Architecture for Temporal Graph Anomaly Detection

Architecture: 256 → 128 → 64 → 128 → 256
- Encoder: Two GraphSAGE layers with ReLU activation
- Latent: 64-dimensional compressed representation
- Decoder: Two GraphSAGE layers with ReLU activation
- Reconstruction: L2 distance from input histogram features

Training: Reconstruction loss on normal training sequences
Inference: High reconstruction error indicates anomaly
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GNNAutoencoder(nn.Module):
    """
    Graph Neural Network Autoencoder for histogram-based anomaly detection
    
    Args:
        in_channels: Input feature dimension (256 for histograms)
        hidden_channels: Hidden layer dimension (128)
        latent_dim: Bottleneck dimension (64)
    """
    
    def __init__(self, in_channels: int = 256, hidden_channels: int = 128, latent_dim: int = 64):
        super(GNNAutoencoder, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        
        # Encoder: 256 → 128 → 64
        self.encoder_conv1 = SAGEConv(in_channels, hidden_channels)
        self.encoder_conv2 = SAGEConv(hidden_channels, latent_dim)
        
        # Decoder: 64 → 128 → 256
        self.decoder_conv1 = SAGEConv(latent_dim, hidden_channels)
        self.decoder_conv2 = SAGEConv(hidden_channels, in_channels)
        
        self.activation = nn.ReLU()
    
    def encode(self, x, edge_index):
        """
        Encode node features to latent space
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Latent embeddings [num_nodes, latent_dim]
        """
        x = self.encoder_conv1(x, edge_index)
        x = self.activation(x)
        x = self.encoder_conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        """
        Decode latent embeddings to reconstruct input features
        
        Args:
            z: Latent embeddings [num_nodes, latent_dim]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Reconstructed features [num_nodes, in_channels]
        """
        x = self.decoder_conv1(z, edge_index)
        x = self.activation(x)
        x = self.decoder_conv2(x, edge_index)
        return x
    
    def forward(self, x, edge_index):
        """
        Full forward pass: encode → decode
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Reconstructed features [num_nodes, in_channels]
        """
        z = self.encode(x, edge_index)
        x_reconstructed = self.decode(z, edge_index)
        return x_reconstructed
    
    def reconstruction_loss(self, x, x_reconstructed):
        """
        Compute L2 reconstruction loss
        
        Args:
            x: Original features
            x_reconstructed: Reconstructed features
        
        Returns:
            Scalar loss value
        """
        return torch.mean((x - x_reconstructed) ** 2)
    
    def anomaly_score(self, x, x_reconstructed):
        """
        Compute per-node anomaly scores (L2 reconstruction error)
        
        Args:
            x: Original features [num_nodes, in_channels]
            x_reconstructed: Reconstructed features [num_nodes, in_channels]
        
        Returns:
            Per-node scores [num_nodes]
        """
        return torch.norm(x - x_reconstructed, p=2, dim=1)

class GNNTrainer:
    """Training utilities for GNN autoencoder"""
    
    def __init__(self, model, learning_rate=0.001, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def train_step(self, x, edge_index):
        """
        Single training step
        
        Args:
            x: Node features
            edge_index: Graph connectivity
        
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        x_reconstructed = self.model(x, edge_index)
        loss = self.model.reconstruction_loss(x, x_reconstructed)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, x, edge_index):
        """
        Evaluate reconstruction loss without gradient computation
        
        Args:
            x: Node features
            edge_index: Graph connectivity
        
        Returns:
            Loss value
        """
        self.model.eval()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        x_reconstructed = self.model(x, edge_index)
        loss = self.model.reconstruction_loss(x, x_reconstructed)
        
        return loss.item()

def create_gnn_autoencoder(in_channels=256, hidden_channels=128, latent_dim=64):
    """
    Factory function to create GNN autoencoder
    
    Args:
        in_channels: Input dimension (default: 256 for histograms)
        hidden_channels: Hidden layer dimension (default: 128)
        latent_dim: Latent space dimension (default: 64)
    
    Returns:
        Initialized GNN autoencoder model
    """
    return GNNAutoencoder(in_channels, hidden_channels, latent_dim)
