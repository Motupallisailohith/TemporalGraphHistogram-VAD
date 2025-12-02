#!/usr/bin/env python3
"""
Ablation Model Classes
Shared model definitions for ablation studies
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch_geometric.nn import GCNConv


class AblationGNNAutoencoder(nn.Module):
    """
    GNN Autoencoder for ablation studies.
    Configurable input dimension based on feature type.
    """
    
    def __init__(self, input_dim, hidden_dim=512, latent_dim=128, dropout=0.1):
        super(AblationGNNAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_conv1 = GCNConv(input_dim, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, latent_dim)
        
        # Decoder  
        self.decoder_conv1 = GCNConv(latent_dim, hidden_dim)
        self.decoder_conv2 = GCNConv(hidden_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, x, edge_index, edge_weight=None):
        """Encoder forward pass."""
        x = self.encoder_conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.encoder_conv2(x, edge_index, edge_weight)
        return x
    
    def decode(self, z, edge_index, edge_weight=None):
        """Decoder forward pass."""
        x = self.decoder_conv1(z, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.decoder_conv2(x, edge_index, edge_weight)
        return x
    
    def forward(self, x, edge_index, edge_weight=None):
        """Full autoencoder forward pass."""
        z = self.encode(x, edge_index, edge_weight)
        x_recon = self.decode(z, edge_index, edge_weight)
        return x_recon, z