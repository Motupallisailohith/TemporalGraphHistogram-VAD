#!/usr/bin/env python3
"""
Component 5.1c: Train Ablation Models
Purpose: Train GNN models on different feature modalities to determine contribution
"""

import os
import json
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


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


class AblationModelTrainer:
    """
    Train GNN models on different feature modalities.
    """
    
    def __init__(self, config):
        """
        Initialize trainer.
        
        Args:
            config: dict with training parameters
                - graph_dir: path to feature graphs
                - output_dir: where to save models
                - device: 'cuda' or 'cpu'
                - feature_types: list of feature types to train
                - model_params: GNN architecture parameters
                - training_params: learning rate, epochs, etc.
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        self.feature_types = config.get('feature_types', ['histogram', 'optical_flow', 'cnn', 'combined'])
        self.model_params = config.get('model_params', {
            'hidden_dim': 512,
            'latent_dim': 128,
            'dropout': 0.1
        })
        self.training_params = config.get('training_params', {
            'learning_rate': 0.001,
            'num_epochs': 50,
            'batch_size': 1  # Process one sequence at a time
        })
        
        print(f"🔧 Initializing AblationModelTrainer")
        print(f"   Device: {self.device}")
        print(f"   Feature types: {self.feature_types}")
        print(f"   Model params: {self.model_params}")
        print(f"   Training params: {self.training_params}")
        
        os.makedirs(config['output_dir'], exist_ok=True)
    
    def load_training_graphs(self, feature_type):
        """
        Load training graphs for a specific feature type.
        
        Args:
            feature_type: str, e.g., 'histogram', 'optical_flow', 'cnn', 'combined'
        
        Returns:
            list: training graphs
        """
        print(f"   Loading {feature_type} training graphs...")
        
        # Get the correct graph directory for this feature type
        if feature_type == 'combined':
            # For combined, we'll load histogram graphs as the base
            graph_dir = Path('data/processed/temporal_graphs_histogram')
        else:
            graph_dir = Path(f'data/processed/temporal_graphs_{feature_type}')
        
        # Get training sequences
        splits_file = 'data/splits/ucsd_ped2_splits.json'
        if os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            
            # Handle different split file formats
            if 'train_sequences' in splits:
                train_sequences = splits['train_sequences']
            elif 'train' in splits:
                # Extract sequence names from paths
                train_paths = splits['train']
                train_sequences = []
                for path in train_paths:
                    seq_name = os.path.basename(path)
                    train_sequences.append(seq_name)
            else:
                raise ValueError("Unknown splits file format")
        else:
            train_sequences = [f'Train{i:03d}' for i in range(1, 17)]
        
        training_graphs = []
        
        for seq_name in train_sequences:
            graph_file = graph_dir / f'{seq_name}_graph.npz'
            
            if graph_file.exists():
                try:
                    graph_data = np.load(graph_file)
                    
                    # Convert to PyTorch Geometric format using correct keys
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
                    
                    training_graphs.append((seq_name, graph))
                    
                except Exception as e:
                    print(f"     Warning: Failed to load {graph_file}: {e}")
        
        print(f"     Loaded {len(training_graphs)} training graphs")
        return training_graphs
    
    def get_input_dimension(self, feature_type):
        """Get input dimension for a feature type."""
        if feature_type == 'histogram':
            return 256
        elif feature_type == 'optical_flow':
            return 64
        elif feature_type == 'cnn':
            return 2048
        elif feature_type == 'combined':
            return 256 + 64 + 2048  # Sum of all features
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def train_single_model(self, feature_type):
        """
        Train GNN autoencoder for a single feature type.
        
        Args:
            feature_type: str, feature type to train on
        
        Returns:
            dict: training results
        """
        print(f"\n🚀 Training {feature_type} model...")
        
        # Load training data
        training_graphs = self.load_training_graphs(feature_type)
        
        if not training_graphs:
            print(f"   ❌ No training data found for {feature_type}")
            return None
        
        # Initialize model
        input_dim = self.get_input_dimension(feature_type)
        
        model = AblationGNNAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.model_params['hidden_dim'],
            latent_dim=self.model_params['latent_dim'],
            dropout=self.model_params['dropout']
        ).to(self.device)
        
        print(f"   Model architecture: {input_dim} → {self.model_params['hidden_dim']} → {self.model_params['latent_dim']}")
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.training_params['learning_rate']
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        epoch_losses = []
        
        num_epochs = self.training_params['num_epochs']
        
        for epoch in tqdm(range(num_epochs), desc=f"  Training {feature_type}"):
            epoch_loss = 0.0
            
            for seq_name, graph in training_graphs:
                # Move to device
                graph = graph.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                x_recon, z = model(graph.x, graph.edge_index, graph.edge_weight)
                
                # Reconstruction loss
                loss = criterion(x_recon, graph.x)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(training_graphs)
            epoch_losses.append(avg_loss)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"     Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save model
        model_path = os.path.join(self.config['output_dir'], f'gnn_ablation_{feature_type}.pth')
        torch.save(model.state_dict(), model_path)
        
        # Save training history
        history = {
            'feature_type': feature_type,
            'input_dim': input_dim,
            'model_params': self.model_params,
            'training_params': self.training_params,
            'epoch_losses': epoch_losses,
            'final_loss': epoch_losses[-1],
            'num_training_graphs': len(training_graphs)
        }
        
        history_path = os.path.join(self.config['output_dir'], f'training_history_{feature_type}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"   ✓ Final loss: {epoch_losses[-1]:.6f}")
        print(f"   ✓ Model saved: {model_path}")
        
        return {
            'model': model,
            'history': history,
            'feature_type': feature_type
        }
    
    def train_all_models(self):
        """Train models for all feature types."""
        print(f"\n📚 Training models for {len(self.feature_types)} feature types...")
        
        results = {}
        training_summary = {
            'trained_models': {},
            'failed_models': [],
            'training_time': 0
        }
        
        start_time = time.time()
        
        for feature_type in self.feature_types:
            try:
                result = self.train_single_model(feature_type)
                
                if result is not None:
                    results[feature_type] = result
                    training_summary['trained_models'][feature_type] = {
                        'final_loss': result['history']['final_loss'],
                        'input_dim': result['history']['input_dim'],
                        'num_epochs': len(result['history']['epoch_losses'])
                    }
                else:
                    training_summary['failed_models'].append(feature_type)
                    
            except Exception as e:
                print(f"   ❌ Failed to train {feature_type}: {e}")
                training_summary['failed_models'].append(feature_type)
        
        training_summary['training_time'] = time.time() - start_time
        
        # Save summary
        summary_path = os.path.join(self.config['output_dir'], 'ablation_training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        return results, training_summary


def main():
    """
    Train ablation models for all feature types.
    """
    print("\n" + "="*70)
    print("🚀 COMPONENT 5.1c: TRAIN ABLATION MODELS")
    print("="*70)
    print("Training GNN models on different feature modalities")
    
    config = {
        'output_dir': 'models/ablation_models',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'feature_types': ['histogram'],  # Start with histogram which is working
        'model_params': {
            'hidden_dim': 512,
            'latent_dim': 128,
            'dropout': 0.1
        },
        'training_params': {
            'learning_rate': 0.001,
            'num_epochs': 50,
            'batch_size': 1
        }
    }
    
    # Check prerequisites - verify histogram graphs exist
    histogram_dir = Path('data/processed/temporal_graphs_histogram')
    if not histogram_dir.exists():
        print(f"❌ Histogram graph directory not found: {histogram_dir}")
        print("   Run Component 5.1b first to build feature graphs")
        return
    
    # Check if any histogram graphs exist
    histogram_files = list(histogram_dir.glob('*.npz'))
    if not histogram_files:
        print(f"❌ No histogram graph files found in {histogram_dir}")
        return
    
    print(f"✓ Found {len(histogram_files)} histogram graph files")
    
    # Initialize trainer
    trainer = AblationModelTrainer(config)
    
    # Train all models
    results, summary = trainer.train_all_models()
    
    # Report results
    print("\n" + "="*70)
    print("📊 ABLATION TRAINING SUMMARY")
    print("="*70)
    
    if summary['trained_models']:
        print(f"✅ Successfully trained {len(summary['trained_models'])} models:")
        
        for feature_type, info in summary['trained_models'].items():
            print(f"   {feature_type}:")
            print(f"     Input dim: {info['input_dim']}")
            print(f"     Final loss: {info['final_loss']:.6f}")
            print(f"     Epochs: {info['num_epochs']}")
    
    if summary['failed_models']:
        print(f"\n❌ Failed to train {len(summary['failed_models'])} models:")
        for feature_type in summary['failed_models']:
            print(f"   {feature_type}")
    
    print(f"\n⏱️  Total training time: {summary['training_time']:.1f}s")
    
    if summary['trained_models']:
        print(f"\n📂 Models saved to: {config['output_dir']}")
        print(f"✅ Ablation model training complete!")
        print(f"🎯 Next: Run Component 5.1d (Evaluate Ablations)")
    else:
        print(f"\n❌ No models trained successfully")


if __name__ == "__main__":
    main()