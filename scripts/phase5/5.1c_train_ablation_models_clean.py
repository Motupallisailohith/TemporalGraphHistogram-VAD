#!/usr/bin/env python3
"""
Component 5.1c: Train & Evaluate Feature Ablations

Train separate GNN autoencoders for different feature modalities to determine
feature contributions in ablation studies.
Inspired by ablation studies in [3][6].
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score  # type: ignore

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore
    print("Warning: wandb not available. Logging disabled.")


class GNNAutoencoder(nn.Module):
    """
    GNN-based autoencoder for feature reconstruction.
    Architecture follows [4][5] reconstruction-based approaches.
    """
    
    def __init__(self, in_dim, hidden_dim=512, latent_dim=128, dropout=0.1):
        super(GNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder1 = GCNConv(in_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder1 = GCNConv(latent_dim, hidden_dim)
        self.decoder2 = GCNConv(hidden_dim, in_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, x, edge_index, edge_weight=None):
        """Encode features to latent space."""
        x = F.relu(self.encoder1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = self.encoder2(x, edge_index, edge_weight)
        return x
    
    def decode(self, z, edge_index, edge_weight=None):
        """Decode from latent space."""
        z = F.relu(self.decoder1(z, edge_index, edge_weight))
        z = self.dropout(z)
        z = self.decoder2(z, edge_index, edge_weight)
        return z
    
    def forward(self, x, edge_index, edge_weight=None):
        """Full autoencoder forward pass."""
        z = self.encode(x, edge_index, edge_weight)
        x_recon = self.decode(z, edge_index, edge_weight)
        return x_recon


class AblationExperiment:
    """
    Run feature ablation experiments.
    Inspired by ablation studies in [3][6].
    """
    
    def __init__(self, use_wandb=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Initialize wandb if requested and available
        if self.use_wandb:
            wandb.init(  # type: ignore
                project="temporalgraph-vad-ablations",
                name="feature-ablation-5.1c",
                tags=["phase5", "feature-ablation", "ucsd-ped2"],
                config={
                    "experiment_type": "feature_ablation",
                    "dataset": "UCSD_Ped2",
                    "device": str(self.device),
                    "feature_types": ["histogram", "optical_flow", "cnn"]
                }
            )
            print("📊 W&B experiment tracking initialized")
        
        print(f"🔧 Initializing AblationExperiment")
        print(f"   Device: {self.device}")
    
    def get_feature_dim(self, feature_type):
        """
        Get input dimension for each feature type.
        
        Args:
            feature_type: str
        
        Returns:
            dim: int
        """
        dims = {
            'histogram': 256,
            'optical_flow': 64,
            'cnn': 2048
        }
        return dims[feature_type]
    
    def create_gnn_model(self, in_dim, hidden_dim=512, latent_dim=128):
        """
        Create GNN autoencoder.
        
        Architecture follows [4][5] reconstruction-based approaches.
        
        Args:
            in_dim: input feature dimension
            hidden_dim: hidden layer size
            latent_dim: bottleneck dimension
        
        Returns:
            model: GNNAutoencoder
        """
        model = GNNAutoencoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        ).to(self.device)
        
        return model
    
    def load_graphs(self, feature_type, split='train'):
        """
        Load graphs for training or testing.
        
        Args:
            feature_type: str
            split: 'train' or 'test'
        
        Returns:
            graphs: list of PyTorch Geometric Data objects
        """
        graph_dir = f'data/processed/temporal_graphs_{feature_type}'
        
        if split == 'train':
            sequences = [f'Train{i:03d}' for i in range(1, 17)]
        else:
            sequences = [f'Test{i:03d}' for i in range(1, 13)]
        
        graphs = []
        for seq in sequences:
            graph_path = f'{graph_dir}/{seq}_graph.npz'
            if os.path.exists(graph_path):
                try:
                    data = np.load(graph_path)
                    
                    # Convert to PyTorch Geometric Data
                    graph = Data(
                        x=torch.FloatTensor(data['node_features']),
                        edge_index=torch.LongTensor(data['edge_index'])
                    )
                    graphs.append(graph)
                except Exception as e:
                    print(f"     Warning: Failed to load {graph_path}: {e}")
        
        return graphs
    
    def train_gnn(self, model, train_graphs, num_epochs=50, lr=1e-4, feature_type="unknown"):
        """
        Train GNN autoencoder on normal training graphs.
        
        Args:
            model: GNNAutoencoder
            train_graphs: list of Data objects
            num_epochs: int
            lr: learning rate
            feature_type: str, type of features for wandb logging
        
        Returns:
            trained model
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for graph in train_graphs:
                graph = graph.to(self.device)
                
                # Forward
                x_recon = model(graph.x, graph.edge_index)
                
                # Loss
                loss = criterion(x_recon, graph.x)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_graphs)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({  # type: ignore
                    f"train_loss_{feature_type}": avg_loss,
                    f"epoch_{feature_type}": epoch
                })
            
            if (epoch + 1) % 10 == 0:
                print(f'    Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}')
        
        return model
    
    def evaluate_gnn(self, model, test_graphs, labels_dict):
        """
        Evaluate trained GNN on test set and compute AUC.
        
        Args:
            model: trained GNNAutoencoder
            test_graphs: list of test Data objects
            labels_dict: dict mapping seq_name to binary labels
        
        Returns:
            auc: float, frame-level AUC
        """
        model.eval()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for i, graph in enumerate(test_graphs):
                seq_name = f'Test{i+1:03d}'
                graph = graph.to(self.device)
                
                # Forward
                x_recon = model(graph.x, graph.edge_index)
                
                # Compute reconstruction error per frame
                errors = torch.mean((x_recon - graph.x) ** 2, dim=1)
                scores = errors.cpu().numpy()
                
                # Get labels
                if seq_name in labels_dict:
                    labels = labels_dict[seq_name]
                    
                    # Handle size mismatch - for single-frame features, repeat the score
                    if len(scores) == 1 and len(labels) > 1:
                        # Single node graph - repeat the score for all frames
                        scores = np.repeat(scores[0], len(labels))
                    elif len(scores) != len(labels):
                        print(f"     Warning: Size mismatch for {seq_name}: {len(scores)} scores vs {len(labels)} labels")
                        continue
                    
                    all_scores.extend(scores)
                    all_labels.extend(labels)
        
        # Compute AUC
        if len(all_scores) > 0 and len(set(all_labels)) > 1:
            all_scores = np.array(all_scores)
            all_labels = np.array(all_labels)
        # Compute AUC
        if len(all_scores) > 0 and len(set(all_labels)) > 1:
            all_scores = np.array(all_scores)
            all_labels = np.array(all_labels)
            auc = roc_auc_score(all_labels, all_scores)
        else:
            auc = 0.5  # Random performance
        
        return auc
    
    def run_single_feature_ablation(self, feature_type):
        """
        Train and evaluate GNN with a single feature type.
        
        Args:
            feature_type: str
        
        Returns:
            auc: float
        """
        print(f'\n📊 Testing {feature_type} features...')
        
        # Get feature dimension
        in_dim = self.get_feature_dim(feature_type)
        
        # Create model
        model = self.create_gnn_model(in_dim)
        
        # Load data
        train_graphs = self.load_graphs(feature_type, 'train')
        test_graphs = self.load_graphs(feature_type, 'test')
        
        if len(train_graphs) == 0:
            print(f'    ❌ No training graphs found for {feature_type}')
            return 0.5
        
        if len(test_graphs) == 0:
            print(f'    ❌ No test graphs found for {feature_type}')
            return 0.5
        
        # Load labels
        try:
            with open('data/splits/ucsd_ped2_labels.json', 'r') as f:
                labels_dict = json.load(f)
        except FileNotFoundError:
            print('    ❌ Labels file not found')
            return 0.5
        
        # Train
        print('    Training...')
        model = self.train_gnn(model, train_graphs, feature_type=feature_type)
        
        # Evaluate
        print('    Evaluating...')
        auc = self.evaluate_gnn(model, test_graphs, labels_dict)
        
        # Log AUC to wandb
        if self.use_wandb:
            wandb.log({  # type: ignore
                f"auc_{feature_type}": auc,
                f"test_samples_{feature_type}": len(test_graphs),
            })
        
        print(f'    ✓ AUC: {auc:.4f}')
        
        return auc
    
    def run_all_ablations(self):
        """
        Run complete feature ablation study.
        
        Returns:
            results: dict mapping feature_type to AUC
        """
        print('='*70)
        print('🔬 FEATURE ABLATION STUDY')
        print('='*70)
        
        feature_types = ['histogram', 'optical_flow', 'cnn']  # Test all available feature types
        
        for feat_type in feature_types:
            auc = self.run_single_feature_ablation(feat_type)
            self.results[feat_type] = auc
        
        # Print summary
        print('\n' + '='*70)
        print('📊 ABLATION RESULTS')
        print('='*70)
        
        for feat, auc in sorted(self.results.items(), key=lambda x: x[1], reverse=True):
            print(f'  {feat:20s}: {auc:.4f}')
        
        # Save results
        os.makedirs('reports/ablations', exist_ok=True)
        with open('reports/ablations/feature_ablation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Log final results summary to wandb
        if self.use_wandb:
            # Create comparison chart
            feature_names = list(self.results.keys())
            auc_values = list(self.results.values())
            
            # Log summary metrics
            wandb.log({  # type: ignore
                "best_feature_auc": max(auc_values),
                "worst_feature_auc": min(auc_values),
                "auc_range": max(auc_values) - min(auc_values)
            })
            
            # Create summary table
            wandb.log({  # type: ignore
                "feature_comparison_table": wandb.Table(  # type: ignore
                    data=[[name, auc] for name, auc in zip(feature_names, auc_values)],
                    columns=["feature_type", "auc"]
                )
            })
            
            wandb.finish()  # type: ignore
            print("📊 Results logged to W&B dashboard")
        
        return self.results


def main():
    """
    Run feature ablation experiments.
    """
    print("\n" + "="*70)
    print("🚀 COMPONENT 5.1c: TRAIN & EVALUATE FEATURE ABLATIONS")
    print("="*70)
    print("Training GNN models on different feature modalities")
    
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
    
    # Run experiment
    experiment = AblationExperiment()
    results = experiment.run_all_ablations()
    
    print('\n✅ Feature ablation complete!')
    print(f'🎯 Results saved to: reports/ablations/feature_ablation_results.json')


if __name__ == "__main__":
    main()