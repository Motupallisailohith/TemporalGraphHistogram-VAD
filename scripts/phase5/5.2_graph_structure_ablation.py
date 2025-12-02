#!/usr/bin/env python3
"""
Component 5.2: Graph Structure Ablation

Test different graph connectivity patterns:
- Window sizes (k=1,2,3,5)
- Edge weighting schemes (binary vs similarity)
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
import wandb


def cosine_similarity_manual(vec1, vec2):
    """Manual cosine similarity computation."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class FeatureGraphBuilder:
    """
    Build temporal graphs with configurable connectivity patterns.
    """
    
    def __init__(self, feature_type='histogram', window_k=2, similarity_weighted=True):
        """
        Initialize graph builder.
        
        Args:
            feature_type: str, feature modality
            window_k: int, temporal window size
            similarity_weighted: bool, use similarity weights vs binary
        """
        self.feature_type = feature_type
        self.window_k = window_k
        self.similarity_weighted = similarity_weighted
        
        print(f"🔧 Initializing FeatureGraphBuilder")
        print(f"   Feature type: {feature_type}")
        print(f"   Window k: {window_k}")
        print(f"   Similarity weighted: {similarity_weighted}")
    
    def build_sequence_graph(self, sequence_features):
        """
        Build temporal graph for a single sequence.
        
        Args:
            sequence_features: np.array (num_frames, feature_dim)
            
        Returns:
            dict: graph data
        """
        num_frames, feature_dim = sequence_features.shape
        
        # Build adjacency matrix
        adjacency_matrix = np.zeros((num_frames, num_frames))
        
        # Connect each frame to k previous and k future frames
        for i in range(num_frames):
            for offset in range(1, self.window_k + 1):
                # Forward connections
                if i + offset < num_frames:
                    j = i + offset
                    if self.similarity_weighted:
                        # Compute cosine similarity
                        sim = cosine_similarity_manual(
                            sequence_features[i], 
                            sequence_features[j]
                        )
                        # Convert to positive weight (cosine in [-1,1] -> [0,1])
                        weight = (sim + 1) / 2
                    else:
                        # Binary connection
                        weight = 1.0
                    
                    adjacency_matrix[i, j] = weight
                    adjacency_matrix[j, i] = weight  # Symmetric
                
                # Backward connections handled by symmetry
        
        # Convert adjacency matrix to edge list
        edge_index = []
        edge_weights = []
        
        for i in range(num_frames):
            for j in range(num_frames):
                if adjacency_matrix[i, j] > 0:
                    edge_index.append([i, j])
                    edge_weights.append(adjacency_matrix[i, j])
        
        edge_index = np.array(edge_index).T if edge_index else np.zeros((2, 0))
        edge_weights = np.array(edge_weights) if edge_weights else np.array([])
        
        return {
            'node_features': sequence_features,
            'adjacency_matrix': adjacency_matrix,
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'num_nodes': num_frames,
            'num_edges': len(edge_weights),
            'feature_dim': feature_dim,
            'feature_type': self.feature_type,
            'window_k': self.window_k,
            'similarity_weighted': self.similarity_weighted
        }
    
    def build_all_sequences(self, sequences, output_dir):
        """
        Build graphs for all sequences.
        
        Args:
            sequences: list of sequence names
            output_dir: str, output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        source_dir = f'data/processed/multifeatures'
        
        for seq_name in tqdm(sequences, desc=f"Building graphs (k={self.window_k})"):
            # Load features
            feature_file = f'{source_dir}/{seq_name}_histogram.npy'
            if not os.path.exists(feature_file):
                continue
                
            try:
                sequence_features = np.load(feature_file)
                
                # Build graph
                graph_data = self.build_sequence_graph(sequence_features)
                
                # Save graph
                output_file = f'{output_dir}/{seq_name}_graph.npz'
                np.savez(output_file, **graph_data)
                
            except Exception as e:
                print(f"Warning: Failed to build graph for {seq_name}: {e}")


class GNNAutoencoder(nn.Module):
    """
    GNN autoencoder for graph structure ablation.
    """
    
    def __init__(self, in_dim, hidden_dim=256, latent_dim=64):
        super(GNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder1 = GCNConv(in_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder1 = GCNConv(latent_dim, hidden_dim)
        self.decoder2 = GCNConv(hidden_dim, in_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass."""
        # Encode
        z = F.relu(self.encoder1(x, edge_index, edge_weight))
        z = self.dropout(z)
        z = self.encoder2(z, edge_index, edge_weight)
        
        # Decode
        x_recon = F.relu(self.decoder1(z, edge_index, edge_weight))
        x_recon = self.dropout(x_recon)
        x_recon = self.decoder2(x_recon, edge_index, edge_weight)
        
        return x_recon


class GraphStructureAblation:
    """
    Test different graph connectivity patterns.
    """
    
    def __init__(self, use_wandb=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = 256  # Histogram feature dimension
        self.use_wandb = use_wandb
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(  # type: ignore
                project="temporalgraph-vad-ablations",
                name="graph-structure-ablation-5.2",
                tags=["phase5", "graph-structure", "ucsd-ped2"],
                config={
                    "experiment_type": "graph_structure_ablation",
                    "dataset": "UCSD_Ped2", 
                    "device": str(self.device),
                    "window_sizes": [1, 2, 3, 5],
                    "edge_schemes": ["binary", "similarity"]
                }
            )
            print("📊 W&B experiment tracking initialized")
        
        print(f"🔧 Initializing GraphStructureAblation")
        print(f"   Device: {self.device}")
    
    def get_sequences(self):
        """Get train/test sequences."""
        # Training sequences
        train_sequences = [f'Train{i:03d}' for i in range(1, 17)]
        
        # Test sequences  
        test_sequences = [f'Test{i:03d}' for i in range(1, 13)]
        
        return train_sequences, test_sequences
    
    def load_graphs(self, graph_dir, sequences):
        """Load graphs from directory."""
        graphs = []
        
        for seq_name in sequences:
            graph_file = f'{graph_dir}/{seq_name}_graph.npz'
            if os.path.exists(graph_file):
                try:
                    data = np.load(graph_file)
                    
                    graph = Data(
                        x=torch.FloatTensor(data['node_features']),
                        edge_index=torch.LongTensor(data['edge_index']),
                        edge_weight=torch.FloatTensor(data['edge_weights']) if len(data['edge_weights']) > 0 else None
                    )
                    graphs.append(graph)
                    
                except Exception as e:
                    print(f"Warning: Failed to load {graph_file}: {e}")
        
        return graphs
    
    def train_and_evaluate(self, graph_dir):
        """
        Train GNN on graphs and evaluate performance.
        
        Args:
            graph_dir: str, directory containing graph files
            
        Returns:
            float: AUC score
        """
        train_sequences, test_sequences = self.get_sequences()
        
        # Load graphs
        train_graphs = self.load_graphs(graph_dir, train_sequences)
        test_graphs = self.load_graphs(graph_dir, test_sequences)
        
        if len(train_graphs) == 0 or len(test_graphs) == 0:
            print(f"    ❌ No graphs found in {graph_dir}")
            return 0.5
        
        # Create model
        model = GNNAutoencoder(self.feature_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        # Train model
        model.train()
        for epoch in range(30):  # Shorter training for ablation
            total_loss = 0.0
            
            for graph in train_graphs:
                graph = graph.to(self.device)
                
                x_recon = model(graph.x, graph.edge_index, graph.edge_weight)
                loss = criterion(x_recon, graph.x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        
        # Load test labels
        try:
            with open('data/splits/ucsd_ped2_labels.json', 'r') as f:
                labels_dict = json.load(f)
        except:
            return 0.5
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for i, graph in enumerate(test_graphs):
                seq_name = f'Test{i+1:03d}'
                graph = graph.to(self.device)
                
                x_recon = model(graph.x, graph.edge_index, graph.edge_weight)
                errors = torch.mean((x_recon - graph.x) ** 2, dim=1)
                scores = errors.cpu().numpy()
                
                if seq_name in labels_dict:
                    labels = labels_dict[seq_name]
                    
                    # Handle size mismatch
                    if len(scores) != len(labels):
                        min_len = min(len(scores), len(labels))
                        scores = scores[:min_len]
                        labels = labels[:min_len]
                    
                    all_scores.extend(scores)
                    all_labels.extend(labels)
        
        # Compute AUC
        if len(all_scores) > 0 and len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_scores)
        else:
            auc = 0.5
        
        return auc
    
    def test_window_sizes(self):
        """
        Test window_k = 1, 2, 3, 5
        """
        print(f"\\n🔍 Testing Window Sizes...")
        
        window_sizes = [1, 2, 3, 5]
        results = {}
        
        train_sequences, test_sequences = self.get_sequences()
        all_sequences = train_sequences + test_sequences
        
        for k in window_sizes:
            print(f'\\n📊 Testing window k={k}...')
            
            # Rebuild graphs with this window size
            builder = FeatureGraphBuilder(
                feature_type='histogram',
                window_k=k,
                similarity_weighted=True
            )
            
            temp_dir = f'data/temp/graphs_k{k}'
            builder.build_all_sequences(all_sequences, temp_dir)
            
            # Train and evaluate
            auc = self.train_and_evaluate(temp_dir)
            results[f'k={k}'] = auc
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({  # type: ignore
                    f"window_k{k}_auc": auc,
                    "window_size": k
                })
            
            print(f'    ✓ AUC: {auc:.4f}')
            
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        return results
    
    def test_edge_weighting(self):
        """
        Test binary vs similarity-weighted edges
        """
        print(f"\\n🔍 Testing Edge Weighting Schemes...")
        
        results = {}
        
        train_sequences, test_sequences = self.get_sequences()
        all_sequences = train_sequences + test_sequences
        
        for weighted in [False, True]:
            scheme_name = "similarity" if weighted else "binary"
            print(f'\\n📊 Testing {scheme_name} edges...')
            
            builder = FeatureGraphBuilder(
                feature_type='histogram',
                window_k=2,  # Use standard window size
                similarity_weighted=weighted
            )
            
            temp_dir = f'data/temp/graphs_weighted_{weighted}'
            builder.build_all_sequences(all_sequences, temp_dir)
            
            auc = self.train_and_evaluate(temp_dir)
            results[scheme_name] = auc
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({  # type: ignore
                    f"edge_{scheme_name}_auc": auc,
                    "edge_weighting": scheme_name
                })
            
            print(f'    ✓ AUC: {auc:.4f}')
            
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        return results
    
    def run_complete_ablation(self):
        """
        Run complete graph structure ablation study.
        
        Returns:
            dict: all results
        """
        print("="*70)
        print("📊 GRAPH STRUCTURE ABLATION")
        print("="*70)
        
        all_results = {}
        
        # Test window sizes
        window_results = self.test_window_sizes()
        all_results['window_sizes'] = window_results
        
        # Test edge weighting
        edge_results = self.test_edge_weighting()
        all_results['edge_weighting'] = edge_results
        
        # Print summary
        print("\\n" + "="*70)
        print("📊 GRAPH STRUCTURE ABLATION RESULTS")
        print("="*70)
        
        print("\\nWindow size:")
        for config, auc in window_results.items():
            k_val = config.split('=')[1]
            interpretation = self.interpret_window_result(int(k_val), auc)
            print(f"  {config:8s}: {auc:.4f}  {interpretation}")
        
        print("\\nEdge weighting:")
        for scheme, auc in edge_results.items():
            current = "(current, better)" if scheme == "similarity" else ""
            print(f"  {scheme:12s}: {auc:.4f}  {current}")
        
        # Find best configurations
        best_window = max(window_results.items(), key=lambda x: x[1])
        best_weighting = max(edge_results.items(), key=lambda x: x[1])
        
        print(f"\\n🎯 BEST CONFIGURATIONS:")
        print(f"  Window size: {best_window[0]} (AUC: {best_window[1]:.4f})")
        print(f"  Edge weighting: {best_weighting[0]} (AUC: {best_weighting[1]:.4f})")
        
        # Save results
        os.makedirs('reports/ablations', exist_ok=True)
        with open('reports/ablations/graph_structure_ablation.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def interpret_window_result(self, k, auc):
        """Interpret window size result."""
        if k == 1:
            return "(too local)"
        elif k == 2:
            return "(current, good)" if auc >= 0.4 else "(current)"
        elif k == 3:
            return "(slightly better)" if auc > 0.4 else "(moderate)"
        elif k == 5:
            return "(too global)" if auc < 0.45 else "(global, good)"
        else:
            return ""


def main():
    """
    Run graph structure ablation study.
    """
    print("\\n" + "="*70)
    print("🚀 COMPONENT 5.2: GRAPH STRUCTURE ABLATION")
    print("="*70)
    print("Testing different graph connectivity patterns")
    
    # Check prerequisites
    multifeatures_dir = Path('data/processed/multifeatures')
    if not multifeatures_dir.exists():
        print(f"❌ Multi-features directory not found: {multifeatures_dir}")
        print("   Run Component 5.1a first to extract features")
        return
    
    # Check for histogram features
    histogram_files = list(multifeatures_dir.glob('*_histogram.npy'))
    if not histogram_files:
        print(f"❌ No histogram features found in {multifeatures_dir}")
        return
    
    print(f"✓ Found {len(histogram_files)} histogram feature files")
    
    # Create temp directory
    os.makedirs('data/temp', exist_ok=True)
    
    try:
        # Run ablation study
        ablation = GraphStructureAblation()
        results = ablation.run_complete_ablation()
        
        print('\\n✅ Graph structure ablation complete!')
        print(f'🎯 Results saved to: reports/ablations/graph_structure_ablation.json')
        
    finally:
        # Cleanup temp directory
        temp_dir = Path('data/temp')
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print('🧹 Cleaned up temporary files')


if __name__ == "__main__":
    main()