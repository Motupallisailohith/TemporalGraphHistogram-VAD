#!/usr/bin/env python3
"""
Component 5.1b: Build Feature Graphs
Purpose: Construct temporal graphs from different feature modalities for ablation
"""

import os
import json
import numpy as np
from pathlib import Path
import torch  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from tqdm import tqdm
import time


class FeatureGraphBuilder:
    """
    Build temporal graphs from different feature modalities.
    Graph structure follows temporal modeling approaches from VAD literature.
    """
    
    def __init__(self, feature_type='cnn', window_k=2, similarity_weighted=True):
        """
        Initialize graph builder.
        
        Args:
            feature_type: 'histogram', 'optical_flow', 'cnn', or 'fusion'
            window_k: temporal window size (edges to ±k neighbors)
            similarity_weighted: bool, weight edges by cosine similarity
        """
        self.feature_type = feature_type
        self.window_k = window_k
        self.similarity_weighted = similarity_weighted
        
        print(f"🔧 Initializing FeatureGraphBuilder")
        print(f"   Feature type: {feature_type}")
        print(f"   Window k: {window_k}")
        print(f"   Similarity weighted: {similarity_weighted}")
    
    def load_features(self, seq_name):
        """
        Load features of specified type.
        
        Args:
            seq_name: str, e.g., 'Train001'
        
        Returns:
            features: np.ndarray (num_frames, feature_dim)
        """
        feature_path = f'data/processed/multifeatures/{seq_name}_{self.feature_type}.npy'
        
        if not os.path.exists(feature_path):
            print(f"   ❌ Feature file not found: {feature_path}")
            return None
            
        features = np.load(feature_path)
        print(f"   ✓ Loaded {self.feature_type} features: {features.shape}")
        return features
    
    def normalize_features(self, features):
        """
        Z-score normalization.
        
        Args:
            features: (num_frames, feature_dim)
        
        Returns:
            normalized: (num_frames, feature_dim)
        """
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        normalized = (features - mean) / std
        return normalized
    
    def build_adjacency_matrix(self, num_frames):
        """
        Build adjacency matrix with temporal window connections.
        
        Args:
            num_frames: int, number of frames in sequence
        
        Returns:
            A: np.ndarray (num_frames, num_frames) adjacency matrix
        """
        A = np.zeros((num_frames, num_frames))
        
        for i in range(num_frames):
            # Connect to neighbors within window
            for offset in range(-self.window_k, self.window_k + 1):
                j = i + offset
                if 0 <= j < num_frames and i != j:
                    A[i, j] = 1.0
        
        return A
    
    def weight_edges_by_similarity(self, A, features):
        """
        Weight edges by cosine similarity between node features.
        
        Args:
            A: (num_frames, num_frames) binary adjacency
            features: (num_frames, feature_dim) normalized features
        
        Returns:
            A_weighted: (num_frames, num_frames) weighted adjacency
        """
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        
        # Compute pairwise cosine similarities
        similarity_matrix = cosine_similarity(features)
        
        # Apply to edges only (where A = 1)
        A_weighted = A * similarity_matrix
        
        return A_weighted
    
    def _adjacency_to_edge_index(self, A):
        """
        Convert adjacency matrix to edge_index format.
        
        Args:
            A: (N, N) adjacency matrix
        
        Returns:
            edge_index: (2, num_edges) edge indices
        """
        rows, cols = np.nonzero(A)
        edge_index = np.vstack([rows, cols])
        return edge_index
    
    def build_graph(self, seq_name):
        """
        Build complete temporal graph for a sequence.
        
        Args:
            seq_name: str
        
        Returns:
            graph_data: dict with keys:
                - node_features: (num_frames, feature_dim)
                - adjacency_matrix: (num_frames, num_frames)
                - edge_index: (2, num_edges)
                - num_nodes, num_edges, feature_dim
        """
        print(f"   Building {self.feature_type} graph for {seq_name}...")
        
        # Load and normalize features
        features = self.load_features(seq_name)
        if features is None:
            return None
            
        features_norm = self.normalize_features(features)
        
        # Build adjacency
        A = self.build_adjacency_matrix(len(features))
        
        # Weight edges
        if self.similarity_weighted:
            A = self.weight_edges_by_similarity(A, features_norm)
        
        # Convert to edge_index format (PyTorch Geometric)
        edge_index = self._adjacency_to_edge_index(A)
        
        graph_data = {
            'node_features': features_norm.astype(np.float32),
            'adjacency_matrix': A.astype(np.float32),
            'edge_index': edge_index.astype(np.int32),
            'num_nodes': len(features),
            'num_edges': edge_index.shape[1],
            'feature_dim': features.shape[1],
            'feature_type': self.feature_type
        }
        
        print(f"     ✓ {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
        
        return graph_data
    
    def build_all_sequences(self, sequences, output_dir):
        """
        Build graphs for all sequences with current feature type.
        
        Args:
            sequences: list of sequence names
            output_dir: where to save graphs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        success_count = 0
        
        for seq in tqdm(sequences, desc=f'Building {self.feature_type} graphs'):
            try:
                graph = self.build_graph(seq)
                
                if graph is not None:
                    save_path = os.path.join(output_dir, f'{seq}_graph.npz')
                    np.savez_compressed(save_path, **graph)
                    success_count += 1
                    print(f'     ✓ Saved: {save_path}')
                else:
                    print(f'     ❌ Failed to build graph for {seq}')
                    
            except Exception as e:
                print(f'     ❌ Error with {seq}: {e}')
        
        print(f"\n   ✅ Built {success_count}/{len(sequences)} {self.feature_type} graphs")


def get_sequence_list():
    """Get available sequences."""
    splits_file = 'data/splits/ucsd_ped2_splits.json'
    
    if os.path.exists(splits_file):
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        # Handle different split file formats
        if 'train_sequences' in splits:
            sequences = splits['train_sequences'] + splits['test_sequences']
        elif 'train' in splits:
            # Extract sequence names from paths
            train_paths = splits['train']
            test_paths = splits['test']
            
            train_seqs = []
            for path in train_paths:
                seq_name = os.path.basename(path)
                train_seqs.append(seq_name)
            
            test_seqs = []
            for path in test_paths:
                seq_name = os.path.basename(path)
                test_seqs.append(seq_name)
            
            sequences = train_seqs + test_seqs
        else:
            raise ValueError("Unknown splits file format")
            
        return sequences
    else:
        # Default sequences
        train_seqs = [f'Train{i:03d}' for i in range(1, 17)]
        test_seqs = [f'Test{i:03d}' for i in range(1, 13)]
        return train_seqs + test_seqs


def main():
    """
    Build graphs for each feature type.
    """
    print("\n" + "="*70)
    print("🚀 COMPONENT 5.1b: BUILD FEATURE-SPECIFIC GRAPHS")
    print("="*70)
    print("Building temporal graphs from different feature modalities")
    
    # Check if features exist
    feature_dir = Path('data/processed/multifeatures')
    if not feature_dir.exists():
        print(f"❌ Feature directory not found: {feature_dir}")
        print("   Run Component 5.1a first to extract features")
        return
    
    # Get sequence list
    sequences = get_sequence_list()
    print(f"\n📝 Processing {len(sequences)} sequences...")
    
    # Feature types to process
    feature_types = ['histogram', 'optical_flow', 'cnn']
    
    overall_success = {}
    
    for feat_type in feature_types:
        print(f'\n=== Building {feat_type} graphs ===')
        
        builder = FeatureGraphBuilder(
            feature_type=feat_type,
            window_k=2,
            similarity_weighted=True
        )
        
        output_dir = f'data/processed/temporal_graphs_{feat_type}'
        
        try:
            builder.build_all_sequences(sequences, output_dir)
            overall_success[feat_type] = True
        except Exception as e:
            print(f"   ❌ Failed to build {feat_type} graphs: {e}")
            overall_success[feat_type] = False
    
    # Summary
    print("\n" + "="*70)
    print("📊 GRAPH BUILDING SUMMARY")
    print("="*70)
    
    successful_types = [ft for ft, success in overall_success.items() if success]
    failed_types = [ft for ft, success in overall_success.items() if not success]
    
    if successful_types:
        print(f"✅ Successfully built graphs for: {successful_types}")
        
        # Count total files
        total_files = 0
        for feat_type in successful_types:
            output_dir = Path(f'data/processed/temporal_graphs_{feat_type}')
            if output_dir.exists():
                graph_files = list(output_dir.glob('*.npz'))
                total_files += len(graph_files)
                print(f"   {feat_type}: {len(graph_files)} graph files")
        
        print(f"\n📂 Total: {total_files} graph files generated")
        
        print(f"\n📁 Output structure:")
        print(f"data/processed/")
        for feat_type in successful_types:
            print(f"├── temporal_graphs_{feat_type}/")
            print(f"│   ├── Train001_graph.npz")
            print(f"│   └── ...")
    
    if failed_types:
        print(f"\n❌ Failed to build graphs for: {failed_types}")
    
    if successful_types:
        print(f"\n✅ Feature graph construction complete!")
        print(f"🎯 Next: Run Component 5.1c (Train Ablation Models)")
    else:
        print(f"\n❌ No graphs built successfully")


if __name__ == "__main__":
    main()