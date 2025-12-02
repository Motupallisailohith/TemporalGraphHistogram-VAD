#!/usr/bin/env python3
"""
PHASE 3 - COMPONENT 3: Temporal Graph Construction
Purpose: Build temporal graph structures from CNN features for graph-based anomaly detection
Input: CNN features (TestXXX_cnn_features.npy) from Component 2
Output: Temporal graphs (TestXXX_temporal_graph.npz) with node features and adjacency matrices

Logical Flow:
1. Load CNN features (num_frames, 2048)
2. Normalize features (zero-mean, unit variance)
3. Build adjacency matrix (window-based temporal connections)
4. Weight edges by cosine similarity (optional)
5. Extract edge indices (sparse format for PyTorch Geometric)
6. Save as .npz file
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import wandb
from datetime import datetime


class TemporalGraphBuilder:
    """
    Build temporal graph structures from frame-level CNN features.
    
    Graph Structure:
      - Nodes: Video frames (each node = one frame)
      - Edges: Temporal connections (window-based)
      - Node features: Normalized CNN features (2048-dim)
      - Edge weights: Cosine similarity between connected frames
    """
    
    def __init__(self,
                 feature_dir='data/processed/cnn_features',
                 output_dir='data/processed/temporal_graphs',
                 window_k=2,
                 use_similarity_weights=True,
                 enable_wandb=True):
        """
        Initialize temporal graph builder.
        
        Args:
            feature_dir (str): Directory containing CNN features
            output_dir (str): Where to save temporal graphs
            window_k (int): Temporal window size (connect to k neighbors before/after)
            use_similarity_weights (bool): Weight edges by cosine similarity
            enable_wandb (bool): Enable W&B tracking
        """
        # STEP 1: Store input parameters
        self.enable_wandb = enable_wandb
        
        if self.enable_wandb:
            wandb.init(  # type: ignore
                project="temporalgraph-vad-complete",
                name=f"phase3_graph_construction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["phase3", "graph_construction", "temporal_graphs"],
                config={
                    "phase": "3_temporal_graphs",
                    "window_size": window_k,
                    "similarity_weights": use_similarity_weights,
                    "feature_type": "cnn",
                    "graph_type": "temporal"
                }
            )
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path(output_dir)
        self.window_k = window_k
        self.use_similarity_weights = use_similarity_weights
        
        # STEP 2: Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # STEP 3: Print configuration
        print(f"🔧 Temporal Graph Builder Configuration")
        print(f"   Feature directory: {self.feature_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Temporal window (k): {self.window_k}")
        print(f"   Similarity weights: {self.use_similarity_weights}")
        
        # Validate feature directory
        if not self.feature_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {self.feature_dir}")
    
    def load_features(self, seq_name):
        """
        FUNCTION 2: Load CNN features for a single sequence.
        
        Logical Steps:
          Step 1: Construct file path
          Step 2: Load numpy file
          Step 3: Handle errors
          Step 4: Return features
        
        Args:
            seq_name (str): Sequence name (e.g., 'Test001')
            
        Returns:
            np.ndarray: Features (num_frames, 2048) or None if error
        """
        # STEP 1: Construct file path
        feature_file = self.feature_dir / f'{seq_name}_cnn_features.npy'
        
        try:
            # STEP 2: Load numpy file
            features = np.load(feature_file)
            return features
        except FileNotFoundError:
            # STEP 3: Handle errors
            print(f"  ✗ {seq_name}: Feature file not found")
            return None
        except Exception as e:
            print(f"  ✗ {seq_name}: Error loading features: {e}")
            return None
    
    def normalize_features(self, features):
        """
        FUNCTION 3: Normalize features to zero-mean, unit variance.
        
        Why normalize?
          - Makes all features have same scale
          - Prevents large values from dominating
          - Standard practice for neural networks
        
        Logical Steps:
          Step 1: Compute mean per feature dimension
          Step 2: Compute std per feature dimension
          Step 3: Normalize (z-score normalization)
          Step 4: Convert to float32
        
        Args:
            features (np.ndarray): Features (num_frames, feature_dim)
            
        Returns:
            np.ndarray: Normalized features (num_frames, feature_dim)
        """
        # STEP 1: Compute mean per feature dimension
        mean = features.mean(axis=0)  # Shape: (feature_dim,)
        
        # STEP 2: Compute std per feature dimension
        std = features.std(axis=0)  # Shape: (feature_dim,)
        
        # STEP 3: Normalize (subtract mean, divide by std)
        # Add epsilon (1e-8) to prevent division by zero
        features_norm = (features - mean) / (std + 1e-8)
        
        # STEP 4: Convert to float32 (reduce memory usage)
        features_norm = features_norm.astype(np.float32)
        
        return features_norm
    
    def build_adjacency_matrix(self, num_frames):
        """
        FUNCTION 4: Build adjacency matrix with window-based temporal connections.
        
        Why window-based?
          - Linear (i→i+1 only) too restrictive
          - Fully connected (i→all) too expensive
          - Window captures local temporal context
        
        Logical Steps:
          Step 1: Initialize empty matrix
          Step 2: Add temporal connections (window-based)
          Step 3: Avoid self-loops
          Step 4: Return adjacency matrix
        
        Example with k=2:
          Frame 0: connects to frames 1, 2
          Frame 1: connects to frames 0, 2, 3
          Frame 89: connects to frames 87, 88, 90, 91
        
        Args:
            num_frames (int): Number of frames in sequence
            
        Returns:
            np.ndarray: Adjacency matrix (num_frames, num_frames)
        """
        # STEP 1: Initialize empty matrix (all zeros)
        A = np.zeros((num_frames, num_frames), dtype=np.float32)
        
        # STEP 2: Add temporal connections
        for i in range(num_frames):
            # Connect to frames in window [i-k, ..., i-1, i+1, ..., i+k]
            for offset in range(-self.window_k, self.window_k + 1):
                j = i + offset
                
                # STEP 3: Avoid self-loops and out-of-bounds
                if j >= 0 and j < num_frames and i != j:
                    A[i, j] = 1.0  # Binary connection
        
        # STEP 4: Return adjacency matrix
        return A
    
    def weight_edges_by_similarity(self, A, features_norm):
        """
        FUNCTION 5: Replace binary edges with cosine similarity scores.
        
        Why weight by similarity?
          - Frames with similar features get stronger connections
          - Helps GNN focus on meaningful temporal patterns
          - Optional: can use binary (0/1) instead
        
        Logical Steps:
          Step 1: For each edge (i, j) where A[i,j] = 1:
            Step 1a: Extract features for frames i and j
            Step 1b: Compute cosine similarity
            Step 1c: Replace edge weight
          Step 2: Return weighted adjacency matrix
        
        Args:
            A (np.ndarray): Binary adjacency matrix (num_frames, num_frames)
            features_norm (np.ndarray): Normalized features (num_frames, feature_dim)
            
        Returns:
            np.ndarray: Weighted adjacency matrix (num_frames, num_frames)
        """
        num_frames = A.shape[0]
        
        # Find all edges
        edges = np.where(A > 0)
        
        # STEP 1: For each edge, compute similarity
        for i, j in zip(edges[0], edges[1]):
            # STEP 1a: Extract features
            feat_i = features_norm[i:i+1, :]  # Shape: (1, feature_dim)
            feat_j = features_norm[j:j+1, :]  # Shape: (1, feature_dim)
            
            # STEP 1b: Compute cosine similarity
            sim = cosine_similarity(feat_i, feat_j)[0, 0]
            # Output: scalar in [-1, 1], typically [0, 1] for normalized features
            
            # STEP 1c: Replace edge weight
            A[i, j] = sim
        
        # STEP 2: Return weighted adjacency matrix
        return A
    
    def extract_edge_index(self, A):
        """
        FUNCTION 6: Convert adjacency matrix to sparse edge index format.
        
        Why this format?
          - PyTorch Geometric expects sparse edge list
          - More efficient than storing full adjacency matrix
          - Standard format: edge_index[0] = sources, edge_index[1] = targets
        
        Logical Steps:
          Step 1: Find all non-zero edges
          Step 2: Stack into edge_index format
          Step 3: Return
        
        Example:
          Adjacency matrix:   [[0, 1, 1, 0],
                               [1, 0, 1, 1],
                               [1, 1, 0, 1],
                               [0, 1, 1, 0]]
          
          Edge index:         [[0, 0, 1, 1, 1, 2, 2, 2, 3, 3],  # source nodes
                               [1, 2, 0, 2, 3, 0, 1, 3, 1, 2]]  # target nodes
        
        Args:
            A (np.ndarray): Adjacency matrix (num_frames, num_frames)
            
        Returns:
            np.ndarray: Edge index (2, num_edges)
        """
        # STEP 1: Find all non-zero edges
        edge_indices = np.where(A > 0)
        # Returns: (row_indices, col_indices)
        
        # STEP 2: Stack into edge_index format
        edge_index = np.array(edge_indices, dtype=np.int64)
        # Shape: (2, num_edges)
        # Row 0: source node IDs
        # Row 1: target node IDs
        
        # STEP 3: Return
        return edge_index
    
    def build_sequence_graph(self, seq_name):
        """
        FUNCTION 7: Build complete temporal graph for one sequence.
        
        Orchestrate all steps:
          Step 1: Load features
          Step 2: Normalize features
          Step 3: Build adjacency matrix
          Step 4: Optionally weight edges
          Step 5: Extract edge indices
          Step 6: Package graph data
          Step 7: Return graph data
        
        Args:
            seq_name (str): Sequence name (e.g., 'Test001')
            
        Returns:
            dict: Graph data or None if error
        """
        # STEP 1: Load features
        features = self.load_features(seq_name)
        if features is None:
            return None
        
        num_frames = features.shape[0]
        feature_dim = features.shape[1]
        
        # STEP 2: Normalize features
        features_norm = self.normalize_features(features)
        
        # STEP 3: Build adjacency matrix (binary connections)
        A = self.build_adjacency_matrix(num_frames)
        
        # STEP 4: Optionally weight edges by similarity
        if self.use_similarity_weights:
            A = self.weight_edges_by_similarity(A, features_norm)
        
        # STEP 5: Extract edge indices
        edge_index = self.extract_edge_index(A)
        num_edges = edge_index.shape[1]
        
        # STEP 6: Package graph data
        graph_data = {
            'node_features': features_norm,       # (num_frames, feature_dim)
            'adjacency_matrix': A,                # (num_frames, num_frames)
            'edge_index': edge_index,             # (2, num_edges)
            'num_nodes': num_frames,
            'num_edges': num_edges,
            'feature_dim': feature_dim,
            'window_k': self.window_k
        }
        
        # STEP 7: Return graph data
        return graph_data
    
    def save_graph(self, seq_name, graph_data):
        """
        FUNCTION 8: Save graph to compressed .npz file.
        
        Logical Steps:
          Step 1: Construct output path
          Step 2: Save using np.savez
          Step 3: Handle errors
        
        Args:
            seq_name (str): Sequence name
            graph_data (dict): Graph data dictionary
            
        Returns:
            bool: Success status
        """
        # STEP 1: Construct output path
        output_path = self.output_dir / f'{seq_name}_temporal_graph.npz'
        
        try:
            # STEP 2: Save using np.savez
            np.savez(output_path, **graph_data)
            return True
        except Exception as e:
            # STEP 3: Handle errors
            print(f"  ✗ {seq_name}: Error saving graph: {e}")
            return False
    
    def build_all_sequences(self):
        """
        FUNCTION 9: Build temporal graphs for all sequences.
        
        Logical Steps:
          Step 1: Find all CNN feature files
          Step 2: Initialize tracking
          Step 3: For each sequence:
            Step 3a: Build graph
            Step 3b: Save graph
            Step 3c: Update summary
            Step 3d: Log progress
          Step 4: Save summary JSON
          Step 5: Print statistics
        
        Returns:
            dict: Processing summary
        """
        # STEP 1: Find all CNN feature files
        feature_files = [f for f in self.feature_dir.iterdir() 
                        if f.suffix == '.npy' and f.stem.endswith('_cnn_features')]
        
        # Extract sequence names
        sequences = sorted([f.stem.replace('_cnn_features', '') for f in feature_files])
        
        print(f"\n🔗 PHASE 3 COMPONENT 3: Temporal Graph Construction")
        print(f"   Feature directory: {self.feature_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Temporal window: k={self.window_k}")
        print(f"   Similarity weights: {self.use_similarity_weights}")
        print(f"   Found sequences: {len(sequences)}")
        print("-" * 70)
        
        # STEP 2: Initialize tracking
        summary = {}
        total_nodes = 0
        total_edges = 0
        
        # STEP 3: Process each sequence
        for seq_name in tqdm(sequences, desc="Building temporal graphs"):
            # STEP 3a: Build graph
            graph_data = self.build_sequence_graph(seq_name)
            
            if graph_data is not None:
                # STEP 3b: Save graph
                success = self.save_graph(seq_name, graph_data)
                
                if success:
                    # STEP 3c: Update summary
                    summary[seq_name] = {
                        'status': 'success',
                        'num_frames': graph_data['num_nodes'],
                        'num_edges': graph_data['num_edges'],
                        'feature_dim': graph_data['feature_dim'],
                        'window_k': graph_data['window_k'],
                        'edge_density': graph_data['num_edges'] / (graph_data['num_nodes'] ** 2)
                    }
                    
                    total_nodes += graph_data['num_nodes']
                    total_edges += graph_data['num_edges']
                    
                    # STEP 3d: Log progress
                    print(f"  ✓ {seq_name}: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
                else:
                    summary[seq_name] = {'status': 'failed_save'}
            else:
                summary[seq_name] = {'status': 'failed_build'}
        
        # STEP 4: Save summary JSON
        summary_data = {
            'graph_summary': summary,
            'total_sequences': len(sequences),
            'successful_sequences': sum(1 for v in summary.values() if v['status'] == 'success'),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'avg_nodes_per_graph': total_nodes / len(sequences) if sequences else 0,
            'avg_edges_per_graph': total_edges / len(sequences) if sequences else 0,
            'window_k': self.window_k,
            'use_similarity_weights': self.use_similarity_weights
        }
        
        summary_path = self.output_dir / 'temporal_graph_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # STEP 5: Print statistics
        print(f"\n" + "=" * 70)
        print(f"✅ TEMPORAL GRAPH CONSTRUCTION COMPLETE")
        print(f"   📊 Processed: {summary_data['successful_sequences']}/{len(sequences)} sequences")
        print(f"   🔵 Total nodes: {total_nodes}")
        print(f"   🔗 Total edges: {total_edges}")
        print(f"   📈 Avg nodes/graph: {summary_data['avg_nodes_per_graph']:.1f}")
        print(f"   📈 Avg edges/graph: {summary_data['avg_edges_per_graph']:.1f}")
        print(f"   💾 Output directory: {self.output_dir}")
        print(f"   📋 Summary: {summary_path}")
        print("=" * 70)
        
        # Log to W&B
        if self.enable_wandb:
            wandb.log({
                "graphs_constructed": summary_data['successful_sequences'],
                "total_sequences": len(sequences),
                "construction_success_rate": summary_data['successful_sequences'] / len(sequences),
                "total_nodes_across_graphs": total_nodes,
                "total_edges_across_graphs": total_edges,
                "avg_nodes_per_graph": summary_data['avg_nodes_per_graph'],
                "avg_edges_per_graph": summary_data['avg_edges_per_graph'],
                "phase3_graphs_complete": True
            })
        
        return summary_data


def main():
    """
    FUNCTION 10: Main entry point.
    
    Execution Order:
      1. Check prerequisites
      2. Create builder
      3. Build all graphs
      4. Report results
    """
    print("🚀 Starting Phase 3 Component 3: Temporal Graph Construction")
    
    # Check prerequisites
    feature_dir = Path('data/processed/cnn_features')
    if not feature_dir.exists():
        print(f"❌ Error: CNN features not found: {feature_dir}")
        print("   Please run Component 2 first (phase3_extract_cnn_features.py)")
        return
    
    # STEP 2: Create builder
    builder = TemporalGraphBuilder(
        feature_dir='data/processed/cnn_features',
        output_dir='data/processed/temporal_graphs',
        window_k=2,                      # Connect to 2 neighbors before/after
        use_similarity_weights=True      # Weight edges by cosine similarity
    )
    
    # STEP 3: Build all graphs
    summary = builder.build_all_sequences()
    
    # STEP 4: Report results
    if summary['successful_sequences'] == summary['total_sequences']:
        print(f"\n🎉 All sequences processed successfully!")
        print(f"📂 Files generated:")
        for seq_name in sorted(summary['graph_summary'].keys()):
            if summary['graph_summary'][seq_name]['status'] == 'success':
                info = summary['graph_summary'][seq_name]
                print(f"   {seq_name}_temporal_graph.npz: "
                      f"{info['num_frames']} nodes, {info['num_edges']} edges")
    else:
        failed = summary['total_sequences'] - summary['successful_sequences']
        print(f"⚠️ {failed} sequences failed. Check logs above for details.")
    
    print(f"\n✅ Phase 3 Component 3 complete!")
    print(f"   Next: Implement temporal graph neural network for anomaly detection")
    
    # Finish W&B
    if builder.enable_wandb:
        wandb.finish()  # type: ignore


if __name__ == "__main__":
    main()
