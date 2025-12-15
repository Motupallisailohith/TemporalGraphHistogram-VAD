#!/usr/bin/env python3
"""
ShanghaiTech Temporal Graph Constructor
Purpose: Build temporal graphs from histogram features for ShanghaiTech dataset
Output: NPZ files containing graph structures with k-NN temporal connectivity

Graph Structure:
- Nodes: Frames with 256-dimensional histogram features
- Edges: k-nearest temporal neighbors (k=5, bidirectional)
- Format: NPZ archive (node_features, edge_index, adjacency_matrix, metadata)

Usage: python scripts/build_shanghaitech_temporal_graphs.py
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import time

class ShanghaiTechGraphBuilder:
    """Build temporal graphs for ShanghaiTech sequences"""
    
    def __init__(self, k_neighbors: int = 5):
        self.k_neighbors = k_neighbors
        
        # Input paths
        self.splits_file = Path('data/splits/shanghaitech_splits.json')
        self.test_hist_dir = Path('data/processed/shanghaitech/test_histograms')
        
        # Output paths
        self.output_dir = Path('data/processed/shanghaitech/temporal_graphs_histogram')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_splits(self) -> Dict:
        """Load dataset splits"""
        if not self.splits_file.exists():
            raise FileNotFoundError(f"Splits file not found: {self.splits_file}")
        
        with open(self.splits_file, 'r') as f:
            return json.load(f)
    
    def build_knn_edges(self, num_nodes: int, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-nearest neighbor temporal edges (bidirectional)
        
        Args:
            num_nodes: Number of frames in sequence
            k: Number of temporal neighbors (default: 5)
        
        Returns:
            edge_index: (2, num_edges) edge connectivity
            adjacency_matrix: (num_nodes, num_nodes) adjacency matrix
        """
        edges = []
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        for i in range(num_nodes):
            # Forward neighbors (up to k frames ahead)
            for j in range(1, k + 1):
                if i + j < num_nodes:
                    edges.append([i, i + j])
                    adjacency[i, i + j] = 1.0
            
            # Backward neighbors (up to k frames behind)
            for j in range(1, k + 1):
                if i - j >= 0:
                    edges.append([i, i - j])
                    adjacency[i, i - j] = 1.0
        
        # Convert to edge_index format (2, num_edges)
        if edges:
            edge_index = np.array(edges, dtype=np.int32).T
        else:
            edge_index = np.array([[], []], dtype=np.int32)
        
        return edge_index, adjacency
    
    def build_temporal_graph(self, sequence_name: str, histograms: np.ndarray) -> Dict:
        """
        Build temporal graph from histogram features
        
        Args:
            sequence_name: Name of sequence
            histograms: (num_frames, 256) histogram features
        
        Returns:
            Dictionary containing graph components
        """
        num_frames = histograms.shape[0]
        
        # Build k-NN temporal edges
        edge_index, adjacency_matrix = self.build_knn_edges(num_frames, self.k_neighbors)
        
        # Create graph dictionary
        graph = {
            'node_features': histograms,
            'edge_index': edge_index,
            'adjacency_matrix': adjacency_matrix,
            'num_nodes': np.array(num_frames, dtype=np.int64),
            'num_edges': np.array(edge_index.shape[1], dtype=np.int64),
            'feature_dim': np.array(256, dtype=np.int64),
            'feature_type': np.array('histogram', dtype='<U9'),
            'k_neighbors': np.array(self.k_neighbors, dtype=np.int32),
            'sequence_name': np.array(sequence_name, dtype='<U20')
        }
        
        return graph
    
    def build_all_graphs(self):
        """Build temporal graphs for all test sequences"""
        print("="*60)
        print("ShanghaiTech Temporal Graph Construction")
        print("="*60)
        print(f"k-neighbors: {self.k_neighbors} (bidirectional)")
        print(f"Edge type: Binary temporal connectivity")
        print(f"Feature dimension: 256 (histogram)")
        
        # Load splits
        print("\n📋 Loading dataset splits...")
        splits = self.load_splits()
        test_sequences = {k: v for k, v in splits.items() if v['split'] == 'test'}
        
        print(f"Test sequences: {len(test_sequences)}")
        
        # Build graphs
        print("\n🔄 Building temporal graphs...")
        print(f"{'─'*60}")
        
        built_count = 0
        skipped_count = 0
        total_nodes = 0
        total_edges = 0
        start_time = time.time()
        
        for idx, (seq_name, seq_info) in enumerate(sorted(test_sequences.items()), 1):
            # Check if already exists
            output_file = self.output_dir / f"{seq_name}_graph.npz"
            if output_file.exists():
                if skipped_count < 3:
                    print(f"   ✓ Skipping {seq_name} (already exists)")
                skipped_count += 1
                continue
            
            # Load histogram features
            hist_file = self.test_hist_dir / f"{seq_name}_histograms.npy"
            if not hist_file.exists():
                print(f"   ⚠️ Histogram file not found: {seq_name}")
                continue
            
            try:
                histograms = np.load(hist_file)
                
                # Build temporal graph
                graph = self.build_temporal_graph(seq_name, histograms)
                
                # Save graph
                np.savez_compressed(output_file, **graph)
                
                built_count += 1
                total_nodes += graph['num_nodes']
                total_edges += graph['num_edges']
                
                # Progress update
                elapsed = time.time() - start_time
                rate = built_count / elapsed if elapsed > 0 else 0
                eta = (len(test_sequences) - idx) / rate if rate > 0 else 0
                
                if built_count % 10 == 0 or built_count <= 5:
                    print(f"   [{idx}/{len(test_sequences)}] {seq_name}: "
                          f"{graph['num_nodes']} nodes, {graph['num_edges']} edges "
                          f"| Rate: {rate:.1f} seq/s | ETA: {eta:.0f}s")
            
            except Exception as e:
                print(f"   ❌ Error processing {seq_name}: {e}")
        
        elapsed = time.time() - start_time
        
        # Summary
        print(f"\n{'─'*60}")
        print(f"✅ Graph construction complete!")
        print(f"\n📊 Statistics:")
        print(f"   Built: {built_count} new graphs")
        print(f"   Skipped: {skipped_count} existing graphs")
        print(f"   Total nodes: {total_nodes:,} frames")
        print(f"   Total edges: {total_edges:,} temporal connections")
        print(f"   Avg edges/node: {total_edges/total_nodes:.1f}" if total_nodes > 0 else "")
        print(f"   Time: {elapsed:.1f} seconds")
        print(f"\n💾 Saved to: {self.output_dir}")
    
    def validate_graphs(self):
        """Validate generated graphs"""
        print("\n🔍 Validating temporal graphs...")
        
        graph_files = list(self.output_dir.glob("*_graph.npz"))
        
        if not graph_files:
            print("⚠️ No graph files found for validation")
            return
        
        print(f"Found {len(graph_files)} graph files")
        
        # Sample validation
        sample_size = min(10, len(graph_files))
        validation_errors = []
        
        for graph_file in graph_files[:sample_size]:
            try:
                with np.load(graph_file) as data:
                    # Check required fields
                    required_fields = ['node_features', 'edge_index', 'adjacency_matrix', 
                                     'num_nodes', 'num_edges', 'feature_dim']
                    
                    for field in required_fields:
                        if field not in data:
                            validation_errors.append(f"{graph_file.name}: Missing {field}")
                    
                    # Validate shapes
                    node_features = data['node_features']
                    edge_index = data['edge_index']
                    
                    if node_features.shape[1] != 256:
                        validation_errors.append(f"{graph_file.name}: Wrong feature dim")
                    
                    if edge_index.shape[0] != 2:
                        validation_errors.append(f"{graph_file.name}: Wrong edge_index shape")
                    
            except Exception as e:
                validation_errors.append(f"{graph_file.name}: Load error - {e}")
        
        if validation_errors:
            print(f"⚠️ Found {len(validation_errors)} validation errors:")
            for error in validation_errors:
                print(f"   {error}")
        else:
            print(f"✅ All sampled graphs ({sample_size}) validated successfully")

def main():
    """Main execution function"""
    # Check prerequisites
    splits_file = Path('data/splits/shanghaitech_splits.json')
    hist_dir = Path('data/processed/shanghaitech/test_histograms')
    
    if not splits_file.exists():
        print("❌ Error: Splits file not found. Run make_shanghaitech_splits.py first.")
        return
    
    if not hist_dir.exists():
        print("❌ Error: Histogram directory not found. Run extract_shanghaitech_histograms.py first.")
        return
    
    # Build graphs
    builder = ShanghaiTechGraphBuilder(k_neighbors=5)
    builder.build_all_graphs()
    
    # Validate
    builder.validate_graphs()
    
    print("\n" + "="*60)
    print("✅ ShanghaiTech temporal graph construction complete!")
    print("="*60)

if __name__ == "__main__":
    main()
