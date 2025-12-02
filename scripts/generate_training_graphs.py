#!/usr/bin/env python3
"""
Generate temporal graphs for TRAINING sequences (Train001-Train016).
This script runs the temporal graph builder on training data for GNN training.
"""

import os
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from scripts.phase3_build_temporal_graphs import TemporalGraphBuilder


def main():
    """Generate training temporal graphs."""
    print("\n🔧 GENERATING TRAINING TEMPORAL GRAPHS")
    print("=" * 70)
    
    # Configure paths for TRAINING data
    feature_dir = repo_root / 'data' / 'processed' / 'cnn_features'
    output_dir = repo_root / 'data' / 'processed' / 'temporal_graphs'
    
    # Check if training CNN features exist
    train_features = list(feature_dir.glob('Train*_cnn_features.npy'))
    
    if not train_features:
        print("\n❌ ERROR: No training CNN features found!")
        print(f"   Expected: Train001-Train016_cnn_features.npy in {feature_dir}")
        print("\n💡 Solution: Run CNN feature extraction on training sequences:")
        print("   python scripts/phase3_extract_cnn_features.py --split train")
        return
    
    print(f"\n✓ Found {len(train_features)} training feature files")
    
    # Build temporal graphs (same settings as test)
    builder = TemporalGraphBuilder(
        feature_dir=str(feature_dir),
        output_dir=str(output_dir),
        window_k=2,  # Same as test sequences
        use_similarity_weights=True  # Same as test sequences
    )
    
    # Process all sequences (will pick up Train* files)
    summary = builder.build_all_sequences()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING GRAPH GENERATION COMPLETE")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   🎯 Ready for GNN training!")
    

if __name__ == '__main__':
    main()
