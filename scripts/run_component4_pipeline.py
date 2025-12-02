#!/usr/bin/env python3
"""
COMPLETE COMPONENT 4 PIPELINE
Runs all steps for GNN training and scoring:
  1. Extract CNN features for training sequences
  2. Build training temporal graphs
  3. Train GNN autoencoder
  4. Score test sequences with trained GNN
"""

import os
import sys
import subprocess
from pathlib import Path
import time


def run_command(cmd, description):
    """
    Run a command and report results.
    
    Args:
        cmd (str): Command to run
        description (str): What the command does
        
    Returns:
        bool: Success status
    """
    print(f"\n{'=' * 70}")
    print(f"▶️  {description}")
    print(f"{'=' * 70}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {description} - COMPLETE")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED")
        print(f"   Error: {e}")
        return False


def check_prerequisites():
    """Check if prerequisite files exist."""
    print("\n🔍 CHECKING PREREQUISITES")
    print("=" * 70)
    
    checks = []
    
    # Check training directory
    train_dir = Path('data/raw/UCSD_Ped2/UCSDped2/Train')
    if train_dir.exists():
        train_count = len(list(train_dir.glob('Train*')))
        print(f"✓ Training directory: {train_count} sequences found")
        checks.append(True)
    else:
        print(f"✗ Training directory not found: {train_dir}")
        checks.append(False)
    
    # Check test temporal graphs
    test_graphs = list(Path('data/processed/temporal_graphs').glob('Test*_temporal_graph.npz'))
    if test_graphs:
        print(f"✓ Test temporal graphs: {len(test_graphs)} found")
        checks.append(True)
    else:
        print(f"✗ Test temporal graphs not found")
        checks.append(False)
    
    # Check PyTorch Geometric
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric installed")
        checks.append(True)
    except ImportError:
        print(f"✗ PyTorch Geometric not installed")
        print(f"  Install: pip install torch-geometric torch-scatter torch-sparse")
        checks.append(False)
    
    print("=" * 70)
    
    return all(checks)


def main():
    """Run complete Component 4 pipeline."""
    print("\n" + "=" * 70)
    print("🚀 COMPONENT 4: GNN TRAINING & SCORING - COMPLETE PIPELINE")
    print("=" * 70)
    print("\nThis pipeline will:")
    print("  1️⃣  Extract CNN features for training sequences (~2 min)")
    print("  2️⃣  Build training temporal graphs (~3 sec)")
    print("  3️⃣  Train GNN autoencoder (~10-20 min)")
    print("  4️⃣  Score test sequences with GNN (~1 min)")
    print("\nEstimated total time: ~15-25 minutes")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix issues above.")
        return
    
    print("\n✅ All prerequisites met. Starting pipeline...")
    
    start_time = time.time()
    
    # STEP 1: Extract training CNN features
    success = run_command(
        "python scripts/extract_train_cnn_features.py",
        "STEP 1/4: Extract CNN features for training sequences"
    )
    if not success:
        print("\n❌ Pipeline stopped at Step 1")
        return
    
    # STEP 2: Build training temporal graphs
    success = run_command(
        "python scripts/generate_training_graphs.py",
        "STEP 2/4: Build training temporal graphs"
    )
    if not success:
        print("\n❌ Pipeline stopped at Step 2")
        return
    
    # STEP 3: Train GNN autoencoder
    success = run_command(
        "python scripts/phase3_4a_train_gnn.py",
        "STEP 3/4: Train GNN autoencoder (this may take 10-20 minutes)"
    )
    if not success:
        print("\n❌ Pipeline stopped at Step 3")
        return
    
    # STEP 4: Score test sequences
    success = run_command(
        "python scripts/phase3_4b_score_gnn.py",
        "STEP 4/4: Score test sequences with trained GNN"
    )
    if not success:
        print("\n❌ Pipeline stopped at Step 4")
        return
    
    # Complete!
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "=" * 70)
    print("🎉 COMPONENT 4 PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"   ⏱️  Total time: {minutes} min {seconds} sec")
    print(f"\n📊 Generated outputs:")
    print(f"   ├── data/processed/cnn_features/Train*.npy")
    print(f"   ├── data/processed/temporal_graphs/Train*.npz")
    print(f"   ├── models/gnn_autoencoder.pth (trained model)")
    print(f"   ├── models/training_history.json")
    print(f"   └── data/processed/gnn_scores/Test*_gnn_scores.npy")
    print(f"\n🎯 NEXT STEP: Evaluate GNN performance")
    print(f"   Expected: GNN ~85% AUC vs Baseline ~51% AUC")
    print(f"   Command: python scripts/evaluate_gnn_scores.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
