#!/usr/bin/env python3
"""
MASTER OPTIMIZATION PIPELINE
Runs Priority 1A and 2A in sequence for maximum AUC improvement

Execution flow:
1. Priority 1A: Hyperparameter tuning (1-2 hours)
   - Tests 162 configurations
   - Finds optimal GNN architecture
   - Target: 65%+ AUC

2. Priority 2A: Ensemble method (<5 minutes)
   - Combines tuned GNN + baseline
   - Tests 7 ensemble strategies
   - Target: 70%+ AUC

Expected final result: 65-75% AUC (vs current 57.74%)
"""

import subprocess
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'=' * 70}")
    print(f"▶️  {description}")
    print(f"{'=' * 70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✅ {description} - COMPLETE ({elapsed/60:.1f} minutes)")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} - FAILED ({elapsed/60:.1f} minutes)")
        print(f"   Error: {e}")
        return False, elapsed


def main():
    """Run complete optimization pipeline."""
    print("\n" + "=" * 70)
    print("🚀 MASTER OPTIMIZATION PIPELINE")
    print("=" * 70)
    
    print("\n📋 EXECUTION PLAN:")
    print("   1️⃣  Priority 1A: Hyperparameter Tuning (~1-2 hours)")
    print("       • Tests 162 GNN configurations")
    print("       • Finds optimal architecture")
    print("       • Target: 65%+ AUC")
    print()
    print("   2️⃣  Priority 2A: Ensemble Method (~5 minutes)")
    print("       • Combines tuned GNN + baseline")
    print("       • Tests ensemble strategies")
    print("       • Target: 70%+ AUC")
    print()
    print("📊 Current Performance: 57.74% AUC")
    print("🎯 Target Performance:  70.00%+ AUC")
    print("💪 Expected Gain:       +12-15 percentage points")
    
    # Confirm execution
    print("\n" + "=" * 70)
    print("⚠️  This will take 1-2 hours. Continue? (y/n): ", end='')
    response = input().strip().lower()
    
    if response != 'y':
        print("Execution cancelled.")
        return
    
    total_start = time.time()
    
    # Step 1: Hyperparameter Tuning
    success, time1 = run_command(
        ".venv312\\Scripts\\python.exe scripts/priority1a_hyperparameter_tuning.py",
        "PRIORITY 1A: Hyperparameter Tuning"
    )
    
    if not success:
        print("\n❌ Pipeline stopped at Priority 1A")
        return
    
    # Step 2: Ensemble Method
    success, time2 = run_command(
        ".venv312\\Scripts\\python.exe scripts/priority2a_ensemble_method.py",
        "PRIORITY 2A: Ensemble Method"
    )
    
    if not success:
        print("\n❌ Pipeline stopped at Priority 2A")
        return
    
    # Complete!
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("🎉 OPTIMIZATION PIPELINE COMPLETE!")
    print("=" * 70)
    
    print(f"\n⏱️  Execution Time:")
    print(f"   Priority 1A: {time1/60:.1f} minutes")
    print(f"   Priority 2A: {time2/60:.1f} minutes")
    print(f"   Total:       {total_time/60:.1f} minutes")
    
    print(f"\n📂 Generated Outputs:")
    print(f"   ├── models/gnn_autoencoder_tuned_best.pth")
    print(f"   ├── models/tuning_results/tuning_results.json")
    print(f"   ├── data/processed/ensemble_scores/Test*_ensemble_scores.npy")
    print(f"   ├── data/processed/ensemble_scores/ensemble_results.json")
    print(f"   └── reports/ensemble_comparison.png")
    
    print(f"\n📊 Check Results:")
    print(f"   1. Tuning results: cat models/tuning_results/tuning_results.json")
    print(f"   2. Ensemble results: cat data/processed/ensemble_scores/ensemble_results.json")
    print(f"   3. Visualization: reports/ensemble_comparison.png")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   ✅ If AUC >= 70%: Excellent! Write up results")
    print(f"   ✅ If AUC >= 65%: Good! Consider testing on Avenue dataset")
    print(f"   ⚠️  If AUC < 65%: Analyze top configurations, try longer training")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
