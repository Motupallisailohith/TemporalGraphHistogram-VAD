#!/usr/bin/env python3
"""
Phase 5 Pipeline Runner
Execute complete ablation study workflow
"""

import os
import subprocess
import sys
from pathlib import Path
import time


def run_script(script_path, description):
    """
    Run a Python script and handle output.
    
    Args:
        script_path: path to script
        description: human-readable description
    
    Returns:
        bool: True if successful
    """
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Running: {script_path}")
    
    start_time = time.time()
    
    try:
        # Run script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=Path(__file__).parent.parent.parent,  # Repository root
            capture_output=False,  # Show output in real time
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully ({elapsed:.1f}s)")
            return True
        else:
            print(f"\n❌ {description} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed with error: {e} ({elapsed:.1f}s)")
        return False


def check_prerequisites():
    """Check if required data exists."""
    print("\n🔍 Checking prerequisites...")
    
    required_paths = [
        'data/splits/ucsd_ped2_splits.json',
        'data/splits/ucsd_ped2_labels.json',
        'data/raw/UCSD_Ped2/UCSDped2'
    ]
    
    missing = []
    for path in required_paths:
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print(f"❌ Missing required data:")
        for path in missing:
            print(f"   - {path}")
        return False
    
    print(f"✅ All prerequisites found")
    return True


def main():
    """
    Run complete Phase 5 ablation study pipeline.
    """
    print("\n" + "="*70)
    print("🚀 PHASE 5: ABLATION STUDIES & CROSS-DATASET VALIDATION")
    print("="*70)
    print("Component 5.1: Feature Ablation Studies")
    print()
    print("Timeline: This will take 30-60 minutes depending on your hardware")
    print("Components:")
    print("  5.1a: Extract multi-modal features (histogram, optical flow, CNN)")
    print("  5.1b: Build feature-specific temporal graphs") 
    print("  5.1c: Train ablation models on different feature types")
    print("  5.1d: Evaluate and analyze feature contributions")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nCannot proceed without required data.")
        print("Please ensure UCSD Ped2 dataset is properly set up.")
        return False
    
    # Define pipeline steps
    steps = [
        ("scripts/phase5/5.1a_extract_all_features.py", "Extract Multi-Modal Features"),
        ("scripts/phase5/5.1b_build_feature_graphs.py", "Build Feature-Specific Graphs"),
        ("scripts/phase5/5.1c_train_ablation_models_clean.py", "Train & Evaluate Feature Ablations")
    ]
    
    # Track progress
    completed_steps = 0
    total_steps = len(steps)
    
    overall_start_time = time.time()
    
    # Execute pipeline
    for i, (script_path, description) in enumerate(steps, 1):
        print(f"\n📋 Step {i}/{total_steps}: {description}")
        
        success = run_script(script_path, description)
        
        if success:
            completed_steps += 1
        else:
            print(f"\n⚠️  Step {i} failed. You can:")
            print(f"   1. Fix the issue and re-run: python {script_path}")
            print(f"   2. Skip this step and continue with remaining steps")
            print(f"   3. Stop the pipeline here")
            
            # Ask user what to do
            while True:
                choice = input("\nContinue with remaining steps? (y/n): ").lower().strip()
                if choice in ['y', 'yes']:
                    print("Continuing with remaining steps...")
                    break
                elif choice in ['n', 'no']:
                    print("Pipeline stopped by user.")
                    return False
                else:
                    print("Please enter 'y' or 'n'")
    
    overall_elapsed = time.time() - overall_start_time
    
    # Final summary
    print("\n" + "="*70)
    print("📊 PHASE 5 COMPLETION SUMMARY")
    print("="*70)
    print(f"Steps completed: {completed_steps}/{total_steps}")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    
    if completed_steps == total_steps:
        print("\n🎉 Phase 5 completed successfully!")
        
        # Show generated outputs
        print(f"\n📂 Generated outputs:")
        
        output_dirs = [
            "data/processed/multifeatures",
            "data/processed/feature_graphs", 
            "models/ablation_models",
            "reports/ablation_study"
        ]
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                num_files = len(list(Path(output_dir).rglob('*.*')))
                print(f"   {output_dir}: {num_files} files")
        
        print(f"\n🎯 Key Results:")
        
        # Try to show ablation results if available
        results_file = "reports/ablation_study/ablation_results.json"
        if os.path.exists(results_file):
            import json
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                if 'summary' in results and 'performance_ranking' in results['summary']:
                    ranking = results['summary']['performance_ranking']
                    print(f"   Feature Performance Ranking:")
                    for i, (feature_type, auc) in enumerate(ranking, 1):
                        print(f"     {i}. {feature_type}: {auc:.4f} AUC")
            except:
                print(f"   Results available in: {results_file}")
        
        print(f"\n✅ Ablation study complete!")
        print(f"📝 Next steps:")
        print(f"   - Review ablation results to understand feature contributions")
        print(f"   - Consider cross-dataset validation (Phase 5.2)")
        print(f"   - Optimize based on findings")
        
    elif completed_steps > 0:
        print(f"\n⚠️  Partial completion. {total_steps - completed_steps} steps failed.")
        print(f"   You can re-run individual failed steps")
    else:
        print(f"\n❌ No steps completed successfully")
    
    print("="*70)
    return completed_steps == total_steps


if __name__ == "__main__":
    main()