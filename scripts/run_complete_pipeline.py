#!/usr/bin/env python3
"""
Complete Anomaly Detection Pipeline Runner
Runs the complete pipeline: baseline scoring → evaluation → reporting

This script orchestrates the complete anomaly detection workflow:
1. Validates dataset integrity
2. Computes baseline anomaly scores  
3. Evaluates performance with ROC/AUC metrics
4. Generates comprehensive reports

Usage: python scripts/run_complete_pipeline.py
"""

import sys
import subprocess
from pathlib import Path
import json

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status"""
    print(f"\n🔄 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, f"scripts/{script_name}"], 
                              capture_output=False, check=True)
        print(f"{script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def check_prerequisites() -> bool:
    """Check if required files exist"""
    required_files = [
        "data/splits/ucsd_ped2_labels.json",
        "data/splits/ucsd_ped2_splits.json"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run the data preparation scripts first:")
        print("   python scripts/make_ucsd_splits.py")
        print("   python scripts/make_ucsd_label_masks.py")
        return False
    
    return True

def print_final_summary():
    """Print final pipeline summary from saved results"""
    print("\n" + "=" * 80)
    print("COMPLETE ANOMALY DETECTION PIPELINE - FINAL SUMMARY")
    print("=" * 80)
    
    # Load evaluation results
    results_file = Path("data/processed/evaluation_results/evaluation_results.json")
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        print(f"PERFORMANCE METRICS:")
        if results.get('mean_auc') is not None:
            print(f"   Mean AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
            print(f"   Overall AUC: {results['overall_auc']:.3f}")
        print(f"   Evaluated: {results['sequences_with_auc']}/{results['total_sequences']} sequences")
        print(f"   Total Frames: {results['total_frames']:,}")
        print(f"   Anomaly Rate: {results['total_anomalies']/results['total_frames']:.1%}")
        
        print(f"\nSEQUENCE-LEVEL RESULTS:")
        for seq_result in results['sequence_results']:
            if seq_result.get('auc') is not None:
                print(f"   {seq_result['sequence']}: AUC={seq_result['auc']:.3f}, "
                      f"F1={seq_result['optimal_f1']:.3f}")
    
    # List generated files
    print(f"\n📁 GENERATED FILES:")
    output_dirs = [
        ("data/processed/anomaly_scores", "Anomaly scores and baseline model"),
        ("data/processed/evaluation_results", "Performance metrics and plots")
    ]
    
    for dir_path, description in output_dirs:
        if Path(dir_path).exists():
            files = list(Path(dir_path).rglob("*"))
            print(f"   {description}: {len(files)} files in {dir_path}")
    
    print(f"\nNext Steps:")
    print("   1. Review ROC plots: data/processed/evaluation_results/roc_analysis.png")
    print("   2. Check detailed results: data/processed/evaluation_results/evaluation_summary.txt")  
    print("   3. Experiment with different scoring methods or parameters")
    print("   4. Implement more sophisticated temporal modeling")
    
    print("=" * 80)

def main():
    """Main pipeline execution"""
    print("COMPLETE ANOMALY DETECTION PIPELINE")
    print("   TemporalGraphHistogram-VAD - Baseline Implementation")
    print("=" * 80)
    
    # Step 1: Check prerequisites
    print("Checking prerequisites...")
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 2: Validate dataset (optional but recommended)
    print("\nStep 1: Dataset Validation")
    if not run_script("validate_ucsd_dataset.py", "Validating dataset integrity"):
        print("Warning: Dataset validation failed. Continuing anyway...")
    
    # Step 3: Baseline anomaly scoring
    print("\nStep 2: Baseline Anomaly Scoring")
    if not run_script("baseline_anomaly_scoring.py", "Computing baseline L2 distance scores"):
        print("Pipeline failed at baseline scoring step")
        sys.exit(1)
    
    # Step 4: Evaluation
    print("\nStep 3: Performance Evaluation") 
    if not run_script("evaluate_anomaly_detection.py", "Evaluating anomaly detection performance"):
        print("Pipeline failed at evaluation step")
        sys.exit(1)
    
    # Step 5: Final summary
    print_final_summary()

if __name__ == "__main__":
    main()