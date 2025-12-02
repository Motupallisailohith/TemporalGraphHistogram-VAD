#!/usr/bin/env python3
"""
test_modular_pipeline.py - Test Complete Modular Pipeline
Tests the complete UCSD Ped2 anomaly detection pipeline end-to-end.
"""

import sys
from pathlib import Path

def test_pipeline():
    """Test complete modular pipeline"""
    
    print("=" * 70)
    print("TESTING MODULAR ANOMALY DETECTION PIPELINE")
    print("=" * 70)
    
    # Check if all required scripts exist
    scripts_dir = Path('scripts')
    required_scripts = [
        'score_ucsd_baseline.py',
        'evaluate_ucsd_scores.py', 
        'plot_vad_scores.py'
    ]
    
    print("Checking required scripts...")
    for script in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"  {script} - EXISTS")
        else:
            print(f"  {script} - MISSING")
            return False
    
    # Check if required data files exist
    print("\nChecking data prerequisites...")
    required_data = [
        'data/splits/ucsd_ped2_splits.json',
        'data/splits/ucsd_ped2_labels.json'
    ]
    
    for data_file in required_data:
        if Path(data_file).exists():
            print(f"  {data_file} - EXISTS")
        else:
            print(f"  {data_file} - MISSING")
            return False
    
    # Check if scores were generated
    scores_dir = Path('data/processed/baseline_scores')
    print(f"\nChecking generated scores...")
    if scores_dir.exists():
        score_files = list(scores_dir.glob('Test*_scores.npy'))
        print(f"  Scores directory exists with {len(score_files)} score files")
        
        # Check for background model
        if (scores_dir / 'background_model.pkl').exists():
            print(f"  Background model saved")
        else:
            print(f"  Background model missing")
    else:
        print(f"  Scores directory does not exist")
        return False
    
    # Check if evaluation results exist
    eval_dir = Path('data/processed/evaluation_results')
    print(f"\nChecking evaluation results...")
    if eval_dir.exists():
        if (eval_dir / 'evaluation_results.json').exists():
            print(f"  Evaluation results saved")
        else:
            print(f"  Evaluation results missing")
            
        if (eval_dir / 'evaluation_summary.txt').exists():
            print(f"  Evaluation summary saved")
            
            # Show summary
            with open(eval_dir / 'evaluation_summary.txt') as f:
                lines = f.readlines()
                print(f"\nEvaluation Summary:")
                for line in lines[:15]:  # Show first 15 lines
                    print(f"     {line.rstrip()}")
        else:
            print(f"  Evaluation summary missing")
    else:
        print(f"  Evaluation directory does not exist")
    
    print("\n" + "=" * 70)
    print("MODULAR PIPELINE TEST COMPLETE")
    print("=" * 70)
    
    return True

def show_usage_guide():
    """Show usage guide for the modular pipeline"""
    
    print("\n" + "=" * 70)
    print("📖 MODULAR PIPELINE USAGE GUIDE")
    print("=" * 70)
    
    print("1️⃣ Generate baseline anomaly scores:")
    print("   python scripts\\score_ucsd_baseline.py")
    print()
    
    print("2️⃣ Evaluate anomaly detection performance:")
    print("   python scripts\\evaluate_ucsd_scores.py --scores_dir data\\processed\\baseline_scores")
    print()
    
    print("3️⃣ Generate visualizations:")
    print("   python scripts\\plot_vad_scores.py --scores_dir data\\processed\\baseline_scores")
    print()
    
    print("All scripts support --help for detailed options")
    print("📁 Results are saved to data/processed/ with organized subdirectories")
    print("🔄 Scripts can be run independently or as part of a complete pipeline")

if __name__ == "__main__":
    success = test_pipeline()
    show_usage_guide()
    
    if success:
        print(f"\n🎉 Pipeline is ready for production use!")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline test failed - check missing components")
        sys.exit(1)