"""
Component 5.4: Refined Ensemble Methods - Complete Pipeline
Purpose: Orchestrate all three steps to achieve 65-70% AUC target

WHAT THIS DOES:
1. Diagnose current ensemble issues (Step 1)
2. Test different normalization methods (Step 2)  
3. Apply advanced ensemble strategies (Step 3)
4. Evaluate and report final results

EXPECTED OUTCOME:
- Target AUC: 65-70%
- Best strategy: Likely stacking with meta-learner
- Comprehensive analysis and recommendations
"""

import os
import json
import numpy as np
from pathlib import Path


def run_component_5_4():
    """
    Execute complete Component 5.4 pipeline.
    
    Pipeline Flow:
    Step 1: Diagnose → Identify scale mismatch, correlation patterns
    Step 2: Normalize → Fix scale issues with proper normalization
    Step 3: Ensemble → Apply advanced strategies for optimal performance
    """
    print("="*80)
    print("COMPONENT 5.4: REFINED ENSEMBLE METHODS")
    print("="*80)
    print("Target: Achieve 65-70% AUC through advanced ensemble optimization")
    print()
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    success_count = 0
    total_steps = 3
    
    # STEP 1: DIAGNOSE ENSEMBLE ISSUES
    print("🔍 STEP 1: DIAGNOSING ENSEMBLE ISSUES")
    print("-" * 50)
    
    try:
        from phase5_4a_diagnose_ensemble import main as diagnose_main
        diagnose_main()
        print("✅ Step 1 completed successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
    
    print()
    
    # STEP 2: TEST NORMALIZATION METHODS
    print("⚖️  STEP 2: TESTING SCORE NORMALIZATION")
    print("-" * 50)
    
    try:
        from phase5_4b_score_normalization import main as normalize_main
        normalize_main()
        print("✅ Step 2 completed successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
    
    print()
    
    # STEP 3: APPLY ADVANCED ENSEMBLE STRATEGIES
    print("🚀 STEP 3: ADVANCED ENSEMBLE STRATEGIES")
    print("-" * 50)
    
    try:
        from phase5_4c_advanced_ensemble import main as ensemble_main
        ensemble_main()
        print("✅ Step 3 completed successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
    
    print()
    
    # FINAL REPORT
    print("="*80)
    print("COMPONENT 5.4 COMPLETION REPORT")
    print("="*80)
    
    print(f"Steps completed: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("🎯 ALL STEPS COMPLETED SUCCESSFULLY!")
        
        # Check if target was achieved
        results_file = 'reports/ensemble_optimization_results.json'
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                best_auc = max(results.values())
                best_strategy = max(results, key=results.get)
                
                print(f"\nFINAL RESULTS:")
                print(f"   Best Strategy: {best_strategy}")
                print(f"   Best AUC: {best_auc:.4f}")
                
                if best_auc >= 0.65:
                    print("🏆 TARGET ACHIEVED: AUC ≥ 65%!")
                    if best_auc >= 0.70:
                        print("🌟 EXCEEDED EXPECTATIONS: AUC ≥ 70%!")
                else:
                    print(f"📊 Close to target. Gap: {0.65 - best_auc:.4f}")
                    
            except Exception as e:
                print(f"Could not read results: {e}")
        
        print(f"\nGenerated Files:")
        print(f"   📊 reports/ensemble_diagnostics.json")
        print(f"   📊 reports/normalization_comparison.png") 
        print(f"   📊 reports/ensemble_optimization_results.json")
        
    else:
        print(f"⚠️  {total_steps - success_count} steps failed.")
        print("Check error messages above for troubleshooting.")
    
    print()
    print("Component 5.4 pipeline complete.")
    return success_count == total_steps


def check_prerequisites():
    """
    Check that required data and dependencies are available.
    """
    print("🔍 CHECKING PREREQUISITES")
    print("-" * 50)
    
    # Check for score files
    score_dirs = [
        'data/processed/baseline_scores',
        'data/processed/gnn_scores'
    ]
    
    missing_data = []
    for score_dir in score_dirs:
        if not os.path.exists(score_dir):
            missing_data.append(score_dir)
    
    # Check for labels
    labels_file = 'data/splits/ucsd_ped2_labels.json'
    if not os.path.exists(labels_file):
        missing_data.append(labels_file)
    
    if missing_data:
        print("❌ Missing required data:")
        for item in missing_data:
            print(f"   - {item}")
        print()
        print("Please run the following commands first:")
        print("   python scripts/score_ucsd_baseline.py")
        print("   python scripts/phase3_4b_score_gnn.py")
        return False
    
    print("✅ All required data found")
    return True


def main():
    """
    Main execution function.
    """
    print("Starting Component 5.4: Refined Ensemble Methods")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("Cannot proceed without required data.")
        return False
    
    # Run complete pipeline
    success = run_component_5_4()
    
    if success:
        print("\n🎉 Component 5.4 completed successfully!")
        print("Your ensemble methods have been optimized.")
        print("Check the reports/ directory for detailed results.")
    else:
        print("\n⚠️  Component 5.4 completed with issues.")
        print("Review error messages and run individual steps if needed.")
    
    return success


if __name__ == "__main__":
    main()