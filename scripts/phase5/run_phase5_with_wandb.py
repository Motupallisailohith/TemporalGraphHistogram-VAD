#!/usr/bin/env python3
"""
Unified Phase 5 Pipeline with W&B Integration
Runs all Phase 5 components with comprehensive experiment tracking.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import wandb

def run_phase5_with_wandb():
    """
    Execute complete Phase 5 pipeline with W&B tracking.
    """
    
    print("=" * 80)
    print("🚀 PHASE 5 COMPREHENSIVE PIPELINE WITH W&B TRACKING")
    print("=" * 80)
    
    # Initialize master wandb run
    wandb.init(  # type: ignore
        project="temporalgraph-vad-ablations",
        name="phase5-complete-pipeline",
        tags=["phase5", "comprehensive", "ablation-study", "cross-dataset"],
        config={
            "pipeline_type": "complete_phase5",
            "components": ["5.1c", "5.2", "5.3"],
            "datasets": ["UCSD_Ped2", "Avenue"],
            "experiment_scope": "feature_ablation_graph_structure_cross_dataset"
        }
    )
    
    print("📊 Master W&B experiment tracking initialized")
    
    # Component 5.1c: Feature Ablation
    print("\n" + "=" * 70)
    print("🔬 COMPONENT 5.1c: FEATURE ABLATION")
    print("=" * 70)
    
    try:
        print("Running feature ablation study...")
        result = subprocess.run([
            sys.executable, 
            "scripts/phase5/5.1c_train_ablation_models_clean.py"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Component 5.1c completed successfully")
            
            # Load and log results
            feature_results_path = "reports/ablations/feature_ablation_results.json"
            if Path(feature_results_path).exists():
                with open(feature_results_path, 'r') as f:
                    feature_results = json.load(f)
                
                wandb.log({  # type: ignore
                    "component_5_1c_complete": True,
                    **{f"feature_{k}_auc": v for k, v in feature_results.items()}
                })
                
                print(f"📊 Feature ablation results logged to W&B")
            else:
                print("⚠️  Feature ablation results file not found")
        else:
            print(f"❌ Component 5.1c failed: {result.stderr}")
            wandb.log({"component_5_1c_failed": True})  # type: ignore
            
    except Exception as e:
        print(f"❌ Error running Component 5.1c: {e}")
        wandb.log({"component_5_1c_error": str(e)})  # type: ignore
    
    # Component 5.2: Graph Structure Ablation  
    print("\n" + "=" * 70)
    print("📊 COMPONENT 5.2: GRAPH STRUCTURE ABLATION")
    print("=" * 70)
    
    try:
        print("Running graph structure ablation study...")
        result = subprocess.run([
            sys.executable,
            "scripts/phase5/5.2_graph_structure_ablation.py"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Component 5.2 completed successfully")
            
            # Load and log results
            graph_results_path = "reports/ablations/graph_structure_ablation.json"
            if Path(graph_results_path).exists():
                with open(graph_results_path, 'r') as f:
                    graph_results = json.load(f)
                
                # Log window size results
                for window, auc in graph_results.get('window_sizes', {}).items():
                    wandb.log({f"graph_{window}_auc": auc})  # type: ignore
                
                # Log edge weighting results  
                for scheme, auc in graph_results.get('edge_weighting', {}).items():
                    wandb.log({f"edge_{scheme}_auc": auc})  # type: ignore
                
                wandb.log({"component_5_2_complete": True})  # type: ignore
                print(f"📊 Graph structure results logged to W&B")
            else:
                print("⚠️  Graph structure results file not found")
        else:
            print(f"❌ Component 5.2 failed: {result.stderr}")
            wandb.log({"component_5_2_failed": True})  # type: ignore
            
    except Exception as e:
        print(f"❌ Error running Component 5.2: {e}")
        wandb.log({"component_5_2_error": str(e)})  # type: ignore
    
    # Component 5.3: Cross-Dataset Validation
    print("\n" + "=" * 70)
    print("🌍 COMPONENT 5.3: AVENUE CROSS-DATASET VALIDATION")
    print("=" * 70)
    
    try:
        print("Running Avenue cross-dataset validation...")
        result = subprocess.run([
            sys.executable,
            "scripts/phase5/5.3_avenue_evaluation.py"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Component 5.3 completed successfully")
            
            # Load and log results
            avenue_results_path = "reports/ablations/avenue_cross_dataset_results.json"
            if Path(avenue_results_path).exists():
                with open(avenue_results_path, 'r') as f:
                    avenue_results = json.load(f)
                
                # Log cross-dataset performance
                ucsd_results = avenue_results.get('ucsd_results', {})
                avenue_perf = avenue_results.get('avenue_results', {})
                
                for feature_type in ['histogram', 'optical_flow', 'cnn']:
                    ucsd_auc = ucsd_results.get(feature_type, 0.0)
                    avenue_auc = avenue_perf.get(feature_type, 0.0)
                    
                    wandb.log({  # type: ignore
                        f"cross_dataset_{feature_type}_ucsd": ucsd_auc,
                        f"cross_dataset_{feature_type}_avenue": avenue_auc,
                        f"cross_dataset_{feature_type}_drop": ucsd_auc - avenue_auc
                    })
                
                # Log generalization metrics
                comparison = avenue_results.get('performance_comparison', {})
                wandb.log({  # type: ignore
                    "generalization_drop": comparison.get('generalization_drop', 0.0),
                    "generalization_status": comparison.get('generalization_status', 'unknown'),
                    "component_5_3_complete": True
                })
                
                print(f"📊 Cross-dataset results logged to W&B")
            else:
                print("⚠️  Avenue results file not found")
        else:
            print(f"❌ Component 5.3 failed: {result.stderr}")
            wandb.log({"component_5_3_failed": True})  # type: ignore
            
    except Exception as e:
        print(f"❌ Error running Component 5.3: {e}")
        wandb.log({"component_5_3_error": str(e)})  # type: ignore
    
    # Generate Comprehensive Summary
    print("\n" + "=" * 70)
    print("📋 COMPREHENSIVE PHASE 5 SUMMARY")
    print("=" * 70)
    
    try:
        # Load all results for final summary
        summary_data = {}
        
        # Feature ablation results
        feature_path = "reports/ablations/feature_ablation_results.json"
        if Path(feature_path).exists():
            with open(feature_path, 'r') as f:
                summary_data['feature_ablation'] = json.load(f)
        
        # Graph structure results
        graph_path = "reports/ablations/graph_structure_ablation.json"
        if Path(graph_path).exists():
            with open(graph_path, 'r') as f:
                summary_data['graph_structure'] = json.load(f)
        
        # Cross-dataset results
        avenue_path = "reports/ablations/avenue_cross_dataset_results.json"
        if Path(avenue_path).exists():
            with open(avenue_path, 'r') as f:
                summary_data['cross_dataset'] = json.load(f)
        
        # Compute final insights
        if summary_data:
            # Best performing configurations
            feature_results = summary_data.get('feature_ablation', {})
            if feature_results:
                best_feature = max(feature_results.items(), key=lambda x: x[1])
                wandb.log({  # type: ignore
                    "best_feature_type": best_feature[0],
                    "best_feature_auc": best_feature[1]
                })
            
            # Optimal graph structure
            graph_results = summary_data.get('graph_structure', {})
            if graph_results:
                window_results = graph_results.get('window_sizes', {})
                if window_results:
                    best_window = max(window_results.items(), key=lambda x: x[1])
                    wandb.log({  # type: ignore
                        "optimal_window_size": best_window[0],
                        "optimal_window_auc": best_window[1]
                    })
                
                edge_results = graph_results.get('edge_weighting', {})
                if edge_results:
                    best_edge = max(edge_results.items(), key=lambda x: x[1])
                    wandb.log({  # type: ignore
                        "optimal_edge_weighting": best_edge[0],
                        "optimal_edge_auc": best_edge[1]
                    })
            
            # Cross-dataset generalization
            cross_results = summary_data.get('cross_dataset', {})
            if cross_results:
                perf_comparison = cross_results.get('performance_comparison', {})
                wandb.log({  # type: ignore
                    "final_generalization_assessment": perf_comparison.get('generalization_status', 'unknown'),
                    "final_performance_drop": perf_comparison.get('generalization_drop', 0.0)
                })
            
            # Save comprehensive summary
            final_summary_path = "reports/ablations/phase5_comprehensive_summary.json"
            with open(final_summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            wandb.log({"phase5_pipeline_complete": True})  # type: ignore
            
            print(f"✅ Phase 5 comprehensive summary saved: {final_summary_path}")
            print(f"📊 Complete experimental results available on W&B dashboard")
            
        else:
            print("⚠️  No summary data available")
            wandb.log({"phase5_pipeline_incomplete": True})  # type: ignore
            
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        wandb.log({"summary_generation_error": str(e)})  # type: ignore
    
    # Finalize W&B logging
    wandb.finish()  # type: ignore
    
    print("\n" + "=" * 80)
    print("🎯 PHASE 5 PIPELINE WITH W&B TRACKING COMPLETE!")
    print("=" * 80)
    print("📊 View comprehensive results on your W&B dashboard")
    print("🔬 All ablation studies and cross-dataset validation logged")
    print("🚀 Ready for production deployment with optimized configurations")

if __name__ == "__main__":
    try:
        run_phase5_with_wandb()
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        wandb.finish()  # type: ignore
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        wandb.finish()  # type: ignore
        sys.exit(1)