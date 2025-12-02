#!/usr/bin/env python3
"""
MASTER W&B TRACKING PIPELINE - Complete Project Lifecycle
Orchestrates all phases of TemporalGraphHistogram-VAD with comprehensive W&B tracking.

This script runs the complete project pipeline from start to finish with full W&B integration:
Phase 1: Data Preparation & Preprocessing
Phase 2: Baseline Methods 
Phase 3: Temporal Graph Generation
Phase 4: GNN Training & Scoring
Phase 5: Comprehensive Ablation Studies 
Phase 6: Ensemble Methods & Optimization

Usage: python scripts/master_wandb_pipeline.py [--phase PHASE]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import wandb
import json

class MasterWandBPipeline:
    """Complete project pipeline with comprehensive W&B tracking."""
    
    def __init__(self, project_name="temporalgraph-vad-complete"):
        self.project_name = project_name
        self.start_time = datetime.now()
        
        # Initialize master W&B run
        wandb.init(  # type: ignore
            project=project_name,
            name=f"master-pipeline-{self.start_time.strftime('%Y%m%d_%H%M%S')}",
            tags=["master-pipeline", "complete-lifecycle", "automated"],
            config={
                "pipeline_type": "complete_project_lifecycle",
                "execution_mode": "automated",
                "start_time": self.start_time.isoformat(),
                "phases": ["data_prep", "baseline", "graphs", "gnn", "ablation", "ensemble"]
            }
        )
        
        self.phase_results = {}
        self.overall_success = True
        
    def log_phase_start(self, phase_name, description):
        """Log the start of a phase."""
        print(f"\\n{'='*80}")
        print(f"🚀 STARTING {phase_name}")
        print(f"   {description}")
        print(f"{'='*80}")
        
        wandb.log({f"{phase_name.lower().replace(' ', '_')}_started": True})  # type: ignore
        
    def log_phase_completion(self, phase_name, success, details=None):
        """Log the completion of a phase."""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\\n{status}: {phase_name}")
        
        log_data = {
            f"{phase_name.lower().replace(' ', '_')}_completed": True,
            f"{phase_name.lower().replace(' ', '_')}_success": success
        }
        
        if details:
            log_data.update({f"{phase_name.lower().replace(' ', '_')}_{k}": v for k, v in details.items()})
            
        wandb.log(log_data)  # type: ignore
        self.phase_results[phase_name] = {"success": success, "details": details}
        
        if not success:
            self.overall_success = False
    
    def run_script(self, script_path, description, timeout=1800):
        """Run a Python script and capture results."""
        print(f"\\n🔄 Running: {description}")
        print(f"   Script: {script_path}")
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd()
            )
            
            success = result.returncode == 0
            
            if success:
                print(f"✅ {description} completed successfully")
            else:
                print(f"❌ {description} failed:")
                print(f"   Error: {result.stderr[:200]}...")
                
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} timed out after {timeout} seconds")
            return False, "", "Timeout"
        except Exception as e:
            print(f"❌ {description} failed with exception: {e}")
            return False, "", str(e)
    
    def phase1_data_preparation(self):
        """Phase 1: Data Preparation & Preprocessing"""
        self.log_phase_start("PHASE 1", "Data Preparation & Preprocessing")
        
        scripts = [
            ("scripts/make_ucsd_splits.py", "Generate train/test splits"),
            ("scripts/make_ucsd_label_masks.py", "Extract anomaly labels"),
            ("scripts/extract_ucsd_histograms.py", "Extract histogram features"),
            ("scripts/validate_ucsd_dataset.py", "Validate dataset integrity")
        ]
        
        phase_success = True
        completed_scripts = 0
        
        for script_path, description in scripts:
            if Path(script_path).exists():
                success, stdout, stderr = self.run_script(script_path, description)
                if success:
                    completed_scripts += 1
                else:
                    phase_success = False
            else:
                print(f"⚠️  Script not found: {script_path}")
        
        details = {
            "scripts_completed": completed_scripts,
            "total_scripts": len(scripts),
            "completion_rate": completed_scripts / len(scripts)
        }
        
        self.log_phase_completion("PHASE 1", phase_success, details)
        return phase_success
    
    def phase2_baseline_methods(self):
        """Phase 2: Baseline Anomaly Detection Methods"""
        self.log_phase_start("PHASE 2", "Baseline Anomaly Detection Methods")
        
        scripts = [
            ("scripts/score_ucsd_baseline.py", "Generate baseline L2 scores"),
            ("scripts/baseline_anomaly_scoring.py", "Advanced baseline scoring"),
            ("scripts/evaluate_ucsd_scores.py", "Evaluate baseline performance")
        ]
        
        phase_success = True
        completed_scripts = 0
        
        for script_path, description in scripts:
            if Path(script_path).exists():
                success, stdout, stderr = self.run_script(script_path, description)
                if success:
                    completed_scripts += 1
                else:
                    phase_success = False
            else:
                print(f"⚠️  Script not found: {script_path}")
        
        details = {
            "scripts_completed": completed_scripts,
            "total_scripts": len(scripts),
            "baseline_methods_tested": completed_scripts
        }
        
        self.log_phase_completion("PHASE 2", phase_success, details)
        return phase_success
    
    def phase3_temporal_graphs(self):
        """Phase 3: Temporal Graph Generation"""
        self.log_phase_start("PHASE 3", "Temporal Graph Generation")
        
        scripts = [
            ("scripts/phase3_extract_cnn_features.py", "Extract CNN features"),
            ("scripts/phase3_build_temporal_graphs.py", "Build temporal graphs"),
            ("scripts/generate_training_graphs.py", "Generate training graphs")
        ]
        
        phase_success = True
        completed_scripts = 0
        
        for script_path, description in scripts:
            if Path(script_path).exists():
                success, stdout, stderr = self.run_script(script_path, description, timeout=3600)  # Longer timeout
                if success:
                    completed_scripts += 1
                else:
                    phase_success = False
            else:
                print(f"⚠️  Script not found: {script_path}")
        
        details = {
            "scripts_completed": completed_scripts,
            "total_scripts": len(scripts),
            "graph_generation_success": phase_success
        }
        
        self.log_phase_completion("PHASE 3", phase_success, details)
        return phase_success
    
    def phase4_gnn_training(self):
        """Phase 4: GNN Training & Scoring"""
        self.log_phase_start("PHASE 4", "GNN Training & Scoring")
        
        scripts = [
            ("scripts/phase3_4a_train_gnn.py", "Train GNN autoencoder"),
            ("scripts/phase3_4b_score_gnn.py", "Generate GNN scores"),
            ("scripts/priority1a_hyperparameter_tuning.py", "Hyperparameter tuning")
        ]
        
        phase_success = True
        completed_scripts = 0
        
        for script_path, description in scripts:
            if Path(script_path).exists():
                # Longer timeout for training
                timeout = 7200 if "train" in script_path else 3600
                success, stdout, stderr = self.run_script(script_path, description, timeout=timeout)
                if success:
                    completed_scripts += 1
                else:
                    phase_success = False
            else:
                print(f"⚠️  Script not found: {script_path}")
        
        details = {
            "scripts_completed": completed_scripts,
            "total_scripts": len(scripts),
            "gnn_training_success": phase_success
        }
        
        self.log_phase_completion("PHASE 4", phase_success, details)
        return phase_success
    
    def phase5_ablation_studies(self):
        """Phase 5: Comprehensive Ablation Studies"""
        self.log_phase_start("PHASE 5", "Comprehensive Ablation Studies")
        
        scripts = [
            ("scripts/phase5/5.1c_train_ablation_models_clean.py", "Feature ablation study"),
            ("scripts/phase5/5.2_graph_structure_ablation.py", "Graph structure ablation"),
            ("scripts/phase5/5.3_avenue_evaluation.py", "Cross-dataset validation"),
            ("scripts/phase5/run_phase5_with_wandb.py", "Unified ablation pipeline")
        ]
        
        phase_success = True
        completed_scripts = 0
        
        for script_path, description in scripts:
            if Path(script_path).exists():
                success, stdout, stderr = self.run_script(script_path, description, timeout=3600)
                if success:
                    completed_scripts += 1
                else:
                    phase_success = False
            else:
                print(f"⚠️  Script not found: {script_path}")
        
        details = {
            "scripts_completed": completed_scripts,
            "total_scripts": len(scripts),
            "ablation_studies_complete": phase_success
        }
        
        self.log_phase_completion("PHASE 5", phase_success, details)
        return phase_success
    
    def phase6_ensemble_optimization(self):
        """Phase 6: Ensemble Methods & Optimization"""
        self.log_phase_start("PHASE 6", "Ensemble Methods & Optimization")
        
        scripts = [
            ("scripts/priority2a_ensemble_method.py", "Ensemble optimization"),
            ("scripts/final_ensemble_real_data.py", "Final ensemble evaluation"),
            ("scripts/run_optimization_pipeline.py", "Complete optimization")
        ]
        
        phase_success = True
        completed_scripts = 0
        
        for script_path, description in scripts:
            if Path(script_path).exists():
                success, stdout, stderr = self.run_script(script_path, description)
                if success:
                    completed_scripts += 1
                else:
                    phase_success = False
            else:
                print(f"⚠️  Script not found: {script_path}")
        
        details = {
            "scripts_completed": completed_scripts,
            "total_scripts": len(scripts),
            "ensemble_optimization_complete": phase_success
        }
        
        self.log_phase_completion("PHASE 6", phase_success, details)
        return phase_success
    
    def run_complete_pipeline(self):
        """Run the complete project pipeline."""
        print(f"🚀 STARTING COMPLETE PROJECT PIPELINE")
        print(f"   Start time: {self.start_time}")
        print(f"   W&B project: {self.project_name}")
        
        # Run all phases
        phases = [
            ("Phase 1", self.phase1_data_preparation),
            ("Phase 2", self.phase2_baseline_methods), 
            ("Phase 3", self.phase3_temporal_graphs),
            ("Phase 4", self.phase4_gnn_training),
            ("Phase 5", self.phase5_ablation_studies),
            ("Phase 6", self.phase6_ensemble_optimization)
        ]
        
        successful_phases = 0
        
        for phase_name, phase_func in phases:
            success = phase_func()
            if success:
                successful_phases += 1
        
        # Final summary
        completion_time = datetime.now()
        total_duration = completion_time - self.start_time
        
        wandb.log({  # type: ignore
            "pipeline_completed": True,
            "overall_success": self.overall_success,
            "successful_phases": successful_phases,
            "total_phases": len(phases),
            "completion_rate": successful_phases / len(phases),
            "total_duration_minutes": total_duration.total_seconds() / 60,
            "completion_time": completion_time.isoformat()
        })
        
        print(f"\\n{'='*80}")
        print(f"🎯 COMPLETE PIPELINE SUMMARY")
        print(f"{'='*80}")
        print(f"📊 Phases completed: {successful_phases}/{len(phases)}")
        print(f"⏱️  Total duration: {total_duration}")
        print(f"✅ Overall success: {self.overall_success}")
        
        if self.overall_success:
            print(f"🎉 COMPLETE PROJECT SUCCESS!")
            print(f"📈 Ready for research publication and production deployment")
        else:
            print(f"⚠️  Some phases failed - check logs for details")
        
        print(f"\\n📊 W&B Dashboard: https://wandb.ai/{wandb.run.entity}/{self.project_name}")
        
        wandb.finish()  # type: ignore
        return self.overall_success

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Master W&B Pipeline for TemporalGraphHistogram-VAD")
    parser.add_argument("--phase", type=int, choices=range(1, 7),
                       help="Run specific phase only (1-6)")
    parser.add_argument("--project", type=str, default="temporalgraph-vad-complete",
                       help="W&B project name")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MasterWandBPipeline(project_name=args.project)
    
    if args.phase:
        # Run specific phase
        phase_methods = {
            1: pipeline.phase1_data_preparation,
            2: pipeline.phase2_baseline_methods,
            3: pipeline.phase3_temporal_graphs,
            4: pipeline.phase4_gnn_training,
            5: pipeline.phase5_ablation_studies,
            6: pipeline.phase6_ensemble_optimization
        }
        
        print(f"🎯 Running Phase {args.phase} only")
        success = phase_methods[args.phase]()
        
        wandb.log({
            "single_phase_execution": True,
            "phase_number": args.phase,
            "phase_success": success
        })  # type: ignore
        
        wandb.finish()  # type: ignore
        sys.exit(0 if success else 1)
    else:
        # Run complete pipeline
        success = pipeline.run_complete_pipeline()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()