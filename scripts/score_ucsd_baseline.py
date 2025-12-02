#!/usr/bin/env python3
"""
score_ucsd_baseline.py - Modular Baseline Scorer (Working Version)
Computes L2 distance baseline anomaly scores for UCSD Ped2 dataset.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import pickle
from PIL import Image
import wandb
from datetime import datetime

class UCSDBaselineScorer:
    """Modular baseline scorer for UCSD Ped2 dataset"""
    
    def __init__(self, base_path: str = 'data/raw/UCSD_Ped2/UCSDped2', enable_wandb: bool = True):
        self.base_path = Path(base_path)
        self.train_dir = self.base_path / 'Train'
        self.test_histograms_dir = self.base_path / 'Test_histograms'
        self.splits_file = Path('data/splits/ucsd_ped2_splits.json')
        self.scores_output_dir = Path('data/processed/baseline_scores')
        self.scores_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.background_model = None
        self.background_stats = {}
        self.enable_wandb = enable_wandb
        
        if self.enable_wandb:
            wandb.init(  # type: ignore
                project="temporalgraph-vad-complete",
                name=f"phase2_baseline_scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["phase2", "baseline_methods", "l2_distance"],
                config={
                    "phase": "2_baseline_methods",
                    "method": "l2_distance_baseline",
                    "feature_type": "histogram",
                    "bins": 256
                }
            )
    
    def load_training_histograms(self) -> np.ndarray:
        """Load all training histogram data"""
        print("📊 Loading training histograms...")
        
        # Load splits
        with open(self.splits_file) as f:
            splits_data = json.load(f)
        
        # Extract sequence names - they're the last part after backslashes/slashes
        train_sequences = []
        for seq_path in splits_data['train']:
            # Handle both forward and back slashes, including Windows paths
            seq_name = seq_path.replace('\\\\', '/').replace('\\', '/').split('/')[-1]
            train_sequences.append(seq_name)
        
        all_train_histograms = []
        sequence_counts = {}
        
        for seq_name in train_sequences:
            seq_folder = self.train_dir / seq_name
            
            if not seq_folder.exists():
                print(f"⚠️ Warning: {seq_name} not found, skipping...")
                continue
            
            # Get frame files
            frame_files = sorted([f for f in seq_folder.iterdir() 
                                if f.suffix.lower() in ['.tif', '.bmp'] and not f.name.startswith('.')])
            
            # Compute histograms
            seq_histograms = []
            for frame_file in frame_files:
                try:
                    img = Image.open(frame_file).convert('L')
                    arr = np.array(img)
                    hist, _ = np.histogram(arr, bins=256, range=(0, 255), density=True)
                    seq_histograms.append(hist)
                except Exception as e:
                    print(f"⚠️ Warning: Could not process {frame_file}: {e}")
                    continue
            
            if seq_histograms:
                all_train_histograms.extend(seq_histograms)
                sequence_counts[seq_name] = len(seq_histograms)
                print(f"  ✓ {seq_name}: {len(seq_histograms)} frames")
        
        if not all_train_histograms:
            raise ValueError("No training histograms found!")
        
        train_hist_array = np.array(all_train_histograms)
        
        # Store statistics
        self.background_stats = {
            'total_frames': len(train_hist_array),
            'sequence_counts': sequence_counts,
            'histogram_shape': train_hist_array.shape
        }
        
        print(f"📈 Loaded {len(train_hist_array)} training frames from {len(sequence_counts)} sequences")
        return train_hist_array
    
    def compute_background_model(self, train_hist_array: np.ndarray) -> np.ndarray:
        """Compute background model (mean histogram)"""
        print("🧮 Computing background model...")
        
        # Compute mean histogram
        mean_hist = np.mean(train_hist_array, axis=0)
        std_hist = np.std(train_hist_array, axis=0)
        
        self.background_model = mean_hist
        self.background_stats.update({
            'mean_histogram': mean_hist,
            'std_histogram': std_hist,
            'histogram_mean': float(np.mean(mean_hist)),
            'histogram_std': float(np.std(mean_hist))
        })
        
        print(f"  ✓ Background model computed: {len(mean_hist)} bins")
        return mean_hist
    
    def score_test_sequences(self) -> Dict[str, np.ndarray]:
        """Score test sequences using L2 distance"""
        if self.background_model is None:
            raise ValueError("Background model not computed!")
        
        print("🎯 Scoring test sequences...")
        
        anomaly_scores = {}
        
        # Get test histogram files
        test_hist_files = [f for f in self.test_histograms_dir.iterdir() 
                          if f.suffix == '.npy' and not f.name.startswith('.') 
                          and not f.name.endswith('_gt_histograms.npy')]
        
        for hist_file in sorted(test_hist_files):
            seq_name = hist_file.stem.replace('_histograms', '')
            
            try:
                # Load test histograms
                test_hist_array = np.load(hist_file)
                
                # Compute L2 distances
                distances = np.linalg.norm(test_hist_array - self.background_model, axis=1)
                
                anomaly_scores[seq_name] = distances
                
                print(f"  ✓ {seq_name}: {len(distances)} frames scored")
                print(f"    📊 Score range: {distances.min():.4f} - {distances.max():.4f}")
                
            except Exception as e:
                print(f"  ✗ Error processing {hist_file}: {e}")
                continue
        
        return anomaly_scores
    
    def save_baseline_scores(self, anomaly_scores: Dict[str, np.ndarray]) -> None:
        """Save baseline anomaly scores"""
        print("💾 Saving baseline scores...")
        
        for seq_name, scores in anomaly_scores.items():
            score_file = self.scores_output_dir / f"{seq_name}_scores.npy"
            np.save(score_file, scores)
            print(f"  ✓ Saved {score_file}")
        
        # Save background model
        model_file = self.scores_output_dir / 'background_model.pkl'
        model_data = {
            'background_model': self.background_model,
            'background_stats': self.background_stats,
            'method': 'l2_distance_to_mean_histogram'
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"  ✓ Saved background model: {model_file}")
    
    def run_scoring_pipeline(self) -> Dict[str, np.ndarray]:
        """Run complete scoring pipeline"""
        print("=" * 60)
        print("🚀 UCSD BASELINE SCORING PIPELINE")
        print("   Method: L2 Distance to Mean Histogram")
        print("=" * 60)
        
        # Step 1: Load training data and compute background model
        train_histograms = self.load_training_histograms()
        self.compute_background_model(train_histograms)
        
        # Log training statistics
        if self.enable_wandb:
            wandb.log({
                "train_frames_used": len(train_histograms),
                "background_model_computed": True,
                **self.background_stats
            })
        
        # Step 2: Score test sequences
        anomaly_scores = self.score_test_sequences()
        
        # Step 3: Save results
        self.save_baseline_scores(anomaly_scores)
        
        # Log final statistics
        total_frames = sum(len(s) for s in anomaly_scores.values())
        if self.enable_wandb:
            wandb.log({
                "test_sequences_scored": len(anomaly_scores),
                "total_test_frames": total_frames,
                "phase2_baseline_complete": True
            })
        
        print()
        print("=" * 60)
        print("✅ BASELINE SCORING COMPLETE")
        print(f"📁 Scores saved to: {self.scores_output_dir}")
        print("=" * 60)
        
        return anomaly_scores

def main():
    """Main execution function"""
    # Check prerequisites
    if not Path('data/splits/ucsd_ped2_splits.json').exists():
        print("❌ Error: ucsd_ped2_splits.json not found. Run make_ucsd_splits.py first.")
        return
    
    # Run scorer
    scorer = UCSDBaselineScorer()
    scores = scorer.run_scoring_pipeline()
    
    # Summary
    total_frames = sum(len(s) for s in scores.values())
    print(f"📊 Summary: Generated scores for {total_frames} frames across {len(scores)} sequences")
    
    # Finish W&B
    if scorer.enable_wandb:
        wandb.finish()  # type: ignore

if __name__ == "__main__":
    main()