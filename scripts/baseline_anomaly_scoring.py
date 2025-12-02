#!/usr/bin/env python3
"""
Baseline Anomaly Scoring Module
Implements Global Histogram Distance (L2 baseline) for unsupervised anomaly detection

This module:
1. Computes background statistics from training data (mean histogram)
2. Scores test frames based on L2 distance to background model
3. Saves frame-level anomaly scores for evaluation

Usage: python scripts/baseline_anomaly_scoring.py
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import wandb

class BaselineAnomalyScorer:
    def __init__(self, base_path: str = 'data/raw/UCSD_Ped2/UCSDped2', use_wandb: bool = True, experiment_name: str = "histogram-l2"):
        self.base_path = Path(base_path)
        self.train_dir = self.base_path / 'Train'
        self.test_histograms_dir = self.base_path / 'Test_histograms'
        self.splits_file = Path('data/splits/ucsd_ped2_splits.json')
        self.output_dir = Path('data/processed/anomaly_scores')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.background_model = None
        self.background_stats = {}
        self.use_wandb = use_wandb
        
        # Initialize W&B if enabled
        if self.use_wandb:
            try:
                wandb.init(  # type: ignore
                    project="vad-baseline-ucsdped2",
                    name=experiment_name,
                    config={
                        "feature_type": "histogram",
                        "scoring_method": "l2_mean_distance",
                        "histogram_bins": 256,
                        "dataset": "UCSD_Ped2",
                        "normalization": "density"
                    }
                )
                print(f"🔗 W&B experiment initialized: {experiment_name}")
            except Exception as e:
                print(f"W&B initialization failed: {e}. Continuing without tracking.")
                self.use_wandb = False
        
    def load_training_histograms(self) -> np.ndarray:
        """
        Step 1a: Gather background statistics
        Load all training histogram files and concatenate into single array
        """
        print("Loading training histograms for background model...")
        
        # Load splits to get training sequences
        with open(self.splits_file) as f:
            splits_data = json.load(f)
        
        # Extract just the sequence names (handle both forward and backslashes)
        train_sequences = []
        for seq_path in splits_data['train']:
            # Split by both / and \ and take the last part
            seq_name = seq_path.replace('\\', '/').split('/')[-1]
            train_sequences.append(seq_name)
        
        all_train_histograms = []
        sequence_counts = {}
        
        for seq_name in train_sequences:
            # Training histograms should be in Train_histograms directory (if they exist)
            # For now, we'll compute them from the Train directory frames
            seq_folder = self.train_dir / seq_name
            
            if not seq_folder.exists():
                print(f"⚠ Warning: Training sequence {seq_name} not found, skipping...")
                continue
            
            # Get frame files and compute histograms on-the-fly
            frame_files = sorted([f for f in seq_folder.iterdir() 
                                if f.suffix.lower() in ['.tif', '.bmp'] and not f.name.startswith('.')])
            
            seq_histograms = []
            for frame_file in frame_files:
                try:
                    from PIL import Image
                    img = Image.open(frame_file).convert('L')
                    arr = np.array(img)
                    hist, _ = np.histogram(arr, bins=256, range=(0, 255), density=True)
                    seq_histograms.append(hist)
                except Exception as e:
                    print(f"⚠ Warning: Could not process {frame_file}: {e}")
                    continue
            
            if seq_histograms:
                all_train_histograms.extend(seq_histograms)
                sequence_counts[seq_name] = len(seq_histograms)
                print(f"  ✓ {seq_name}: {len(seq_histograms)} frames")
        
        if not all_train_histograms:
            raise ValueError("No training histograms found! Check data paths.")
        
        train_hist_array = np.array(all_train_histograms)
        print(f"\nLoaded {len(train_hist_array)} training frames from {len(sequence_counts)} sequences")
        print(f"   Shape: {train_hist_array.shape} (frames × bins)")
        
        # Store statistics
        self.background_stats = {
            'total_frames': len(train_hist_array),
            'sequence_counts': sequence_counts,
            'histogram_shape': train_hist_array.shape
        }
        
        return train_hist_array
    
    def compute_background_model(self, train_hist_array: np.ndarray) -> np.ndarray:
        """
        Step 1a (continued): Compute mean histogram as background model
        """
        print("🧮 Computing background model (mean histogram)...")
        
        # Compute mean vector per bin
        mean_hist = np.mean(train_hist_array, axis=0)
        
        # Compute additional statistics for analysis
        std_hist = np.std(train_hist_array, axis=0)
        
        self.background_model = mean_hist
        self.background_stats.update({
            'mean_histogram': mean_hist,
            'std_histogram': std_hist,
            'histogram_mean': float(np.mean(mean_hist)),
            'histogram_std': float(np.std(mean_hist))
        })
        
        print(f"  ✓ Background model computed: {len(mean_hist)} bins")
        print(f"  Histogram statistics: mean={self.background_stats['histogram_mean']:.6f}, "
              f"std={self.background_stats['histogram_std']:.6f}")
        
        return mean_hist
    
    def score_test_sequences(self) -> Dict[str, np.ndarray]:
        """
        Step 1b-c: Score each test frame using L2 distance to background model
        """
        if self.background_model is None:
            raise ValueError("Background model not computed! Run compute_background_model() first.")
        
        print("\nScoring test sequences...")
        
        anomaly_scores = {}
        
        # Get all test histogram files
        test_hist_files = [f for f in self.test_histograms_dir.iterdir() 
                          if f.suffix == '.npy' and not f.name.startswith('.') 
                          and not f.name.endswith('_gt_histograms.npy')]
        
        for hist_file in sorted(test_hist_files):
            # Extract sequence name (e.g., Test001_histograms.npy -> Test001)
            seq_name = hist_file.stem.replace('_histograms', '')
            
            try:
                # Load test histogram array
                test_hist_array = np.load(hist_file)
                
                # Compute L2 distance for every frame
                # Broadcasting: test_hist_array [frames, 256] - background_model [256]
                distances = np.linalg.norm(test_hist_array - self.background_model, axis=1)
                
                anomaly_scores[seq_name] = distances
                
                print(f"  ✓ {seq_name}: {len(distances)} frames scored")
                print(f"    Score range: {distances.min():.4f} - {distances.max():.4f} "
                      f"(mean: {distances.mean():.4f})")
                
            except Exception as e:
                print(f"  ✗ Error processing {hist_file}: {e}")
                continue
        
        print(f"\nScored {len(anomaly_scores)} test sequences")
        return anomaly_scores
    
    def save_results(self, anomaly_scores: Dict[str, np.ndarray]) -> None:
        """
        Save anomaly scores and background model for later use
        """
        print("\nSaving results...")
        
        # Save individual sequence scores as .npy files
        scores_dir = self.output_dir / 'baseline_l2_scores'
        scores_dir.mkdir(exist_ok=True)
        
        for seq_name, scores in anomaly_scores.items():
            score_file = scores_dir / f"{seq_name}_anomaly_scores.npy"
            np.save(score_file, scores)
            print(f"  ✓ Saved {score_file}")
        
        # Save background model and metadata
        model_file = self.output_dir / 'baseline_l2_model.pkl'
        model_data = {
            'background_model': self.background_model,
            'background_stats': self.background_stats,
            'anomaly_scores': anomaly_scores,
            'method': 'L2_distance_to_mean_histogram'
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"  ✓ Saved background model: {model_file}")
        
        # Save summary statistics as JSON
        summary_file = self.output_dir / 'baseline_l2_summary.json'
        summary_data = {
            'method': 'L2_distance_to_mean_histogram',
            'background_stats': {
                k: v for k, v in self.background_stats.items() 
                if k not in ['mean_histogram', 'std_histogram']  # Skip large arrays
            },
            'test_sequences': {
                seq_name: {
                    'num_frames': len(scores),
                    'score_min': float(scores.min()),
                    'score_max': float(scores.max()),
                    'score_mean': float(scores.mean()),
                    'score_std': float(scores.std())
                }
                for seq_name, scores in anomaly_scores.items()
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"  ✓ Saved summary: {summary_file}")
        
        # Log to W&B if enabled
        if self.use_wandb:
            try:
                # Log background model statistics
                wandb.log({
                    "train_sequences": len(self.background_stats['sequence_counts']),
                    "train_frames": self.background_stats['total_frames'],
                    "background_histogram_mean": self.background_stats['histogram_mean'],
                    "background_histogram_std": self.background_stats['histogram_std']
                })
                
                # Log per-sequence score statistics
                for seq_name, scores in anomaly_scores.items():
                    wandb.log({
                        f"score_min_{seq_name}": float(scores.min()),
                        f"score_max_{seq_name}": float(scores.max()),
                        f"score_mean_{seq_name}": float(scores.mean()),
                        f"score_std_{seq_name}": float(scores.std()),
                        f"num_frames_{seq_name}": len(scores)
                    })
                
                print("  ✓ Logged results to W&B")
            except Exception as e:
                print(f"  W&B logging failed: {e}")
    
    def run_baseline_scoring(self) -> Dict[str, np.ndarray]:
        """
        Complete pipeline: compute background model and score test sequences
        """
        print("=" * 60)
        print("BASELINE ANOMALY SCORING PIPELINE")
        print("   Method: L2 Distance to Mean Histogram")
        print("=" * 60)
        
        # Step 1a: Load training data and compute background model
        train_histograms = self.load_training_histograms()
        self.compute_background_model(train_histograms)
        
        # Step 1b-c: Score test sequences
        anomaly_scores = self.score_test_sequences()
        
        # Save results
        self.save_results(anomaly_scores)
        
        print("\n" + "=" * 60)
        print("BASELINE SCORING COMPLETE")
        print(f"📁 Results saved to: {self.output_dir}")
        print("=" * 60)
        
        return anomaly_scores

def main(use_wandb: bool = True, experiment_name: str = "histogram-l2"):
    """Main execution function"""
    # Check if required files exist
    if not Path('data/splits/ucsd_ped2_splits.json').exists():
        print("Error: ucsd_ped2_splits.json not found. Run make_ucsd_splits.py first.")
        return
    
    # Initialize and run baseline scorer
    scorer = BaselineAnomalyScorer(use_wandb=use_wandb, experiment_name=experiment_name)
    anomaly_scores = scorer.run_baseline_scoring()
    
    # Print summary
    total_frames = sum(len(scores) for scores in anomaly_scores.values())
    print(f"\nSummary: Scored {total_frames} frames across {len(anomaly_scores)} sequences")
    
    # Finish W&B run
    if scorer.use_wandb:
        try:
            wandb.finish()  # type: ignore
            print("W&B experiment completed")
        except:
            pass

if __name__ == "__main__":
    main()