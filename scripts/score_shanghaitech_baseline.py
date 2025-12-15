#!/usr/bin/env python3
"""
Baseline Anomaly Scoring for ShanghaiTech Dataset
Purpose: Generate simple histogram-based anomaly scores without GNN

Baseline Method: L2 Distance from Mean Normal Histogram
1. Compute mean histogram from all available frames
2. For each test frame, compute L2 distance from mean
3. Higher distance = more anomalous

This provides a simple baseline to compare against GNN methods.

Usage: python scripts/score_shanghaitech_baseline.py
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict
import time

class ShanghaiTechBaselineScorer:
    """Generate baseline anomaly scores for ShanghaiTech"""
    
    def __init__(self):
        self.histograms_dir = Path('data/processed/shanghaitech/test_histograms')
        self.labels_file = Path('data/splits/shanghaitech_labels.json')
        self.output_dir = Path('data/processed/shanghaitech/anomaly_scores')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_histograms(self) -> Dict[str, np.ndarray]:
        """Load histogram features for all test sequences"""
        with open(self.labels_file, 'r') as f:
            labels_dict = json.load(f)
        
        sequences = sorted(labels_dict.keys())
        histograms_dict = {}
        
        print(f"📂 Loading histograms for {len(sequences)} sequences...")
        start_time = time.time()
        
        for seq_name in sequences:
            hist_file = self.histograms_dir / f'{seq_name}_histograms.npy'
            
            if not hist_file.exists():
                print(f"   ⚠️ Missing histograms for {seq_name}")
                continue
            
            histograms = np.load(hist_file)
            histograms_dict[seq_name] = histograms
        
        elapsed = time.time() - start_time
        total_frames = sum(len(h) for h in histograms_dict.values())
        print(f"   ✓ Loaded {total_frames:,} frames in {elapsed:.2f}s")
        
        return histograms_dict
    
    def compute_mean_histogram(self, histograms_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute mean histogram across all frames
        
        This represents the "normal" pattern
        """
        all_histograms = np.vstack([h for h in histograms_dict.values()])
        mean_hist = np.mean(all_histograms, axis=0)
        
        print(f"\n📊 Computed mean histogram from {len(all_histograms):,} frames")
        
        return mean_hist
    
    def compute_anomaly_scores(self, 
                               histograms_dict: Dict[str, np.ndarray],
                               mean_hist: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute L2 distances from mean histogram
        
        Args:
            histograms_dict: Frame histograms per sequence
            mean_hist: Reference "normal" histogram
        
        Returns:
            Dictionary of anomaly scores per sequence
        """
        scores_dict = {}
        
        print(f"\n🔍 Computing anomaly scores...")
        start_time = time.time()
        
        for seq_name, histograms in histograms_dict.items():
            # L2 distance from mean for each frame
            distances = np.linalg.norm(histograms - mean_hist, axis=1)
            scores_dict[seq_name] = distances
        
        elapsed = time.time() - start_time
        total_frames = sum(len(s) for s in scores_dict.values())
        
        print(f"   ✓ Scored {total_frames:,} frames in {elapsed:.2f}s")
        print(f"   Processing rate: {total_frames/elapsed:.0f} frames/sec")
        
        return scores_dict
    
    def save_scores(self, scores_dict: Dict[str, np.ndarray]):
        """Save baseline anomaly scores"""
        output_file = self.output_dir / 'baseline_anomaly_scores.npy'
        np.save(output_file, scores_dict)
        print(f"\n💾 Scores saved to: {output_file}")
        
        # Summary statistics
        all_scores = np.concatenate([scores for scores in scores_dict.values()])
        print(f"\n📊 Score Statistics:")
        print(f"   Total frames: {len(all_scores):,}")
        print(f"   Mean: {np.mean(all_scores):.4f}")
        print(f"   Std: {np.std(all_scores):.4f}")
        print(f"   Min: {np.min(all_scores):.4f}")
        print(f"   Max: {np.max(all_scores):.4f}")

def main():
    """Main execution function"""
    print("="*60)
    print("Baseline Anomaly Scoring - ShanghaiTech Dataset")
    print("="*60)
    print("\nMethod: L2 Distance from Mean Histogram")
    
    scorer = ShanghaiTechBaselineScorer()
    
    # Load all histograms
    histograms_dict = scorer.load_all_histograms()
    
    # Compute reference "normal" histogram
    mean_hist = scorer.compute_mean_histogram(histograms_dict)
    
    # Score all frames
    scores_dict = scorer.compute_anomaly_scores(histograms_dict, mean_hist)
    
    # Save results
    scorer.save_scores(scores_dict)
    
    print("\n" + "="*60)
    print("✅ Baseline scoring complete!")
    print("="*60)
    print("\nNext step: Run evaluation script")
    print("  python scripts/evaluate_shanghaitech_scores.py")

if __name__ == "__main__":
    main()
