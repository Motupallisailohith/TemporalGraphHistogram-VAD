#!/usr/bin/env python3
"""
Generate dummy anomaly scores for Avenue dataset testing
Uses extracted features to create realistic anomaly scores for evaluation pipeline testing
"""

import os
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

def generate_dummy_avenue_scores():
    """Generate dummy anomaly scores for Avenue test sequences"""
    print("🎯 Generating dummy Avenue anomaly scores...")
    
    # Paths
    features_dir = Path('data/processed/avenue/features')
    scores_dir = Path('data/processed/avenue/anomaly_scores')
    
    # Create score directories
    for feature_type in ['histogram', 'cnn', 'optical_flow']:
        (scores_dir / f'{feature_type}_scores').mkdir(parents=True, exist_ok=True)
    
    # Load sequences info
    with open('data/processed/avenue/avenue_sequences.json') as f:
        sequences_data = json.load(f)
    
    # Process each feature type
    for feature_type in ['histogram', 'cnn', 'optical_flow']:
        print(f"\n📊 Processing {feature_type} features...")
        
        all_scores = []
        
        # Process test sequences
        for seq_info in tqdm(sequences_data['test_sequences'], desc=f"Generating {feature_type} scores"):
            seq_name = seq_info['name']
            feature_file = features_dir / f'test_{seq_name}_{feature_type}s.npy'
            
            if feature_file.exists():
                # Load features
                features = np.load(feature_file)
                
                # Generate dummy anomaly scores
                # Use distance from mean as anomaly score + some noise
                mean_feature = np.mean(features, axis=0)
                scores = []
                
                for i, feature in enumerate(features):
                    # Compute anomaly score as distance from mean + noise
                    if feature_type == 'histogram':
                        # For histograms, use L1 distance
                        score = np.sum(np.abs(feature - mean_feature))
                    else:
                        # For other features, use L2 distance
                        score = np.linalg.norm(feature - mean_feature)
                    
                    # Add some temporal structure and noise
                    temporal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * i / len(features))
                    noise = np.random.normal(0, 0.1)
                    final_score = score * temporal_factor + noise
                    
                    scores.append(final_score)
                
                all_scores.extend(scores)
                
                print(f"  ✓ {seq_name}: {len(scores)} scores (range: {np.min(scores):.3f}-{np.max(scores):.3f})")
            else:
                print(f"  ⚠️ Missing feature file: {feature_file}")
        
        # Save combined scores
        if all_scores:
            scores_array = np.array(all_scores)
            scores_file = scores_dir / f'{feature_type}_scores' / 'avenue_test_anomaly_scores.npy'
            np.save(scores_file, scores_array)
            print(f"  ✅ Saved {len(scores_array)} {feature_type} scores to {scores_file}")
        
        print(f"  📈 {feature_type} scores summary:")
        print(f"     Total frames: {len(all_scores)}")
        if all_scores:
            print(f"     Score range: {np.min(all_scores):.3f} - {np.max(all_scores):.3f}")
            print(f"     Mean ± Std: {np.mean(all_scores):.3f} ± {np.std(all_scores):.3f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    generate_dummy_avenue_scores()
    
    print("\n✅ Dummy Avenue anomaly scores generation complete!")
    print("🔄 Ready to run Avenue evaluation pipeline")