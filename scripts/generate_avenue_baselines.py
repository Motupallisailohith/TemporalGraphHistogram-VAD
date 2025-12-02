#!/usr/bin/env python3
"""
Generate Simple Baseline Anomaly Scores for Avenue Dataset
Uses simple statistical distance measures on existing features
"""

import numpy as np
from pathlib import Path
import json

def generate_avenue_baselines():
    """Generate simple baseline scores for Avenue comparison"""
    print("📊 Generating Avenue baseline anomaly scores...")
    
    features_dir = Path("data/processed/avenue/features") 
    scores_dir = Path("data/processed/avenue/anomaly_scores")
    
    # Load sequences info
    with open("data/processed/avenue/avenue_sequences.json") as f:
        sequences_data = json.load(f)
    
    # Create baseline scores directory
    baseline_dir = scores_dir / "baseline_scores"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🎯 Processing feature types...")
    
    for feature_type in ['histogram', 'cnn', 'optical_flow']:
        print(f"\n📈 {feature_type.upper()} baseline...")
        
        # Load training features to compute background model
        train_features = []
        for seq_info in sequences_data['train_sequences']:
            seq_name = seq_info['name']
            if feature_type == 'histogram':
                feature_file = features_dir / f"train_{seq_name}_histograms.npy"
            else:
                feature_file = features_dir / f"train_{seq_name}_{feature_type}.npy"
            
            if feature_file.exists():
                features = np.load(feature_file)
                train_features.append(features)
        
        if not train_features:
            print(f"  ⚠️  No training features found for {feature_type}")
            continue
            
        # Compute background model (mean of training features)
        all_train = np.vstack(train_features)
        background_mean = np.mean(all_train, axis=0)
        background_std = np.std(all_train, axis=0) + 1e-8  # Add small epsilon
        
        print(f"  ✓ Background model: {background_mean.shape}, mean range [{background_mean.min():.4f}, {background_mean.max():.4f}]")
        
        # Load test features and compute anomaly scores
        test_features = []
        for seq_info in sequences_data['test_sequences']:
            seq_name = seq_info['name']
            if feature_type == 'histogram':
                feature_file = features_dir / f"test_{seq_name}_histograms.npy"
            else:
                feature_file = features_dir / f"test_{seq_name}_{feature_type}.npy"
            
            if feature_file.exists():
                features = np.load(feature_file)
                test_features.append(features)
        
        if not test_features:
            print(f"  ⚠️  No test features found for {feature_type}")
            continue
            
        all_test = np.vstack(test_features)
        
        # Generate different baseline scores
        baselines = {
            'l2_distance': np.linalg.norm(all_test - background_mean, axis=1),
            'normalized_l2': np.linalg.norm((all_test - background_mean) / background_std, axis=1),
            'cosine_distance': 1 - np.sum(all_test * background_mean, axis=1) / (
                np.linalg.norm(all_test, axis=1) * np.linalg.norm(background_mean) + 1e-8),
            'mahalanobis_simple': np.sum(((all_test - background_mean) / background_std) ** 2, axis=1)
        }
        
        # Save baseline scores
        for baseline_name, scores in baselines.items():
            score_file = baseline_dir / f"avenue_{feature_type}_{baseline_name}_scores.npy"
            np.save(score_file, scores)
            print(f"    💾 {baseline_name}: {len(scores)} scores, range [{scores.min():.4f}, {scores.max():.4f}]")
    
    print(f"\n✅ Avenue baselines generated!")
    print(f"📁 Saved to: {baseline_dir}")

if __name__ == "__main__":
    generate_avenue_baselines()