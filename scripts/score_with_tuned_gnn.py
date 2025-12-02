"""
Score test sequences with tuned GNN model
"""

import os
import sys
import json
import torch  # type: ignore
import numpy as np
from pathlib import Path

# Add scripts to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'scripts'))

from phase3_4b_score_gnn import GNNScorer

def main():
    print("\n" + "="*70)
    print("🎯 SCORING WITH TUNED GNN MODEL")
    print("="*70)
    
    # Load tuning results to get optimal configuration
    tuning_results_path = 'models/tuning_results/tuning_results.json'
    if os.path.exists(tuning_results_path):
        with open(tuning_results_path, 'r') as f:
            tuning_results = json.load(f)
        
        best_config = tuning_results['best_config']
        print(f"📋 Using tuned configuration:")
        print(f"   Hidden dim: {best_config['hidden_dim']}")
        print(f"   Latent dim: {best_config['latent_dim']}")
        print(f"   Best AUC: {tuning_results['best_auc']:.4f}")
        
        # Initialize scorer with tuned parameters
        scorer = GNNScorer(
            model_path='models/gnn_autoencoder_tuned_best.pth',
            graph_dir='data/processed/temporal_graphs',
            output_dir='data/processed/gnn_scores_tuned',
            input_dim=best_config['input_dim'],
            hidden_dim=best_config['hidden_dim'],
            latent_dim=best_config['latent_dim'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        print("⚠️ Tuning results not found. Using default tuned parameters.")
        # Use known good parameters from tuning results
        scorer = GNNScorer(
            model_path='models/gnn_autoencoder_tuned_best.pth',
            graph_dir='data/processed/temporal_graphs',
            output_dir='data/processed/gnn_scores_tuned',
            input_dim=2048,
            hidden_dim=1024,  # From tuning results
            latent_dim=256,   # From tuning results
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    # Score all test sequences
    print(f"\n🚀 Scoring with tuned model...")
    
    try:
        scorer.score_all_sequences()
        
        print(f"\n✅ TUNED SCORING COMPLETE!")
        print(f"📂 Scores saved to: data/processed/gnn_scores_tuned/")
        
        # Now copy to main gnn_scores directory for ensemble use
        import shutil
        
        tuned_dir = Path('data/processed/gnn_scores_tuned')
        main_dir = Path('data/processed/gnn_scores')
        
        print(f"\n📁 Copying tuned scores to main directory...")
        
        if tuned_dir.exists():
            # Copy all .npy files
            for npy_file in tuned_dir.glob('*.npy'):
                shutil.copy2(npy_file, main_dir / npy_file.name)
                print(f"   ✓ {npy_file.name}")
            
            # Copy summary file
            summary_file = tuned_dir / 'gnn_scores_summary.json'
            if summary_file.exists():
                shutil.copy2(summary_file, main_dir / 'gnn_scores_summary.json')
                print(f"   ✓ gnn_scores_summary.json")
        
        print(f"\n🎯 Ready for ensemble with tuned GNN scores!")
        
    except Exception as e:
        print(f"❌ Error during scoring: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()