#!/usr/bin/env python3
"""
test_wandb_integration.py - Test W&B Integration
Quick test to verify W&B is working and push sample data to dashboard.
"""

import wandb
import numpy as np
from pathlib import Path
import json

def test_wandb_connection():
    """Test basic W&B connection and logging"""
    print("🔗 Testing W&B integration...")
    
    try:
        # Initialize W&B
        run = wandb.init(
            project="temporalgraph-vad-test",
            name="integration-test",
            tags=["test", "ucsd-ped2"],
            config={
                "dataset": "UCSD Ped2", 
                "method": "histogram-l2",
                "test_purpose": "verify_wandb_connection"
            }
        )
        
        print("✅ W&B initialized successfully!")
        
        # Log some sample metrics
        wandb.log({
            "test_metric": 0.85,
            "epoch": 1,
            "accuracy": 0.92
        })
        
        # Log our actual evaluation results if available
        eval_file = Path("data/processed/evaluation_results/evaluation_results.json")
        if eval_file.exists():
            with open(eval_file) as f:
                results = json.load(f)
            
            stats = results['aggregate_stats']
            wandb.log({
                "mean_auc": stats['mean_auc'],
                "std_auc": stats['std_auc'],
                "num_sequences": stats['num_sequences_evaluated'],
                "total_frames": stats['total_frames'],
                "total_anomalies": stats['total_anomalies']
            })
            
            print(f"✅ Logged evaluation results to W&B!")
            print(f"   Mean AUC: {stats['mean_auc']:.4f}")
        
        # Create a sample plot
        import matplotlib.pyplot as plt
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, 'b-', alpha=0.7, label='Sample Data')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_title('Sample Plot for W&B Test')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Log the plot
        wandb.log({"sample_plot": wandb.Image(fig)})  # type: ignore
        plt.close(fig)
        
        print("✅ Logged sample plot to W&B!")
        
        # Finish the run
        wandb.finish()  # type: ignore
        
        print("\n🎉 W&B integration test completed successfully!")
        print(f"🌐 View your experiment at: https://wandb.ai/{wandb.api.default_entity or 'your-username'}/temporalgraph-vad-test")
        return True
        
    except Exception as e:
        print(f"❌ W&B test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_wandb_connection()
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("1. Check your API key: wandb login --relogin")
        print("2. Check internet connection")
        print("3. Verify W&B account is active")