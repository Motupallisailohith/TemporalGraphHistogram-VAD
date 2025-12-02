#!/usr/bin/env python3
"""
Extract CNN features for TRAINING sequences (Train001-Train016).
This prepares training data for GNN training.
"""

import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class TrainingCNNExtractor:
    """Extract CNN features specifically for TRAINING sequences."""
    
    def __init__(self, train_dir='data/raw/UCSD_Ped2/UCSDped2/Train',
                 output_dir='data/processed/cnn_features',
                 batch_size=32,
                 device=None):
        self.train_dir = Path(train_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load ResNet50
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove final FC
        self.model = self.model.to(self.device)
        self.model.eval()
        self.feature_dim = 2048
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_sequence_features(self, seq_name):
        """Extract features for one training sequence."""
        seq_path = self.train_dir / seq_name
        
        # Get all frame files
        frame_files = sorted([f for f in seq_path.iterdir() 
                            if f.suffix in ['.tif', '.jpg', '.png'] 
                            and not f.name.startswith('.')])
        
        if not frame_files:
            return None
        
        # Process in batches
        all_features = []
        for i in range(0, len(frame_files), self.batch_size):
            batch_files = frame_files[i:i+self.batch_size]
            batch_tensors = []
            
            for frame_file in batch_files:
                try:
                    img = Image.open(frame_file).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"  Error loading {frame_file.name}: {e}")
                    continue
            
            if batch_tensors:
                batch = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    features = self.model(batch)
                    features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims
                
                all_features.append(features.cpu().numpy())
        
        # Concatenate all features
        features_array = np.concatenate(all_features, axis=0).astype(np.float32)
        
        # Save
        output_path = self.output_dir / f'{seq_name}_cnn_features.npy'
        np.save(output_path, features_array)
        
        return features_array
    
    def extract_all_sequences(self):
        """Extract features for all training sequences."""
        # Find all training sequences
        train_sequences = sorted([d.name for d in self.train_dir.iterdir() 
                                if d.is_dir() and d.name.startswith('Train')])
        
        print(f"\n🧠 Extracting CNN Features for TRAINING Sequences")
        print(f"   Device: {self.device}")
        print(f"   Found sequences: {len(train_sequences)}")
        print("-" * 70)
        
        summary = {}
        total_frames = 0
        
        for seq_name in tqdm(train_sequences, desc="Processing"):
            features = self.extract_sequence_features(seq_name)
            
            if features is not None:
                summary[seq_name] = {
                    'num_frames': len(features),
                    'feature_shape': list(features.shape),
                    'status': 'success'
                }
                total_frames += len(features)
                print(f"  ✓ {seq_name}: {features.shape}")
            else:
                summary[seq_name] = {'status': 'failed'}
                print(f"  ✗ {seq_name}: Failed")
        
        # Save summary
        summary_data = {
            'training_summary': summary,
            'total_sequences': len(train_sequences),
            'successful_sequences': sum(1 for v in summary.values() if v['status'] == 'success'),
            'total_frames': total_frames,
            'feature_dim': self.feature_dim
        }
        
        summary_path = self.output_dir / 'training_cnn_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        return summary_data


def main():
    """Extract CNN features for training sequences."""
    print("\n" + "=" * 70)
    print("EXTRACTING CNN FEATURES FOR TRAINING SEQUENCES")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("\nNo GPU detected. Running on CPU (slower).")
        device = 'cpu'
    
    # Initialize extractor
    extractor = TrainingCNNExtractor(
        train_dir='data/raw/UCSD_Ped2/UCSDped2/Train',
        output_dir='data/processed/cnn_features',
        batch_size=32,
        device=device
    )
    
    # Extract features
    summary = extractor.extract_all_sequences()
    
    # Report results
    print(f"\n" + "=" * 70)
    if summary['successful_sequences'] == summary['total_sequences']:
        print(f"ALL TRAINING SEQUENCES PROCESSED!")
        print(f"   Sequences: {summary['successful_sequences']}/{summary['total_sequences']}")
        print(f"   🎞️ Total frames: {summary.get('total_frames', 0)}")
        print(f"   Output: data/processed/cnn_features/")
        print(f"\nNext step: Build training temporal graphs")
        print(f"   Command: python scripts/generate_training_graphs.py")
    else:
        failed = summary['total_sequences'] - summary['successful_sequences']
        print(f"{failed} sequences failed. Check logs above.")
    print("=" * 70)


if __name__ == '__main__':
    main()
