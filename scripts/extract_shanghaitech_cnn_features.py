#!/usr/bin/env python3
"""
CNN Feature Extraction for ShanghaiTech Dataset
Purpose: Extract ResNet50 features for GNN-based anomaly detection

Feature Extraction:
1. Load pretrained ResNet50 (ImageNet weights)
2. Extract 2048-dim features from final pooling layer
3. Process all test sequences
4. Save features for temporal graph construction

Usage: python scripts/extract_shanghaitech_cnn_features.py
"""

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import json
import time
from typing import Dict

class ShanghaiTechCNNExtractor:
    """Extract ResNet50 features from ShanghaiTech frames"""
    
    def __init__(self):
        self.dataset_root = Path('data/raw/ShanghaiTech/shanghaitech/testing/frames')
        self.output_dir = Path('data/processed/shanghaitech/cnn_features')
        self.splits_file = Path('data/splits/shanghaitech_splits.json')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_resnet50(self):
        """Load pretrained ResNet50 model"""
        print(f"📦 Loading ResNet50 model...")
        print(f"   Device: {self.device}")
        
        # Load pretrained ResNet50
        model = models.resnet50(pretrained=True)
        
        # Remove final classification layer to get 2048-dim features
        model = torch.nn.Sequential(*list(model.children())[:-1])
        
        model = model.to(self.device)
        model.eval()
        
        print(f"   ✓ Model loaded successfully")
        return model
    
    def extract_frame_features(self, model, frame_path: Path) -> np.ndarray:
        """
        Extract CNN features from a single frame
        
        Args:
            model: ResNet50 feature extractor
            frame_path: Path to image frame
        
        Returns:
            2048-dim feature vector
        """
        try:
            # Load and preprocess image
            img = Image.open(frame_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = model(img_tensor)
            
            # Flatten to 2048-dim vector
            features = features.squeeze().cpu().numpy()
            
            return features
            
        except Exception as e:
            print(f"   ⚠️ Error processing {frame_path.name}: {e}")
            return np.zeros(2048)
    
    def extract_sequence_features(self, model, seq_name: str, frame_paths: list) -> np.ndarray:
        """
        Extract CNN features for all frames in a sequence
        
        Args:
            model: ResNet50 feature extractor
            seq_name: Sequence name
            frame_paths: List of frame file paths
        
        Returns:
            Array of shape (num_frames, 2048)
        """
        features_list = []
        
        for frame_path in frame_paths:
            features = self.extract_frame_features(model, frame_path)
            features_list.append(features)
        
        return np.array(features_list)
    
    def get_frame_paths(self, seq_name: str) -> list:
        """Get sorted list of frame paths for a sequence"""
        seq_dir = self.dataset_root / seq_name
        
        if not seq_dir.exists():
            return []
        
        # Get all image files
        frame_files = sorted([
            f for f in seq_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.png', '.bmp', '.tif']
        ])
        
        return frame_files
    
    def process_all_sequences(self, model):
        """Process all test sequences"""
        # Load splits to get sequence list
        with open(self.splits_file, 'r') as f:
            splits_data = json.load(f)
        
        sequences = sorted(splits_data.keys())
        
        print(f"\n🎬 Processing {len(sequences)} sequences...")
        print(f"   Output: {self.output_dir}")
        
        start_time = time.time()
        total_frames = 0
        processed_seqs = 0
        skipped_seqs = 0
        
        for idx, seq_name in enumerate(sequences, 1):
            # Check if already processed
            output_file = self.output_dir / f'{seq_name}_cnn_features.npy'
            if output_file.exists():
                skipped_seqs += 1
                continue
            
            # Get frame paths
            frame_paths = self.get_frame_paths(seq_name)
            
            if not frame_paths:
                print(f"   ⚠️ No frames found for {seq_name}")
                continue
            
            # Extract features
            features = self.extract_sequence_features(model, seq_name, frame_paths)
            
            # Save features
            np.save(output_file, features)
            
            total_frames += len(features)
            processed_seqs += 1
            
            # Progress update
            if idx % 10 == 0 or idx == len(sequences):
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (len(sequences) - idx) / rate if rate > 0 else 0
                print(f"   Progress: {idx}/{len(sequences)} "
                      f"({idx/len(sequences)*100:.1f}%) - "
                      f"Frames: {total_frames:,} - "
                      f"ETA: {remaining:.1f}s")
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Feature extraction complete!")
        print(f"   Processed: {processed_seqs} sequences")
        print(f"   Skipped: {skipped_seqs} sequences (already exist)")
        print(f"   Total frames: {total_frames:,}")
        print(f"   Time: {total_time:.2f}s ({total_frames/total_time:.1f} frames/s)")
    
    def validate_features(self):
        """Validate extracted features"""
        print(f"\n🔍 Validating CNN features...")
        
        feature_files = list(self.output_dir.glob('*_cnn_features.npy'))
        
        if not feature_files:
            print(f"   ⚠️ No feature files found")
            return False
        
        print(f"   Found {len(feature_files)} feature files")
        
        # Check dimensions and statistics
        all_features = []
        for feature_file in feature_files[:5]:  # Sample first 5
            features = np.load(feature_file)
            all_features.append(features)
            
            if features.shape[1] != 2048:
                print(f"   ❌ Invalid dimensions in {feature_file.name}: {features.shape}")
                return False
        
        # Overall statistics
        all_features = np.vstack(all_features)
        print(f"   ✓ All features have correct dimensions (2048)")
        print(f"   Sample statistics (n={len(all_features):,}):")
        print(f"     Mean: {np.mean(all_features):.4f}")
        print(f"     Std: {np.std(all_features):.4f}")
        print(f"     Min: {np.min(all_features):.4f}")
        print(f"     Max: {np.max(all_features):.4f}")
        
        return True

def main():
    """Main execution function"""
    print("="*60)
    print("CNN Feature Extraction - ShanghaiTech Dataset")
    print("="*60)
    print("\nModel: ResNet50 (ImageNet pretrained)")
    print("Output: 2048-dimensional feature vectors")
    
    extractor = ShanghaiTechCNNExtractor()
    
    # Load model
    model = extractor.load_resnet50()
    
    # Extract features for all sequences
    extractor.process_all_sequences(model)
    
    # Validate results
    extractor.validate_features()
    
    print("\n" + "="*60)
    print("✅ ShanghaiTech CNN feature extraction complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Build temporal graphs: python scripts/build_shanghaitech_temporal_graphs.py --features cnn")
    print("  2. Score with GNN: python scripts/score_shanghaitech_gnn.py --features cnn")
    print("  3. Evaluate: python scripts/evaluate_shanghaitech_scores.py")

if __name__ == "__main__":
    main()
