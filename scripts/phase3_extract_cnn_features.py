#!/usr/bin/env python3
"""
PHASE 3 - COMPONENT 2: Deep CNN Feature Extraction
Purpose: Extract deep learning features from UCSD Ped2 video frames using pre-trained ResNet50
Input: Test sequences (Test001-Test012 folders with .tif frames)
Output: CNN feature vectors (TestXXX_cnn_features.npy)

GPU Optimized: Uses CUDA acceleration with RTX 4050 GPU

Data Structure Compliance:
- Input: data/raw/UCSD_Ped2/UCSDped2/Test/TestXXX/*.tif
- Output: data/processed/cnn_features/TestXXX_cnn_features.npy
"""

import os
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torchvision.models as models  # type: ignore
import torchvision.transforms as transforms  # type: ignore
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import wandb
from datetime import datetime

class CNNFeatureExtractor:
    """
    Extract deep CNN features using pre-trained ResNet50.
    
    Architecture:
      - ResNet50 pre-trained on ImageNet
      - Remove final classification layer
      - Extract 2048-dim features from avgpool layer
      - Process frames in batches for GPU efficiency
    """
    
    def __init__(self, 
                 test_dir='data/raw/UCSD_Ped2/UCSDped2/Test',
                 output_dir='data/processed/cnn_features',
                 model_name='resnet50',
                 batch_size=32,
                 device=None,
                 enable_wandb=True):
        """
        Initialize CNN feature extractor.
        
        Args:
            test_dir (str): Path to test sequences
            output_dir (str): Where to save CNN features
            model_name (str): Pre-trained model ('resnet50', 'resnet18', 'vgg16')
            batch_size (int): Batch size for GPU processing
            device (str): Device for computation ('cuda', 'cpu', None=auto)
            enable_wandb (bool): Enable W&B tracking
        """
        self.enable_wandb = enable_wandb
        
        if self.enable_wandb:
            wandb.init(  # type: ignore
                project="temporalgraph-vad-complete",
                name=f"phase3_cnn_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["phase3", "feature_extraction", "cnn", "resnet50"],
                config={
                    "phase": "3_temporal_graphs",
                    "model_name": model_name,
                    "batch_size": batch_size,
                    "feature_dims": 2048,
                    "preprocessing": "imagenet_normalization"
                }
            )
            device (str): 'cuda' or 'cpu' (auto-detected if None)
        """
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Initializing CNN Feature Extractor")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size}")
        
        # Validate test directory
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")
        
        # Load pre-trained model
        self.model = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms (ResNet preprocessing)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def _load_model(self):
        """
        Load pre-trained CNN model and remove classification head.
        
        Returns:
            nn.Module: Feature extraction model (outputs 2048-dim for ResNet50)
        """
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # Remove final FC layer, keep avgpool output (2048-dim)
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 2048
        elif self.model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 512
        elif self.model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # VGG features before classifier
            model = model.features
            self.feature_dim = 512 * 7 * 7  # After pooling
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model
    
    def get_frame_paths(self, seq_folder):
        """
        Get sorted list of frame paths for a sequence.
        
        Args:
            seq_folder (Path): Path to sequence folder
            
        Returns:
            list: Sorted frame paths
        """
        frame_files = [f for f in seq_folder.iterdir() 
                      if f.suffix.lower() == '.tif' and not f.name.startswith('.')]
        frame_files = sorted(frame_files, key=lambda x: int(x.stem))
        return frame_files
    
    def extract_features_batch(self, frame_batch):
        """
        Extract CNN features from a batch of frames.
        
        Args:
            frame_batch (list): List of PIL Images
            
        Returns:
            np.ndarray: Features (batch_size, feature_dim)
        """
        # Transform images to tensors
        tensors = torch.stack([self.transform(img) for img in frame_batch])
        tensors = tensors.to(self.device)
        
        # Extract features (no gradients needed)
        with torch.no_grad():
            features = self.model(tensors)
            features = features.squeeze()  # Remove spatial dimensions if present
            
            # Handle single image case
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            # Flatten if VGG
            if self.model_name == 'vgg16':
                features = features.view(features.size(0), -1)
        
        return features.cpu().numpy()
    
    def extract_sequence_features(self, seq_name):
        """
        Extract CNN features for entire sequence with batch processing.
        
        Args:
            seq_name (str): Sequence name (e.g., 'Test001')
            
        Returns:
            tuple: (success: bool, num_frames: int, feature_array: np.ndarray)
        """
        seq_folder = self.test_dir / seq_name
        
        if not seq_folder.exists():
            print(f"  ✗ {seq_name}: Sequence folder not found")
            return False, 0, None
        
        # Get sorted frame paths
        frame_paths = self.get_frame_paths(seq_folder)
        if len(frame_paths) == 0:
            print(f"  ✗ {seq_name}: No frames found")
            return False, 0, None
        
        # Process frames in batches
        all_features = []
        num_batches = (len(frame_paths) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(frame_paths), self.batch_size):
            batch_paths = frame_paths[i:i + self.batch_size]
            
            try:
                # Load batch of images
                batch_images = []
                for frame_path in batch_paths:
                    img = Image.open(frame_path).convert('RGB')
                    batch_images.append(img)
                
                # Extract features for batch
                batch_features = self.extract_features_batch(batch_images)
                all_features.append(batch_features)
                
            except Exception as e:
                print(f"  ✗ {seq_name}: Error processing batch {i//self.batch_size + 1}: {e}")
                return False, len(frame_paths), None
        
        # Concatenate all batch features
        features_array = np.concatenate(all_features, axis=0).astype(np.float32)
        
        # Save to file
        output_path = self.output_dir / f'{seq_name}_cnn_features.npy'
        np.save(output_path, features_array)
        
        return True, len(frame_paths), features_array
    
    def extract_all_sequences(self):
        """
        Extract CNN features for all test sequences with GPU acceleration.
        """
        # Find all test sequences
        test_sequences = []
        for item in self.test_dir.iterdir():
            if item.is_dir() and item.name.startswith('Test') and not item.name.endswith('_gt'):
                test_sequences.append(item.name)
        
        test_sequences = sorted(test_sequences)
        
        print(f"\n🧠 PHASE 3 COMPONENT 2: Deep CNN Feature Extraction")
        print(f"   Input directory: {self.test_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Model: {self.model_name}")
        print(f"   Feature dimension: {self.feature_dim}")
        print(f"   Device: {self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print(f"   Found sequences: {len(test_sequences)}")
        print("-" * 70)
        
        summary = {}
        total_frames = 0
        total_features = 0
        
        # Process each sequence
        for seq_name in tqdm(test_sequences, desc="Extracting CNN features"):
            success, num_frames, features = self.extract_sequence_features(seq_name)
            
            if success and features is not None:
                summary[seq_name] = {
                    'num_frames': num_frames,
                    'feature_shape': list(features.shape),
                    'feature_dim': self.feature_dim,
                    'feature_range': [float(features.min()), float(features.max())],
                    'feature_mean': float(features.mean()),
                    'feature_std': float(features.std()),
                    'status': 'success'
                }
                total_frames += num_frames
                total_features += len(features)
                print(f"  ✓ {seq_name}: {num_frames} frames → {features.shape}")
            else:
                summary[seq_name] = {
                    'num_frames': num_frames,
                    'status': 'failed'
                }
                print(f"  ✗ {seq_name}: Failed to extract features")
        
        # Save summary
        summary_data = {
            'extraction_summary': summary,
            'total_sequences': len(test_sequences),
            'successful_sequences': sum(1 for v in summary.values() if v['status'] == 'success'),
            'total_frames': total_frames,
            'total_features': total_features,
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'batch_size': self.batch_size,
            'device': str(self.device),
            'feature_type': 'deep_cnn_features'
        }
        
        summary_path = self.output_dir / 'cnn_extraction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n" + "=" * 70)
        print(f"✅ CNN FEATURE EXTRACTION COMPLETE")
        print(f"   📊 Processed: {summary_data['successful_sequences']}/{len(test_sequences)} sequences")
        print(f"   🎞️ Total frames: {total_frames}")
        print(f"   🧠 Total CNN features: {total_features}")
        print(f"   📏 Feature dimension: {self.feature_dim}")
        print(f"   💾 Output files: {self.output_dir}")
        print(f"   📋 Summary: {summary_path}")
        print("=" * 70)
        
        return summary_data


def main():
    """
    Main execution function for Phase 3 Component 2.
    """
    print("🚀 Starting Phase 3 Component 2: Deep CNN Feature Extraction")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   PyTorch version: {torch.__version__}")
    else:
        print("⚠️ No GPU detected. Running on CPU (will be slower).")
    
    # Check prerequisites
    test_dir = Path('data/raw/UCSD_Ped2/UCSDped2/Test')
    if not test_dir.exists():
        print(f"❌ Error: Test directory not found: {test_dir}")
        return
    
    # Initialize extractor with GPU acceleration
    extractor = CNNFeatureExtractor(
        test_dir='data/raw/UCSD_Ped2/UCSDped2/Test',
        output_dir='data/processed/cnn_features',
        model_name='resnet50',  # 2048-dim features
        batch_size=32,  # Adjust based on GPU memory
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Extract all sequences
    summary = extractor.extract_all_sequences()
    
    # Report results
    if summary['successful_sequences'] == summary['total_sequences']:
        print(f"\n🎉 All sequences processed successfully!")
        print(f"📂 Files generated:")
        for seq_name in sorted(summary['extraction_summary'].keys()):
            if summary['extraction_summary'][seq_name]['status'] == 'success':
                shape = summary['extraction_summary'][seq_name]['feature_shape']
                print(f"   {seq_name}_cnn_features.npy: {shape[0]} × {shape[1]}")
    else:
        failed = summary['total_sequences'] - summary['successful_sequences']
        print(f"⚠️ {failed} sequences failed. Check logs above for details.")
    
    print(f"\n✅ Phase 3 Component 2 complete!")
    print(f"   Next: Combine histogram + optical flow + CNN features for advanced model")


if __name__ == "__main__":
    main()
