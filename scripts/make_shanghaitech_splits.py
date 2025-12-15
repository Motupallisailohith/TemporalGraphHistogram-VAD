#!/usr/bin/env python3
"""
ShanghaiTech Dataset Splits Generator
Purpose: Create train/test splits for ShanghaiTech video anomaly detection dataset
Output: shanghaitech_splits.json with sequence metadata

ShanghaiTech Dataset Structure:
- Training: 330 sequences (normal only)
- Testing: 107 sequences (with anomalies)
- Format: JPG frames in sequence folders

Usage: python scripts/make_shanghaitech_splits.py
"""

import os
import json
from pathlib import Path
from typing import Dict, List

class ShanghaiTechSplitsGenerator:
    """Generate train/test splits for ShanghaiTech dataset"""
    
    def __init__(self, base_path: str = 'data/raw/ShanghaiTech/shanghaitech'):
        self.base_path = Path(base_path)
        self.training_path = self.base_path / 'training' / 'frames'
        self.testing_path = self.base_path / 'testing' / 'frames'
        self.output_file = Path('data/splits/shanghaitech_splits.json')
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def count_frames_in_sequence(self, sequence_path: Path) -> int:
        """Count number of frames (.jpg files) in a sequence"""
        if not sequence_path.exists():
            return 0
        return len([f for f in sequence_path.iterdir() if f.suffix.lower() == '.jpg'])
    
    def generate_training_splits(self) -> Dict[str, Dict]:
        """Generate splits for training sequences"""
        print("\n📊 Processing training sequences...")
        training_splits = {}
        
        if not self.training_path.exists():
            print(f"⚠️ Training path not found: {self.training_path}")
            return training_splits
        
        # Get all training sequence folders
        sequence_folders = sorted([f for f in self.training_path.iterdir() if f.is_dir()])
        
        print(f"Found {len(sequence_folders)} training sequences")
        
        for seq_folder in sequence_folders:
            seq_name = seq_folder.name
            frame_count = self.count_frames_in_sequence(seq_folder)
            
            if frame_count > 0:
                training_splits[seq_name] = {
                    'split': 'train',
                    'num_frames': frame_count,
                    'has_anomaly': False,  # Training sequences are normal only
                    'scene_id': seq_name.split('_')[0],  # Extract scene ID
                    'clip_id': seq_name.split('_')[1] if '_' in seq_name else '0000'
                }
        
        print(f"✓ Processed {len(training_splits)} training sequences")
        return training_splits
    
    def generate_testing_splits(self) -> Dict[str, Dict]:
        """Generate splits for testing sequences"""
        print("\n📊 Processing testing sequences...")
        testing_splits = {}
        
        if not self.testing_path.exists():
            print(f"⚠️ Testing path not found: {self.testing_path}")
            return testing_splits
        
        # Get all testing sequence folders
        sequence_folders = sorted([f for f in self.testing_path.iterdir() if f.is_dir()])
        
        print(f"Found {len(sequence_folders)} testing sequences")
        
        for seq_folder in sequence_folders:
            seq_name = seq_folder.name
            frame_count = self.count_frames_in_sequence(seq_folder)
            
            if frame_count > 0:
                testing_splits[seq_name] = {
                    'split': 'test',
                    'num_frames': frame_count,
                    'has_anomaly': True,  # Test sequences may contain anomalies
                    'scene_id': seq_name.split('_')[0],
                    'clip_id': seq_name.split('_')[1] if '_' in seq_name else '0000'
                }
        
        print(f"✓ Processed {len(testing_splits)} testing sequences")
        return testing_splits
    
    def generate_splits(self):
        """Generate complete train/test splits"""
        print("="*60)
        print("ShanghaiTech Dataset Splits Generator")
        print("="*60)
        
        # Generate training and testing splits
        training_splits = self.generate_training_splits()
        testing_splits = self.generate_testing_splits()
        
        # Combine splits
        all_splits = {**training_splits, **testing_splits}
        
        # Statistics
        print("\n📈 Dataset Statistics:")
        print(f"   Training sequences: {len(training_splits)}")
        print(f"   Testing sequences: {len(testing_splits)}")
        print(f"   Total sequences: {len(all_splits)}")
        
        # Calculate total frames
        total_train_frames = sum(s['num_frames'] for s in training_splits.values())
        total_test_frames = sum(s['num_frames'] for s in testing_splits.values())
        
        print(f"\n   Training frames: {total_train_frames:,}")
        print(f"   Testing frames: {total_test_frames:,}")
        print(f"   Total frames: {total_train_frames + total_test_frames:,}")
        
        # Save to JSON
        print(f"\n💾 Saving splits to: {self.output_file}")
        with open(self.output_file, 'w') as f:
            json.dump(all_splits, f, indent=2)
        
        print(f"✅ Splits saved successfully!")
        print(f"\n📁 Output file: {self.output_file}")
        
        return all_splits

def main():
    """Main execution function"""
    # Check if dataset exists
    dataset_path = Path('data/raw/ShanghaiTech/shanghaitech')
    if not dataset_path.exists():
        print(f"❌ Error: ShanghaiTech dataset not found at {dataset_path}")
        print("Please ensure the dataset is extracted to the correct location.")
        return
    
    # Generate splits
    generator = ShanghaiTechSplitsGenerator()
    splits = generator.generate_splits()
    
    print("\n" + "="*60)
    print("✅ ShanghaiTech splits generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
