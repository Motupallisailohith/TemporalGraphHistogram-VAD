#!/usr/bin/env python3
"""
ShanghaiTech Label Masks Generator
Purpose: Extract frame-level binary anomaly labels from ShanghaiTech ground truth masks
Output: shanghaitech_labels.json with binary anomaly annotations

ShanghaiTech Ground Truth Format:
- Frame-level masks: test_frame_mask/*.npy (binary 0/1 per frame)
- Pixel-level masks: test_pixel_mask/*.npy (spatial anomaly maps)

Usage: python scripts/make_shanghaitech_label_masks.py
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List

class ShanghaiTechLabelGenerator:
    """Generate frame-level anomaly labels from ShanghaiTech ground truth"""
    
    def __init__(self, base_path: str = 'data/raw/ShanghaiTech/shanghaitech'):
        self.base_path = Path(base_path)
        self.test_frame_mask_path = self.base_path / 'testing' / 'test_frame_mask'
        self.splits_file = Path('data/splits/shanghaitech_splits.json')
        self.output_file = Path('data/splits/shanghaitech_labels.json')
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def load_splits(self) -> Dict:
        """Load previously generated splits"""
        if not self.splits_file.exists():
            raise FileNotFoundError(
                f"Splits file not found: {self.splits_file}\n"
                "Please run make_shanghaitech_splits.py first."
            )
        
        with open(self.splits_file, 'r') as f:
            return json.load(f)
    
    def extract_frame_labels(self, sequence_name: str) -> List[int]:
        """
        Extract binary frame-level labels from ground truth mask
        
        Args:
            sequence_name: Name of test sequence (e.g., '01_0014')
        
        Returns:
            List of binary labels (0=normal, 1=anomaly)
        """
        # Ground truth file path
        gt_file = self.test_frame_mask_path / f"{sequence_name}.npy"
        
        if not gt_file.exists():
            print(f"⚠️ Ground truth not found for {sequence_name}, assuming all normal")
            return []
        
        try:
            # Load frame-level mask (binary array)
            frame_mask = np.load(gt_file)
            
            # Convert to binary labels (0 or 1)
            binary_labels = [int(label) for label in frame_mask]
            
            return binary_labels
            
        except Exception as e:
            print(f"❌ Error processing {sequence_name}: {e}")
            return []
    
    def generate_labels(self):
        """Generate frame-level labels for all test sequences"""
        print("="*60)
        print("ShanghaiTech Frame-Level Label Generator")
        print("="*60)
        
        # Load splits
        print("\n📋 Loading dataset splits...")
        splits = self.load_splits()
        
        # Filter test sequences only
        test_sequences = {k: v for k, v in splits.items() if v['split'] == 'test'}
        print(f"Found {len(test_sequences)} test sequences")
        
        # Generate labels
        labels = {}
        total_frames = 0
        total_anomalous_frames = 0
        
        print("\n🔍 Extracting frame-level labels...")
        
        for seq_name, seq_info in sorted(test_sequences.items()):
            frame_labels = self.extract_frame_labels(seq_name)
            
            if frame_labels:
                labels[seq_name] = frame_labels
                anomaly_count = sum(frame_labels)
                total_frames += len(frame_labels)
                total_anomalous_frames += anomaly_count
                
                # Progress indicator
                if len(labels) % 20 == 0:
                    print(f"   Processed {len(labels)}/{len(test_sequences)} sequences...")
        
        # Statistics
        print(f"\n📈 Label Statistics:")
        print(f"   Sequences with labels: {len(labels)}")
        print(f"   Total frames: {total_frames:,}")
        print(f"   Normal frames: {total_frames - total_anomalous_frames:,}")
        print(f"   Anomalous frames: {total_anomalous_frames:,}")
        
        if total_frames > 0:
            anomaly_ratio = (total_anomalous_frames / total_frames) * 100
            print(f"   Anomaly ratio: {anomaly_ratio:.2f}%")
        
        # Save to JSON
        print(f"\n💾 Saving labels to: {self.output_file}")
        with open(self.output_file, 'w') as f:
            json.dump(labels, f, indent=2)
        
        print(f"✅ Labels saved successfully!")
        print(f"\n📁 Output file: {self.output_file}")
        
        return labels
    
    def validate_labels(self, labels: Dict):
        """Validate generated labels against splits"""
        print("\n🔍 Validating labels...")
        
        splits = self.load_splits()
        test_sequences = {k: v for k, v in splits.items() if v['split'] == 'test'}
        
        mismatches = []
        
        for seq_name, split_info in test_sequences.items():
            if seq_name in labels:
                expected_frames = split_info['num_frames']
                actual_frames = len(labels[seq_name])
                
                if expected_frames != actual_frames:
                    mismatches.append({
                        'sequence': seq_name,
                        'expected': expected_frames,
                        'actual': actual_frames
                    })
        
        if mismatches:
            print(f"⚠️ Found {len(mismatches)} frame count mismatches:")
            for mismatch in mismatches[:5]:  # Show first 5
                print(f"   {mismatch['sequence']}: expected {mismatch['expected']}, got {mismatch['actual']}")
        else:
            print("✅ All label counts match split information")

def main():
    """Main execution function"""
    # Check if dataset exists
    dataset_path = Path('data/raw/ShanghaiTech/shanghaitech')
    if not dataset_path.exists():
        print(f"❌ Error: ShanghaiTech dataset not found at {dataset_path}")
        return
    
    # Generate labels
    generator = ShanghaiTechLabelGenerator()
    labels = generator.generate_labels()
    
    # Validate
    generator.validate_labels(labels)
    
    print("\n" + "="*60)
    print("✅ ShanghaiTech label generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
