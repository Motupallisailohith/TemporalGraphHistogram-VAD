#!/usr/bin/env python3
"""
ShanghaiTech Histogram Feature Extractor
Purpose: Extract 256-bin grayscale histograms from ShanghaiTech video frames
Output: NPY files containing histogram features for each sequence

ShanghaiTech Frame Format:
- JPG images in sequence folders
- Resolution: 856×480 pixels (varies by scene)
- Processing: Convert to grayscale → compute 256-bin histogram

Usage: python scripts/extract_shanghaitech_histograms.py
"""

import numpy as np
import json
from PIL import Image
from pathlib import Path
from typing import Dict, List
import sys
import time

class ShanghaiTechHistogramExtractor:
    """Extract histogram features from ShanghaiTech sequences"""
    
    def __init__(self, base_path: str = 'data/raw/ShanghaiTech/shanghaitech'):
        self.base_path = Path(base_path)
        self.training_frames_path = self.base_path / 'training' / 'frames'
        self.testing_frames_path = self.base_path / 'testing' / 'frames'
        self.splits_file = Path('data/splits/shanghaitech_splits.json')
        
        # Output directories
        self.train_hist_output = Path('data/processed/shanghaitech/train_histograms')
        self.test_hist_output = Path('data/processed/shanghaitech/test_histograms')
        
        # Create output directories
        self.train_hist_output.mkdir(parents=True, exist_ok=True)
        self.test_hist_output.mkdir(parents=True, exist_ok=True)
    
    def load_splits(self) -> Dict:
        """Load dataset splits"""
        if not self.splits_file.exists():
            raise FileNotFoundError(
                f"Splits file not found: {self.splits_file}\n"
                "Please run make_shanghaitech_splits.py first."
            )
        
        with open(self.splits_file, 'r') as f:
            return json.load(f)
    
    def extract_histogram_from_frame(self, frame_path: Path) -> np.ndarray:
        """
        Extract 256-bin grayscale histogram from a single frame
        
        Args:
            frame_path: Path to frame image file
        
        Returns:
            256-dimensional histogram (probability distribution)
        """
        try:
            # Load image and convert to grayscale
            img = Image.open(frame_path).convert('L')
            img_array = np.array(img)
            
            # Compute normalized histogram (256 bins, density=True for probability)
            histogram, _ = np.histogram(img_array, bins=256, range=(0, 256), density=True)
            
            return histogram.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error processing {frame_path}: {e}")
            return np.zeros(256, dtype=np.float32)
    
    def extract_sequence_histograms(self, sequence_name: str, split: str) -> np.ndarray:
        """
        Extract histograms for all frames in a sequence
        
        Args:
            sequence_name: Name of sequence (e.g., '01_0014')
            split: 'train' or 'test'
        
        Returns:
            Array of shape (num_frames, 256)
        """
        # Determine frames path based on split
        if split == 'train':
            sequence_path = self.training_frames_path / sequence_name
        else:
            sequence_path = self.testing_frames_path / sequence_name
        
        if not sequence_path.exists():
            return np.array([])
        
        # Get all frame files (sorted by name)
        frame_files = sorted([f for f in sequence_path.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
        
        if not frame_files:
            return np.array([])
        
        # Pre-allocate array for efficiency
        num_frames = len(frame_files)
        histograms = np.zeros((num_frames, 256), dtype=np.float32)
        
        # Extract histograms with batch processing
        for idx, frame_file in enumerate(frame_files):
            histograms[idx] = self.extract_histogram_from_frame(frame_file)
        
        return histograms
    
    def extract_all_histograms(self):
        """Extract histograms for all sequences"""
        print("="*60)
        print("ShanghaiTech Histogram Feature Extraction")
        print("="*60)
        
        # Load splits
        print("\n📋 Loading dataset splits...")
        splits = self.load_splits()
        
        # Separate training and testing
        train_sequences = {k: v for k, v in splits.items() if v['split'] == 'train'}
        test_sequences = {k: v for k, v in splits.items() if v['split'] == 'test'}
        
        print(f"Training sequences: {len(train_sequences)}")
        print(f"Testing sequences: {len(test_sequences)}")
        
        # Extract training histograms
        if train_sequences:
            print(f"\n🔄 Extracting training histograms...")
            self._extract_split_histograms(train_sequences, 'train', self.train_hist_output)
        
        # Extract testing histograms
        if test_sequences:
            print(f"\n🔄 Extracting testing histograms...")
            self._extract_split_histograms(test_sequences, 'test', self.test_hist_output)
        
        print("\n✅ Histogram extraction complete!")
    
    def _extract_split_histograms(self, sequences: Dict, split: str, output_dir: Path):
        """Extract histograms for a split (train or test)"""
        total = len(sequences)
        processed = 0
        skipped = 0
        total_frames = 0
        start_time = time.time()
        
        print(f"   Total sequences to process: {total}")
        print(f"   {'─' * 50}")
        
        for idx, (seq_name, seq_info) in enumerate(sorted(sequences.items()), 1):
            # Check if already processed
            output_file = output_dir / f"{seq_name}_histograms.npy"
            if output_file.exists():
                skipped += 1
                if skipped <= 3:  # Show first 3 skips
                    print(f"   ✓ Skipping {seq_name} (already exists)")
                continue
            
            # Extract histograms
            try:
                histograms = self.extract_sequence_histograms(seq_name, split)
                
                if histograms.size > 0:
                    # Save to NPY file
                    np.save(output_file, histograms)
                    
                    processed += 1
                    total_frames += len(histograms)
                    
                    # Progress update every sequence
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total - idx) / rate if rate > 0 else 0
                    
                    print(f"   [{idx}/{total}] {seq_name}: {len(histograms)} frames "
                          f"| Rate: {rate:.1f} seq/s | ETA: {eta/60:.1f}m")
            
            except KeyboardInterrupt:
                print(f"\n   ⚠️ Interrupted at sequence {idx}/{total}")
                print(f"   ✓ Processed: {processed} sequences ({total_frames:,} frames)")
                print(f"   ✓ Skipped: {skipped} sequences")
                print(f"   ℹ️ Resume by running the script again")
                sys.exit(0)
            except Exception as e:
                print(f"   ❌ Error processing {seq_name}: {e}")
        
        elapsed = time.time() - start_time
        print(f"\n   {'─' * 50}")
        print(f"   ✓ Processed: {processed} new sequences ({total_frames:,} frames)")
        print(f"   ✓ Skipped: {skipped} existing sequences")
        print(f"   ✓ Total time: {elapsed/60:.1f} minutes")
        print(f"   ✓ Saved to: {output_dir}")
    
    def validate_histograms(self):
        """Validate extracted histograms"""
        print("\n🔍 Validating histogram features...")
        
        # Load splits
        splits = self.load_splits()
        
        validation_errors = []
        
        for seq_name, seq_info in splits.items():
            split = seq_info['split']
            expected_frames = seq_info['num_frames']
            
            # Determine output file
            if split == 'train':
                hist_file = self.train_hist_output / f"{seq_name}_histograms.npy"
            else:
                hist_file = self.test_hist_output / f"{seq_name}_histograms.npy"
            
            # Check if file exists
            if not hist_file.exists():
                validation_errors.append(f"Missing: {seq_name}")
                continue
            
            # Load and validate
            try:
                histograms = np.load(hist_file)
                
                # Check shape
                if histograms.shape[1] != 256:
                    validation_errors.append(f"{seq_name}: Wrong feature dimension {histograms.shape[1]}")
                
                # Check frame count
                if histograms.shape[0] != expected_frames:
                    validation_errors.append(
                        f"{seq_name}: Frame mismatch (expected {expected_frames}, got {histograms.shape[0]})"
                    )
                
                # Check normalization (should be probability distributions)
                if not np.allclose(np.sum(histograms[0]), 1.0, atol=0.01):
                    validation_errors.append(f"{seq_name}: Histogram not normalized")
                    
            except Exception as e:
                validation_errors.append(f"{seq_name}: Load error - {e}")
        
        # Report results
        if validation_errors:
            print(f"⚠️ Found {len(validation_errors)} validation errors:")
            for error in validation_errors[:10]:  # Show first 10
                print(f"   {error}")
        else:
            print("✅ All histograms validated successfully")

def main():
    """Main execution function"""
    # Check if dataset exists
    dataset_path = Path('data/raw/ShanghaiTech/shanghaitech')
    if not dataset_path.exists():
        print(f"❌ Error: ShanghaiTech dataset not found at {dataset_path}")
        return
    
    # Extract histograms
    extractor = ShanghaiTechHistogramExtractor()
    extractor.extract_all_histograms()
    
    # Validate
    extractor.validate_histograms()
    
    print("\n" + "="*60)
    print("✅ ShanghaiTech histogram extraction complete!")
    print("="*60)

if __name__ == "__main__":
    main()
