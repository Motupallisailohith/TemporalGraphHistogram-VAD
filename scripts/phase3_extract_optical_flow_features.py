#!/usr/bin/env python3
"""
PHASE 3 - COMPONENT 1: Optical Flow Feature Extraction
Purpose: Extract motion-based features from UCSD Ped2 video frames
Input: Test sequences (Test001-Test012 folders with .tif frames)  
Output: Optical flow histograms (TestXXX_optical_flow.npy)

Data Structure Compliance:
- Input: data/raw/UCSD_Ped2/UCSDped2/Test/TestXXX/*.tif (001.tif, 002.tif, ...)
- Output: data/processed/optical_flow/TestXXX_optical_flow.npy 
- Naming matches existing: TestXXX_histograms.npy → TestXXX_optical_flow.npy
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

class OpticalFlowExtractor:
    """
    Extract optical flow features (motion patterns) from frame sequences.
    
    Algorithm:
      - For each consecutive frame pair (t, t+1):
        1. Compute optical flow using Farneback algorithm
        2. Extract motion magnitude and direction
        3. Create histogram of motion magnitude
        4. Normalize and save
    """
    
    def __init__(self, 
                 test_dir='data/raw/UCSD_Ped2/UCSDped2/Test',
                 output_dir='data/processed/optical_flow',
                 bins=64):
        """
        Initialize optical flow extractor.
        
        Args:
            test_dir (str): Path to test sequences (follows existing structure)
            output_dir (str): Where to save optical flow features
            bins (int): Number of histogram bins for motion magnitude
        """
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.bins = bins
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate test directory exists
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")
    
    def get_frame_paths(self, seq_folder):
        """
        Get sorted list of frame paths for a sequence.
        
        Logic:
          - List all .tif files in folder (matching your nomenclature: 001.tif, 002.tif, ...)
          - Sort numerically (ensures frame order: 001, 002, ..., 180)
          - Return full paths
        
        Args:
            seq_folder (Path): Path to sequence folder
            
        Returns:
            list: Sorted frame paths
        """
        # Get all .tif files and filter out system files
        frame_files = [f for f in seq_folder.iterdir() 
                      if f.suffix.lower() == '.tif' and not f.name.startswith('.')]
        
        # Sort numerically by filename (001.tif, 002.tif, ...)
        frame_files = sorted(frame_files, key=lambda x: int(x.stem))
        
        return frame_files
    
    def compute_optical_flow_histogram(self, frame_curr, frame_next):
        """
        Compute optical flow between two frames and return histogram.
        
        Logical Steps:
          Step 1: Convert frames to grayscale (if needed)
          Step 2: Apply Farneback optical flow algorithm
          Step 3: Extract motion magnitude (strength of motion)
          Step 4: Normalize magnitude to [0, 1]
          Step 5: Create histogram of normalized magnitudes
          Step 6: Normalize histogram to sum=1 (probability distribution)
        
        Args:
            frame_curr (np.ndarray): Current frame (H × W × 3 or H × W)
            frame_next (np.ndarray): Next frame (H × W × 3 or H × W)
            
        Returns:
            np.ndarray: Histogram of optical flow magnitude (bins,)
        """
        
        # Step 1: Convert to grayscale
        if len(frame_curr.shape) == 3:
            frame_curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        else:
            frame_curr_gray = frame_curr
            
        if len(frame_next.shape) == 3:
            frame_next_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
        else:
            frame_next_gray = frame_next
        
        # Step 2: Compute optical flow using Farneback algorithm
        # Parameters tuned for surveillance video
        flow = cv2.calcOpticalFlowFarneback(
            frame_curr_gray,
            frame_next_gray,
            None,
            pyr_scale=0.5,           # Pyramid scale
            levels=3,                # Number of pyramid levels
            winsize=15,              # Averaging window
            iterations=3,            # Iterations per level
            poly_n=5,                # Polynomial neighborhood size
            poly_sigma=1.2,          # Gaussian sigma
            flags=0
        )
        # flow.shape = (height, width, 2) where flow[i,j] = [u, v] (x, y displacement)
        
        # Step 3: Extract motion magnitude from flow vectors
        # For each pixel, compute: magnitude = sqrt(u^2 + v^2)
        u = flow[..., 0]  # x-displacement
        v = flow[..., 1]  # y-displacement
        mag = np.sqrt(u**2 + v**2)  # Euclidean norm
        # mag.shape = (height, width)
        
        # Step 4: Normalize magnitude to [0, 1]
        mag_min = mag.min()
        mag_max = mag.max()
        if mag_max > mag_min:
            mag_normalized = (mag - mag_min) / (mag_max - mag_min)
        else:
            mag_normalized = np.zeros_like(mag)
        # mag_normalized ∈ [0, 1]
        
        # Step 5: Create histogram of normalized magnitudes
        # Divide [0, 1] into 'bins' equal intervals
        # Count how many pixels fall into each bin
        hist, _ = np.histogram(mag_normalized, bins=self.bins, range=(0, 1), density=True)
        
        # Step 6: Normalize histogram to probability distribution
        # Ensure it's normalized (density=True should handle this, but double-check)
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist = hist / hist.sum()
        else:
            # If no motion detected, uniform histogram
            hist = np.ones(self.bins, dtype=np.float32) / self.bins
        
        return hist
    
    def extract_sequence_features(self, seq_name):
        """
        Extract optical flow features for entire sequence.
        
        Logical Steps:
          For each test sequence:
            1. Get sorted frame paths
            2. Initialize list for features
            3. For consecutive frame pairs (i, i+1):
               - Load both frames
               - Compute optical flow histogram
               - Append to features list
            4. Stack features into 2D array (num_frames-1, bins)
            5. Save to .npy file (matching existing nomenclature)
        
        Args:
            seq_name (str): Sequence name (e.g., 'Test001')
            
        Returns:
            tuple: (success: bool, num_frames: int, feature_array: np.ndarray)
        """
        
        seq_folder = self.test_dir / seq_name
        
        if not seq_folder.exists():
            print(f"  ✗ {seq_name}: Sequence folder not found")
            return False, 0, None
        
        # Step 1: Get sorted frame paths
        frame_paths = self.get_frame_paths(seq_folder)
        if len(frame_paths) < 2:
            print(f"  ✗ {seq_name}: Not enough frames ({len(frame_paths)})")
            return False, len(frame_paths), None
        
        # Step 2: Initialize feature list
        optical_flow_features = []
        
        # Step 3: Process consecutive frame pairs
        for i in range(len(frame_paths) - 1):
            try:
                # Load current and next frames
                frame_curr = cv2.imread(str(frame_paths[i]))
                frame_next = cv2.imread(str(frame_paths[i+1]))
                
                if frame_curr is None or frame_next is None:
                    print(f"  ✗ {seq_name}: Failed to load frames {frame_paths[i].name}, {frame_paths[i+1].name}")
                    return False, len(frame_paths), None
                
                # Compute optical flow histogram
                hist = self.compute_optical_flow_histogram(frame_curr, frame_next)
                optical_flow_features.append(hist)
                
            except Exception as e:
                print(f"  ✗ {seq_name}: Error processing frames {i}, {i+1}: {e}")
                return False, len(frame_paths), None
        
        # Step 4: Stack into 2D array
        features_array = np.array(optical_flow_features, dtype=np.float32)
        # features_array.shape = (num_frames - 1, bins)
        
        # Step 5: Save to file (matching existing nomenclature TestXXX_histograms.npy)
        output_path = self.output_dir / f'{seq_name}_optical_flow.npy'
        np.save(output_path, features_array)
        
        return True, len(frame_paths), features_array
    
    def extract_all_sequences(self):
        """
        Extract optical flow features for all test sequences.
        
        Logical Steps:
          1. List all test sequence folders (Test001-Test012)
          2. For each sequence:
             - Extract features
             - Log progress
          3. Save summary statistics
          4. Return processing summary
        """
        
        # Step 1: Find all test sequences (following existing pattern)
        test_sequences = []
        for item in self.test_dir.iterdir():
            if item.is_dir() and item.name.startswith('Test') and not item.name.endswith('_gt'):
                test_sequences.append(item.name)
        
        test_sequences = sorted(test_sequences)
        
        print(f"\n🎬 PHASE 3 COMPONENT 1: Optical Flow Feature Extraction")
        print(f"   Input directory: {self.test_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Histogram bins: {self.bins}")
        print(f"   Found sequences: {len(test_sequences)}")
        print("-" * 70)
        
        summary = {}
        total_frames = 0
        total_features = 0
        
        # Step 2: Process each sequence
        for seq_name in tqdm(test_sequences, desc="Extracting optical flow"):
            success, num_frames, features = self.extract_sequence_features(seq_name)
            
            if success and features is not None:
                # Log to summary
                summary[seq_name] = {
                    'num_frames': num_frames,
                    'num_features': len(features),
                    'feature_shape': list(features.shape),
                    'feature_range': [float(features.min()), float(features.max())],
                    'status': 'success'
                }
                total_frames += num_frames
                total_features += len(features)
                print(f"  ✓ {seq_name}: {num_frames} frames → {len(features)} flow features")
            else:
                summary[seq_name] = {
                    'num_frames': num_frames,
                    'num_features': 0,
                    'status': 'failed'
                }
                print(f"  ✗ {seq_name}: Failed to extract features")
        
        # Step 3: Save summary (matching existing pattern)
        summary_data = {
            'extraction_summary': summary,
            'total_sequences': len(test_sequences),
            'successful_sequences': sum(1 for v in summary.values() if v['status'] == 'success'),
            'total_frames': total_frames,
            'total_features': total_features,
            'histogram_bins': self.bins,
            'algorithm': 'farneback_optical_flow',
            'feature_type': 'motion_magnitude_histogram'
        }
        
        summary_path = self.output_dir / 'optical_flow_extraction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n" + "=" * 70)
        print(f"✅ OPTICAL FLOW EXTRACTION COMPLETE")
        print(f"   📊 Processed: {summary_data['successful_sequences']}/{len(test_sequences)} sequences")
        print(f"   🎞️ Total frames: {total_frames}")
        print(f"   🌊 Total flow features: {total_features}")
        print(f"   📁 Output files: {self.output_dir}")
        print(f"   📋 Summary: {summary_path}")
        print("=" * 70)
        
        return summary_data


def main():
    """
    Main execution function for Phase 3 Component 1.
    """
    
    print("🚀 Starting Phase 3 Component 1: Optical Flow Feature Extraction")
    
    # Check prerequisites
    test_dir = Path('data/raw/UCSD_Ped2/UCSDped2/Test')
    if not test_dir.exists():
        print(f"❌ Error: Test directory not found: {test_dir}")
        print("   Please ensure UCSD Ped2 dataset is properly extracted.")
        return
    
    # Initialize extractor with paths matching your data structure
    extractor = OpticalFlowExtractor(
        test_dir='data/raw/UCSD_Ped2/UCSDped2/Test',
        output_dir='data/processed/optical_flow',
        bins=64
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
                print(f"   {seq_name}_optical_flow.npy: {shape[0]} × {shape[1]}")
    else:
        failed = summary['total_sequences'] - summary['successful_sequences']
        print(f"⚠️ {failed} sequences failed. Check logs above for details.")
    
    print(f"\n✅ Phase 3 Component 1 complete!")
    print(f"   Next: Component 2 (Deep CNN Feature Extraction)")


if __name__ == "__main__":
    main()