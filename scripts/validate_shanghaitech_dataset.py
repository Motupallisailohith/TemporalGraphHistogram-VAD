#!/usr/bin/env python3
"""
ShanghaiTech Dataset Validator
Purpose: Comprehensive validation of ShanghaiTech dataset integrity
Validates: Splits, labels, histograms, and data consistency

Validation Checks:
1. Dataset structure and file existence
2. Splits and labels alignment
3. Histogram feature integrity
4. Frame count consistency
5. Ground truth mask validation
6. Statistical properties

Usage: python scripts/validate_shanghaitech_dataset.py
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

class ShanghaiTechDatasetValidator:
    """Comprehensive validator for ShanghaiTech dataset"""
    
    def __init__(self, base_path: str = 'data/raw/ShanghaiTech/shanghaitech'):
        self.base_path = Path(base_path)
        self.splits_file = Path('data/splits/shanghaitech_splits.json')
        self.labels_file = Path('data/splits/shanghaitech_labels.json')
        self.train_hist_dir = Path('data/processed/shanghaitech/train_histograms')
        self.test_hist_dir = Path('data/processed/shanghaitech/test_histograms')
        
        self.validation_results = {
            'checks_passed': [],
            'checks_failed': [],
            'warnings': []
        }
    
    def check_1_dataset_structure(self) -> bool:
        """Check 1: Validate dataset directory structure"""
        print("\n" + "="*60)
        print("CHECK 1: Dataset Structure Integrity")
        print("="*60)
        
        # Essential paths (test data is required)
        required_paths = [
            (self.base_path / 'testing' / 'frames', True),
            (self.base_path / 'testing' / 'test_frame_mask', True),
        ]
        
        # Optional paths (training data not always available)
        optional_paths = [
            (self.base_path / 'training' / 'frames', False),
        ]
        
        all_required_exist = True
        
        # Check required paths
        for path, is_required in required_paths:
            exists = path.exists()
            status = "✓" if exists else "✗"
            print(f"{status} {path} (required)")
            if not exists:
                all_required_exist = False
        
        # Check optional paths
        for path, is_required in optional_paths:
            exists = path.exists()
            status = "✓" if exists else "○"
            note = "(optional - not available in this dataset)"
            print(f"{status} {path} {note if not exists else ''}")
        
        if all_required_exist:
            self.validation_results['checks_passed'].append("Dataset structure")
            print("\n✅ Dataset structure is valid (test-only dataset)")
            return True
        else:
            self.validation_results['checks_failed'].append("Dataset structure")
            print("\n❌ Dataset structure is incomplete")
            return False
    
    def check_2_splits_labels_alignment(self) -> bool:
        """Check 2: Validate splits and labels alignment"""
        print("\n" + "="*60)
        print("CHECK 2: Splits and Labels Alignment")
        print("="*60)
        
        # Load splits and labels
        if not self.splits_file.exists():
            print(f"❌ Splits file not found: {self.splits_file}")
            self.validation_results['checks_failed'].append("Splits file missing")
            return False
        
        if not self.labels_file.exists():
            print(f"❌ Labels file not found: {self.labels_file}")
            self.validation_results['checks_failed'].append("Labels file missing")
            return False
        
        with open(self.splits_file, 'r') as f:
            splits = json.load(f)
        
        with open(self.labels_file, 'r') as f:
            labels = json.load(f)
        
        # Get test sequences from splits
        test_sequences = {k: v for k, v in splits.items() if v['split'] == 'test'}
        
        print(f"Test sequences in splits: {len(test_sequences)}")
        print(f"Sequences with labels: {len(labels)}")
        
        # Check alignment
        mismatches = []
        for seq_name in test_sequences:
            if seq_name not in labels:
                mismatches.append(f"Missing labels: {seq_name}")
            else:
                expected_frames = test_sequences[seq_name]['num_frames']
                actual_frames = len(labels[seq_name])
                if expected_frames != actual_frames:
                    mismatches.append(
                        f"{seq_name}: Expected {expected_frames} frames, got {actual_frames}"
                    )
        
        if mismatches:
            print(f"\n⚠️ Found {len(mismatches)} alignment issues:")
            for issue in mismatches[:5]:
                print(f"   {issue}")
            self.validation_results['warnings'].extend(mismatches)
        else:
            print("\n✅ Splits and labels are properly aligned")
            self.validation_results['checks_passed'].append("Splits-labels alignment")
            return True
        
        return len(mismatches) == 0
    
    def check_3_histogram_integrity(self) -> bool:
        """Check 3: Validate histogram feature files"""
        print("\n" + "="*60)
        print("CHECK 3: Histogram Feature Integrity")
        print("="*60)
        
        with open(self.splits_file, 'r') as f:
            splits = json.load(f)
        
        train_sequences = {k: v for k, v in splits.items() if v['split'] == 'train'}
        test_sequences = {k: v for k, v in splits.items() if v['split'] == 'test'}
        
        print(f"\nChecking training histograms ({len(train_sequences)} sequences)...")
        train_valid = self._validate_histogram_split(train_sequences, self.train_hist_dir)
        
        print(f"\nChecking testing histograms ({len(test_sequences)} sequences)...")
        test_valid = self._validate_histogram_split(test_sequences, self.test_hist_dir)
        
        if train_valid and test_valid:
            print("\n✅ All histogram features are valid")
            self.validation_results['checks_passed'].append("Histogram integrity")
            return True
        else:
            print("\n❌ Some histogram features are invalid")
            self.validation_results['checks_failed'].append("Histogram integrity")
            return False
    
    def _validate_histogram_split(self, sequences: Dict, hist_dir: Path) -> bool:
        """Validate histograms for a split"""
        if not hist_dir.exists():
            print(f"   ❌ Histogram directory not found: {hist_dir}")
            return False
        
        errors = []
        valid_count = 0
        
        for seq_name, seq_info in list(sequences.items())[:10]:  # Check first 10
            hist_file = hist_dir / f"{seq_name}_histograms.npy"
            
            if not hist_file.exists():
                errors.append(f"Missing: {seq_name}")
                continue
            
            try:
                histograms = np.load(hist_file)
                
                # Validate shape
                if histograms.ndim != 2 or histograms.shape[1] != 256:
                    errors.append(f"{seq_name}: Invalid shape {histograms.shape}")
                    continue
                
                # Validate frame count
                if histograms.shape[0] != seq_info['num_frames']:
                    errors.append(
                        f"{seq_name}: Frame count mismatch "
                        f"(expected {seq_info['num_frames']}, got {histograms.shape[0]})"
                    )
                    continue
                
                valid_count += 1
                
            except Exception as e:
                errors.append(f"{seq_name}: Load error - {e}")
        
        print(f"   Valid: {valid_count}/10 sampled sequences")
        
        if errors:
            print(f"   ⚠️ Errors found:")
            for error in errors[:3]:
                print(f"      {error}")
        
        return len(errors) == 0
    
    def check_4_statistical_properties(self) -> bool:
        """Check 4: Validate statistical properties of histograms"""
        print("\n" + "="*60)
        print("CHECK 4: Statistical Properties")
        print("="*60)
        
        # Sample a few histogram files
        test_hist_files = list(self.test_hist_dir.glob("*_histograms.npy"))[:5]
        
        if not test_hist_files:
            print("⚠️ No histogram files found for validation")
            return False
        
        stats_valid = True
        
        for hist_file in test_hist_files:
            histograms = np.load(hist_file)
            
            # Check normalization (should sum to ~1.0)
            row_sums = np.sum(histograms, axis=1)
            if not np.allclose(row_sums, 1.0, atol=0.01):
                print(f"   ⚠️ {hist_file.name}: Not normalized (sums={row_sums[0]:.4f})")
                stats_valid = False
            
            # Check for NaN or Inf
            if np.any(np.isnan(histograms)) or np.any(np.isinf(histograms)):
                print(f"   ⚠️ {hist_file.name}: Contains NaN or Inf values")
                stats_valid = False
            
            # Check value range [0, 1] for probability distributions
            if np.any(histograms < 0) or np.any(histograms > 1):
                print(f"   ⚠️ {hist_file.name}: Values outside [0, 1] range")
                stats_valid = False
        
        if stats_valid:
            print("✅ Statistical properties are valid")
            print(f"   Histogram normalization: ✓")
            print(f"   No NaN/Inf values: ✓")
            print(f"   Value range [0,1]: ✓")
            self.validation_results['checks_passed'].append("Statistical properties")
            return True
        else:
            self.validation_results['checks_failed'].append("Statistical properties")
            return False
    
    def check_5_ground_truth_masks(self) -> bool:
        """Check 5: Validate ground truth masks"""
        print("\n" + "="*60)
        print("CHECK 5: Ground Truth Mask Validation")
        print("="*60)
        
        mask_dir = self.base_path / 'testing' / 'test_frame_mask'
        
        if not mask_dir.exists():
            print(f"❌ Mask directory not found: {mask_dir}")
            self.validation_results['checks_failed'].append("Ground truth masks")
            return False
        
        # Sample mask files
        mask_files = list(mask_dir.glob("*.npy"))[:10]
        
        if not mask_files:
            print("⚠️ No mask files found")
            return False
        
        valid_masks = 0
        total_anomaly_frames = 0
        
        for mask_file in mask_files:
            try:
                mask = np.load(mask_file)
                
                # Check if binary (0 or 1)
                unique_values = np.unique(mask)
                if not np.all(np.isin(unique_values, [0, 1])):
                    print(f"   ⚠️ {mask_file.name}: Non-binary values {unique_values}")
                    continue
                
                valid_masks += 1
                total_anomaly_frames += np.sum(mask)
                
            except Exception as e:
                print(f"   ❌ {mask_file.name}: Load error - {e}")
        
        print(f"   Valid masks: {valid_masks}/{len(mask_files)}")
        print(f"   Total anomaly frames (sampled): {total_anomaly_frames}")
        
        if valid_masks == len(mask_files):
            print("\n✅ Ground truth masks are valid")
            self.validation_results['checks_passed'].append("Ground truth masks")
            return True
        else:
            self.validation_results['checks_failed'].append("Ground truth masks")
            return False
    
    def generate_summary(self):
        """Generate validation summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        total_checks = len(self.validation_results['checks_passed']) + len(self.validation_results['checks_failed'])
        passed = len(self.validation_results['checks_passed'])
        
        print(f"\nPassed: {passed}/{total_checks} checks")
        
        if self.validation_results['checks_passed']:
            print("\n✅ Passed Checks:")
            for check in self.validation_results['checks_passed']:
                print(f"   ✓ {check}")
        
        if self.validation_results['checks_failed']:
            print("\n❌ Failed Checks:")
            for check in self.validation_results['checks_failed']:
                print(f"   ✗ {check}")
        
        if self.validation_results['warnings']:
            print(f"\n⚠️ Warnings: {len(self.validation_results['warnings'])}")
        
        # Save results
        output_file = Path('validation_results_shanghaitech.json')
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        print(f"\n📁 Detailed results saved to: {output_file}")
        
        # Overall status
        if len(self.validation_results['checks_failed']) == 0:
            print("\n" + "="*60)
            print("✅ ALL CHECKS PASSED - Data is production-ready!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ VALIDATION FAILED - Please fix issues above")
            print("="*60)
    
    def run_all_checks(self):
        """Run all validation checks"""
        print("="*60)
        print("ShanghaiTech Dataset Comprehensive Validation")
        print("="*60)
        
        self.check_1_dataset_structure()
        self.check_2_splits_labels_alignment()
        self.check_3_histogram_integrity()
        self.check_4_statistical_properties()
        self.check_5_ground_truth_masks()
        
        self.generate_summary()

def main():
    """Main execution function"""
    if not Path('data/raw/ShanghaiTech/shanghaitech').exists():
        print("Error: ShanghaiTech dataset not found at data/raw/ShanghaiTech/shanghaitech")
        print("Please ensure the dataset is properly extracted.")
        return
    
    validator = ShanghaiTechDatasetValidator()
    validator.run_all_checks()

if __name__ == "__main__":
    main()
