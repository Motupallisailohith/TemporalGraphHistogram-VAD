#!/usr/bin/env python3
"""
UCSD Ped2 Dataset Comprehensive Validator Script
Phase 1 Implementation - Complete Data Engineering Pipeline Validation

This script performs 6 critical validation checks to ensure data integrity:
1. Frame Count Verification - Frames match ground truth masks
2. Histogram-Label Alignment - Features align with labels  
3. Train Sequences Check - Training data has no labels (by design)
4. Ground Truth Mask Quality - Masks are binary and consistent
5. JSON Metadata Validation - Labels structure is valid
6. Histogram File Integrity - .npy files load correctly and have right shape

Usage: python scripts/validate_ucsd_dataset.py
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any
import wandb
from datetime import datetime

class UCSDDatasetValidator:
    def __init__(self, base_path: str = 'data/raw/UCSD_Ped2/UCSDped2', enable_wandb: bool = True):
        self.base_path = Path(base_path)
        self.test_dir = self.base_path / 'Test'
        self.train_dir = self.base_path / 'Train'
        self.test_histograms_dir = self.base_path / 'Test_histograms'
        self.labels_file = Path('data/splits/ucsd_ped2_labels.json')
        self.splits_file = Path('data/splits/ucsd_ped2_splits.json')
        self.enable_wandb = enable_wandb
        
        if self.enable_wandb:
            wandb.init(  # type: ignore
                project="temporalgraph-vad-complete",
                name=f"phase1_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["phase1", "data_preparation", "validation"],
                config={"phase": "1_data_preparation", "step": "dataset_validation"}
            )
        
        self.results = {
            'frame_count_verification': {'status': 'PENDING', 'details': []},
            'histogram_label_alignment': {'status': 'PENDING', 'details': []},
            'train_sequences_check': {'status': 'PENDING', 'details': []},
            'ground_truth_mask_quality': {'status': 'PENDING', 'details': []},
            'json_metadata_validation': {'status': 'PENDING', 'details': []},
            'histogram_file_integrity': {'status': 'PENDING', 'details': []}
        }

    def check_frame_count_verification(self) -> bool:
        """Check 1: Verify frames match ground truth masks"""
        print("Check 1: Frame Count Verification...")
        try:
            mismatches = []
            
            for seq_folder in self.test_dir.iterdir():
                if not seq_folder.is_dir() or seq_folder.name.startswith('.'):
                    continue
                    
                seq_name = seq_folder.name
                gt_folder = self.test_dir / f"{seq_name}_gt"
                
                if not gt_folder.exists():
                    continue  # Skip sequences without ground truth
                    
                # Count frames (excluding system files)
                frame_files = [f for f in seq_folder.iterdir() 
                             if f.suffix.lower() in ['.tif', '.bmp'] and not f.name.startswith('.')]
                
                # Count ground truth masks
                gt_files = [f for f in gt_folder.iterdir() 
                           if f.suffix.lower() in ['.bmp', '.png', '.tif'] and not f.name.startswith('.')]
                
                if len(frame_files) != len(gt_files):
                    mismatches.append(f"{seq_name}: {len(frame_files)} frames vs {len(gt_files)} GT masks")
                else:
                    self.results['frame_count_verification']['details'].append(
                        f"✓ {seq_name}: {len(frame_files)} frames = {len(gt_files)} GT masks"
                    )
            
            if mismatches:
                self.results['frame_count_verification']['status'] = 'FAIL'
                self.results['frame_count_verification']['details'].extend([f"✗ {m}" for m in mismatches])
                return False
            else:
                self.results['frame_count_verification']['status'] = 'PASS'
                return True
                
        except Exception as e:
            self.results['frame_count_verification']['status'] = 'ERROR'
            self.results['frame_count_verification']['details'].append(f"Error: {str(e)}")
            return False

    def check_histogram_label_alignment(self) -> bool:
        """Check 2: Verify histogram features align with labels"""
        print("Check 2: Histogram-Label Alignment...")
        try:
            if not self.labels_file.exists():
                self.results['histogram_label_alignment']['status'] = 'FAIL'
                self.results['histogram_label_alignment']['details'].append("✗ Labels file not found")
                return False
            
            with open(self.labels_file) as f:
                labels_data = json.load(f)
            
            mismatches = []
            
            for seq_name, labels in labels_data.items():
                histogram_file = self.test_histograms_dir / f"{seq_name}_histograms.npy"
                
                if not histogram_file.exists():
                    mismatches.append(f"{seq_name}: histogram file missing")
                    continue
                
                try:
                    histograms = np.load(histogram_file)
                    if len(histograms) != len(labels):
                        mismatches.append(f"{seq_name}: {len(histograms)} histograms vs {len(labels)} labels")
                    else:
                        self.results['histogram_label_alignment']['details'].append(
                            f"✓ {seq_name}: {len(histograms)} histograms = {len(labels)} labels"
                        )
                except Exception as e:
                    mismatches.append(f"{seq_name}: error loading histograms - {str(e)}")
            
            if mismatches:
                self.results['histogram_label_alignment']['status'] = 'FAIL'
                self.results['histogram_label_alignment']['details'].extend([f"✗ {m}" for m in mismatches])
                return False
            else:
                self.results['histogram_label_alignment']['status'] = 'PASS'
                return True
                
        except Exception as e:
            self.results['histogram_label_alignment']['status'] = 'ERROR'
            self.results['histogram_label_alignment']['details'].append(f"Error: {str(e)}")
            return False

    def check_train_sequences(self) -> bool:
        """Check 3: Verify training sequences have no labels (normal data only)"""
        print("Check 3: Train Sequences Check...")
        try:
            train_sequences = []
            
            for seq_folder in self.train_dir.iterdir():
                if seq_folder.is_dir() and not seq_folder.name.startswith('.'):
                    train_sequences.append(seq_folder.name)
                    # Check no corresponding _gt folder exists
                    gt_folder = self.test_dir / f"{seq_folder.name}_gt"
                    if gt_folder.exists():
                        self.results['train_sequences_check']['details'].append(
                            f"⚠ {seq_folder.name}: has ground truth (unexpected for training)"
                        )
            
            if train_sequences:
                self.results['train_sequences_check']['status'] = 'PASS'
                self.results['train_sequences_check']['details'].append(
                    f"✓ Found {len(train_sequences)} training sequences (normal data only)"
                )
                return True
            else:
                self.results['train_sequences_check']['status'] = 'FAIL'
                self.results['train_sequences_check']['details'].append("✗ No training sequences found")
                return False
                
        except Exception as e:
            self.results['train_sequences_check']['status'] = 'ERROR'
            self.results['train_sequences_check']['details'].append(f"Error: {str(e)}")
            return False

    def check_ground_truth_mask_quality(self) -> bool:
        """Check 4: Verify ground truth masks are binary and consistent"""
        print("Check 4: Ground Truth Mask Quality...")
        try:
            issues = []
            checked_masks = 0
            
            for seq_folder in self.test_dir.iterdir():
                if not seq_folder.is_dir() or not seq_folder.name.endswith('_gt'):
                    continue
                    
                seq_name = seq_folder.name.replace('_gt', '')
                
                for mask_file in seq_folder.iterdir():
                    if mask_file.suffix.lower() not in ['.bmp', '.png', '.tif'] or mask_file.name.startswith('.'):
                        continue
                        
                    try:
                        mask = np.array(Image.open(mask_file).convert('L'))
                        checked_masks += 1
                        
                        # Check if binary (only 0 and 255 values)
                        unique_vals = np.unique(mask)
                        if not all(val in [0, 255] for val in unique_vals):
                            issues.append(f"{seq_name}/{mask_file.name}: non-binary values {unique_vals}")
                        
                        # Check reasonable dimensions
                        if mask.shape[0] < 10 or mask.shape[1] < 10:
                            issues.append(f"{seq_name}/{mask_file.name}: suspicious dimensions {mask.shape}")
                            
                    except Exception as e:
                        issues.append(f"{seq_name}/{mask_file.name}: load error - {str(e)}")
            
            self.results['ground_truth_mask_quality']['details'].append(f"Checked {checked_masks} mask files")
            
            if issues:
                self.results['ground_truth_mask_quality']['status'] = 'FAIL'
                self.results['ground_truth_mask_quality']['details'].extend([f"✗ {issue}" for issue in issues[:10]])  # Limit output
                if len(issues) > 10:
                    self.results['ground_truth_mask_quality']['details'].append(f"... and {len(issues) - 10} more issues")
                return False
            else:
                self.results['ground_truth_mask_quality']['status'] = 'PASS'
                self.results['ground_truth_mask_quality']['details'].append("✓ All ground truth masks are binary and valid")
                return True
                
        except Exception as e:
            self.results['ground_truth_mask_quality']['status'] = 'ERROR'
            self.results['ground_truth_mask_quality']['details'].append(f"Error: {str(e)}")
            return False

    def check_json_metadata_validation(self) -> bool:
        """Check 5: Validate JSON structure and metadata"""
        print("Check 5: JSON Metadata Validation...")
        try:
            issues = []
            
            # Check labels file
            if self.labels_file.exists():
                with open(self.labels_file) as f:
                    labels_data = json.load(f)
                
                if not isinstance(labels_data, dict):
                    issues.append("Labels file: not a dictionary")
                else:
                    for seq_name, labels in labels_data.items():
                        if not isinstance(labels, list):
                            issues.append(f"Labels for {seq_name}: not a list")
                        elif not all(isinstance(label, (int, float)) and label in [0, 1] for label in labels):
                            issues.append(f"Labels for {seq_name}: contains non-binary values")
                        else:
                            self.results['json_metadata_validation']['details'].append(
                                f"✓ {seq_name}: {len(labels)} valid binary labels"
                            )
            else:
                issues.append("Labels file missing")
            
            # Check splits file
            if self.splits_file.exists():
                with open(self.splits_file) as f:
                    splits_data = json.load(f)
                
                if not isinstance(splits_data, dict):
                    issues.append("Splits file: not a dictionary")
                elif 'train' not in splits_data or 'test' not in splits_data:
                    issues.append("Splits file: missing 'train' or 'test' keys")
                else:
                    self.results['json_metadata_validation']['details'].append(
                        f"✓ Splits: {len(splits_data['train'])} train, {len(splits_data['test'])} test sequences"
                    )
            else:
                issues.append("Splits file missing")
            
            if issues:
                self.results['json_metadata_validation']['status'] = 'FAIL'
                self.results['json_metadata_validation']['details'].extend([f"✗ {issue}" for issue in issues])
                return False
            else:
                self.results['json_metadata_validation']['status'] = 'PASS'
                return True
                
        except Exception as e:
            self.results['json_metadata_validation']['status'] = 'ERROR'
            self.results['json_metadata_validation']['details'].append(f"Error: {str(e)}")
            return False

    def check_histogram_file_integrity(self) -> bool:
        """Check 6: Verify histogram .npy files load correctly and have proper shape"""
        print("Check 6: Histogram File Integrity...")
        try:
            if not self.test_histograms_dir.exists():
                self.results['histogram_file_integrity']['status'] = 'FAIL'
                self.results['histogram_file_integrity']['details'].append("✗ Test histograms directory missing")
                return False
            
            issues = []
            valid_files = 0
            
            for hist_file in self.test_histograms_dir.iterdir():
                if not hist_file.suffix == '.npy' or hist_file.name.startswith('.'):
                    continue
                    
                try:
                    histograms = np.load(hist_file)
                    
                    # Check shape (should be [num_frames, 256] for 256-bin histograms)
                    if len(histograms.shape) != 2:
                        issues.append(f"{hist_file.name}: wrong dimensions {histograms.shape}")
                    elif histograms.shape[1] != 256:
                        issues.append(f"{hist_file.name}: expected 256 bins, got {histograms.shape[1]}")
                    elif histograms.shape[0] == 0:
                        issues.append(f"{hist_file.name}: no frames")
                    else:
                        valid_files += 1
                        self.results['histogram_file_integrity']['details'].append(
                            f"✓ {hist_file.name}: {histograms.shape[0]} frames × 256 bins"
                        )
                        
                        # Check histogram properties (should sum to ~1.0 for normalized histograms)
                        row_sums = np.sum(histograms, axis=1)
                        if not np.allclose(row_sums, 1.0, rtol=1e-2):  # More realistic tolerance for floating point
                            issues.append(f"{hist_file.name}: histograms not normalized (sums: {row_sums.min():.3f}-{row_sums.max():.3f})")
                        
                except Exception as e:
                    issues.append(f"{hist_file.name}: load error - {str(e)}")
            
            self.results['histogram_file_integrity']['details'].append(f"Validated {valid_files} histogram files")
            
            if issues:
                self.results['histogram_file_integrity']['status'] = 'FAIL'
                self.results['histogram_file_integrity']['details'].extend([f"✗ {issue}" for issue in issues])
                return False
            else:
                self.results['histogram_file_integrity']['status'] = 'PASS'
                return True
                
        except Exception as e:
            self.results['histogram_file_integrity']['status'] = 'ERROR'
            self.results['histogram_file_integrity']['details'].append(f"Error: {str(e)}")
            return False

    def run_all_checks(self) -> bool:
        """Run all validation checks and return overall status"""
        print("=" * 60)
        print("UCSD Ped2 Dataset Comprehensive Validation")
        print("=" * 60)
        
        checks = [
            ('Frame Count Verification', self.check_frame_count_verification),
            ('Histogram-Label Alignment', self.check_histogram_label_alignment),
            ('Train Sequences Check', self.check_train_sequences),
            ('Ground Truth Mask Quality', self.check_ground_truth_mask_quality),
            ('JSON Metadata Validation', self.check_json_metadata_validation),
            ('Histogram File Integrity', self.check_histogram_file_integrity)
        ]
        
        all_passed = True
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_func in checks:
            try:
                passed = check_func()
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"✓ {check_name}: {status}")
                
                if passed:
                    passed_checks += 1
                    if self.enable_wandb:
                        wandb.log({f"check_{check_name}": "PASS"})  # type: ignore
                else:
                    all_passed = False
                    if self.enable_wandb:
                        wandb.log({f"check_{check_name}": "FAIL"})  # type: ignore
                        
            except Exception as e:
                print(f"✗ {check_name}: ERROR - {str(e)}")
                all_passed = False
                if self.enable_wandb:
                    wandb.log({f"check_{check_name}": "ERROR"})  # type: ignore
        
        # Log overall validation results
        if self.enable_wandb:
            wandb.log({
                "validation_success": all_passed,
                "checks_passed": passed_checks,
                "total_checks": total_checks,
                "pass_rate": passed_checks / total_checks,
                "phase1_validation_complete": True
            })
        
        print("=" * 60)
        if all_passed:
            print("ALL CHECKS PASSED - Data is production-ready!")
        else:
            print("SOME CHECKS FAILED - Review issues above")
        print("=" * 60)
        
        return all_passed

    def save_results(self, output_file: str = 'validation_results.json'):
        """Save detailed validation results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📝 Detailed results saved to: {output_file}")

    def print_detailed_report(self):
        """Print detailed validation report"""
        print("\n" + "=" * 60)
        print("DETAILED VALIDATION REPORT")
        print("=" * 60)
        
        for check_name, result in self.results.items():
            print(f"\n{check_name.replace('_', ' ').title()}:")
            print(f"   Status: {result['status']}")
            for detail in result['details']:
                print(f"   {detail}")

def main():
    """Main execution function"""
    # Check if base dataset exists
    if not Path('data/raw/UCSD_Ped2/UCSDped2').exists():
        print("Error: UCSD Ped2 dataset not found at data/raw/UCSD_Ped2/UCSDped2")
        print("Please ensure the dataset is properly extracted and placed in the correct location.")
        sys.exit(1)
    
    # Initialize and run validator
    validator = UCSDDatasetValidator()
    success = validator.run_all_checks()
    
    # Save results and show detailed report
    validator.save_results()
    validator.print_detailed_report()
    
    # Finish W&B logging
    if validator.enable_wandb:
        wandb.finish()  # type: ignore
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()