# TemporalGraphHistogram-VAD - AI Coding Agent Instructions

## Project Overview
Temporal Graph Networks for detection-free video anomaly detection using histogram-based scene modeling. This repository implements HistoGraph, which learns normal scene dynamics through global histogram evolution analysis.

## Key Architecture & Data Flow

### Dataset Structure
```
data/
├── raw/                    # Original datasets (Avenue, ShanghaiTech, UCSD_Ped2)
├── splits/                 # Generated train/test splits and labels
│   ├── ucsd_ped2_labels.json      # Frame-level binary anomaly labels
│   └── ucsd_ped2_splits.json      # Train/test sequence splits
└── processed/              # Generated features (histograms, embeddings)
```

### Core Data Pipeline (Phase 1)
1. **Frame extraction** → `.tif` image sequences from datasets
2. **Histogram computation** → 256-bin grayscale histograms per frame  
3. **Label generation** → Binary anomaly labels from ground truth masks
4. **Validation** → Comprehensive integrity checks via `validate_ucsd_dataset.py`

## Essential Scripts & Usage

### Dataset Preparation
```bash
# Generate train/test splits
python scripts/make_ucsd_splits.py

# Extract frame-level anomaly labels from ground truth masks  
python scripts/make_ucsd_label_masks.py

# Compute 256-bin histograms for all frames
python scripts/extract_ucsd_histograms.py

# Validate complete data pipeline integrity (6 checks)
python scripts/validate_ucsd_dataset.py
```

### Key File Patterns
- **Input frames**: `data/raw/UCSD_Ped2/UCSDped2/Test/Test001/*.tif`
- **Ground truth**: `data/raw/UCSD_Ped2/UCSDped2/Test/Test001_gt/*.bmp` 
- **Generated features**: `data/raw/UCSD_Ped2/UCSDped2/Test_histograms/Test001_histograms.npy`
- **Labels**: Binary arrays in `ucsd_ped2_labels.json` (`0`=normal, `1`=anomaly)

## Development Conventions

### Data Handling
- Use **relative paths** from repository root (never hardcode absolute paths)
- Filter out system files (`.DS_Store`, `._.DS_Store`) when processing directories
- Normalize histograms using `density=True` in `np.histogram()` 
- Validate data integrity after any pipeline changes using the validator script

### Error Prevention
- Always check file existence before processing: `if Path(file).exists()`
- Filter image files explicitly: `f.endswith(('.tif', '.bmp', '.png'))`
- Handle edge cases: empty folders, missing ground truth, malformed images

### Code Organization
- **`scripts/`**: Standalone utilities for data preparation and validation
- **`src/`**: Core model implementation (placeholder - to be developed)
- **`notebooks/`**: Experimental analysis and visualization

## Integration Points

### External Dependencies
- **PIL/Pillow**: Image loading and processing
- **NumPy**: Histogram computation and array operations  
- **JSON**: Metadata serialization for splits and labels
- **Pathlib**: Cross-platform file path handling

### Dataset Requirements
- UCSD Ped2: Frame images (`.tif`) + ground truth masks (`.bmp`)
- Avenue: MATLAB volumes (`.mat` files) - integration pending
- ShanghaiTech: Video files + frame/pixel masks - integration pending

## Quality Assurance

### Before Committing Changes
1. **Run validator**: `python scripts/validate_ucsd_dataset.py` → must show "ALL CHECKS PASSED"
2. **Check paths**: Ensure all file paths use `os.path.join()` or `pathlib.Path`
3. **Test edge cases**: Empty directories, missing files, corrupted images
4. **Verify outputs**: JSON files should be valid, `.npy` files should load correctly

### Critical Files to Preserve
- `data/splits/*.json` - Generated metadata (commit these)
- `data/raw/` - Original datasets (do NOT commit, too large)
- `validation_results.json` - Validation reports (useful for debugging)

## Quick Start for New Contributors
```bash
# 1. Verify dataset is properly placed
ls data/raw/UCSD_Ped2/UCSDped2/

# 2. Run complete data pipeline  
python scripts/make_ucsd_splits.py
python scripts/make_ucsd_label_masks.py  
python scripts/extract_ucsd_histograms.py

# 3. Validate everything works
python scripts/validate_ucsd_dataset.py

# Expected: "ALL CHECKS PASSED - Data is production-ready!"
```