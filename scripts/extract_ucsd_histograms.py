import os
from PIL import Image
import numpy as np
import wandb
from datetime import datetime

# Initialize W&B for histogram extraction tracking
wandb.init(  # type: ignore
    project="temporalgraph-vad-complete",
    name=f"phase1_histogram_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    tags=["phase1", "data_preparation", "histogram_features"],
    config={"phase": "1_data_preparation", "step": "histogram_features"}
)

# Set base directories
base_folder = 'data/raw/UCSD_Ped2/UCSDped2/Test'  # or 'Train'
output_dir = 'data/raw/UCSD_Ped2/UCSDped2/Test_histograms'  # or 'Train_histograms'
os.makedirs(output_dir, exist_ok=True)

total_sequences = 0
total_frames = 0
processed_sequences = []

for seq_name in os.listdir(base_folder):
    seq_folder = os.path.join(base_folder, seq_name)
    if not os.path.isdir(seq_folder):
        continue  # skip files
    frame_files = sorted([f for f in os.listdir(seq_folder) if f.endswith('.tif') or f.endswith('.bmp')])
    histograms = []
    for fname in frame_files:
        img_path = os.path.join(seq_folder, fname)
        img = Image.open(img_path).convert('L')
        arr = np.array(img)
        hist, _ = np.histogram(arr, bins=256, range=(0,255), density=True)
        histograms.append(hist)
    
    # Save all histograms for this sequence
    out_path = os.path.join(output_dir, f'{seq_name}_histograms.npy')
    np.save(out_path, np.array(histograms))
    print(f"Saved {out_path}")
    
    # Track progress
    total_sequences += 1
    total_frames += len(frame_files)
    processed_sequences.append({"sequence": seq_name, "frames": len(frame_files)})
    
    # Log intermediate progress
    wandb.log({
        f"sequence_{seq_name}_frames": len(frame_files),
        "sequences_processed": total_sequences,
        "total_frames_processed": total_frames
    })

# Log final histogram extraction statistics
wandb.log({
    "total_sequences_processed": total_sequences,
    "total_frames_processed": total_frames,
    "histogram_bins": 256,
    "feature_type": "grayscale_histogram",
    "phase1_histograms_created": True
})

print(f"\nHistogram extraction complete:")
print(f"Processed {total_sequences} sequences with {total_frames} total frames")
wandb.finish()  # type: ignore
