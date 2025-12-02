import os
import json
from PIL import Image
import numpy as np
import wandb
from datetime import datetime

# Initialize W&B for label extraction tracking
wandb.init(  # type: ignore
    project="temporalgraph-vad-complete",
    name=f"phase1_label_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    tags=["phase1", "data_preparation", "label_extraction"],
    config={"phase": "1_data_preparation", "step": "anomaly_labels"}
)

gt_root = 'data/raw/UCSD_Ped2/UCSDped2/Test'  # contains Test001_gt folders

output_labels = {}

gt_folders = [f for f in os.listdir(gt_root) if f.endswith('_gt')]
for gt_folder in gt_folders:
    video_id = gt_folder.replace('_gt', '')  # e.g., Test001
    folder_path = os.path.join(gt_root, gt_folder)
    # Filter out non-image files
    frame_files = sorted(f for f in os.listdir(folder_path) if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')))
    labels = []
    for fname in frame_files:
        img_path = os.path.join(folder_path, fname)
        img = Image.open(img_path)
        img_arr = np.array(img)
        # If any white pixel exists, label anomaly=1, else normal=0.
        label = int(np.any(img_arr > 0))
        labels.append(label)
    output_labels[video_id] = labels

os.makedirs('data/splits', exist_ok=True)
with open('data/splits/ucsd_ped2_labels.json', 'w') as f:
    json.dump(output_labels, f, indent=2)

# Log label statistics to W&B
total_frames = sum(len(labels) for labels in output_labels.values())
total_anomaly = sum(sum(labels) for labels in output_labels.values())
anomalyrate = total_anomaly / total_frames if total_frames > 0 else 0

wandb.log({
    "test_sequences": len(output_labels),
    "total_frames": total_frames,
    "anomaly_frames": total_anomaly,
    "normal_frames": total_frames - total_anomaly,
    "anomaly_rate": anomalyrate,
    "phase1_labels_created": True
})

print('Frame-level labels written to data/splits/ucsd_ped2_labels.json')
print(f'Processed {len(output_labels)} sequences, {total_frames} total frames')
print(f'Anomaly rate: {anomalyrate:.2%}')
wandb.finish()  # type: ignore
