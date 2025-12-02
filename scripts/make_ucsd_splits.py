import os
import json
import wandb
from datetime import datetime

# Initialize W&B for data preparation tracking
wandb.init(  # type: ignore
    project="temporalgraph-vad-complete",
    name=f"phase1_data_splits_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    tags=["phase1", "data_preparation", "ucsd_splits"],
    config={"phase": "1_data_preparation", "step": "train_test_splits"}
)

# Adjust these paths to match your true locations (relative to project root)
train_dir = 'data/raw/UCSD_Ped2/UCSDped2/Train'
test_dir = 'data/raw/UCSD_Ped2/UCSDped2/Test'

print(os.path.abspath(train_dir))
print(os.listdir(train_dir))


# List all normal training sequences (each is a folder)
train_videos = [os.path.join('UCSDped2/Train', d)
                for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d))]

# List all test sequences (each is a folder)
test_videos = [os.path.join('UCSDped2/Test', d)
               for d in os.listdir(test_dir)
               if os.path.isdir(os.path.join(test_dir, d))]

splits = {
    "train": train_videos,
    "test": test_videos
}

os.makedirs('data/splits', exist_ok=True)
with open('data/splits/ucsd_ped2_splits.json', 'w') as f:
    json.dump(splits, f, indent=2)

# Log split statistics to W&B
wandb.log({  # type: ignore
    "train_sequences": len(train_videos),
    "test_sequences": len(test_videos),
    "total_sequences": len(train_videos) + len(test_videos),
    "phase1_splits_created": True
})

print(f"Created {len(train_videos)} train and {len(test_videos)} test sequences")
wandb.finish()  # type: ignore
