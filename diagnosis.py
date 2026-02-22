import os
import shutil
from collections import defaultdict
import random

# =====================
# CONFIG
# =====================
IMAGE_DIR = "scripts/yolo/images"
LABEL_DIR = "scripts/yolo/labels/train"  # All labels currently in train

# Output structure
OUTPUT_BASE = "scripts/yolo/dataset"
TRAIN_IMG = os.path.join(OUTPUT_BASE, "images/train")
VAL_IMG = os.path.join(OUTPUT_BASE, "images/val")
TEST_IMG = os.path.join(OUTPUT_BASE, "images/test")

TRAIN_LABEL = os.path.join(OUTPUT_BASE, "labels/train")
VAL_LABEL = os.path.join(OUTPUT_BASE, "labels/val")
TEST_LABEL = os.path.join(OUTPUT_BASE, "labels/test")

# Split ratios (70% train, 20% val, 10% test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42

# =====================
# Create directory structure
# =====================
for dir_path in [TRAIN_IMG, VAL_IMG, TEST_IMG, TRAIN_LABEL, VAL_LABEL, TEST_LABEL]:
    os.makedirs(dir_path, exist_ok=True)

# =====================
# Group frames by video
# =====================
print("Scanning files...")
video_frames = defaultdict(list)

for label_file in os.listdir(LABEL_DIR):
    if not label_file.endswith('.txt'):
        continue
    
    base_name = label_file.replace('.txt', '')
    
    # Check if corresponding image exists
    img_file = f"{base_name}.jpg"
    img_path = os.path.join(IMAGE_DIR, img_file)
    
    if not os.path.exists(img_path):
        print(f"⚠ Warning: Label without image: {label_file}")
        continue
    
    # Extract video name (everything before _frame_)
    video_name = base_name.rsplit('_frame_', 1)[0]
    video_frames[video_name].append(base_name)

print(f"\nFound {len(video_frames)} videos with {sum(len(f) for f in video_frames.values())} frames total")

# =====================
# Split by video (important for generalization)
# =====================
random.seed(RANDOM_SEED)

videos = list(video_frames.keys())
random.shuffle(videos)

# Compute split points
n_videos = len(videos)
n_train = int(n_videos * TRAIN_RATIO)
n_val = int(n_videos * VAL_RATIO)

train_videos = videos[:n_train]
val_videos = videos[n_train:n_train + n_val]
test_videos = videos[n_train + n_val:]

print(f"\nVideo split:")
print(f"  Train: {len(train_videos)} videos")
print(f"  Val:   {len(val_videos)} videos")
print(f"  Test:  {len(test_videos)} videos")

# =====================
# Copy files
# =====================
def copy_files(video_list, img_dest, label_dest, split_name):
    """Copy images and labels for a list of videos to the target split."""
    frame_count = 0
    
    for video in video_list:
        frames = video_frames[video]
        
        for frame_base in frames:
            # Copy image
            src_img = os.path.join(IMAGE_DIR, f"{frame_base}.jpg")
            dst_img = os.path.join(img_dest, f"{frame_base}.jpg")
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            src_label = os.path.join(LABEL_DIR, f"{frame_base}.txt")
            dst_label = os.path.join(label_dest, f"{frame_base}.txt")
            shutil.copy2(src_label, dst_label)
            
            frame_count += 1
    
    print(f"  {split_name}: {frame_count} frames from {len(video_list)} videos")
    return frame_count

print("\nCopying files...")
train_count = copy_files(train_videos, TRAIN_IMG, TRAIN_LABEL, "Train")
val_count = copy_files(val_videos, VAL_IMG, VAL_LABEL, "Val")
test_count = copy_files(test_videos, TEST_IMG, TEST_LABEL, "Test")