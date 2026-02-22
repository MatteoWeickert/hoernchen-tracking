import json
import os
import subprocess
from collections import defaultdict
import unicodedata

# =====================
# CONFIG
# =====================
NDJSON_PATH = "Squirrels_in_town_annotations_12_12_2025_hoernchen.ndjson"
VIDEO_DIR = "scripts/yolo/videos"
IMAGE_DIR = "scripts/yolo/images"

os.makedirs(IMAGE_DIR, exist_ok=True)

# =====================
# HELPER: Fix encoding
# =====================
def fix_encoding(text):
    """Fix mojibake in filenames (e.g. HÃ¶rnchen -> Hörnchen)."""
    try:
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return unicodedata.normalize('NFC', text)

# =====================
# STEP 1: collect frames per video
# =====================
video_frames = defaultdict(set)

with open(NDJSON_PATH) as f:
    for line in f:
        data = json.loads(line)
        video_name = data["data_row"]["external_id"]

        # Fix encoding issues in video name
        video_name = fix_encoding(video_name)

        for project in data["projects"].values():
            for label in project["labels"]:
                for frame_idx in label["annotations"]["frames"].keys():
                    video_frames[video_name].add(int(frame_idx))

# =====================
# STEP 2: extract frames
# =====================
for video_name, frames in video_frames.items():
    video_path = os.path.join(VIDEO_DIR, video_name)

    if not os.path.exists(video_path):
        print(f"⚠ Video not found: {video_name}")
        print(f"  Looking for: {repr(video_name)}")
        available_videos = os.listdir(VIDEO_DIR)
        print(f"  Available videos:")
        for v in available_videos:
            print(f"    - {v} (repr: {repr(v)})")
        continue

    frame_set = set(frames)
    base_name = video_name.replace('.mp4', '')
    
    print(f"Extracting {len(frame_set)} frames from {video_name}")
    
    temp_pattern = os.path.join(IMAGE_DIR, f"{base_name}_temp_%06d.jpg")
    
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", video_path,
        "-vsync", "0",
        temp_pattern
    ]
    
    subprocess.run(cmd, check=True)
    
    # Keep only annotated frames, delete the rest
    kept_count = 0
    removed_count = 0
    
    for temp_file in os.listdir(IMAGE_DIR):
        if temp_file.startswith(f"{base_name}_temp_"):
            frame_num = int(temp_file.split('_')[-1].replace('.jpg', ''))
            
            temp_path = os.path.join(IMAGE_DIR, temp_file)
            
            if frame_num in frame_set:
                final_name = f"{base_name}_frame_{frame_num:06d}.jpg"
                final_path = os.path.join(IMAGE_DIR, final_name)
                
                if os.path.exists(final_path):
                    os.remove(final_path)
                
                os.rename(temp_path, final_path)
                kept_count += 1
            else:
                os.remove(temp_path)
                removed_count += 1
    
    print(f"✓ Kept {kept_count} frames, removed {removed_count} frames")