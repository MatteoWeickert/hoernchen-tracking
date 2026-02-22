# Randomly select 10% of training images + labels and move them to a validation directory.

import os
import random
import shutil

VAL_SPLIT = 0.1

train_images_dir = "YOLO/images/train"
train_labels_dir = "YOLO/labels/train"
val_images_dir = "YOLO/images/val"
val_labels_dir = "YOLO/labels/val"

os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]

val_count = int(len(image_files) * VAL_SPLIT)
val_files = random.sample(image_files, val_count)

for image_file in val_files:

    label_file = os.path.splitext(image_file)[0] + ".txt"

    src_image_path = os.path.join(train_images_dir, image_file)
    src_label_path = os.path.join(train_labels_dir, label_file)
    dest_image_path = os.path.join(val_images_dir, image_file)
    dest_label_path = os.path.join(val_labels_dir, label_file)

    if os.path.exists(src_image_path):
        shutil.move(src_image_path, dest_image_path)
        print(f"Moved image: {image_file}")

    if os.path.exists(src_label_path):
        shutil.move(src_label_path, dest_label_path)
        print(f"Moved label: {label_file}")