# This script converts annotations from NDJSON format to YOLO format.

import json
import os

# =====================
# CONFIG
# =====================
NDJSON_PATH = "Squirrels_in_town_annotations_12_12_2025.ndjson"
LABEL_OUT_DIR = "scripts/yolo/labels/train"

os.makedirs(LABEL_OUT_DIR, exist_ok=True)

CLASS_MAP = {
    "squirrel_head_entry": 0,
    "squirrel_halfway_entry": 1,
    "squirrel_full_entry": 2,
    "cup_full": 3,
    "cup_empty": 4,
    "nut": 5,
    "disco_ball": 6
}

# =====================
# HELPER FUNCTIONS
# =====================
def yolo_bbox(box, img_w, img_h):
    """
    Labelbox -> YOLO
    """
    x_center = (box["left"] + box["width"] / 2) / img_w
    y_center = (box["top"] + box["height"] / 2) / img_h
    w = box["width"] / img_w
    h = box["height"] / img_h
    return x_center, y_center, w, h


def get_squirrel_class(obj):
    """
    Resolve squirrel Entry_level
    """
    if not obj.get("classifications"):
        return None

    for cls in obj["classifications"]:
        if cls["name"] == "Entry_level":
            entry = cls["radio_answer"]["value"]
            return f"squirrel_{entry}"

    return None


# =====================
# MAIN
# =====================
with open(NDJSON_PATH, "r") as f:
    for line in f:
        data = json.loads(line)

        # Video-Name extrahieren
        video_name = data["data_row"]["external_id"]
        base_name = video_name.replace('.mp4', '')  # z.B. "Squirrels_new_cups1"

        media = data["media_attributes"]
        IMAGE_WIDTH = media["width"]
        IMAGE_HEIGHT = media["height"]

        projects = data.get("projects", {})
        for project in projects.values():
            labels = project.get("labels", [])

            for label in labels:
                frames = label["annotations"]["frames"]

                for frame_idx, frame_data in frames.items():
                    yolo_lines = []

                    for obj in frame_data["objects"].values():
                        name = obj["name"]

                        # ---- Class handling ----
                        if name == "squirrel":
                            class_name = get_squirrel_class(obj)
                            if class_name not in CLASS_MAP:
                                continue
                            class_id = CLASS_MAP[class_name]
                        else:
                            if name not in CLASS_MAP:
                                continue
                            class_id = CLASS_MAP[name]

                        # ---- Bounding box ----
                        box = obj["bounding_box"]
                        x, y, w, h = yolo_bbox(
                            box,
                            IMAGE_WIDTH,
                            IMAGE_HEIGHT
                        )

                        yolo_lines.append(
                            f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                        )

                    # ---- Write label file ----
                    if yolo_lines:
                        # Speichere Videonamen und Frame-Index im Dateinamen
                        label_path = os.path.join(
                            LABEL_OUT_DIR,
                            f"{base_name}_frame_{int(frame_idx):06d}.txt"
                        )
                        with open(label_path, "w") as out:
                            out.write("\n".join(yolo_lines))

print("✓ Label conversion complete!")