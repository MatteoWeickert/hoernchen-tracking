from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import deque

# Load YOLO model
model = YOLO('C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\runs\\detect\\best_andi.pt')

# Video paths
video_path1 = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\box_analysis_videos\\20241015_TrepS_02_in (1)_clip-1.mp4"
video_path2 = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\box_analysis_videos\\20241015_TrepS_02_in (1)_clip-3.mp4"
video_path3 = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\box_analysis_videos\\20241107_TrepS_01_in (2)_cut_updated.mp4"
video_path4 = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\box_analysis_videos\\20241107_TrepS_01_in (3)_cut.mp4"
video_path5 = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\box_analysis_videos\\20241108_TrepS_01_in (4)_cut_updated.mp4"
video_path = video_path4


class TrackedObject:
    def __init__(self, obj_id, class_id, class_name, initial_box):
        self.id = obj_id
        self.class_id = class_id
        self.class_name = class_name
        self.box = initial_box
        self.center = self._get_center(initial_box)
        
        # Status: 0=visible, 1=occluded (squirrel still in box), 2=gone (likely interacted)
        self.history_status = [] 
        self.history_frames = []
        
        self.squirrel_absent_counter = 0
        self.is_active = True

    def _get_center(self, box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    def update(self, new_box, frame_id):
        self.box = new_box
        self.center = self._get_center(new_box)
        self.history_status.append(0)
        self.history_frames.append(frame_id)

    def mark_missing(self, frame_id, is_squirrel_present=False):
        """Mark object as missing; status depends on squirrel presence."""
        
        current_status = 1
        if is_squirrel_present:
            current_status = 1
            self.squirrel_absent_counter = 0

        else:
            self.squirrel_absent_counter += 1
            if self.squirrel_absent_counter > 10:
                current_status = 2
            else:
                current_status = 1

        if self.class_name == 'squirrel':
            current_status = 2

        self.history_status.append(current_status)
        self.history_frames.append(frame_id)
        

# --- HELPER FUNCTIONS ---
def calculate_iou(boxA, boxB):
    """Calculate IoU between two [x_center, y_center, w, h] boxes."""
    def to_coords(b):
        x, y, w, h = b
        return [x - w/2, y - h/2, x + w/2, y + h/2]

    b1 = to_coords(boxA)
    b2 = to_coords(boxB)

    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (b1[2] - b1[0]) * (b1[3] - b1[1])
    boxBArea = (b2[2] - b2[0]) * (b2[3] - b2[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# --- INITIALIZATION ---
def initialize_objects(video_path, model, num_init_frames=1):
    """Detect static objects in the first frame for tracking."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not load video")

    print(f"Initializing tracking (scanning first {num_init_frames} frames)...")
    
    # Speichere alle Detections des ersten Frames
        
    detections = []

    ret, frame = cap.read()
    if not ret: 
        raise ValueError("Could not read first frame")
        
    results = model(frame, verbose=False)
    results = results[0]
    if results.boxes:
        for box, cls in zip(results.boxes.xywh.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            class_name = model.names[int(cls)]
            if class_name != 'squirrel':
                detections.append({'box': box, 'cls': int(cls), 'name': class_name})

    cap.release()

    obj_id_counter = 1
    final_objects = []

    for det in detections:
        new_obj = TrackedObject(obj_id_counter, det['cls'], det['name'], det['box'])
        final_objects.append(new_obj)
        obj_id_counter += 1

    print(f"Initialization complete. {len(final_objects)} static objects found.")
    print("Objects: ", [(obj.id, obj.class_name) for obj in final_objects])
    return final_objects

def run_tracking(video_path, model, tracked_objects):
    """Run frame-by-frame tracking with Hungarian matching and live visualization."""
    cap = cv2.VideoCapture(video_path)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.show()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ax.set_xlim(0, total_frames)
    ax.set_xlabel("Frame")
    ax.set_title("Live Object Interaction Timeline")
    plot_state = {}
    
    frame_count = 0
    squirrel_obj = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # YOLO detection
        results = model(frame, verbose=False, conf=0.25)
        
        current_boxes = []
        current_squirrel = None
        is_squirrel_present = False
        
        # Parse detections
        if results[0].boxes:
            for box, cls, conf in zip(results[0].boxes.xywh.cpu().numpy(), results[0].boxes.cls.cpu().numpy(), results[0].boxes.conf.cpu().numpy()):
                name = model.names[int(cls)]
                if name == 'squirrel':
                    if current_squirrel is None or conf > current_squirrel['conf']:
                        current_squirrel = {'box': box, 'conf': conf}
                        is_squirrel_present = True
                else:
                    current_boxes.append(box)

        # Squirrel tracking (ID: 99)
        if current_squirrel:
            if squirrel_obj is None:
                squirrel_obj = TrackedObject(99, -1, 'squirrel', current_squirrel['box'])
            squirrel_obj.update(current_squirrel['box'], frame_count)
        else:
            if squirrel_obj:
                squirrel_obj.mark_missing(frame_count, is_squirrel_present)

        # Static object matching via Hungarian algorithm (IoU cost matrix)
        cost_matrix = np.ones((len(tracked_objects), len(current_boxes)))
        
        for i, obj in enumerate(tracked_objects):
            for j, box in enumerate(current_boxes):
                iou = calculate_iou(obj.box, box)
                cost_matrix[i, j] = 1 - iou

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        print(f"Frame {frame_count}: Cost matrix:\n{cost_matrix}\nAssignments (obj -> box): {[(int(r), int(c)) for r, c in zip(row_ind, col_ind)]}")
        
        assigned_objs = set()
        assigned_boxes = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.9:  # IoU > 0.1
                tracked_objects[r].update(current_boxes[c], frame_count)
                assigned_objs.add(r)
                assigned_boxes.add(c)
        
        # Handle unmatched objects (may be occluded)
        for i, obj in enumerate(tracked_objects):
            if i not in assigned_objs:
                obj.mark_missing(frame_count, is_squirrel_present)
                
        # --- VISUALIZATION ---
        for obj in tracked_objects:
            x, y, w, h = map(int, obj.box)
            color = (0, 255, 0) if obj.history_status[-1] == 0 else (0, 0, 255)
            
            if obj.history_status[-1] == 0:
                cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 2)
                cv2.putText(frame, f"ID {obj.id} ({obj.class_name})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.circle(frame, (int(obj.center[0]), int(obj.center[1])), 5, (0,0,255), -1)

        # Draw squirrel
        if squirrel_obj and squirrel_obj.history_status[-1] == 0:
            sx, sy, sw, sh = map(int, squirrel_obj.box)
            cv2.rectangle(frame, (sx - sw//2, sy - sh//2), (sx + sw//2, sy + sh//2), (255, 100, 0), 2)
            cv2.putText(frame, "Squirrel", (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        # cv2.imshow('Tracking Live', frame)
        
        # --- PLOT UPDATE (every 10 frames for performance) ---
        if frame_count % 10 == 0:
            update_plot(fig, ax, plot_state, tracked_objects, squirrel_obj)
            plot_img = fig_to_image(fig)

            # Resize plot to match frame height
            h_frame = frame.shape[0]
            plot_img = cv2.resize(plot_img, (int(plot_img.shape[1] * h_frame / plot_img.shape[0]), h_frame))

            combined = cv2.hconcat([frame, plot_img])

            cv2.imshow("Tracking + Timeline", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

def update_plot(fig, ax, plot_state, tracked_objects, squirrel_obj):
    """Update the live timeline plot with object status history."""
        
    all_objs = tracked_objects + ([squirrel_obj] if squirrel_obj else [])
    
    for i, obj in enumerate(all_objs):
        if not obj: continue
        
        key=obj.id
        
        frames = np.array(obj.history_frames)
        statuses = np.array(obj.history_status)

        if key not in plot_state:
            visible_scatter = ax.scatter([], [], c='green', s=10, marker='|')
            missing_scatter = ax.scatter([], [], c='orange', s=5, marker='.')
            gone_scatter = ax.scatter([], [], c='red', s=5, marker='x')

            plot_state[key] = {
                "visible": visible_scatter,
                "missing": missing_scatter,
                "gone": gone_scatter,
                "y": i
            }

        y_pos = plot_state[key]["y"]

        mask_visible = statuses == 0
        mask_missing = statuses == 1
        mask_gone = statuses == 2

        # Update scatter offsets
        plot_state[key]["visible"].set_offsets(
            np.column_stack((frames[mask_visible], np.full(np.sum(mask_visible), y_pos)))
            if np.any(mask_visible) else np.empty((0, 2))
        )

        plot_state[key]["missing"].set_offsets(
            np.column_stack((frames[mask_missing], np.full(np.sum(mask_missing), y_pos)))
            if np.any(mask_missing) else np.empty((0, 2))
        )

        plot_state[key]["gone"].set_offsets(
            np.column_stack((frames[mask_gone], np.full(np.sum(mask_gone), y_pos)))
            if np.any(mask_gone) else np.empty((0, 2))
        )

    # Update y-axis labels
    ax.set_yticks(range(len(all_objs)))
    ax.set_yticklabels([f"{obj.class_name} {obj.id}" for obj in all_objs if obj])
    ax.set_ylim(-1, len(all_objs))

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def fig_to_image(fig):
    """Convert matplotlib figure to OpenCV BGR image."""
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape((height, width, 4))

    # ARGB -> RGB -> BGR
    buf = buf[:, :, [1, 2, 3, 0]]
    rgb = buf[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return bgr

if __name__ == "__main__":
    VIDEO_PATH = video_path
    
    model = model
    
    # Initialize static objects from first frame
    objects = initialize_objects(VIDEO_PATH, model)
    
    # Run tracking loop
    run_tracking(VIDEO_PATH, model, objects)