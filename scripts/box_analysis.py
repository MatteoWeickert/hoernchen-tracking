from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import deque

# 1. Lade das vortrainierte YOLOv8-Modell
model = YOLO('C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\runs\\detect\\best_andi.pt')

# 2. Öffne deine Videodatei
video_path1 = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\box_analysis_videos\\20241015_TrepS_02_in (1)_clip-1.mp4"
video_path2 = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\box_analysis_videos\\20241015_TrepS_02_in (1)_clip-3.mp4"
video_path3 = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\box_analysis_videos\\20241107_TrepS_01_in (2)_cut_updated.mp4"
video_path4 = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\box_analysis_videos\\20241107_TrepS_01_in (3)_cut.mp4"
video_path5 = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\box_analysis_videos\\20241108_TrepS_01_in (4)_cut_updated.mp4"
video_path = video_path3 # Hier kannst du zwischen den Videos wechseln


class TrackedObject:
    def __init__(self, obj_id, class_id, class_name, initial_box):
        self.id = obj_id
        self.class_id = class_id
        self.class_name = class_name
        self.box = initial_box  # [x, y, w, h]
        self.center = self._get_center(initial_box)
        
        # Status History: 0=Sichtbar, 1=Nicht sichtbar & Hörnchen noch in Box, 2=Nicht Sichtbar & Hörnchen weg (vermutlich interagiert)
        self.history_status = [] 
        self.history_frames = []
        
        self.squirrel_absent_counter = 0 # Seit wie vielen Frames ist das Eichhörnchen weg?
        self.is_active = True     # Ist das Objekt noch Teil der Simulation?

    def _get_center(self, box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    def update(self, new_box, frame_id):
        self.box = new_box
        self.center = self._get_center(new_box)
        self.history_status.append(0) # 0 = Grün (Sichtbar)
        self.history_frames.append(frame_id)

    def mark_missing(self, frame_id, is_squirrel_present=False):
        """
        Docstring for mark_missing
        
        :param self: Description
        :param frame_id: Description
        :param is_squirrel_present: Description
        """
        
        current_status = 1 # Orange, vermutlich verdecktes Objekt
        if is_squirrel_present:
            current_status = 1
            self.squirrel_absent_counter = 0 # Reset counter when squirrel is present

        else:
            self.squirrel_absent_counter += 1
            if self.squirrel_absent_counter > 10:
                current_status = 2 # Rot, vermutlich interagiert oder weg
            else:
                current_status = 1 # Orange, Hörnchen für ein paar Frames nicht erkannt

        if self.class_name == 'squirrel':
            current_status = 2

        self.history_status.append(current_status)
        self.history_frames.append(frame_id)
        

# --- HILFSFUNKTIONEN ---
def calculate_iou(boxA, boxB):
    # box: [x_center, y_center, w, h] -> umrechnen in [x1, y1, x2, y2]
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

# --- INITIALISIERUNG ---
def initialize_objects(video_path, model, num_init_frames=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video konnte nicht geladen werden")

    print(f"Initialisiere Tracking (scanne erste {num_init_frames} Frames)...")
    
    # Speichere alle Detections des ersten Frames
        
    detections = []

    ret, frame = cap.read()
    if not ret: 
        raise ValueError("Konnte den ersten Frame nicht lesen")
        
    # YOLO Inferenz
    results = model(frame, verbose=False)
    results = results[0]
    if results.boxes:
        for box, cls in zip(results.boxes.xywh.cpu().numpy(), results.boxes.cls.cpu().numpy()): # verbinde jede Box mit ihrer Klasse
            class_name = model.names[int(cls)]
            if class_name != 'squirrel': # Eichhörnchen ignorieren bei Init
                detections.append({'box': box, 'cls': int(cls), 'name': class_name})

    cap.release()

    obj_id_counter = 1
    final_objects = []

    for det in detections:
        # Erstelle festes Objekt
        new_obj = TrackedObject(obj_id_counter, det['cls'], det['name'], det['box'])
        final_objects.append(new_obj)
        obj_id_counter += 1

    print(f"Initialisierung abgeschlossen. {len(final_objects)} statische Objekte gefunden.")
    print("Objekte: ", [(obj.id, obj.class_name) for obj in final_objects])
    return final_objects

def run_tracking(video_path, model, tracked_objects):
    cap = cv2.VideoCapture(video_path)
    
    # Plot Setup (Matplotlib interaktiv)
    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.show()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ax.set_xlim(0, total_frames)
    ax.set_xlabel("Frame")
    ax.set_title("Live Object Interaction Timeline")
    plot_state = {}
    
    frame_count = 0
    squirrel_obj = None # Spezial-Objekt für das Eichhörnchen, da es im initialen Frame nicht getrackt wird
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # 1. YOLO Detection
        results = model(frame, verbose=False, conf=0.25)
        
        current_boxes = []     # Alle nicht-Squirrel Boxen
        current_squirrel = None # Die Squirrel Box
        is_squirrel_present = False
        
        # Finde aktuelle Boxen im Frame und setze Squirrel Box
        if results[0].boxes:
            for box, cls, conf in zip(results[0].boxes.xywh.cpu().numpy(), results[0].boxes.cls.cpu().numpy(), results[0].boxes.conf.cpu().numpy()):
                name = model.names[int(cls)]
                if name == 'squirrel':
                    # Nimm das Eichhörnchen mit der höchsten Confidence
                    if current_squirrel is None or conf > current_squirrel['conf']:
                        current_squirrel = {'box': box, 'conf': conf}
                        is_squirrel_present = True
                else:
                    current_boxes.append(box)

        # 2. Squirrel Tracking (ID: 99)
        if current_squirrel:
            if squirrel_obj is None:
                squirrel_obj = TrackedObject(99, -1, 'squirrel', current_squirrel['box'])
            squirrel_obj.update(current_squirrel['box'], frame_count)
        else:
            if squirrel_obj:
                squirrel_obj.mark_missing(frame_count, is_squirrel_present)

        # 3. Static Object Matching (Hungarian Algorithm via IoU)
        # Wir bauen eine Kostenmatrix: Zeilen = Alte Objekte, Spalten = Neue Boxen
        # Kosten = 1 - IoU (damit Maximieren von IoU = Minimieren von Kosten)
        
        cost_matrix = np.ones((len(tracked_objects), len(current_boxes)))
        
        for i, obj in enumerate(tracked_objects):
            for j, box in enumerate(current_boxes):
                iou = calculate_iou(obj.box, box)
                cost_matrix[i, j] = 1 - iou

        # Hungarian Assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        print(f"Frame {frame_count}: Kostenmatrix:\n{cost_matrix}\nZuweisungen (Objekt -> Box): {[(int(r), int(c)) for r, c in zip(row_ind, col_ind)]}")
        
        assigned_objs = set()
        assigned_boxes = set()

        for r, c in zip(row_ind, col_ind):
            # Prüfe ob der Match gut genug ist (Threshold)
            # IoU muss > 0.1 sein
            if cost_matrix[r, c] < 0.9: # Entspricht IoU > 0.1
                tracked_objects[r].update(current_boxes[c], frame_count)
                assigned_objs.add(r)
                assigned_boxes.add(c)
        
        # 4. Nicht gematchte Objekte behandeln (weniger Boxen als Objekte -> einige könnten verdeckt sein)
        for i, obj in enumerate(tracked_objects):
            if i not in assigned_objs:
                obj.mark_missing(frame_count, is_squirrel_present)
                
        # --- VISUALISIERUNG (VIDEO) ---
        # Zeichne statische Objekte
        for obj in tracked_objects:
            x, y, w, h = map(int, obj.box)
            # Farbe je nach Status (Grün=Da, Rot=Weg)
            color = (0, 255, 0) if obj.history_status[-1] == 0 else (0, 0, 255)
            
            # Zeichne Box nur wenn aktuell sichtbar (Status 0)
            if obj.history_status[-1] == 0:
                cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 2)
                cv2.putText(frame, f"ID {obj.id} ({obj.class_name})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Zeichne "Ghost" Position (wo es zuletzt war)
                cv2.circle(frame, (int(obj.center[0]), int(obj.center[1])), 5, (0,0,255), -1)

        # Zeichne Squirrel
        if squirrel_obj and squirrel_obj.history_status[-1] == 0:
            sx, sy, sw, sh = map(int, squirrel_obj.box)
            cv2.rectangle(frame, (sx - sw//2, sy - sh//2), (sx + sw//2, sy + sh//2), (255, 100, 0), 2)
            cv2.putText(frame, "Squirrel", (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        cv2.imshow('Tracking Live', frame)
        
        # --- PLOT UPDATE (Alle X Frames um Performance zu sparen) ---
        if frame_count % 10 == 0:
            update_plot(fig, ax, plot_state, tracked_objects, squirrel_obj)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show() # Plot am Ende offen lassen

def update_plot(fig, ax, plot_state, tracked_objects, squirrel_obj):
        
    all_objs = tracked_objects + ([squirrel_obj] if squirrel_obj else [])
    
    for i, obj in enumerate(all_objs):
        if not obj: continue
        
        key=obj.id
        
        # Arrays für Plotting vorbereiten
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

        # Masken
        mask_visible = statuses == 0
        mask_missing = statuses == 1
        mask_gone = statuses == 2

        # Offsets aktualisieren
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

    # Y-Achse nur aktualisieren wenn neue Objekte dazu kamen
    ax.set_yticks(range(len(all_objs)))
    ax.set_yticklabels([f"{obj.class_name} {obj.id}" for obj in all_objs if obj])
    ax.set_ylim(-1, len(all_objs))

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

if __name__ == "__main__":
    VIDEO_PATH = video_path # Dein Video
    
    # 1. Modell laden
    model = model
    
    # 2. Initialisierung (Phase 1) - Objekte finden
    objects = initialize_objects(VIDEO_PATH, model)
    
    # 3. Tracking Loop (Phase 2-Ende)
    run_tracking(VIDEO_PATH, model, objects)