from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Required for headless server
import matplotlib.pyplot as plt
import os
import io
import base64
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

try:
    yolo_model = YOLO(r"best.pt")
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    yolo_model = None

def get_video_properties(cap):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    return fps, total_frames, f"{minutes}:{seconds:02d}"

def generate_plot(data, label_name, color='blue', title="Analysis"):
    """Generate a base64-encoded PNG plot from time-series data."""
    plt.figure(figsize=(10, 5))
    plt.plot(data, label=label_name, color=color, alpha=0.7)
    
    if len(data) > 20:
        window_size = 15
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data, kernel, mode='valid')
        plt.plot(range(len(smoothed)), smoothed, color='red', label='Smoothed', linewidth=2)

    plt.title(title)
    plt.xlabel("Frame Number")
    plt.ylabel("Intensity / Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    plot_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()
    return plot_base64


def background_subtraction_analysis(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError("Video Error")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fgbg = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=True)
    
    pixel_counts = []
    fps, _, duration_str = get_video_properties(cap)
    frame_idx = 0
    scale_percent = 50 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (width, height))

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        count = cv2.countNonZero(fgmask)
        pixel_counts.append(count)
        frame_idx += 1

    cap.release()

    plot_b64 = generate_plot(pixel_counts, "Changed Pixels", "green", "Motion Detection (Background Sub)")

    return {
        'frames_processed': frame_idx,
        'duration': duration_str,
        'avg_movement': float(np.mean(pixel_counts)) if pixel_counts else 0,
        'max_movement': int(np.max(pixel_counts)) if pixel_counts else 0,
        'peak_frame': int(np.argmax(pixel_counts)) if pixel_counts else 0,
        'total_detections': len([x for x in pixel_counts if x > 1000]),
        'plot': plot_b64
    }

def yolo_analysis(video_path):
    """Analyze video with YOLO object detection."""
    if yolo_model is None:
        raise ValueError("YOLO model is not loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError("Video Error")
    
    fps, _, duration_str = get_video_properties(cap)
    
    confidence_sums = []
    detection_counts = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Optional: analyze every Nth frame for performance
        # if frame_idx % 3 != 0:
        #     frame_idx += 1
        #     confidence_sums.append(confidence_sums[-1] if confidence_sums else 0)
        #     continue

        results = yolo_model(frame, verbose=False)
        
        frame_conf_sum = 0.0
        obj_count = 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                frame_conf_sum += float(box.conf[0])
                obj_count += 1
        
        confidence_sums.append(frame_conf_sum * 100)
        detection_counts.append(obj_count)
        frame_idx += 1

    cap.release()

    plot_b64 = generate_plot(confidence_sums, "Total Confidence Score", "purple", "YOLO Object Detection Strength")

    return {
        'frames_processed': frame_idx,
        'duration': duration_str,
        'avg_movement': float(np.mean(confidence_sums)) if confidence_sums else 0,
        'max_movement': int(np.max(confidence_sums)) if confidence_sums else 0,
        'peak_frame': int(np.argmax(confidence_sums)) if confidence_sums else 0,
        'total_detections': sum(1 for x in detection_counts if x > 0),
        'plot': plot_b64
    }

def time_analysis_function(video_path):
    """
    Estimate time spent in box using background subtraction.
    Uses static ROI split since cv2.selectROI is unavailable on server.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError("Video Error")
    
    fps, _, duration_str = get_video_properties(cap)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Static ROI definition (server workaround, ideally sent from frontend)
    roi_entrance = (0, 0, int(width * 0.25), height) 
    roi_inside = (int(width * 0.25), 0, width, height) 

    fgbg = cv2.createBackgroundSubtractorKNN(history=300, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    inside_activity = []
    
    frame_idx = 0
    scale = 0.5
    
    # Timer logic
    timer_running = False
    time_in_box_frames = 0
    start_frame = 0
    THRESHOLD_INSIDE = 1000
    cooldown_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Resize for performance
        frame_small = cv2.resize(frame, None, fx=scale, fy=scale)
        
        fgmask = fgbg.apply(frame_small)
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # Extract scaled ROI
        r_in = [int(c * scale) for c in roi_inside]
        mask_inside = fgmask[r_in[1]:r_in[3], r_in[0]:r_in[2]]
        
        pixels_inside = cv2.countNonZero(mask_inside)
        inside_activity.append(pixels_inside)
        
        # Simple timer algorithm
        if cooldown_counter > 0:
            cooldown_counter -= 1

        if pixels_inside > THRESHOLD_INSIDE:
            if not timer_running and cooldown_counter == 0:
                timer_running = True
                start_frame = frame_idx
        else:
            if timer_running:
                timer_running = False
                time_in_box_frames += (frame_idx - start_frame)
                cooldown_counter = int(fps * 1.0)  # 1s cooldown

        frame_idx += 1

    cap.release()
    
    if timer_running:
        time_in_box_frames += (frame_idx - start_frame)
        
    total_seconds_in_box = time_in_box_frames / fps if fps > 0 else 0
    
    plot_b64 = generate_plot(inside_activity, "Activity Inside Box", "red", "Presence in Box")

    return {
        'frames_processed': frame_idx,
        'duration': duration_str,
        'avg_movement': float(np.mean(inside_activity)) if inside_activity else 0,
        'max_movement': int(np.max(inside_activity)) if inside_activity else 0,
        'peak_frame': int(np.argmax(inside_activity)) if inside_activity else 0,
        'total_detections': int(total_seconds_in_box),
        'plot': plot_b64
    }




@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400
        
        video_file = request.files['video']
        method = request.form.get('method', 'background_sub')
        
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(video_file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        video_file.save(video_path)

        results = {}
        
        try:
            if method == 'background_sub':
                results = background_subtraction_analysis(video_path)
            elif method == 'yolo':
                results = yolo_analysis(video_path)
            elif method == 'time_analysis':
                results = time_analysis_function(video_path)
            else:
                results = {'error': f'Method {method} not implemented'}
        except Exception as e:
            print(f"Processing Error: {e}")
            raise e
        finally:
            # Clean up uploaded video
            if os.path.exists(video_path):
                os.remove(video_path)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'yolo_loaded': yolo_model is not None})

if __name__ == '__main__':
    app.run(debug=True, port=5000)