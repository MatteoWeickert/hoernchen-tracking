import cv2
import numpy as np

video_path = "sam3_outside_bw.mp4"
output_path = "sam3_with_timer.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

WHITE_THRESHOLD = 200
MOTION_THRESHOLD = 25

WHITE_PIXEL_MIN = 500                 # Eichhörnchen sichtbar
NO_WHITE_FRAMES = int(fps * 0.5)      # 0.5 s kein Weiß = Ende

timer_running = False
start_frame = 0
current_frame_idx = 0
no_white_counter = 0

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- Weiß erkennen ----
    white_pixels = np.sum(gray > WHITE_THRESHOLD)

    # ---- Bewegung erkennen (nur für Start) ----
    diff = cv2.absdiff(gray, prev_gray)
    motion_pixels = np.sum(diff > MOTION_THRESHOLD)

    # ---- Start-Bedingung ----
    if not timer_running:
        if white_pixels > WHITE_PIXEL_MIN and motion_pixels > 300:
            timer_running = True
            start_frame = current_frame_idx
            print("Timer gestartet")

    # ---- Stop-Bedingung (NUR Weiß!) ----
    if timer_running:
        if white_pixels < WHITE_PIXEL_MIN:
            no_white_counter += 1
        else:
            no_white_counter = 0

    # ---- Zeit berechnen ----
    elapsed_time = 0
    if timer_running:
        elapsed_time = (current_frame_idx - start_frame) / fps

    # ---- Overlay ----
    cv2.putText(
        frame,
        f"Time: {elapsed_time:.2f} s",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )

    out.write(frame)

    # ---- Abbruch ----
    if timer_running and no_white_counter >= NO_WHITE_FRAMES:
        print(f"Timer gestoppt bei {elapsed_time:.2f} Sekunden")
        break

    prev_gray = gray
    current_frame_idx += 1

cap.release()
out.release()
