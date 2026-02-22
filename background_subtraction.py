import cv2
import numpy as np

# Load video
video_path = ''
cap = cv2.VideoCapture(video_path)

# Elliptical kernel for morphological operations (7x7)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Background subtractor – learns background over time, flags sudden movement as foreground
fgbg = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=True) 

# Fraction of frame to ignore at top/bottom
crop_percent_top = 0.0
crop_percent_bottom = 0.0

# Downscale factor
scale_percent = 60  # 60 % 

# Get original dimensions and FPS
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

resized_width = int(original_width * scale_percent / 100)
resized_height = int(original_height * scale_percent / 100)

# Output video: two frames side by side
combined_width = resized_width * 2
combined_height = resized_height

output_path = 'output_background_subtraction.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

while cap.isOpened():
    ret, frame = cap.read() # ret: success bool, frame: BGR image

    if not ret:
        break
    
    # crop and resize
    height, width = frame.shape[:2] # shape returns (height, width, channels)

    top_crop = int(height * crop_percent_top)
    bottom_crop = int(height * (1 - crop_percent_bottom))

    cropped_frame = frame[top_crop:bottom_crop, :] # vertical crop only 

    width_resized = int(cropped_frame.shape[1] * scale_percent / 100)
    height_resized = int(cropped_frame.shape[0] * scale_percent / 100)
    cropped_frame_resized = cv2.resize(cropped_frame, (width_resized, height_resized))

    # Background subtraction
    fgmask = fgbg.apply(cropped_frame_resized) # pixels that don't fit the background model: foreground

    # Remove noise: opening removes small blobs, closing fills gaps
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel) 

    # Convert mask to 3 channels so we can stack it next to the color frame
    fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)  
    fgmask_resized = cv2.resize(fgmask_color, (width_resized, height_resized))

    # Contours
    # RETR_EXTERNAL: outer contours only, CHAIN_APPROX_SIMPLE: compressed points
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in contours:
        if 100 < cv2.contourArea(contour):  # skip tiny blobs
            cv2.drawContours(cropped_frame_resized, [contour], -1, (0, 0, 255), 2) 

    # Combine and write
    combined_frame = cv2.hconcat([cropped_frame_resized, fgmask_resized])

    out.write(combined_frame)

    cv2.imshow('Combined Frame', combined_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

print(f"Video erfolgreich gespeichert unter: {output_path}")
cap.release()
out.release()
cv2.destroyAllWindows()
