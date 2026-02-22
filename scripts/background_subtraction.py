import cv2
import numpy as np

video_path = 'videos\Squirrels_new_cups1.mp4'
cap = cv2.VideoCapture(video_path)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

fgbg = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=20, detectShadows=True)

#fgbg = cv2.createBackgroundSubtractorKNN(history=300, detectShadows=True)

# Crop percentages (top/bottom)
crop_percent_top = 0.0
crop_percent_bottom = 0.0

scale_percent = 20

# Video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

resized_width = int(original_width * scale_percent / 100)
resized_height = int(original_height * scale_percent / 100)

combined_width = resized_width * 2
combined_height = resized_height

# VideoWriter setup
output_path = 'output_background_subtraction.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    ############ <Crop region> #############
    height, width = frame.shape[:2]

    top_crop = int(height * crop_percent_top)
    bottom_crop = int(height * (1 - crop_percent_bottom))

    cropped_frame = frame[top_crop:bottom_crop, :]

    # Resize
    width_resized = int(cropped_frame.shape[1] * scale_percent / 100)
    height_resized = int(cropped_frame.shape[0] * scale_percent / 100)
    cropped_frame_resized = cv2.resize(cropped_frame, (width_resized, height_resized))

    # Apply background subtraction
    fgmask = fgbg.apply(cropped_frame_resized)

    # Reduce noise with morphological operations
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Convert mask to 3 channels for display
    fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    fgmask_resized = cv2.resize(fgmask_color, (width_resized, height_resized))

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sortiere die Konturen nach ihrer Fläche und wähle die größten 5 Konturen aus
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # for contour in contours:
    #     if 100 < cv2.contourArea(contour):
    #         cv2.drawContours(cropped_frame_resized, [contour], -1, (0, 0, 255), 2)

    # cv2.drawContours(cropped_frame_resized, contours, -1, (0, 255, 0), 2)

    # Combine side by side
    combined_frame = cv2.hconcat([cropped_frame_resized, fgmask_resized])

    out.write(combined_frame)

    cv2.imshow('Combined Frame', combined_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

print(f"Video saved to: {output_path}")
cap.release()
out.release()
cv2.destroyAllWindows()
