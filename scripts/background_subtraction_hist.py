import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = 'videos/Squirrels_new_leaf2.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Error: Could not open video.")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
fgbg = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=True)

foreground_pixel_counts = []
frame_idx = 0
scale_percent = 60

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
resized_width = int(original_width * scale_percent / 100)
resized_height = int(original_height * scale_percent / 100)

# Optional: VideoWriter
# output_path = 'output_background_subtraction.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# out = cv2.VideoWriter(output_path, fourcc, fps, (resized_width * 2, resized_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    width_resized = int(frame.shape[1] * scale_percent / 100)
    height_resized = int(frame.shape[0] * scale_percent / 100)
    frame_resized = cv2.resize(frame, (width_resized, height_resized))

    fgmask = fgbg.apply(frame_resized)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    count_t = cv2.countNonZero(fgmask)
    foreground_pixel_counts.append(count_t)

    # Optional: live visualization (uncomment to enable)
    # ------------------------------------------------------------------
    # fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    # contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     if 100 < cv2.contourArea(contour):
    #         cv2.drawContours(frame_resized, [contour], -1, (0, 0, 255), 2)
    #
    # combined_frame = cv2.hconcat([frame_resized, fgmask_color])
    # out.write(combined_frame)
    # cv2.imshow('Combined Frame', combined_frame)
    #
    # if cv2.waitKey(30) & 0xFF == ord('q'):
    #     break
    # ------------------------------------------------------------------
    
    frame_idx += 1


cap.release()
# out.release()
cv2.destroyAllWindows()
print(f"Done – {frame_idx} frames processed.")

# Smoothing (moving average)
window_size = 10
kernel = np.ones(window_size) / window_size
smoothed_counts = np.convolve(foreground_pixel_counts, kernel, mode='valid')


# Plot
plt.figure(figsize=(10, 5))
plt.plot(foreground_pixel_counts, label='Raw Data', alpha=0.5)
plt.plot(range(len(smoothed_counts)), smoothed_counts, color='red', label=f'Smoothed (Window={window_size})')
plt.title("Movement / Change Over Time (Background Subtraction)")
plt.xlabel("Frame Number")
plt.ylabel("Number of foreground pixels")
plt.legend()
plt.grid(True)
plt.show()