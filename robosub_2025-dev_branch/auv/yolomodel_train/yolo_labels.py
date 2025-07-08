""" this script processes a video file, detects objects based on specified HSV color ranges,
    and saves the detected objects as YOLO format labels in a specified output directory."""
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# ---- User Settings ----
show_preview = True      # Set False to skip showing window
preview_wait = 1      # ms to show each frame (increase if you want to look longer)

# ---- Select Output Folder ----
root = tk.Tk()
root.withdraw()
save_base = filedialog.askdirectory(title='Select folder to sa ve YOLO dataset')
if not save_base:
    print("No folder selected. Exiting.")
    exit(0)

out_img_dir = os.path.join(save_base, "images")
out_label_dir = os.path.join(save_base, "labels")
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)
print(f"Images will be saved to: {out_img_dir}")
print(f"Labels will be saved to: {out_label_dir}")

# ---- Video and HSV Config ----
video_path = "C:/Users/HOME/Documents/GitHub/CV_data/Image_video_test_2.mp4"
frame_skip = 1
min_box_area = 400

HSV_RANGES = [
    #(0, [0, 80, 50], [12, 255, 255]), # Lower red range
    #(0, [168, 80, 50], [179, 255, 255]), # Upper red range
    # (1, [0, 0, 200], [180, 30, 255]),
    (1, [22, 100, 100], [32, 255, 255]), # bright yellow range
    # Add more if needed
]

def get_color_masks(hsv_img, hsv_ranges):
    masks = []
    for class_id, lower, upper in hsv_ranges:
        mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
        masks.append((class_id, mask))
    return masks

cap = cv2.VideoCapture(video_path)
frame_idx = 0
save_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_skip == 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_labels = []
        used = np.zeros(frame.shape[:2], dtype=np.uint8)
        h, w, _ = frame.shape

        # Draw preview image (copy of frame)
        preview_img = frame.copy()

        for class_id, mask in get_color_masks(hsv, HSV_RANGES):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(used))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh
                if area < min_box_area:
                    continue
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                combined_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                used[y:y+bh, x:x+bw] = 255
                # Draw box for preview
                color = (0, 0, 255) if class_id == 0 else (0, 255, 0)
                 # Draw the bounding rectangle (optional, for reference)
                cv2.rectangle(preview_img, (x, y), (x+bw, y+bh), color, 1)

                # Draw the actual contour as a polygon!
                cv2.drawContours(preview_img, [cnt], -1, (0, 255, 255), 2)

                cv2.putText(preview_img, f"Class {class_id}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if combined_labels:
            out_img_path = os.path.join(out_img_dir, f"frame_{save_idx:04d}.jpg")
            out_label_path = os.path.join(out_label_dir, f"frame_{save_idx:04d}.txt")
            cv2.imwrite(out_img_path, frame)
            with open(out_label_path, "w") as f:
                f.write("\n".join(combined_labels))
            save_idx += 1

        # ---- SHOW PREVIEW ----
        if show_preview:
            cv2.imshow("Box & Polygon Preview", preview_img)
            key = cv2.waitKey(preview_wait) & 0xFF
            if key == ord('q'):
                print("User quit early.")
                break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"Done! Saved {save_idx} labeled frames in '{out_img_dir}' and '{out_label_dir}'.")
