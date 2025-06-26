import cv2
import numpy as np
import os

# ---- CONFIG ----
video_path = "C:/Users/HOME/Documents/GitHub/CV_data/poles_test_4.mp4"  # <-- Your video filename here!
out_img_dir = "images"
out_label_dir = "labels"
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)
frame_skip = 1  # 1 = every frame, 5 = every 5th frame, etc.
min_box_area = 400  # ignore noise; adjust as needed

# HSV values from your last step:
lower_red1 = np.array([0, 75, 40])
upper_red1 = np.array([12, 255, 255])
lower_red2 = np.array([129, 70, 20])
upper_red2 = np.array([179, 255, 255])

cap = cv2.VideoCapture(video_path)
frame_idx = 0
save_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_skip == 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morph close to reduce noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w, _ = frame.shape
        labels = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < min_box_area:
                continue  # skip small noise

            # YOLO format: class x_center y_center width height (all normalized)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # Only save frames with at least one detection!
        if labels:
            out_img_path = os.path.join(out_img_dir, f"frame_{save_idx:04d}.jpg")
            out_label_path = os.path.join(out_label_dir, f"frame_{save_idx:04d}.txt")
            cv2.imwrite(out_img_path, frame)
            with open(out_label_path, "w") as f:
                f.write("\n".join(labels))
            save_idx += 1

    frame_idx += 1

cap.release()
print(f"Done! Saved {save_idx} labeled frames in '{out_img_dir}' and '{out_label_dir}'.")
