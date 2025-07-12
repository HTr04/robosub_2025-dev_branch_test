import os
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# --- SETTINGS ---
show_preview = True     # Show live preview with bounding boxes
preview_wait = 2         # ms to show each frame
frame_skip = 5           # Save every Nth frame
CONF_THRESHOLD = 0.3      # Minimum confidence to keep a detection

# --- Select Output Folder ---
root = tk.Tk()
root.withdraw()
save_base = filedialog.askdirectory(title='Select folder to save YOLO dataset')
if not save_base:
    print("No folder selected. Exiting.")
    exit(0)

out_img_dir = os.path.join(save_base, "images")
out_label_dir = os.path.join(save_base, "labels")
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)
print(f"Images will be saved to: {out_img_dir}")
print(f"Labels will be saved to: {out_label_dir}")

# --- Select Video File ---
video_path = filedialog.askopenfilename(title='Select input video', filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
if not video_path:
    print("No video selected. Exiting.")
    exit(0)

# --- YOLO Model Path ---
MODEL_PATH = 'C:/Users/HOME/Documents/GitHub/CV_data/Trained model/yolov8-custom/weights/best.pt'  # <- Change to your model path

# --- Load YOLO Model ---
model = YOLO(MODEL_PATH)
class_names = model.model.names  # Class names from the model

cap = cv2.VideoCapture(video_path)
frame_idx = 0
save_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_skip == 0:
        h, w, _ = frame.shape
        results = model(frame)
        boxes = results[0].boxes

        label_lines = []
        preview_img = frame.copy()
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf)
                if conf < CONF_THRESHOLD:
                    continue  # Skip low-confidence boxes
                class_id = int(box.cls)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Normalize to YOLO format
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
                # Draw rectangle and class name/conf on preview
                label_text = f"{class_names[class_id] if class_id < len(class_names) else class_id} {conf:.2f}"
                cv2.rectangle(preview_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(preview_img, label_text, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if label_lines:
            out_img_path = os.path.join(out_img_dir, f"frame_{save_idx:04d}.jpg")
            out_label_path = os.path.join(out_label_dir, f"frame_{save_idx:04d}.txt")
            cv2.imwrite(out_img_path, frame)
            with open(out_label_path, "w") as f:
                f.write("\n".join(label_lines))
            save_idx += 1

        # --- SHOW PREVIEW ---
        if show_preview:
            cv2.imwrite(os.path.join(save_base, f"preview_{save_idx:04d}.jpg"), preview_img)

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"Done! Saved {save_idx} labeled frames in '{out_img_dir}' and '{out_label_dir}'.")
