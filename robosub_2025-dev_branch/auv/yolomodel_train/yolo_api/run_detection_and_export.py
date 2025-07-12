import os
import cv2
import json

from .video import Video, Frame
from .detectors import HSVDetector, YOLODetector
from .exporter import LabelExporter

# ----------- CONFIGURATION -----------
VIDEO_PATH = "C:/Users/HOME/Documents/GitHub/CV_data/20250704_all_1.mp4"
FRAMES_DIR = "C:/Users/HOME/Documents/GitHub/CV_data/test/frames"
IMAGES_DIR = "C:/Users/HOME/Documents/GitHub/CV_data/test/images"
LABELS_DIR = "C:/Users/HOME/Documents/GitHub/CV_data/test/labels"
CLASS_MAP = {
    "bin": 0,
    "bin_sawfish": 1,
    "bin_shark": 2,
    "gate": 3,
    "gate_sawfish": 4,
    "gate_shark": 5,
    "octagon_table": 6,
    "path": 7,
    "slalom_red": 8,
    "torpedo_sawfish": 9,
    "torpedo_shark": 10,
    "torpedo_target": 11,
    "slalom_white": 12,
    "octagon": 13
}
DETECTION_METHOD = "YOLO"     # "HSV" or "YOLO"
YOLO_MODEL_PATH = "C:/Users/HOME/Documents/GitHub/CV_data/Trained model/yolov8-custom_v1/weights/best.pt"
HSV_LOWER = [0, 100, 100]
HSV_UPPER = [10, 255, 255]
SHOW_VIDEO = True
# -------------------------------------

def main():
    print("Extracting frames from video...")
    video = Video(VIDEO_PATH)
    frames = []
    cap = video.cap
    frame_num = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frames.append(Frame(img, frame_num))
        frame_num += 1
    cap.release()
    print(f"Extracted {len(frames)} frames.")

    print("Running object detection...")
    if DETECTION_METHOD == "HSV":
        detector = HSVDetector(HSV_LOWER, HSV_UPPER)
    elif DETECTION_METHOD == "YOLO":
        detector = YOLODetector(YOLO_MODEL_PATH)
    else:
        raise ValueError("Invalid DETECTION_METHOD specified.")
    for frame in frames:
        frame.bounding_boxes = detector.detect_objects(frame) or []

    if SHOW_VIDEO:
        print("Showing detections for verification (press q to quit, any key to step)...")
        for frame in frames:
            display = frame.image_data.copy()
            if hasattr(frame, "bounding_boxes"):
                for box in frame.bounding_boxes:
                    color = (0, 255, 0)
                    box.draw_on_image(display, color=color, show_label=True)
            cv2.imshow("Verification", display)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

    print("Exporting images and labels...")
    exporter = LabelExporter(IMAGES_DIR, LABELS_DIR)
    exporter.export_images_and_labels(frames, CLASS_MAP)

    with open(os.path.join(LABELS_DIR, "class_map.txt"), "w") as f:
        for k, v in CLASS_MAP.items():
            f.write(f"{v}: {k}\n")
    with open(os.path.join(LABELS_DIR, "class_map.json"), "w") as f:
        json.dump(CLASS_MAP, f, indent=2)

    print(f"Done! Upload contents of '{IMAGES_DIR}' and '{LABELS_DIR}' to Roboflow. Use 'class_map.txt' or 'class_map.json' for label reference.")

if __name__ == "__main__":
    main()
