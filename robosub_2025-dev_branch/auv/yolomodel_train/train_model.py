import os
import shutil

# Install required libraries if not already present
try:
    from roboflow import Roboflow
except ImportError:
    os.system("pip install roboflow ultralytics")

from roboflow import Roboflow
from ultralytics import YOLO

# ==== USER EDIT: Output Folder for results ====
OUTPUT_FOLDER = r"E:/CV_data/YOLO model"   # Change this to wherever you want your results

# ==== Roboflow Download Code (YOURS, CLEANED UP) ====
rf = Roboflow(api_key="YdDfLLnAGyRUYI27R1aa")
project = rf.workspace("inspirationrs25").project("rs25_lab_20250704")
version = project.version(1)
dataset = version.download("yolov8")  # dataset.location is the path

# ==== Fix possible folder structure issues (val vs valid, etc) ====
yaml_path = os.path.join(dataset.location, "data.yaml")
train_images = os.path.join(dataset.location, "train", "images")
train_labels = os.path.join(dataset.location, "train", "labels")
valid_images = os.path.join(dataset.location, "valid", "images")
valid_labels = os.path.join(dataset.location, "valid", "labels")

# If valid folders are missing, copy a few from train for quick validation
os.makedirs(valid_images, exist_ok=True)
os.makedirs(valid_labels, exist_ok=True)
if not os.listdir(valid_images):
    img_files = os.listdir(train_images)[:10]
    for img in img_files:
        shutil.copy(os.path.join(train_images, img), valid_images)
        label = os.path.splitext(img)[0] + '.txt'
        shutil.copy(os.path.join(train_labels, label), valid_labels)
    print(f"Copied {len(img_files)} train images/labels to valid for validation.")

# ==== Start YOLOv8 Training ====
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
model = YOLO("yolov8s.pt")  # Use another model here if you want
results = model.train(
    data=yaml_path,
    epochs=25,
    imgsz=640,
    project=OUTPUT_FOLDER,
    name="yolov8-custom"
)

print(f"\n✅ Training complete! Results saved in {os.path.join(OUTPUT_FOLDER, 'yolov8-custom')}")
