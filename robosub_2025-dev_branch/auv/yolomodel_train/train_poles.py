from ultralytics import YOLO
import os

# ----- SET YOUR PATHS HERE -----
# Path to the data.yaml file Roboflow gave you
DATA_YAML = r"C:\Users\HOME\Documents\GitHub\CV_data\Red_pole_detection.v1i.yolov8\data.yaml"
# Output directory for model weights and results
RESULTS_DIR = r"C:\Users\HOME\Documents\GitHub\CV_data\Red_pole_detection.v1i.yolov8\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----- 1. Train the Model -----
# You can use 'yolov8n.pt' for Nano (fast, small) or 'yolov8s.pt' for Small (better)
model = YOLO('yolov8n.pt')  # You can also try 'yolov8s.pt' for better accuracy

results = model.train(
    data=DATA_YAML,       # Your data.yaml path
    epochs=50,            # You can adjust this as you like
    imgsz=640,            # Image size
    project=RESULTS_DIR,  # Where results will go
    name="redpole_exp1",  # Name of this run/experiment
    workers=4             # Speed up data loading (optional)
)

# Path to the best trained model
BEST_MODEL = os.path.join(RESULTS_DIR, "redpole_exp1", "weights", "best.pt")

# ----- 2. Run Inference on a Test Image -----
# Pick an image to test (update the path below)
TEST_IMAGE = r"C:\Users\HOME\Documents\GitHub\CV_data\Red_pole_detection.v1i.yolov8\valid\images\YOUR_TEST_IMAGE.jpg"

model = YOLO(BEST_MODEL)
results = model(TEST_IMAGE)  # Runs prediction

# Show result in a window (press any key to close)
results[0].show()

# ----- 3. Save Results (optional) -----
# Save prediction with boxes to disk
results[0].save(filename=os.path.join(RESULTS_DIR, "prediction.jpg"))

print(f"Training and prediction complete! Model saved at: {BEST_MODEL}")
