"""This script tests a trained YOLOv8 model on a video file.
It loads the model, runs object detection on the video,
and saves the output video with detected objects highlighted."""
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('D:/CV_data/YOLO model/Model_red_pole_3692.pt')  # Change to your actual model path

# Run detection on your video
results = model.predict(
    source='D:/CV_data/Video_test/poles_test_1.mp4',    # Path to your video
    save=True,             # Save the output video with boxes drawn
    conf=0.25,             # Confidence threshold (adjust as needed)
    show=True              # Show live results (works locally, not in Colab)
)

# The output video will be saved in 'runs/detect/predict' by default
