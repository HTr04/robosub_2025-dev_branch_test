from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('C:/Users/HOME/Downloads/best_1.pt')  # Change to your actual model path

# Run detection on your video
results = model.predict(
    source='C:/Users/HOME/Documents/GitHub/CV_data/Image_video_test_4.mp4',    # Path to your video
    save=True,             # Save the output video with boxes drawn
    conf=0.25,             # Confidence threshold (adjust as needed)
    show=True              # Show live results (works locally, not in Colab)
)

# The output video will be saved in 'runs/detect/predict' by default
