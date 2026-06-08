import torch
from ultralytics import YOLO

def train_plate_detector():
    print("Initializing YOLO26 model for license plate detection...")
    model = YOLO("yolo26n.pt") 

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device for training: {device}")

    # Train model
    print("Starting license plate training...")
    results = model.train(
        data="config/dataset_plate.yaml",
        epochs=3,                     # Set to 3 epochs for fast CPU training
        imgsz=320,                    # Image size 320 is much faster on CPU
        batch=8,                      # Batch size 8 is safe for CPU/low RAM
        device=device,
        workers=2,
        project="yolo26_plate",       # Project directory
        name="plate_detector",        # Run name
    )
    print("License plate model training completed!")
    
    # Run validation
    print("Evaluating plate detector model...")
    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50}")

if __name__ == "__main__":
    train_plate_detector()
