import torch
from ultralytics import YOLO
from pathlib import Path

def train_yolo26():
    # Resolve absolute paths relative to project root (parent of src folder)
    project_root = Path(__file__).resolve().parent.parent
    
    # Load model (yolo26n is the Nano model - fast and efficient)
    print("Initializing YOLO26 model...")
    model_path = project_root / "yolo26n.pt"
    model = YOLO(str(model_path)) 

    # Automatically detect if CUDA GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device for training: {device}")

    # Train model
    print("Starting training process...")
    dataset_yaml = project_root / "config" / "dataset.yaml"
    
    results = model.train(
        data=str(dataset_yaml),       # Path to dataset yaml config
        epochs=50,                    # Set to 5 epochs for quick CPU testing/demonstration
        imgsz=320,                   # Image size 320 is much faster on CPU
        batch=4,                     # Small batch size for CPU/low RAM safety
        device=device,               # Dynamic device selection
        workers=2,                   # Fewer workers to prevent resource clogging
        project=str(project_root / "runs" / "detect" / "yolo26_traffic"), # Save runs in the project root's runs directory
        name="vehicle_detector",     # Run folder name
    )
    print("Training finished!")
    
    # Run validation
    print("Running validation...")
    metrics = model.val()
    print(f"Validation mAP50-95: {metrics.box.map}")

if __name__ == "__main__":
    train_yolo26()
