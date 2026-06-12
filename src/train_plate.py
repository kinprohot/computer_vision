import torch
from ultralytics import YOLO
from pathlib import Path
import shutil

def train_plate_detector():
    # Resolve absolute paths relative to project root (parent of src folder)
    project_root = Path(__file__).resolve().parent.parent
    
    print("Initializing YOLO26 model for license plate detection...")
    model_path = project_root / "yolo26n.pt"
    model = YOLO(str(model_path)) 

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device for training: {device}")

    # Train model
    print("Starting license plate training...")
    dataset_yaml = project_root / "config" / "dataset_plate.yaml"
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=10,                     # Set to 10 epochs for training
        imgsz=320,                    # Image size 320 is much faster on CPU
        batch=8,                      # Batch size 8 is safe for CPU/low RAM
        device=device,
        workers=2,
        project=str(project_root / "runs" / "detect" / "yolo26_plate"), # Save runs in project root's runs directory
        name="plate_detector",        # Run name
        exist_ok=True,                # Allow overwriting existing run folder
    )
    print("License plate model training completed!")
    
    # Copy the best trained model to project root as plate_best.pt
    best_weights_path = project_root / "runs" / "detect" / "yolo26_plate" / "plate_detector" / "weights" / "best.pt"
    dest_path = project_root / "plate_best.pt"
    if best_weights_path.exists():
        print(f"Copying best model weights from {best_weights_path} to {dest_path}")
        shutil.copy(best_weights_path, dest_path)
    else:
        print(f"Warning: Trained weights not found at {best_weights_path}")
    
    # Run validation
    print("Evaluating plate detector model...")
    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50}")

if __name__ == "__main__":
    train_plate_detector()
