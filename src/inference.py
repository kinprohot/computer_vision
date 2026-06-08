import cv2
import glob
from ultralytics import YOLO
from pathlib import Path

def run_inference(model_path, source_path, output_dir="runs/detect/inference_results"):
    print(f"Loading custom model: {model_path}...")
    model = YOLO(model_path)
    
    # Resolve output directory relative to project root if it is relative
    project_root = Path(__file__).resolve().parent.parent
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Class colors:
    # 0: car (Green), 1: motorcycle (Blue), 2: truck (Yellow), 3: bus (Orange), 4: license_plate (Red)
    colors = {
        0: (0, 255, 0),    
        1: (255, 0, 0),   
        2: (0, 255, 255), 
        3: (0, 165, 255), 
        4: (0, 0, 255)    
    }
    
    # Run prediction
    print(f"Running prediction on: {source_path}...")
    results = model(source_path)
    
    for i, result in enumerate(results):
        img = result.orig_img.copy()
        boxes = result.boxes
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            
            # Bounding box coordinates in pixels
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            class_name = model.names[cls_id]
            color = colors.get(cls_id, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw text
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Save output image
        source_name = Path(result.path).name
        output_path = out_path / f"pred_{source_name}"
        cv2.imwrite(str(output_path), img)
        print(f"Saved inference result to: {output_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    
    # Dynamically find the latest trained weights file in runs directory
    weight_pattern = (project_root / "runs" / "detect" / "yolo26_traffic" / "vehicle_detector*" / "weights" / "best.pt").as_posix()
    weight_files = glob.glob(weight_pattern)
    
    if weight_files:
        # Sort to get the latest run folder
        trained_model = sorted(weight_files)[-1]
    else:
        trained_model = str(project_root / "runs" / "detect" / "yolo26_traffic" / "vehicle_detector" / "weights" / "best.pt")
        
    # Source path (can be image file, folder of images, or video file)
    test_source = str(project_root / "dataset" / "images" / "val")
    
    # Check if model exists before running
    if Path(trained_model).exists():
        run_inference(trained_model, test_source)
    else:
        print(f"Model path '{trained_model}' not found. Please train the model first.")
