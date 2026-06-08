import os
from pathlib import Path
from ultralytics import YOLO
import cv2

# Map COCO classes to our custom classes
# COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_MAP = {
    2: 0, # car
    3: 1, # motorcycle
    7: 2, # truck
    5: 3  # bus
}

def auto_label_dataset(dataset_dir="dataset", model_name="yolo26n.pt"):
    """Auto-label vehicles using pre-trained YOLO26 model."""
    print(f"Loading pretrained model: {model_name}...")
    model = YOLO(model_name)
    dataset_path = Path(dataset_dir)
    
    for split in ["train", "val", "test"]:
        img_dir = dataset_path / "images" / split
        label_dir = dataset_path / "labels" / split
        
        if not img_dir.exists():
            print(f"Directory {img_dir} does not exist, skipping.")
            continue
            
        label_dir.mkdir(parents=True, exist_ok=True)
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"Processing {len(images)} images in '{split}' split...")
        
        for img_path in images:
            results = model(img_path, verbose=False)
            label_file_path = label_dir / f"{img_path.stem}.txt"
            
            yolo_labels = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    
                    if cls_id in COCO_MAP:
                        target_cls = COCO_MAP[cls_id]
                        # Get normalized center-x, center-y, width, height (xywhn)
                        xywh = box.xywhn[0].tolist() 
                        
                        yolo_labels.append(f"{target_cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}")
            
            # Write to file
            with open(label_file_path, "w") as f:
                f.write("\n".join(yolo_labels))
                
    print("Auto-labeling completed successfully!")
    print("NOTE: License plates (class 4) must be annotated manually (e.g. using CVAT or LabelImg) since they are not in the default COCO dataset.")

if __name__ == "__main__":
    auto_label_dataset("dataset", "yolo26n.pt")
