import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_voc_to_yolo(voc_root, yolo_root):
    voc_path = Path(voc_root)
    yolo_path = Path(yolo_root)
    
    # Define splits mapping
    # VOC split name -> YOLO split name
    splits = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }
    
    classes = ["plate_long", "plate_square", "plate_yellow"]
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    print(f"Starting VOC to YOLO conversion with multi-class labeling...")
    print(f"Classes: {classes}")
    print(f"Source: {voc_path}")
    print(f"Destination: {yolo_path}")
    
    import cv2
    import numpy as np
    
    for voc_split, yolo_split in splits.items():
        src_dir = voc_path / voc_split
        if not src_dir.exists():
            print(f"Warning: split directory {src_dir} not found, skipping.")
            continue
            
        dest_img_dir = yolo_path / "images" / yolo_split
        dest_lbl_dir = yolo_path / "labels" / yolo_split
        
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        xml_files = list(src_dir.glob("*.xml"))
        print(f"Converting split '{voc_split}' -> '{yolo_split}' ({len(xml_files)} files)...")
        
        count = 0
        for xml_file in xml_files:
            # Parse XML
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                continue
                
            # Get size
            size_elem = root.find("size")
            if size_elem is None:
                continue
            width = int(size_elem.find("width").text)
            height = int(size_elem.find("height").text)
            
            if width == 0 or height == 0:
                continue
                
            # Find corresponding image
            image_name = xml_file.stem + ".jpg"
            image_path = src_dir / image_name
            if not image_path.exists():
                image_name = xml_file.stem + ".png"
                image_path = src_dir / image_name
                
            if not image_path.exists():
                filename_elem = root.find("filename")
                if filename_elem is not None:
                    image_name = filename_elem.text
                    image_path = src_dir / image_name
            
            if not image_path.exists():
                continue
                
            # Read image to check color and crop
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                continue
                
            yolo_annotations = []
            
            # Find all objects
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name != "license_plate":
                    continue
                
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                xmax = float(bndbox.find("xmax").text)
                ymin = float(bndbox.find("ymin").text)
                ymax = float(bndbox.find("ymax").text)
                
                # Check for yellow plate using HSV color space on cropped region
                h_img, w_img = img_bgr.shape[:2]
                x1_c = max(0, int(xmin))
                y1_c = max(0, int(ymin))
                x2_c = min(w_img, int(xmax))
                y2_c = min(h_img, int(ymax))
                
                plate_crop = img_bgr[y1_c:y2_c, x1_c:x2_c]
                is_yellow = False
                if plate_crop.size > 0:
                    hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
                    # Broad range for yellow detection in HSV
                    lower_yellow = np.array([12, 45, 45])
                    upper_yellow = np.array([35, 255, 255])
                    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                    yellow_ratio = np.sum(mask > 0) / mask.size
                    if yellow_ratio > 0.15:
                        is_yellow = True
                
                w_box = xmax - xmin
                h_box = ymax - ymin
                aspect_ratio = w_box / h_box if h_box > 0 else 1.0
                
                # Assign class ID based on color and aspect ratio
                if is_yellow:
                    class_id = 2  # plate_yellow
                elif aspect_ratio > 2.2:
                    class_id = 0  # plate_long
                else:
                    class_id = 1  # plate_square
                
                # Convert to YOLO coordinates
                x_center, y_center, w, h = convert_box((width, height), (xmin, xmax, ymin, ymax))
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
            # Write label file
            label_file_path = dest_lbl_dir / f"{xml_file.stem}.txt"
            with open(label_file_path, "w") as f:
                f.write("\n".join(yolo_annotations))
                
            # Copy image
            shutil.copy(str(image_path), str(dest_img_dir / image_path.name))
            count += 1
            
        print(f"Split '{yolo_split}' completed. Converted {count} images/labels.")
        
    print("Conversion completed successfully!")

if __name__ == "__main__":
    voc_dir = "dataset/License Plate Recognition.v11i.voc"
    yolo_dir = "dataset_plate"
    convert_voc_to_yolo(voc_dir, yolo_dir)
