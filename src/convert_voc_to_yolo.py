import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

import cv2
import numpy as np

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

def read_xml(xml_file):
    try:
        tree = ET.parse(xml_file)
        return tree.getroot()
    except Exception as exc:
        print(f"Error parsing {xml_file}: {exc}")
        return None


def get_annotation_size(root):
    size_elem = root.find("size")
    if size_elem is None:
        return None

    width_elem = size_elem.find("width")
    height_elem = size_elem.find("height")
    if width_elem is None or height_elem is None:
        return None

    width = int(width_elem.text)
    height = int(height_elem.text)
    return (width, height) if width > 0 and height > 0 else None


def find_image_path(src_dir, root, xml_file):
    for suffix in [".jpg", ".png"]:
        candidate = src_dir / f"{xml_file.stem}{suffix}"
        if candidate.exists():
            return candidate

    filename_elem = root.find("filename")
    if filename_elem is not None:
        candidate = src_dir / filename_elem.text
        if candidate.exists():
            return candidate

    return None


def detect_yellow_plate(img_bgr, xmin, xmax, ymin, ymax):
    h_img, w_img = img_bgr.shape[:2]
    x1_c = max(0, int(xmin))
    y1_c = max(0, int(ymin))
    x2_c = min(w_img, int(xmax))
    y2_c = min(h_img, int(ymax))
    plate_crop = img_bgr[y1_c:y2_c, x1_c:x2_c]

    if plate_crop.size == 0:
        return False

    hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([12, 45, 45])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(mask > 0) / mask.size
    return yellow_ratio > 0.15


def classify_plate(is_yellow, xmin, xmax, ymin, ymax):
    if is_yellow:
        return 2

    w_box = xmax - xmin
    h_box = ymax - ymin
    aspect_ratio = w_box / h_box if h_box > 0 else 1.0
    return 0 if aspect_ratio > 2.2 else 1


def build_yolo_annotation(width, height, xmin, xmax, ymin, ymax, class_id):
    x_center, y_center, w, h = convert_box((width, height), (xmin, xmax, ymin, ymax))
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def parse_plate_object(obj, img_bgr, width, height):
    name_elem = obj.find("name")
    if name_elem is None or name_elem.text != "license_plate":
        return None

    bndbox = obj.find("bndbox")
    if bndbox is None:
        return None

    xmin = float(bndbox.find("xmin").text)
    xmax = float(bndbox.find("xmax").text)
    ymin = float(bndbox.find("ymin").text)
    ymax = float(bndbox.find("ymax").text)

    is_yellow = detect_yellow_plate(img_bgr, xmin, xmax, ymin, ymax)
    class_id = classify_plate(is_yellow, xmin, xmax, ymin, ymax)
    return build_yolo_annotation(width, height, xmin, xmax, ymin, ymax, class_id)


def write_labels(label_file_path, yolo_labels):
    with open(label_file_path, "w") as f:
        f.write("\n".join(yolo_labels))


def process_xml_file(xml_file, src_dir, dest_lbl_dir, dest_img_dir):
    root = read_xml(xml_file)
    if root is None:
        return False

    size = get_annotation_size(root)
    if size is None:
        return False

    image_path = find_image_path(src_dir, root, xml_file)
    if image_path is None:
        return False

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return False

    width, height = size
    yolo_annotations = []
    for obj in root.findall("object"):
        annotation = parse_plate_object(obj, img_bgr, width, height)
        if annotation is not None:
            yolo_annotations.append(annotation)

    label_file_path = dest_lbl_dir / f"{xml_file.stem}.txt"
    write_labels(label_file_path, yolo_annotations)
    shutil.copy(str(image_path), str(dest_img_dir / image_path.name))
    return True


def process_split(voc_path, yolo_path, voc_split, yolo_split):
    src_dir = voc_path / voc_split
    if not src_dir.exists():
        print(f"Warning: split directory {src_dir} not found, skipping.")
        return

    dest_img_dir = yolo_path / "images" / yolo_split
    dest_lbl_dir = yolo_path / "labels" / yolo_split
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    xml_files = list(src_dir.glob("*.xml"))
    print(f"Converting split '{voc_split}' -> '{yolo_split}' ({len(xml_files)} files)...")

    count = 0
    for xml_file in xml_files:
        if process_xml_file(xml_file, src_dir, dest_lbl_dir, dest_img_dir):
            count += 1

    print(f"Split '{yolo_split}' completed. Converted {count} images/labels.")


def convert_voc_to_yolo(voc_root, yolo_root):
    voc_path = Path(voc_root)
    yolo_path = Path(yolo_root)

    splits = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }

    print("Starting VOC to YOLO conversion with multi-class labeling...")
    print(f"Source: {voc_path}")
    print(f"Destination: {yolo_path}")

    for voc_split, yolo_split in splits.items():
        process_split(voc_path, yolo_path, voc_split, yolo_split)

    print("Conversion completed successfully!")

if __name__ == "__main__":
    voc_dir = "dataset/License Plate Recognition.v11i.voc"
    yolo_dir = "dataset_plate"
    convert_voc_to_yolo(voc_dir, yolo_dir)
