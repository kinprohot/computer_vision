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

def get_image_paths(img_dir):
    return list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))


def write_labels(label_file_path, yolo_labels):
    with open(label_file_path, "w") as f:
        f.write("\n".join(yolo_labels))


def extract_yolo_labels(results):
    yolo_labels = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in COCO_MAP:
                continue
            target_cls = COCO_MAP[cls_id]
            xywh = box.xywhn[0].tolist()
            yolo_labels.append(
                f"{target_cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}"
            )
    return yolo_labels


def process_split(model, dataset_path, split):
    img_dir = dataset_path / "images" / split
    label_dir = dataset_path / "labels" / split

    if not img_dir.exists():
        print(f"Directory {img_dir} does not exist, skipping.")
        return

    label_dir.mkdir(parents=True, exist_ok=True)
    images = get_image_paths(img_dir)
    print(f"Processing {len(images)} images in '{split}' split...")

    for img_path in images:
        process_image(model, img_path, label_dir)


def process_image(model, img_path, label_dir):
    results = model(img_path, verbose=False)
    label_file_path = label_dir / f"{img_path.stem}.txt"
    yolo_labels = extract_yolo_labels(results)
    write_labels(label_file_path, yolo_labels)


def auto_label_dataset(dataset_dir="dataset", model_name="yolo26n.pt"):
    """Auto-label vehicles using pre-trained YOLO26 model."""
    print(f"Loading pretrained model: {model_name}...")
    model = YOLO(model_name)
    dataset_path = Path(dataset_dir)

    for split in ["train", "val", "test"]:
        process_split(model, dataset_path, split)

    print("Auto-labeling completed successfully!")
    print("NOTE: License plates (class 4) must be annotated manually (e.g. using CVAT or LabelImg) since they are not in the default COCO dataset.")

if __name__ == "__main__":
    auto_label_dataset("dataset", "weights/yolo26n.pt")
