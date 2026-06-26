import os
from pathlib import Path

# Project Roots
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Ensure weights directory exists
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# YOLO Models Configuration
VEHICLE_MODEL_PATH = WEIGHTS_DIR / "vehicle_best.pt"
if not VEHICLE_MODEL_PATH.exists():
    VEHICLE_MODEL_PATH = WEIGHTS_DIR / "yolo26n.pt"

PLATE_MODEL_PATH = WEIGHTS_DIR / "plate_best.pt"
if not PLATE_MODEL_PATH.exists():
    PLATE_MODEL_PATH = WEIGHTS_DIR / "yolo26n.pt"

# Classes Configuration
COCO_MAP = {
    2: 0,  # car
    3: 1,  # motorcycle
    7: 2,  # truck
    5: 3   # bus
}

CLASS_NAMES_VI = {
    0: "O to",
    1: "Xe may",
    2: "Xe tai",
    3: "Xe buyt",
    4: "Bien so"
}

# Stream Configuration
STREAMS = [
    {"id": "G_G8A6JU_LI", "title": "Camera 1"},
    {"id": "sJvEFrG0wq0", "title": "Camera 2"},
    {"id": "oif_zZFIfB4", "title": "Camera 3"},
    {"id": "1EamsYw_Xyo", "title": "Camera 4"},
    {"id": "NeJGBQAY-bE", "title": "Camera 5"},
    {"id": "x8tUUv-NGXs", "title": "Camera 6"}
]
