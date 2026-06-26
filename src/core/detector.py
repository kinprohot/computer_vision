import time
import cv2
import numpy as np
from ultralytics import YOLO
from config import settings
from src.utils.logger import logger
from src.utils.image import deskew_plate, is_plate_yellow
from src.core.ocr import perform_ocr
from src.core.tracker import get_smoothed_class, get_smoothed_box, tracking_history

# Load YOLO models
logger.info(f"Loading custom vehicle model: {settings.VEHICLE_MODEL_PATH}")
vehicle_model = YOLO(str(settings.VEHICLE_MODEL_PATH))

logger.info(f"Loading custom license plate model: {settings.PLATE_MODEL_PATH}")
plate_model = YOLO(str(settings.PLATE_MODEL_PATH))

def _get_cached_or_ocr_plate(track_id, frame, py1, py2, px1, px2, is_square):
    current_time = time.time()
    # Check tracking history cache first to completely eliminate text flickering
    if track_id is not None and tracking_history[track_id]["plate_text"] is not None:
        # Keep valid plates cached for 10.0 seconds, but retry "N/A" after 1.5 seconds
        cache_duration = 10.0 if tracking_history[track_id]["plate_text"] != "N/A" else 1.5
        if current_time - tracking_history[track_id]["plate_time"] < cache_duration:
            return tracking_history[track_id]["plate_text"]
            
    # Crop the plate from original frame for OCR
    plate_crop = frame[py1:py2, px1:px2]
    # Rotate and run VLM OCR
    plate_no = perform_ocr(plate_crop, is_square)
        
    # Default to N/A if OCR fails
    if not plate_no:
        plate_no = "N/A"
        
    # Save to tracking history cache
    if track_id is not None:
        tracking_history[track_id]["plate_text"] = plate_no
        tracking_history[track_id]["plate_time"] = current_time
        
    return plate_no

def _process_found_plate(best_plate_box, x1, y1, track_id, img, frame, counts, colors):
    px1_c, py1_c, px2_c, py2_c = map(int, best_plate_box.xyxy[0].tolist())
    
    # Convert relative coordinates to absolute coordinates on main frame
    px1 = x1 + px1_c
    py1 = y1 + py1_c
    px2 = x1 + px2_c
    py2 = y1 + py2_c
    
    # Use Aspect Ratio
    pw = px2 - px1
    ph = py2 - py1
    ar = pw / ph if ph > 0 else 1.0
    
    # Standard Vietnamese square plate is 280x200 (AR ~1.4), long plate is 470x110 (AR ~4.27)
    is_square = ar <= 1.7
    
    # Detect if yellow plate by checking color in HSV space
    plate_crop_temp = frame[py1:py2, px1:px2]
    is_yellow = is_plate_yellow(plate_crop_temp)
        
    # Determine display color
    plate_color = (0, 255, 255) if is_yellow else colors[4] # Yellow or Red box
    
    # Draw plate box
    cv2.rectangle(img, (px1, py1), (px2, py2), plate_color, 2)
    
    plate_no = _get_cached_or_ocr_plate(track_id, frame, py1, py2, px1, px2, is_square)
    
    # Draw plate text overlay
    plate_prefix = "Bien vang" if is_yellow else "Bien so"
    plate_label = f"{plate_prefix}: {plate_no}"
    cv2.putText(img, plate_label, (px1, max(py1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, plate_color, 1)
    counts[4] += 1

def _process_fallback_plate(x1, y1, x2, y2, w, h, track_id, mapped_cls_id, img, counts, colors):
    if mapped_cls_id not in [0, 2, 3] or w <= 75 or h <= 75:
        return
        
    plate_w = int(w * 0.32)
    plate_h = int(h * 0.14)
    plate_x = x1 + int(w * 0.5)
    plate_y = y1 + int(h * 0.72)
    
    px1 = max(x1, plate_x - plate_w // 2)
    py1 = max(y1, plate_y)
    px2 = min(x2, plate_x + plate_w // 2)
    py2 = min(y2, plate_y + plate_h)
    
    cv2.rectangle(img, (px1, py1), (px2, py2), colors[4], 1)
    
    current_time = time.time()
    plate_no = None
    
    # Check tracking history cache for fallback
    if track_id is not None and tracking_history[track_id]["plate_text"] is not None:
        cache_duration = 10.0 if tracking_history[track_id]["plate_text"] != "N/A" else 1.5
        if current_time - tracking_history[track_id]["plate_time"] < cache_duration:
            plate_no = tracking_history[track_id]["plate_text"]
            
    if plate_no is None:
        plate_no = "N/A"
        if track_id is not None:
            tracking_history[track_id]["plate_text"] = plate_no
            tracking_history[track_id]["plate_time"] = current_time
            
    plate_label = f"Bien so: {plate_no}"
    cv2.putText(img, plate_label, (px1, max(py1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[4], 1)
    counts[4] += 1

def _detect_license_plate(x1, y1, x2, y2, w, h, track_id, mapped_cls_id, img, frame, counts, colors):
    # Crop the vehicle from original frame
    vehicle_crop = frame[y1:y2, x1:x2]
    if vehicle_crop.size == 0:
        return
        
    # Run plate detector on crop
    plate_crop_results = plate_model(vehicle_crop, verbose=False, imgsz=320)
    plate_boxes = plate_crop_results[0].boxes
    
    best_plate_box = None
    best_plate_conf = 0.0
    for pbox in plate_boxes:
        pconf = pbox.conf[0].item()
        if pconf > best_plate_conf:
            best_plate_conf = pconf
            best_plate_box = pbox
            
    if best_plate_box is not None and best_plate_conf > 0.45:
        _process_found_plate(best_plate_box, x1, y1, track_id, img, frame, counts, colors)
    else:
        _process_fallback_plate(x1, y1, x2, y2, w, h, track_id, mapped_cls_id, img, counts, colors)

def _process_single_vehicle(box, track_id, img, frame, counts, colors, is_coco):
    raw_cls_id = int(box.cls[0].item())
    confidence = box.conf[0].item()
    
    is_vehicle = (raw_cls_id in settings.COCO_MAP) if is_coco else (raw_cls_id in [0, 1, 2, 3])
    if confidence <= 0.30 or not is_vehicle:
        return
        
    mapped_cls_id = settings.COCO_MAP[raw_cls_id] if is_coco else raw_cls_id
    
    # Apply class smoothing using majority voting
    if track_id is not None:
        mapped_cls_id = get_smoothed_class(track_id, mapped_cls_id)
        
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    # Apply bounding box smoothing
    if track_id is not None:
        x1, y1, x2, y2 = get_smoothed_box(track_id, (x1, y1, x2, y2))
        
    color = colors.get(mapped_cls_id, (255, 255, 255))
    
    # Draw vehicle box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Draw vehicle label text in Vietnamese
    class_name = settings.CLASS_NAMES_VI.get(mapped_cls_id, "Khong xac dinh")
    label = f"{class_name} {confidence:.2f}"
    cv2.putText(img, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if mapped_cls_id in counts:
        counts[mapped_cls_id] += 1
        
    # Cascade: Crop vehicle and detect license plate inside it
    w = x2 - x1
    h = y2 - y1
    
    if w > 40 and h > 40:
        _detect_license_plate(x1, y1, x2, y2, w, h, track_id, mapped_cls_id, img, frame, counts, colors)

def process_stream_detections(vehicle_results, img, frame, counts, colors, is_coco):
    vehicle_boxes = vehicle_results[0].boxes
    if vehicle_boxes is not None and len(vehicle_boxes) > 0:
        if vehicle_boxes.id is not None:
            track_ids = vehicle_boxes.id.int().tolist()
        else:
            track_ids = [None] * len(vehicle_boxes)
            
        for idx, box in enumerate(vehicle_boxes):
            _process_single_vehicle(box, track_ids[idx], img, frame, counts, colors, is_coco)
