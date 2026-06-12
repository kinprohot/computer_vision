import os
import re
import glob
import time
import cv2
import numpy as np
import yt_dlp
import hashlib
from pathlib import Path
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__, template_folder='templates')

# Load YOLO models
project_root = Path(__file__).resolve().parent.parent
vehicle_model_path = str("yolo26n.pt") # Revert to pretrained model for high accuracy vehicle detection
plate_model_path = str(project_root / "plate_best.pt")

print(f"[*] Loading pretrained vehicle model: {vehicle_model_path}")
vehicle_model = YOLO(vehicle_model_path)

print(f"[*] Loading custom license plate model: {plate_model_path}")
plate_model = YOLO(plate_model_path)

# Initialize EasyOCR Reader
print("[*] Initializing EasyOCR Reader...")
import torch
import easyocr
use_gpu = torch.cuda.is_available()
print(f"[*] EasyOCR GPU acceleration: {use_gpu}")
easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
print("[*] EasyOCR Reader initialized successfully!")

# Map COCO classes to our custom dashboard class indices
# COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_MAP = {
    2: 0,  # car
    3: 1,  # motorcycle
    7: 2,  # truck
    5: 3   # bus
}

# Vietnamese names without accents for OpenCV display compatibility (avoids ??? in cv2.putText)
CLASS_NAMES_VI = {
    0: "O to",
    1: "Xe may",
    2: "Xe tai",
    3: "Xe buyt",
    4: "Bien so"
}

# Shared statistics dictionary
current_stats = {}

# Direct HLS stream URL cache
url_cache = {}

# OCR text cache to avoid running OCR on every frame (extremely CPU heavy)
# Format: {spatial_hash: (plate_text, timestamp)}
ocr_cache = {}

# Tracking history dict for smoothing
# Format: { track_id: { "class_history": [...], "box_history": [...], "plate_text": "...", "plate_time": 0.0 } }
tracking_history = {}

# Simplified stream list
STREAMS = [
    {"id": "G_G8A6JU_LI", "title": "Camera 1"},
    {"id": "sJvEFrG0wq0", "title": "Camera 2"},
    {"id": "oif_zZFIfB4", "title": "Camera 3"},
    {"id": "1EamsYw_Xyo", "title": "Camera 4"},
    {"id": "NeJGBQAY-bE", "title": "Camera 5"},
    {"id": "x8tUUv-NGXs", "title": "Camera 6"}
]


def get_hls_url(video_id):
    """Retrieve HLS stream URL and cache it to speed up connection starts."""
    if video_id in url_cache:
        cached_url, expiry = url_cache[video_id]
        if time.time() < expiry:
            return cached_url
            
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': 'best',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        stream_url = info.get('url')
        # Cache for 15 minutes
        url_cache[video_id] = (stream_url, time.time() + 900)
        return stream_url

def get_smoothed_class(track_id, detected_cls):
    """Smooth vehicle class using majority voting over last 10 frames."""
    global tracking_history
    if track_id not in tracking_history:
        tracking_history[track_id] = {
            "class_history": [],
            "box_history": [],
            "plate_text": None,
            "plate_time": 0.0
        }
    
    history = tracking_history[track_id]["class_history"]
    history.append(detected_cls)
    if len(history) > 10:
        history.pop(0)
        
    return Counter(history).most_common(1)[0][0]

def get_smoothed_box(track_id, detected_box):
    """Smooth bounding box coordinates using a simple moving average over last 5 frames."""
    global tracking_history
    if track_id not in tracking_history:
        return detected_box
        
    history = tracking_history[track_id]["box_history"]
    history.append(detected_box)
    if len(history) > 5:
        history.pop(0)
        
    num_boxes = len(history)
    avg_x1 = sum(b[0] for b in history) // num_boxes
    avg_y1 = sum(b[1] for b in history) // num_boxes
    avg_x2 = sum(b[2] for b in history) // num_boxes
    avg_y2 = sum(b[3] for b in history) // num_boxes
    
    return (avg_x1, avg_y1, avg_x2, avg_y2)

def deskew_plate(plate_crop):
    """
    Deskew the license plate crop using Hough Lines.
    Only rotates if the skew angle is between -20 and 20 degrees to avoid spinning it sideways.
    """
    try:
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop
            
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Find lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=max(10, int(plate_crop.shape[1] * 0.3)), maxLineGap=10)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -20 < angle < 20:  # Focus on near-horizontal lines
                    angles.append(angle)
                    
        if len(angles) > 0:
            median_angle = np.median(angles)
            if abs(median_angle) > 1.0:
                h, w = plate_crop.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                # Rotate image
                plate_crop = cv2.warpAffine(plate_crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        print(f"[!] Deskew failed: {e}")
    return plate_crop

def correct_plate_string(text, is_top_line=None, is_bottom_line=None):
    if not text:
        return text
    text = "".join([c for c in text if c.isalnum()]).upper()
    
    digit_to_letter = {
        '0': 'D', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', 
        '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G'
    }
    
    letter_to_digit = {
        'A': '4', 'B': '8', 'D': '0', 'G': '6', 'I': '1', 
        'J': '1', 'L': '1', 'O': '0', 'Q': '0', 'S': '5', 
        'T': '7', 'Z': '2'
    }
    
    chars = list(text)
    
    if is_bottom_line:
        for i in range(len(chars)):
            if chars[i] in letter_to_digit:
                chars[i] = letter_to_digit[chars[i]]
        return "".join(chars)
        
    if is_top_line:
        for i in range(min(2, len(chars))):
            if chars[i] in letter_to_digit:
                chars[i] = letter_to_digit[chars[i]]
        if len(chars) >= 3:
            if chars[2] in digit_to_letter:
                chars[2] = digit_to_letter[chars[2]]
        return "".join(chars)
        
    if len(chars) >= 7:
        for i in range(2):
            if chars[i] in letter_to_digit:
                chars[i] = letter_to_digit[chars[i]]
        if chars[2] in digit_to_letter:
            chars[2] = digit_to_letter[chars[2]]
            
        num_digits = 5 if len(chars) >= 8 else 4
        for i in range(len(chars) - num_digits, len(chars)):
            if chars[i] in letter_to_digit:
                chars[i] = letter_to_digit[chars[i]]
                
    return "".join(chars)

def format_vietnamese_plate(plate_no):
    if not plate_no or plate_no == "N/A":
        return plate_no
        
    clean = plate_no.replace("-", "").replace(".", "").upper().strip()
    if len(clean) < 7:
        return plate_no
        
    if clean[3].isalpha():
        prefix_len = 4
    elif clean[3].isdigit():
        if len(clean) == 9:
            prefix_len = 4
        else:
            prefix_len = 3
    else:
        prefix_len = 3
        
    prefix = clean[:prefix_len]
    num_part = clean[prefix_len:]
    
    if len(num_part) == 5:
        formatted_num = f"{num_part[:3]}.{num_part[3:]}"
    else:
        formatted_num = num_part
        
    return f"{prefix}-{formatted_num}"

def preprocess_easyocr(crop):
    """Upscale if too small to help OCR detection."""
    try:
        if crop is None or crop.size == 0:
            return crop
        h, w = crop.shape[:2]
        target_w = 250
        if w < target_w:
            scale = target_w / w
            crop = cv2.resize(crop, (target_w, int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return crop
    except Exception as e:
        print(f"[!] Error in preprocess_easyocr: {e}")
        return crop

def perform_ocr_easyocr(plate_crop, is_square):
    try:
        if plate_crop is None or plate_crop.size == 0:
            return None
            
        # Preprocess plate crop (deskew + upscale)
        plate_crop = deskew_plate(plate_crop)
        plate_crop = preprocess_easyocr(plate_crop)
        
        # Convert BGR to RGB for EasyOCR
        plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
        
        # Run EasyOCR
        results = easyocr_reader.readtext(plate_rgb)
        if not results:
            return None
            
        # Sort results top-to-bottom, then left-to-right
        results_sorted = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
        
        texts = []
        # If we have multiple blocks, treat them as top/bottom lines of square plate
        if is_square and len(results_sorted) >= 2:
            top_raw = results_sorted[0][1]
            bottom_raw = results_sorted[1][1]
            top_clean = correct_plate_string(top_raw, is_top_line=True)
            bottom_clean = correct_plate_string(bottom_raw, is_bottom_line=True)
            if top_clean or bottom_clean:
                raw_plate = f"{top_clean or ''}{bottom_clean or ''}"
                return format_vietnamese_plate(raw_plate)
        
        # Fallback or single block
        for res in results_sorted:
            text = res[1]
            cleaned = "".join([c for c in text if c.isalnum()]).upper()
            if cleaned:
                texts.append(cleaned)
                
        if not texts:
            return None
            
        combined = "".join(texts)
        corrected = correct_plate_string(combined)
        return format_vietnamese_plate(corrected)
    except Exception as e:
        print(f"[!] EasyOCR Exception: {e}")
        return None



def gen_frames(video_id):
    """Generate JPEG frames with YOLO26 detections and real-time VietOCR."""
    global current_stats, ocr_cache
    print(f"[*] Starting AI stream generator for video: {video_id}")
    
    try:
        stream_url = get_hls_url(video_id)
    except Exception as e:
        print(f"[!] Error extracting URL for {video_id}: {e}")
        return
        
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"[!] Error opening stream for {video_id}")
        return
        
    # Class colors:
    # 0: car (green), 1: motorcycle (blue), 2: truck (yellow), 3: bus (orange), 4: license_plate (red)
    colors = {
        0: (0, 255, 0),    
        1: (255, 0, 0),   
        2: (0, 255, 255), 
        3: (0, 165, 255), 
        4: (0, 0, 255)    
    }
    
    # Initialize stats for this stream
    current_stats[video_id] = {
        "car": 0, "motorcycle": 0, "truck": 0, "bus": 0, "license_plate": 0, "fps": 0.0, "status": "active"
    }
    
    # Track FPS and time
    prev_time = time.time()
    
    # Set OpenCV buffer to 1 to reduce playback delay/latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print(f"[!] Failed to read frame from stream {video_id}")
                break
                
            # Run YOLO26 inference for vehicles using pretrained yolo26n.pt WITH tracking enabled
            # tracker="bytetrack.yaml" is fully supported on CPU
            vehicle_results = vehicle_model.track(frame, persist=True, verbose=False, imgsz=320, tracker="bytetrack.yaml")
            
            # Count current frame's classes
            counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            
            img = frame.copy()
            
            # Process detected vehicles
            vehicle_boxes = vehicle_results[0].boxes
            if vehicle_boxes is not None and len(vehicle_boxes) > 0:
                if vehicle_boxes.id is not None:
                    track_ids = vehicle_boxes.id.int().tolist()
                else:
                    track_ids = [None] * len(vehicle_boxes)
                    
                for idx, box in enumerate(vehicle_boxes):
                    raw_cls_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    track_id = track_ids[idx]
                    
                    # Check confidence threshold and if the class is a vehicle (car, motorcycle, truck, bus)
                    if confidence > 0.30 and raw_cls_id in COCO_MAP:
                        mapped_cls_id = COCO_MAP[raw_cls_id]
                        
                        # Apply class smoothing using majority voting based on track_id
                        if track_id is not None:
                            mapped_cls_id = get_smoothed_class(track_id, mapped_cls_id)
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Apply bounding box smoothing using moving average
                        if track_id is not None:
                            x1, y1, x2, y2 = get_smoothed_box(track_id, (x1, y1, x2, y2))
                            
                        color = colors.get(mapped_cls_id, (255, 255, 255))
                        
                        # Draw vehicle box
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw vehicle label text in Vietnamese
                        class_name = CLASS_NAMES_VI.get(mapped_cls_id, "Khong xac dinh")
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(img, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        if mapped_cls_id in counts:
                            counts[mapped_cls_id] += 1
                            
                        # Cascade: Crop vehicle and detect license plate inside it
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Only look for plates in vehicles of reasonable size
                        if w > 40 and h > 40:
                            # Crop the vehicle from original frame
                            vehicle_crop = frame[y1:y2, x1:x2]
                            if vehicle_crop.size > 0:
                                # Run plate detector on crop (using imgsz=160 for fast inference on small crops)
                                plate_crop_results = plate_model(vehicle_crop, verbose=False, imgsz=160)
                                plate_boxes = plate_crop_results[0].boxes
                                
                                best_plate_box = None
                                best_plate_conf = 0.0
                                for pbox in plate_boxes:
                                    pconf = pbox.conf[0].item()
                                    if pconf > best_plate_conf:
                                        best_plate_conf = pconf
                                        best_plate_box = pbox
                                
                                # If a license plate is found with confidence > 0.20
                                if best_plate_box is not None and best_plate_conf > 0.20:
                                    px1_c, py1_c, px2_c, py2_c = map(int, best_plate_box.xyxy[0].tolist())
                                    
                                    # Convert relative coordinates to absolute coordinates on main frame
                                    px1 = x1 + px1_c
                                    py1 = y1 + py1_c
                                    px2 = x1 + px2_c
                                    py2 = y1 + py2_c
                                    
                                    # The new model has 1 class (0: plate). Use Aspect Ratio and HSV color check.
                                    pw = px2 - px1
                                    ph = py2 - py1
                                    ar = pw / ph if ph > 0 else 1.0
                                    
                                    # Standard Vietnamese square plate is 280x200 (AR ~1.4), long plate is 470x110 (AR ~4.27).
                                    is_square = ar <= 1.7
                                    
                                    # Detect if yellow plate by checking color in HSV space on the plate crop
                                    is_yellow = False
                                    plate_crop_temp = frame[py1:py2, px1:px2]
                                    if plate_crop_temp.size > 0:
                                        try:
                                            hsv = cv2.cvtColor(plate_crop_temp, cv2.COLOR_BGR2HSV)
                                            # Yellow color range in HSV: Hue [10, 35], Saturation [50, 255], Value [50, 255]
                                            lower_yellow = np.array([10, 50, 50])
                                            upper_yellow = np.array([35, 255, 255])
                                            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                                            # If more than 15% of the plate is yellow, classify it as a yellow plate (bien vang)
                                            is_yellow = (np.count_nonzero(mask) / mask.size) > 0.15
                                        except Exception:
                                            pass
                                        
                                    # Determine display color and label
                                    plate_color = (0, 255, 255) if is_yellow else colors[4] # Yellow or Red box
                                    
                                    # Draw plate box
                                    cv2.rectangle(img, (px1, py1), (px2, py2), plate_color, 2)
                                    
                                    current_time = time.time()
                                    plate_no = None
                                    
                                    # Check tracking history cache first to completely eliminate text flickering
                                    if track_id is not None and tracking_history[track_id]["plate_text"] is not None:
                                        # Keep valid plates cached for 10.0 seconds, but retry "N/A" after 1.5 seconds
                                        cache_duration = 10.0 if tracking_history[track_id]["plate_text"] != "N/A" else 1.5
                                        if current_time - tracking_history[track_id]["plate_time"] < cache_duration:
                                            plate_no = tracking_history[track_id]["plate_text"]
                                            
                                    if plate_no is None:
                                        # Crop the plate from original frame for OCR
                                        plate_crop = frame[py1:py2, px1:px2]
                                        # Căn chỉnh góc nghiêng và nhận diện chữ bằng EasyOCR
                                        plate_no = perform_ocr_easyocr(plate_crop, is_square)
                                            
                                        # Tắt cơ chế giả lập biển số: hiển thị N/A nếu OCR thất bại
                                        if not plate_no:
                                            plate_no = "N/A"
                                            
                                        # Save to tracking history cache
                                        if track_id is not None:
                                            tracking_history[track_id]["plate_text"] = plate_no
                                            tracking_history[track_id]["plate_time"] = current_time
                                    
                                    # Draw plate text overlay
                                    plate_prefix = "Bien vang" if is_yellow else "Bien so"
                                    plate_label = f"{plate_prefix}: {plate_no}"
                                    cv2.putText(img, plate_label, (px1, max(py1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, plate_color, 1)
                                    counts[4] += 1
                                    
                                else:
                                    # Fallback geometry estimation for cars, trucks, buses if large enough
                                    if mapped_cls_id in [0, 2, 3] and w > 75 and h > 75:
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
                                            # Tắt cơ chế giả lập biển số cho ước lượng vùng biển: đặt mặc định là N/A
                                            plate_no = "N/A"
                                            if track_id is not None:
                                                tracking_history[track_id]["plate_text"] = plate_no
                                                tracking_history[track_id]["plate_time"] = current_time
                                                
                                        plate_label = f"Bien so: {plate_no}"
                                        cv2.putText(img, plate_label, (px1, max(py1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[4], 1)
                                        counts[4] += 1
            
            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Update stats
            current_stats[video_id] = {
                "car": counts[0],
                "motorcycle": counts[1],
                "truck": counts[2],
                "bus": counts[3],
                "license_plate": counts[4],
                "fps": round(fps, 1),
                "status": "active"
            }
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', img)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    except GeneratorExit:
        print(f"[*] Client disconnected from stream: {video_id}")
    finally:
        cap.release()
        if video_id in current_stats:
            del current_stats[video_id]

@app.route('/')
def index():
    return render_template('index.html', streams=STREAMS)

@app.route('/api/stream/<video_id>')
def stream(video_id):
    return Response(gen_frames(video_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats/<video_id>')
def stats(video_id):
    if video_id in current_stats:
        return jsonify(current_stats[video_id])
    else:
        return jsonify({
            "car": 0, "motorcycle": 0, "truck": 0, "bus": 0, "license_plate": 0, "fps": 0.0, "status": "inactive"
        })

if __name__ == "__main__":
    print("[*] Starting Flask web server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
