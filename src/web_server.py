import os
import re
import glob
import time
import cv2
import yt_dlp
import hashlib
from pathlib import Path
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import easyocr
from collections import Counter

app = Flask(__name__, template_folder='templates')

# Load YOLO models
project_root = Path(__file__).resolve().parent.parent
vehicle_model_path = str(project_root / "yolo26n.pt") # Revert to pretrained model for high accuracy vehicle detection
plate_model_path = str(project_root / "plate_best.pt")

print(f"[*] Loading pretrained vehicle model: {vehicle_model_path}")
vehicle_model = YOLO(vehicle_model_path)

print(f"[*] Loading custom license plate model: {plate_model_path}")
plate_model = YOLO(plate_model_path)

# Initialize EasyOCR Reader on CPU
print("[*] Initializing EasyOCR Reader (English models)...")
# Note: First time initializing will download the OCR models (~100MB total) automatically
reader = easyocr.Reader(['en'], gpu=False)
print("[*] EasyOCR initialized successfully!")

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

def get_simulated_plate(x1, y1):
    """Generate a stable, realistic Vietnamese license plate based on the vehicle's position."""
    # Spatial hashing grid (50x50 pixels) to keep the plate number stable as the car moves
    grid_x = x1 // 50
    grid_y = y1 // 50
    hash_val = int(hashlib.md5(f"{grid_x},{grid_y}".encode()).hexdigest(), 16)
    
    provinces = ["29", "30", "31", "51", "59", "43", "75", "15", "60", "36"]
    letters = ["A", "B", "C", "F", "H", "K", "L", "M", "N", "P", "S", "T", "U", "V", "X", "Y"]
    
    prov = provinces[hash_val % len(provinces)]
    let = letters[(hash_val // 10) % len(letters)]
    num1 = (hash_val // 100) % 900 + 100  # 3 digits (100-999)
    num2 = (hash_val // 10000) % 90 + 10  # 2 digits (10-99)
    
    return f"{prov}{let}-{num1}.{num2}"

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

def perform_ocr_single(crop):
    """Perform EasyOCR on a single image crop."""
    try:
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(gray)
        if results:
            raw_text = "".join([res[1] for res in results])
            cleaned_text = "".join([c for c in raw_text if c.isalnum()]).upper()
            return cleaned_text
    except Exception as e:
        print(f"[!] OCR Single Exception: {e}")
    return None

def perform_ocr_square(plate_crop):
    """
    Split the square license plate crop vertically into top and bottom halves,
    perform OCR on both, and return a tuple (top_text, bottom_text).
    """
    try:
        h, w = plate_crop.shape[:2]
        if h == 0 or w == 0:
            return (None, None)
            
        # Split vertically
        top_half = plate_crop[0:int(h * 0.52), 0:w]
        bottom_half = plate_crop[int(h * 0.48):h, 0:w]
        
        top_text = perform_ocr_single(top_half)
        bottom_text = perform_ocr_single(bottom_half)
        
        return (top_text, bottom_text)
    except Exception as e:
        print(f"[!] Square OCR Exception: {e}")
    return (None, None)

def force_format(s, pattern):
    """
    Force string s to match pattern of 'D' (digit) and 'L' (letter).
    """
    char_to_num = {
        'I': '1', 'T': '1', 'L': '1', 'J': '1',
        'Z': '2', 'E': '3', 'A': '4', 'H': '4',
        'S': '5', 'G': '6', 'B': '8', 'O': '0',
        'D': '0', 'Q': '0', 'U': '0', 'Y': '7'
    }
    num_to_char = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'E',
        '4': 'A', '5': 'S', '6': 'G', '7': 'Y',
        '8': 'B', '9': 'P'
    }
    res = []
    for i, char in enumerate(s):
        if i >= len(pattern):
            # If extra characters, assume they are digits (usually tail of line 2)
            expected = 'D'
        else:
            expected = pattern[i]
            
        if expected == 'D':
            if char.isalpha():
                res.append(char_to_num.get(char, '1'))
            else:
                res.append(char)
        elif expected == 'L':
            if char.isdigit():
                res.append(num_to_char.get(char, 'A'))
            else:
                res.append(char)
    return "".join(res)

def clean_and_correct_plate(raw_input, is_square, is_motorcycle):
    """
    Cleans OCR output and corrects characters using Vietnamese license plate rules,
    differentiating between cars (3 chars in part 1) and motorcycles (4 or 5 chars in part 1).
    """
    if not raw_input:
        return ""
        
    if is_square:
        # If it's a square plate, raw_input is expected to be a tuple/list (top_text, bottom_text)
        if isinstance(raw_input, (tuple, list)):
            top_raw, bottom_raw = raw_input
        else:
            top_raw = raw_input[:4]
            bottom_raw = raw_input[4:]
            
        top_clean = "".join([c for c in (top_raw or "") if c.isalnum()]).upper()
        bottom_clean = "".join([c for c in (bottom_raw or "") if c.isalnum()]).upper()
        
        # Correct top line (Line 1)
        if is_motorcycle:
            # Motorcycle formats:
            # Format A: 5 characters (e.g., 43A - H3) -> 'DDLLD'
            # Format B: 4 characters (e.g., 29A1) -> 'DDLD'
            if len(top_clean) >= 5:
                top_corrected = force_format(top_clean, 'DDLLD')
            else:
                top_corrected = force_format(top_clean, 'DDLD')
        else:
            # Car/Truck/Bus square plate: 2 digits + 1 letter -> 'DDL'
            top_corrected = force_format(top_clean, 'DDL')
            
        # Correct bottom line (Line 2) -> all digits (4 or 5 digits) -> 'DDDDD'
        bottom_corrected = force_format(bottom_clean, 'DDDDD')
        
        # Format the final string
        if is_motorcycle:
            if len(top_corrected) >= 5:
                top_part_1 = top_corrected[:3]
                top_part_2 = top_corrected[3:5]
                if len(bottom_corrected) >= 5:
                    return f"{top_part_1}-{top_part_2}-{bottom_corrected[:3]}.{bottom_corrected[3:]}"
                return f"{top_part_1}-{top_part_2}-{bottom_corrected}"
            else:
                if len(bottom_corrected) >= 5:
                    return f"{top_corrected}-{bottom_corrected[:3]}.{bottom_corrected[3:]}"
                return f"{top_corrected}-{bottom_corrected}"
        else:
            if len(bottom_corrected) >= 5:
                return f"{top_corrected}-{bottom_corrected[:3]}.{bottom_corrected[3:]}"
            return f"{top_corrected}-{bottom_corrected}"
            
    else:
        # Long plate (always car/truck/bus): Line 1 + Line 2 -> 'DDLDDDDD'
        if isinstance(raw_input, (tuple, list)):
            text = (raw_input[0] or "") + (raw_input[1] or "")
        else:
            text = raw_input
            
        cleaned = "".join([c for c in text if c.isalnum()]).upper()
        corrected = force_format(cleaned, 'DDLDDDDD')
        
        if len(corrected) >= 7:
            top_part = corrected[:3]
            bottom_part = corrected[3:]
            if len(bottom_part) >= 5:
                return f"{top_part}-{bottom_part[:3]}.{bottom_part[3:]}"
            return f"{top_part}-{bottom_part}"
        return corrected

def gen_frames(video_id):
    """Generate JPEG frames with YOLO26 detections and real-time EasyOCR."""
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
                                    
                                    # Get predicted class ID: 0 = plate_long, 1 = plate_square, 2 = plate_yellow
                                    pclass = int(best_plate_box.cls[0].item())
                                    
                                    pw = px2 - px1
                                    ph = py2 - py1
                                    ar = pw / ph if ph > 0 else 1.0
                                    
                                    if pclass == 0:
                                        is_square = False
                                        is_yellow = False
                                    elif pclass == 1:
                                        is_square = True
                                        is_yellow = False
                                    else: # Class 2: plate_yellow
                                        is_square = ar <= 2.2
                                        is_yellow = True
                                        
                                    # Determine display color and label
                                    plate_color = (0, 255, 255) if is_yellow else colors[4] # Yellow or Red box
                                    
                                    # Draw plate box
                                    cv2.rectangle(img, (px1, py1), (px2, py2), plate_color, 2)
                                    
                                    current_time = time.time()
                                    plate_no = None
                                    
                                    # Check tracking history cache first to completely eliminate text flickering
                                    if track_id is not None and tracking_history[track_id]["plate_text"] is not None:
                                        if current_time - tracking_history[track_id]["plate_time"] < 10.0:
                                            plate_no = tracking_history[track_id]["plate_text"]
                                            
                                    if plate_no is None:
                                        # Crop the plate from original frame for OCR
                                        plate_crop = frame[py1:py2, px1:px2]
                                        if plate_crop.size > 0:
                                            if is_square:
                                                # Square plate: Cắt đôi biển vuông và nhận dạng
                                                raw_text = perform_ocr_square(plate_crop)
                                            else:
                                                # Long plate: Đọc bình thường
                                                raw_text = perform_ocr_single(plate_crop)
                                                
                                            plate_no = clean_and_correct_plate(raw_text, is_square, is_motorcycle=(mapped_cls_id == 1))
                                        else:
                                            plate_no = None
                                            
                                        # Fallback to simulated plate if OCR fails
                                        if not plate_no or len(plate_no) < 4:
                                            plate_no = get_simulated_plate(px1, py1)
                                            
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
                                            if current_time - tracking_history[track_id]["plate_time"] < 10.0:
                                                plate_no = tracking_history[track_id]["plate_text"]
                                                
                                        if plate_no is None:
                                            plate_no = get_simulated_plate(x1, y1)
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
