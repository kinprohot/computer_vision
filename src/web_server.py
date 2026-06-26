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
vehicle_model_path = project_root / "vehicle_best.pt"
if not vehicle_model_path.exists():
    vehicle_model_path = project_root / "yolo26n.pt"
    print(f"[!] Custom vehicle model not found. Falling back to pretrained model: {vehicle_model_path}")
else:
    print(f"[*] Loading custom vehicle model: {vehicle_model_path}")
vehicle_model = YOLO(str(vehicle_model_path))

plate_model_path = project_root / "plate_best.pt"
if not plate_model_path.exists():
    # Try looking in runs directory
    run_weights = project_root / "runs" / "detect" / "yolo26_plate" / "plate_detector" / "weights" / "best.pt"
    if run_weights.exists():
        plate_model_path = run_weights
        print(f"[*] Custom license plate model not in root, loading from runs: {plate_model_path}")
    else:
        # Fallback to base model
        plate_model_path = project_root / "yolo26n.pt"
        print(f"[!] Custom license plate model not found. Falling back to base model: {plate_model_path}")
else:
    print(f"[*] Loading custom license plate model: {plate_model_path}")
plate_model = YOLO(str(plate_model_path))

# Try loading API key from .env file if it exists
env_path = Path(__file__).resolve().parent.parent / ".env"
if not env_path.exists():
    env_path = Path(".env")
if env_path.exists():
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str and not line_str.startswith("#") and "=" in line_str:
                    key, val = line_str.split("=", 1)
                    os.environ[key.strip()] = val.strip()
    except Exception as e_env:
        print(f"[!] Warning: Failed to parse .env file: {e_env}")

import google.generativeai as genai
import PIL.Image

# Initialize Gemini Model
gemini_api_key = os.environ.get("GEMINI_API_KEY")
model = None
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    # Use gemini-2.5-flash for fast vision tasks
    model = genai.GenerativeModel('gemini-2.5-flash')
    print("[*] Gemini VLM API initialized successfully!")
else:
    print("[!] WARNING: GEMINI_API_KEY is not set in environment or .env file. Cloud VLM OCR will fail until set.")


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
        'js_runtimes': {
            'node': {}
        },
        'remote_components': {'ejs:github'}
    }
    project_root = Path(__file__).resolve().parent.parent
    cookies_path = project_root / 'cookies.txt'
    if cookies_path.exists():
        ydl_opts['cookiefile'] = str(cookies_path)
    elif Path('cookies.txt').exists():
        ydl_opts['cookiefile'] = 'cookies.txt'

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

def _correct_bottom_line(chars, letter_to_digit):
    for i in range(len(chars)):
        if chars[i] in letter_to_digit:
            chars[i] = letter_to_digit[chars[i]]
    return chars

def _correct_top_line(chars, letter_to_digit, digit_to_letter):
    for i in range(min(2, len(chars))):
        if chars[i] in letter_to_digit:
            chars[i] = letter_to_digit[chars[i]]
    if len(chars) >= 3 and chars[2] in digit_to_letter:
        chars[2] = digit_to_letter[chars[2]]
    return chars

def _correct_standard_line(chars, letter_to_digit, digit_to_letter):
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
    return chars

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
        chars = _correct_bottom_line(chars, letter_to_digit)
    elif is_top_line:
        chars = _correct_top_line(chars, letter_to_digit, digit_to_letter)
    else:
        chars = _correct_standard_line(chars, letter_to_digit, digit_to_letter)
        
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

def perform_ocr(plate_crop, is_square):
    global model
    if model is None:
        return None
        
    try:
        if plate_crop is None or plate_crop.size == 0:
            return None
            
        # Preprocess plate crop (deskew)
        plate_crop = deskew_plate(plate_crop)
        
        # Convert BGR to RGB
        plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = PIL.Image.fromarray(plate_rgb)
        
        # Configure prompt
        prompt = (
            "Read the text on this Vietnamese license plate. "
            "Return ONLY the alphanumeric characters of the plate, formatted as standard plate text (e.g., 29A-123.45 or 30H-999.99). "
            "Do not include any other words, markdown, or punctuation unless it is part of the plate."
        )
        
        # Call Gemini API
        response = model.generate_content([prompt, pil_img])
        text = response.text.strip() if response.text else None
        if not text:
            return None
            
        # Clean text
        cleaned = text.replace("\n", "").replace(" ", "").upper().replace("`", "")
        
        # Format Vietnamese license plate structure
        return format_vietnamese_plate(cleaned)
    except Exception as e:
        print(f"[!] Gemini VLM OCR failed: {e}")
        return None

def _is_plate_yellow(plate_crop_temp):
    if plate_crop_temp.size == 0:
        return False
    try:
        hsv = cv2.cvtColor(plate_crop_temp, cv2.COLOR_BGR2HSV)
        # Yellow color range in HSV: Hue [10, 35], Saturation [50, 255], Value [50, 255]
        lower_yellow = np.array([10, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # If more than 15% of the plate is yellow, classify it as a yellow plate (bien vang)
        return (np.count_nonzero(mask) / mask.size) > 0.15
    except Exception:
        return False

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
    # Căn chỉnh góc nghiêng và nhận diện chữ bằng OCR (Paddle/EasyOCR)
    plate_no = perform_ocr(plate_crop, is_square)
        
    # Tắt cơ chế giả lập biển số: hiển thị N/A nếu OCR thất bại
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
    
    # The new model has 1 class (0: plate). Use Aspect Ratio and HSV color check.
    pw = px2 - px1
    ph = py2 - py1
    ar = pw / ph if ph > 0 else 1.0
    
    # Standard Vietnamese square plate is 280x200 (AR ~1.4), long plate is 470x110 (AR ~4.27).
    is_square = ar <= 1.7
    
    # Detect if yellow plate by checking color in HSV space on the plate crop
    plate_crop_temp = frame[py1:py2, px1:px2]
    is_yellow = _is_plate_yellow(plate_crop_temp)
        
    # Determine display color and label
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
        # Tắt cơ chế giả lập biển số cho ước lượng vùng biển: đặt mặc định là N/A
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
        
    # Run plate detector on crop (using imgsz=320 for better accuracy on details)
    plate_crop_results = plate_model(vehicle_crop, verbose=False, imgsz=320)
    plate_boxes = plate_crop_results[0].boxes
    
    best_plate_box = None
    best_plate_conf = 0.0
    for pbox in plate_boxes:
        pconf = pbox.conf[0].item()
        if pconf > best_plate_conf:
            best_plate_conf = pconf
            best_plate_box = pbox
            
    # If a license plate is found with confidence > 0.45
    if best_plate_box is not None and best_plate_conf > 0.45:
        _process_found_plate(best_plate_box, x1, y1, track_id, img, frame, counts, colors)
    else:
        # Fallback geometry estimation for cars, trucks, buses if large enough
        _process_fallback_plate(x1, y1, x2, y2, w, h, track_id, mapped_cls_id, img, counts, colors)

def _process_single_vehicle(box, track_id, img, frame, counts, colors, is_coco):
    raw_cls_id = int(box.cls[0].item())
    confidence = box.conf[0].item()
    
    is_vehicle = (raw_cls_id in COCO_MAP) if is_coco else (raw_cls_id in [0, 1, 2, 3])
    if confidence <= 0.30 or not is_vehicle:
        return
        
    mapped_cls_id = COCO_MAP[raw_cls_id] if is_coco else raw_cls_id
    
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
        _detect_license_plate(x1, y1, x2, y2, w, h, track_id, mapped_cls_id, img, frame, counts, colors)

def _process_stream_detections(vehicle_results, img, frame, counts, colors, is_coco):
    vehicle_boxes = vehicle_results[0].boxes
    if vehicle_boxes is not None and len(vehicle_boxes) > 0:
        if vehicle_boxes.id is not None:
            track_ids = vehicle_boxes.id.int().tolist()
        else:
            track_ids = [None] * len(vehicle_boxes)
            
        for idx, box in enumerate(vehicle_boxes):
            _process_single_vehicle(box, track_ids[idx], img, frame, counts, colors, is_coco)

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
    
    is_coco = (len(vehicle_model.names) > 10)
    
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
            _process_stream_detections(vehicle_results, img, frame, counts, colors, is_coco)
            
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
