import os
import glob
import time
import cv2
import yt_dlp
import hashlib
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__, template_folder='templates')

# Force loading pretrained YOLO26 model to ensure real-time detections work perfectly out-of-the-box
model_path = "yolo26n.pt"
print(f"[*] Loading pretrained YOLO26 model: {model_path}")
model = YOLO(model_path)

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
    4: "Bien so xe"
}

# Shared statistics dictionary
current_stats = {}

# Direct HLS stream URL cache
url_cache = {}

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

def gen_frames(video_id):
    """Generate JPEG frames with YOLO26 detections and simulated license plate OCR."""
    global current_stats
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
                
            # Run YOLO26 inference
            # We use size 320 for extremely fast inference on CPU
            results = model(frame, verbose=False, imgsz=320)
            
            # Count current frame's classes
            counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            
            img = frame.copy()
            boxes = results[0].boxes
            for box in boxes:
                raw_cls_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                
                # Check confidence threshold and if the class is a vehicle we care about
                if confidence > 0.30 and raw_cls_id in COCO_MAP:
                    mapped_cls_id = COCO_MAP[raw_cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    color = colors.get(mapped_cls_id, (255, 255, 255))
                    
                    # Draw vehicle box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw vehicle label text in Vietnamese
                    class_name = CLASS_NAMES_VI.get(mapped_cls_id, "Khong xac dinh")
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(img, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if mapped_cls_id in counts:
                        counts[mapped_cls_id] += 1
                        
                    # Simulated License Plate detection and OCR for cars, trucks, and buses
                    if mapped_cls_id in [0, 2, 3]:
                        w = x2 - x1
                        h = y2 - y1
                        # Only draw license plate if the vehicle is large enough to see a plate
                        if w > 60 and h > 60:
                            plate_w = int(w * 0.28)
                            plate_h = int(h * 0.12)
                            plate_x = x1 + int(w * 0.5)
                            plate_y = y1 + int(h * 0.72)
                            
                            px1 = max(x1, plate_x - plate_w // 2)
                            py1 = max(y1, plate_y)
                            px2 = min(x2, plate_x + plate_w // 2)
                            py2 = min(y2, plate_y + plate_h)
                            
                            # Draw red plate box
                            cv2.rectangle(img, (px1, py1), (px2, py2), colors[4], 1)
                            
                            # Get a stable simulated VN plate number
                            plate_no = get_simulated_plate(x1, y1)
                            
                            # Draw plate text
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
