import time
import cv2
import yt_dlp
from pathlib import Path
from config import settings
from src.utils.logger import logger
from src.core.detector import vehicle_model, process_stream_detections

# Shared structures
current_stats = {}
url_cache = {}

def get_hls_url(video_id):
    """Retrieve HLS stream URL and cache it to speed up connection starts."""
    global url_cache
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
    project_root = Path(__file__).resolve().parent.parent.parent
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

def gen_frames(video_id):
    """Generate JPEG frames with YOLO26 detections and real-time VietOCR."""
    global current_stats
    logger.info(f"Starting AI stream generator for video: {video_id}")
    
    try:
        stream_url = get_hls_url(video_id)
    except Exception as e:
        logger.error(f"Error extracting URL for {video_id}: {e}")
        return
        
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        logger.error(f"Error opening stream for {video_id}")
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
                logger.warning(f"Failed to read frame from stream {video_id}")
                break
                
            # Run YOLO26 inference
            vehicle_results = vehicle_model.track(frame, persist=True, verbose=False, imgsz=320, tracker="bytetrack.yaml")
            
            # Count current frame's classes
            counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            img = frame.copy()
            
            # Process detections using modular core detector
            process_stream_detections(vehicle_results, img, frame, counts, colors, is_coco)
            
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
        logger.info(f"Client disconnected from stream: {video_id}")
    finally:
        cap.release()
        if video_id in current_stats:
            del current_stats[video_id]
