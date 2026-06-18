import os
import cv2
import time
import argparse
import threading
from pathlib import Path
from ultralytics import YOLO
import yt_dlp

# YouTube streams from web_server.py
STREAMS = [
    {"id": "G_G8A6JU_LI", "title": "Camera 1"},
    {"id": "sJvEFrG0wq0", "title": "Camera 2"},
    {"id": "oif_zZFIfB4", "title": "Camera 3"},
    {"id": "1EamsYw_Xyo", "title": "Camera 4"},
    {"id": "NeJGBQAY-bE", "title": "Camera 5"},
    {"id": "x8tUUv-NGXs", "title": "Camera 6"}
]

class FreshFrameReader:
    """
    Thread-safe reader that continuously grabs frames from a live stream
    to prevent buffering latency. Returns the freshest frame upon request.
    """
    def __init__(self, stream_url):
        self.cap = cv2.VideoCapture(stream_url)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
    def _update(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    with self.lock:
                        self.ret = False
                    self.running = False
                    break
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            except Exception as e:
                print(f"[!] OpenCV error in stream grabber thread: {e}")
                with self.lock:
                    self.ret = False
                self.running = False
                break
            time.sleep(0.01) # Sleep slightly to avoid 100% CPU thread loop
            
    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return self.ret, self.frame.copy()
            
    def release(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()

def get_stream_info(video_id):
    """Retrieve direct HLS stream URL and determine if it's a live stream using yt-dlp."""
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': 'best',
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            stream_url = info.get('url')
            is_live = info.get('is_live', False) or (info.get('live_status') == 'is_live')
            return stream_url, is_live
    except Exception as e:
        print(f"[!] yt-dlp failed to get stream info for ID {video_id}: {e}")
        return None, False

def calculate_sharpness(img_crop):
    """Calculate the image sharpness using Laplacian variance."""
    if img_crop is None or img_crop.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception:
        return 0.0

def verify_plate_crop(context_img, plate_box, plate_model, filter_threshold=0.55):
    """
    Verify if the candidate plate box is indeed a license plate using the plate model.
    Pads the crop by 30% of its size to retain spatial context for the YOLO model.
    """
    if context_img is None or context_img.size == 0:
        return False
        
    try:
        px1, py1, px2, py2 = plate_box
        ph, pw = py2 - py1, px2 - px1
        
        # Add 30% padding around the plate candidate
        pad_y = int(ph * 0.3)
        pad_x = int(pw * 0.3)
        
        py1_pad = max(0, py1 - pad_y)
        py2_pad = min(context_img.shape[0], py2 + pad_y)
        px1_pad = max(0, px1 - pad_x)
        px2_pad = min(context_img.shape[1], px2 + pad_x)
        
        padded_crop = context_img[py1_pad:py2_pad, px1_pad:px2_pad]
        if padded_crop.size == 0:
            return False
            
        results = plate_model(padded_crop, verbose=False, imgsz=320)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                if cls == 0 and conf >= filter_threshold:
                    return True
    except Exception as e:
        print(f"[!] Verification failed: {e}")
    return False

def extract_plates_from_image(img, vehicle_model, plate_model, detect_type="cascade", detect_threshold=0.4, filter_threshold=0.55, min_sharpness=80.0):
    """
    Run detection on a single frame, crop plate candidates, verify using the filter model,
    check image sharpness (xử lý ảnh rõ nét), and return a list of verified clean plate crops.
    Returns list of tuples: (plate_crop, sharpness_value)
    """
    h_orig, w_orig = img.shape[:2]
    verified_plates = []
    candidates = [] # list of (px1, py1, px2, py2, context_img, vx1, vy1)
    
    if detect_type == "cascade":
        # Step 1: Detect vehicles
        vehicle_results = vehicle_model(img, verbose=False, imgsz=320)
        vehicle_boxes = vehicle_results[0].boxes
        
        if vehicle_boxes is not None:
            for box in vehicle_boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                if conf >= 0.35 and cls_id in [0, 1, 2, 3]:
                    vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())
                    vx1, vy1 = max(0, vx1), max(0, vy1)
                    vx2, vy2 = min(w_orig, vx2), min(h_orig, vy2)
                    
                    vw, vh = vx2 - vx1, vy2 - vy1
                    if vw > 40 and vh > 40:
                        vehicle_crop = img[vy1:vy2, vx1:vx2]
                        if vehicle_crop.size > 0:
                            # Step 2: Detect plates inside vehicle crop
                            plate_results = plate_model(vehicle_crop, verbose=False, imgsz=320)
                            plate_boxes = plate_results[0].boxes
                            
                            if plate_boxes is not None:
                                for pbox in plate_boxes:
                                    pconf = pbox.conf[0].item()
                                    if pconf >= detect_threshold:
                                        px1_c, py1_c, px2_c, py2_c = map(int, pbox.xyxy[0].tolist())
                                        candidates.append((px1_c, py1_c, px2_c, py2_c, vehicle_crop, vx1, vy1))
                                            
    elif detect_type == "plate_direct":
        plate_results = plate_model(img, verbose=False, imgsz=640)
        plate_boxes = plate_results[0].boxes
        
        if plate_boxes is not None:
            for pbox in plate_boxes:
                pconf = pbox.conf[0].item()
                if pconf >= detect_threshold:
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                    candidates.append((px1, py1, px2, py2, img, 0, 0))
                        
    elif detect_type == "vehicle_class4":
        vehicle_results = vehicle_model(img, verbose=False, imgsz=320)
        vehicle_boxes = vehicle_results[0].boxes
        
        if vehicle_boxes is not None:
            for box in vehicle_boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                if cls_id == 4 and conf >= detect_threshold:
                    px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
                    candidates.append((px1, py1, px2, py2, img, 0, 0))

    # Step 3: Run filtering model and sharpness checks on all candidate crops
    for px1_c, py1_c, px2_c, py2_c, context_img, offset_x, offset_y in candidates:
        if verify_plate_crop(context_img, (px1_c, py1_c, px2_c, py2_c), plate_model, filter_threshold):
            # Calculate global coordinates for final crop
            px1 = offset_x + px1_c
            py1 = offset_y + py1_c
            px2 = offset_x + px2_c
            py2 = offset_y + py2_c
            
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w_orig, px2), min(h_orig, py2)
            
            plate_crop = img[py1:py2, px1:px2]
            if plate_crop.size > 0:
                # Sharpness check
                sharpness_val = calculate_sharpness(plate_crop)
                if sharpness_val >= min_sharpness:
                    verified_plates.append((plate_crop, sharpness_val))
                else:
                    print(f"      [Candidate Rejected] Blurry plate image (Sharpness: {sharpness_val:.1f} < {min_sharpness})")
            
    return verified_plates

def run_plate_extraction(source="youtube", detect_type="cascade", detect_threshold=0.35, filter_threshold=0.5, 
                         max_frames=200, interval_seconds=10.0, min_sharpness=80.0):
    """
    Main runner function.
    Loads models, loops through sources, and extracts sharp verified plate crops.
    """
    project_root = Path(__file__).resolve().parent.parent
    
    # 1. Load YOLO models
    print("[*] Loading models...")
    vehicle_model_path = project_root / "vehicle_best.pt"
    if not vehicle_model_path.exists():
        vehicle_model_path = project_root / "yolo26n.pt"
        print(f"[!] Custom vehicle model not found. Using default: {vehicle_model_path}")
    else:
        print(f"[*] Loaded vehicle detector model: {vehicle_model_path}")
    vehicle_model = YOLO(str(vehicle_model_path))
    
    plate_model_path = project_root / "plate_best.pt"
    if not plate_model_path.exists():
        plate_model_path = project_root / "runs" / "detect" / "yolo26_plate" / "plate_detector" / "weights" / "best.pt"
    if not plate_model_path.exists():
        plate_model_path = project_root / "yolo26n.pt"
        print(f"[!] Custom plate model not found. Using default: {plate_model_path}")
    else:
        print(f"[*] Loaded plate detector model: {plate_model_path}")
    plate_model = YOLO(str(plate_model_path))
    
    # Define output directories
    output_dir = project_root / "runs" / "detect" / "extracted_plates"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # ==================== PROCESS DATASET IMAGES ====================
    if source in ["dataset", "both"]:
        print("\n=== Processing Dataset Images ===")
        dataset_path = project_root / "dataset"
        if not dataset_path.exists():
            print(f"[!] Dataset path does not exist: {dataset_path}")
        else:
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(dataset_path.rglob(ext))
                
            total_images = len(image_paths)
            print(f"[*] Found {total_images} images in dataset.")
            
            saved_count = 0
            for idx, img_path in enumerate(image_paths, 1):
                parent_name = img_path.parent.name
                grandparent_name = img_path.parent.parent.name
                
                if parent_name in ["train", "val", "test"]:
                    split_out_dir = output_dir / "dataset" / parent_name
                elif grandparent_name in ["train", "val", "test"]:
                    split_out_dir = output_dir / "dataset" / grandparent_name
                else:
                    split_out_dir = output_dir / "dataset"
                    
                split_out_dir.mkdir(parents=True, exist_ok=True)
                
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                verified_plates = extract_plates_from_image(
                    img, vehicle_model, plate_model, 
                    detect_type=detect_type, 
                    detect_threshold=detect_threshold, 
                    filter_threshold=filter_threshold,
                    min_sharpness=min_sharpness
                )
                
                for p_idx, (plate, sharp_val) in enumerate(verified_plates):
                    filename = f"plate_{img_path.stem}_{p_idx}_s{int(sharp_val)}.jpg"
                    cv2.imwrite(str(split_out_dir / filename), plate)
                    saved_count += 1
                    
                if idx % 10 == 0 or idx == total_images:
                    print(f"[*] Dataset progress: {idx}/{total_images} images processed. Saved {saved_count} plates so far.")
            
            print(f"[✓] Finished Dataset: Processed {total_images} images. Extracted & saved {saved_count} verified plates.")
            
    # ==================== PROCESS YOUTUBE CAMERA STREAMS ====================
    if source in ["youtube", "both"]:
        print("\n=== Processing YouTube Camera Streams ===")
        youtube_out_dir = output_dir / "youtube"
        youtube_out_dir.mkdir(parents=True, exist_ok=True)
        
        for camera in STREAMS:
            cam_title = camera["title"].replace(" ", "_")
            cam_id = camera["id"]
            cam_out_dir = youtube_out_dir / cam_title
            cam_out_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n[*] Connecting to {camera['title']} (ID: {cam_id})...")
            stream_url, is_live = get_stream_info(cam_id)
            if not stream_url:
                print(f"[!] Could not retrieve stream URL for {camera['title']}. Skipping.")
                continue
                
            print(f"[*] Stream Type: {'LIVE' if is_live else 'VOD'}")
            
            if is_live:
                # For LIVE streams, read frames in real-time to avoid lag
                print(f"[*] Starting background frame grabber for live stream: {camera['title']}")
                reader = FreshFrameReader(stream_url)
                # Wait 2 seconds to initialize thread and buffer first frame
                time.sleep(2.0)
                
                processed_count = 0
                cam_saved_count = 0
                
                print(f"[*] Processing {camera['title']}: capturing {max_frames} frames with {interval_seconds}s intervals...")
                
                while processed_count < max_frames:
                    success, frame = reader.read()
                    if not success:
                        print(f"[!] Stream ended or frame grabber failed for {camera['title']}.")
                        break
                        
                    verified_plates = extract_plates_from_image(
                        frame, vehicle_model, plate_model, 
                        detect_type=detect_type, 
                        detect_threshold=detect_threshold, 
                        filter_threshold=filter_threshold,
                        min_sharpness=min_sharpness
                    )
                    
                    for p_idx, (plate, sharp_val) in enumerate(verified_plates):
                        filename = f"plate_{cam_title}_cap{processed_count}_{p_idx}_s{int(sharp_val)}.jpg"
                        cv2.imwrite(str(cam_out_dir / filename), plate)
                        cam_saved_count += 1
                        print(f"      [Plate Saved] {filename} (Sharpness: {sharp_val:.1f})")
                        
                    processed_count += 1
                    if processed_count % 5 == 0 or processed_count == max_frames:
                        print(f"    - {camera['title']}: processed {processed_count}/{max_frames} frames. Saved {cam_saved_count} plates.")
                        
                    # Sleep for the interval before capturing the next frame
                    if processed_count < max_frames:
                        time.sleep(interval_seconds)
                        
                reader.release()
                print(f"[✓] Finished LIVE {camera['title']}: processed {processed_count} frames, saved {cam_saved_count} verified plates.")
                
            else:
                # For VOD (offline videos), fast-forward by seeking to skip waiting in real-time!
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    print(f"[!] Could not open VOD VideoCapture for {camera['title']}. Skipping.")
                    continue
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                    
                frame_step = int(fps * interval_seconds)
                processed_count = 0
                cam_saved_count = 0
                frame_idx = 0
                
                print(f"[*] Processing VOD {camera['title']}: capturing {max_frames} frames, jumping {interval_seconds}s ({frame_step} frames) forward each step...")
                
                while processed_count < max_frames:
                    if frame_idx > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        
                    success, frame = cap.read()
                    if not success:
                        break
                        
                    verified_plates = extract_plates_from_image(
                        frame, vehicle_model, plate_model, 
                        detect_type=detect_type, 
                        detect_threshold=detect_threshold, 
                        filter_threshold=filter_threshold,
                        min_sharpness=min_sharpness
                    )
                    
                    for p_idx, (plate, sharp_val) in enumerate(verified_plates):
                        filename = f"plate_{cam_title}_f{frame_idx}_{p_idx}_s{int(sharp_val)}.jpg"
                        cv2.imwrite(str(cam_out_dir / filename), plate)
                        cam_saved_count += 1
                        print(f"      [Plate Saved] {filename} (Sharpness: {sharp_val:.1f})")
                        
                    processed_count += 1
                    frame_idx += frame_step
                    
                    if processed_count % 10 == 0 or processed_count == max_frames:
                        print(f"    - {camera['title']}: processed {processed_count}/{max_frames} frames. Saved {cam_saved_count} plates.")
                        
                cap.release()
                print(f"[✓] Finished VOD {camera['title']}: processed {processed_count} frames, saved {cam_saved_count} verified plates.")
            
    total_time = time.time() - start_time
    print(f"\n[✓] All tasks finished in {total_time:.2f} seconds.")
    print(f"[*] Saved license plate crops folder: {output_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect, crop, and verify sharp license plates from YouTube camera streams.")
    parser.add_argument("--source", type=str, default="youtube", choices=["dataset", "youtube", "both"], help="Source type to process.")
    parser.add_argument("--detect_type", type=str, default="cascade", choices=["cascade", "plate_direct", "vehicle_class4"], help="License plate detection strategy.")
    parser.add_argument("--detect_threshold", type=float, default=0.35, help="Confidence threshold for candidate plate detection.")
    parser.add_argument("--filter_threshold", type=float, default=0.5, help="Confidence threshold for verifying if crop is plate.")
    parser.add_argument("--max_frames", type=int, default=200, help="Maximum number of frames to process per YouTube stream.")
    parser.add_argument("--interval", type=float, default=10.0, help="Interval in seconds between captured frames.")
    parser.add_argument("--min_sharpness", type=float, default=80.0, help="Minimum sharpness score using Laplacian variance to accept crop.")
    
    args = parser.parse_args()
    
    run_plate_extraction(
        source=args.source,
        detect_type=args.detect_type,
        detect_threshold=args.detect_threshold,
        filter_threshold=args.filter_threshold,
        max_frames=args.max_frames,
        interval_seconds=args.interval,
        min_sharpness=args.min_sharpness
    )
