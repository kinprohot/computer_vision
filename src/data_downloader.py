import os
import cv2
import yt_dlp
import time
import random
from pathlib import Path

def get_live_stream_url(youtube_url):
    """Get direct HLS stream URL (.m3u8) from YouTube Live Stream without downloading the file."""
    print(f"Analyzing YouTube link: {youtube_url}...")
    ydl_opts = {
        'format': 'best',  # Get best quality available
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        stream_url = info.get('url')
        is_live = info.get('is_live', False)
        print(f"Video type: {'LIVE (Streaming)' if is_live else 'VOD (Standard Video)'}")
        return stream_url, is_live

def download_youtube_video(url, output_path="temp_video.mp4"):
    """Download standard YouTube video (VOD) to disk."""
    print(f"Downloading standard video from: {url}...")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Download completed! Saved to: {output_path}")
    return output_path

def capture_from_live_stream(stream_url, output_dir, capture_duration_seconds=300, interval_seconds=2.0, val_ratio=0.2):
    """Read HLS Live stream URL and capture frames in real-time."""
    train_img_dir = Path(output_dir) / "images" / "train"
    val_img_dir = Path(output_dir) / "images" / "val"
    
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "labels" / "val").mkdir(parents=True, exist_ok=True)

    print("Connecting to live stream using OpenCV...")
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Error: Could not connect to the live stream. Please verify the URL.")
        return

    print(f"Successfully connected! Capturing one frame every {interval_seconds} seconds for {capture_duration_seconds} seconds...")
    print("Press 'q' on the display window or press Ctrl+C in terminal to stop capture manually.")

    start_time = time.time()
    last_saved_time = 0
    saved_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame or stream disconnected.")
                break
                
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check duration limit
            if elapsed_time > capture_duration_seconds:
                print(f"Duration limit reached ({capture_duration_seconds}s). Stopping capture.")
                break
                
            # Capture frame based on interval
            if current_time - last_saved_time >= interval_seconds:
                is_val = random.random() < val_ratio
                dest_dir = val_img_dir if is_val else train_img_dir
                
                # Use timestamp for unique file names
                timestamp = int(current_time * 1000)
                img_name = f"live_frame_{timestamp}.jpg"
                img_path = dest_dir / img_name
                
                cv2.imwrite(str(img_path), frame)
                saved_count += 1
                last_saved_time = current_time
                print(f"[{int(elapsed_time)}s / {capture_duration_seconds}s] Saved frame {saved_count}: {img_name} to {'validation' if is_val else 'training'} set")

            # Show stream preview (can be disabled in headless servers)
            try:
                cv2.imshow('YouTube Live Stream - Press q to quit', cv2.resize(frame, (854, 480)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User requested stop.")
                    break
            except cv2.error:
                # If running on headless environment (no GUI window), cv2.imshow will fail
                pass
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Capture finished. Total frames saved: {saved_count}")

def extract_frames_from_offline_video(video_path, output_dir, interval_seconds=2.0, val_ratio=0.2):
    """Extract frames from offline video file."""
    train_img_dir = Path(output_dir) / "images" / "train"
    val_img_dir = Path(output_dir) / "images" / "val"
    
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "labels" / "val").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        print("Error: Could not read video file.")
        return

    frame_step = int(fps * interval_seconds)
    frame_count = 0
    saved_count = 0

    print(f"Extracting frames from {video_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_step == 0:
            is_val = random.random() < val_ratio
            dest_dir = val_img_dir if is_val else train_img_dir
            img_name = f"{Path(video_path).stem}_frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(dest_dir / img_name), frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"Extraction completed. Total frames saved: {saved_count}")

if __name__ == "__main__":
    # User provided live stream URL
    youtube_url = "https://www.youtube.com/live/sJvEFrG0wq0?si=c1WR9fRq6lkxdGYJ"
    dataset_root = "dataset"
    
    stream_url, is_live = get_live_stream_url(youtube_url)
    
    if is_live:
        # For live stream, capture for 5 minutes (300 seconds)
        # Capture a frame every 2 seconds
        capture_from_live_stream(
            stream_url=stream_url,
            output_dir=dataset_root,
            capture_duration_seconds=300,
            interval_seconds=2.0,
            val_ratio=0.2
        )
    else:
        # For offline video
        video_file = "temp_video.mp4"
        if not os.path.exists(video_file):
            download_youtube_video(youtube_url, video_file)
        extract_frames_from_offline_video(video_file, dataset_root, interval_seconds=2.0, val_ratio=0.2)
