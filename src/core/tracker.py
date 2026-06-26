from collections import Counter

# Tracking history database
# Format: { track_id: { "class_history": [...], "box_history": [...], "plate_text": "...", "plate_time": 0.0 } }
tracking_history = {}

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
