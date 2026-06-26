import cv2
import numpy as np
from src.utils.logger import logger

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
        logger.warning(f"Deskew failed: {e}")
    return plate_crop

def is_plate_yellow(plate_crop_temp):
    """Detect if yellow plate by checking color in HSV space on the plate crop."""
    if plate_crop_temp is None or plate_crop_temp.size == 0:
        return False
    try:
        hsv = cv2.cvtColor(plate_crop_temp, cv2.COLOR_BGR2HSV)
        # Yellow color range in HSV: Hue [10, 35], Saturation [50, 255], Value [50, 255]
        lower_yellow = np.array([10, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # If more than 15% of the plate is yellow, classify it as a yellow plate
        return (np.count_nonzero(mask) / mask.size) > 0.15
    except Exception:
        return False
