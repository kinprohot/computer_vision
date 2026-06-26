import os
import cv2
import PIL.Image
from pathlib import Path
import google.generativeai as genai
from src.utils.logger import logger

# Load .env file
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str and not line_str.startswith("#") and "=" in line_str:
                    key, val = line_str.split("=", 1)
                    os.environ[key.strip()] = val.strip()
    except Exception as e_env:
        logger.error(f"Failed to parse .env file: {e_env}")

# Initialize Gemini Model
gemini_api_key = os.environ.get("GEMINI_API_KEY")
model = None
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    logger.info("Gemini VLM API initialized successfully (using gemini-2.5-pro)!")
else:
    logger.warning("GEMINI_API_KEY is not set in environment or .env file. Cloud VLM OCR will fail until set.")

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

def perform_ocr(plate_crop, is_square=False):
    global model
    if model is None:
        return None
        
    try:
        if plate_crop is None or plate_crop.size == 0:
            return None
            
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
        logger.error(f"Gemini VLM OCR failed: {e}")
        return None
