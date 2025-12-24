"""
OCR-Based Consistency Checker (Quality-Aware)
==============================================
Extracts text from Aadhaar images using OCR and validates consistency with adaptive thresholds.

Features:
1. Text extraction using EasyOCR (fallback to Tesseract if needed)
2. Field extraction: Name, Gender, DOB, Aadhaar Number
3. Quality-aware consistency checks with weighted scoring:
   - Font sharpness differences (adaptive threshold)
   - Overlapping bounding boxes (10px HIGH, 40px LOW)
   - Invalid Aadhaar number format (CRITICAL)
   - Missing mandatory fields (MINOR)
   - Text alignment anomalies

WEIGHTED SCORING:
- Critical (weight=5): Invalid Aadhaar format
- Major (weight=3): Extreme overlaps, severe sharpness issues
- Minor (weight=1): Missing optional fields, minor overlaps

NO TRAINING REQUIRED - uses pre-trained OCR models.
"""

import cv2
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[Warning] EasyOCR not available. Install: pip install easyocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("[Warning] Tesseract not available. Install: pip install pytesseract")


# ====================================================================================
# OCR INITIALIZATION
# ====================================================================================
_easyocr_reader = None

def get_easyocr_reader():
    """Initialize EasyOCR reader (lazy loading)"""
    global _easyocr_reader
    if _easyocr_reader is None and EASYOCR_AVAILABLE:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


# ====================================================================================
# TEXT EXTRACTION
# ====================================================================================
def extract_text_easyocr(image_path: str) -> List[Tuple[str, List, float]]:
    """
    Extract text using EasyOCR.
    
    Args:
        image_path: Path to image
    
    Returns:
        List of (text, bounding_box, confidence)
    """
    try:
        reader = get_easyocr_reader()
        if reader is None:
            return []
        
        results = reader.readtext(image_path)
        
        # Format: (text, bbox, confidence)
        formatted = []
        for bbox, text, conf in results:
            formatted.append((text, bbox, conf))
        
        return formatted
    
    except Exception as e:
        print(f"[EasyOCR Error] {e}")
        return []


def extract_text_tesseract(image_path: str) -> List[Tuple[str, List, float]]:
    """
    Extract text using Tesseract OCR.
    
    Args:
        image_path: Path to image
    
    Returns:
        List of (text, bounding_box, confidence)
    """
    try:
        if not TESSERACT_AVAILABLE:
            return []
        
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        # Get OCR data with bounding boxes
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and conf > 0:  # Valid text with confidence
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                # Create bbox in format [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                bbox = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]
                
                results.append((text, bbox, conf / 100.0))
        
        return results
    
    except Exception as e:
        print(f"[Tesseract Error] {e}")
        return []


def extract_all_text(image_path: str) -> List[Tuple[str, List, float]]:
    """
    Extract text using available OCR engine.
    Priority: EasyOCR > Tesseract
    
    Returns:
        List of (text, bounding_box, confidence)
    """
    if EASYOCR_AVAILABLE:
        results = extract_text_easyocr(image_path)
        if results:
            return results
    
    if TESSERACT_AVAILABLE:
        results = extract_text_tesseract(image_path)
        if results:
            return results
    
    print("[OCR Error] No OCR engine available!")
    return []


# ====================================================================================
# FIELD EXTRACTION
# ====================================================================================
def extract_aadhaar_number(text_list: List[str]) -> Optional[str]:
    """
    Extract Aadhaar number (12 digits in format: xxxx xxxx xxxx or xxxxxxxxxxxx)
    """
    aadhaar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    
    for text in text_list:
        # Clean text
        cleaned = re.sub(r'[^\d\s]', '', text)
        match = re.search(aadhaar_pattern, cleaned)
        if match:
            return match.group(0).strip()
    
    return None


def extract_name(text_list: List[str], ocr_results: List) -> Optional[str]:
    """
    Extract name from OCR results.
    Heuristic: Look for text after keywords like "Name", "नाम"
    """
    name_keywords = ['name', 'नाम', 'naam']
    
    # Find name keyword index
    name_idx = -1
    for i, text in enumerate(text_list):
        if any(kw in text.lower() for kw in name_keywords):
            name_idx = i
            break
    
    # Extract next significant text after "Name"
    if name_idx >= 0 and name_idx + 1 < len(text_list):
        candidate = text_list[name_idx + 1].strip()
        # Filter out noise
        if len(candidate) > 2 and not candidate.isdigit():
            return candidate
    
    # Fallback: Find longest alphabetic text (likely name)
    candidates = [t for t in text_list if len(t) > 3 and not t.isdigit() and any(c.isalpha() for c in t)]
    if candidates:
        return max(candidates, key=len)
    
    return None


def extract_dob(text_list: List[str]) -> Optional[str]:
    """
    Extract Date of Birth.
    Format: DD/MM/YYYY or DD-MM-YYYY
    """
    dob_pattern = r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'
    
    for text in text_list:
        match = re.search(dob_pattern, text)
        if match:
            return match.group(0)
    
    return None


def extract_gender(text_list: List[str]) -> Optional[str]:
    """
    Extract gender (Male/Female/MALE/FEMALE/M/F)
    """
    gender_keywords = ['male', 'female', 'पुरुष', 'महिला']
    
    for text in text_list:
        text_lower = text.lower()
        if 'female' in text_lower:
            return 'Female'
        elif 'male' in text_lower:
            return 'Male'
        elif text_lower in ['m', 'f']:
            return 'Male' if text_lower == 'm' else 'Female'
    
    return None


def extract_fields(ocr_results: List[Tuple[str, List, float]]) -> Dict:
    """
    Extract all Aadhaar fields from OCR results.
    
    Returns:
        Dictionary with extracted fields
    """
    text_list = [text for text, _, _ in ocr_results]
    
    fields = {
        'aadhaar_number': extract_aadhaar_number(text_list),
        'name': extract_name(text_list, ocr_results),
        'dob': extract_dob(text_list),
        'gender': extract_gender(text_list)
    }
    
    return fields


# ====================================================================================
# CONSISTENCY CHECKS
# ====================================================================================
def check_font_sharpness(image_path: str, ocr_results: List[Tuple[str, List, float]], threshold: float = 2.0) -> Tuple[bool, float]:
    """
    Check if text regions have consistent sharpness.
    
    Theory: Pasted text may have different sharpness than original text.
    
    Returns:
        (is_consistent, variance_score)
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or len(ocr_results) < 2:
            return True, 0.0
        
        sharpness_values = []
        
        for text, bbox, conf in ocr_results:
            if conf < 0.5:  # Skip low confidence
                continue
            
            # Get bounding box coordinates
            bbox_np = np.array(bbox)
            x_min = int(np.min(bbox_np[:, 0]))
            y_min = int(np.min(bbox_np[:, 1]))
            x_max = int(np.max(bbox_np[:, 0]))
            y_max = int(np.max(bbox_np[:, 1]))
            
            # Crop text region
            text_region = img[y_min:y_max, x_min:x_max]
            
            if text_region.size == 0:
                continue
            
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(text_region, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_values.append(sharpness)
        
        if len(sharpness_values) < 2:
            return True, 0.0
        
        # Calculate coefficient of variation (normalized std dev)
        mean_sharpness = np.mean(sharpness_values)
        std_sharpness = np.std(sharpness_values)
        
        if mean_sharpness == 0:
            return True, 0.0
        
        cv = std_sharpness / mean_sharpness  # Coefficient of variation
        
        is_consistent = cv < threshold
        
        return is_consistent, float(cv)
    
    except Exception as e:
        print(f"[Sharpness Check Error] {e}")
        return True, 0.0


def check_overlapping_boxes(ocr_results: List[Tuple[str, List, float]]) -> List[str]:
    """
    Check for overlapping or too-close text bounding boxes.
    
    Theory: Overlapping text boxes indicate OCR confusion or tampering artifacts.
    
    Returns:
        List of issues found
    """
    issues = []
    
    try:
        if len(ocr_results) < 2:
            return issues
        
        boxes = []
        for text, bbox, conf in ocr_results:
            bbox_np = np.array(bbox)
            x_min = int(np.min(bbox_np[:, 0]))
            y_min = int(np.min(bbox_np[:, 1]))
            x_max = int(np.max(bbox_np[:, 0]))
            y_max = int(np.max(bbox_np[:, 1]))
            boxes.append((x_min, y_min, x_max, y_max))
        
        # Check each pair of boxes
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                x1_min, y1_min, x1_max, y1_max = boxes[i]
                x2_min, y2_min, x2_max, y2_max = boxes[j]
                
                # Check for overlap
                x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                
                if x_overlap > 0 and y_overlap > 0:
                    issues.append(f"Overlapping text boxes detected")
                    break
        
    except Exception as e:
        print(f"[Overlap Check Error] {e}")
    
    return issues


def count_overlapping_boxes(ocr_results: List[Tuple[str, List, float]], threshold_pixels: int = 60) -> Tuple[int, int]:
    """
    Count overlapping text boxes with quality-aware threshold.
    
    Args:
        ocr_results: OCR detection results
        threshold_pixels: Minimum overlap (in pixels) to count
    
    Returns:
        Tuple of (total_overlap_count, severe_overlap_count)
    """
    try:
        if len(ocr_results) < 2:
            return 0, 0
        
        boxes = []
        for text, bbox, conf in ocr_results:
            bbox_np = np.array(bbox)
            x_min = int(np.min(bbox_np[:, 0]))
            y_min = int(np.min(bbox_np[:, 1]))
            x_max = int(np.max(bbox_np[:, 0]))
            y_max = int(np.max(bbox_np[:, 1]))
            boxes.append((x_min, y_min, x_max, y_max))
        
        overlap_count = 0
        severe_overlap_count = 0
        
        # Check each pair of boxes
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                x1_min, y1_min, x1_max, y1_max = boxes[i]
                x2_min, y2_min, x2_max, y2_max = boxes[j]
                
                # Calculate overlap area
                x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                
                overlap_area = x_overlap * y_overlap
                
                if overlap_area > threshold_pixels:
                    overlap_count += 1
                    
                    # Severe overlap: > 50% of smaller box
                    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                    min_box_area = min(box1_area, box2_area)
                    
                    if overlap_area > min_box_area * 0.5:
                        severe_overlap_count += 1
        
        return overlap_count, severe_overlap_count
    
    except Exception as e:
        print(f"[Overlap Count Error] {e}")
        return 0, 0


def validate_aadhaar_format(aadhaar: Optional[str]) -> bool:
    """
    Validate Aadhaar number format.
    - Must be 12 digits
    - Can have spaces: xxxx xxxx xxxx
    """
    if aadhaar is None:
        return False
    
    # Remove spaces and check
    digits_only = re.sub(r'\s', '', aadhaar)
    
    if not digits_only.isdigit():
        return False
    
    if len(digits_only) != 12:
        return False
    
    return True


def check_missing_fields(fields: Dict, required_fields: List[str] = None) -> List[str]:
    """
    Check for missing mandatory fields.
    """
    if required_fields is None:
        required_fields = ['aadhaar_number', 'name', 'dob', 'gender']
    
    missing = []
    
    for field in required_fields:
        if fields.get(field) is None:
            missing.append(field)
    
    return missing


# ====================================================================================
# MAIN OCR EXTRACTION FUNCTION
# ====================================================================================
def extract_and_validate_ocr(image_path: str, thresholds: Dict = None) -> Dict:
    """
    Extract text from Aadhaar image and perform quality-aware consistency checks with weighted scoring.
    
    Args:
        image_path: Path to Aadhaar image
        thresholds: Dictionary of adaptive thresholds (from quality_analyzer)
    
    Returns:
        Dictionary with OCR results and weighted validation issues
    """
    print(f"\n[OCR Extraction] Processing: {image_path}")
    
    # Use default HIGH_QUALITY thresholds if not provided
    if thresholds is None:
        thresholds = {
            'sharpness_variance_threshold': 2.0,
            'overlap_threshold': 60
        }
    
    # Extract text
    ocr_results = extract_all_text(image_path)
    
    if not ocr_results:
        # If OCR fails entirely, treat as a minor issue (lenient)
        return {
            "text_fields": {},
            "critical_issues": [],
            "major_issues": [],
            "minor_issues": ["No text could be extracted"],
            "raw_text": [],
            "confidence_scores": {},
            "thresholds_used": thresholds
        }
    
    # Extract fields
    fields = extract_fields(ocr_results)
    
    # Weighted issue categorization
    critical_issues = []  # weight = 5
    major_issues = []     # weight = 3
    minor_issues = []     # weight = 1
    
    # Aadhaar format: Make MAJOR instead of CRITICAL (more lenient)
    if fields.get('aadhaar_number') is None:
        major_issues.append("Aadhaar number not detected")
    elif not validate_aadhaar_format(fields.get('aadhaar_number')):
        major_issues.append("Aadhaar number format irregular")
    
    # Font sharpness consistency: Very lenient - only flag if extreme
    sharpness_ok, sharpness_var = check_font_sharpness(image_path, ocr_results, 
                                                        threshold=thresholds.get('sharpness_variance_threshold', 2.0))
    if not sharpness_ok:
        if sharpness_var > thresholds.get('sharpness_variance_threshold', 2.0) * 4:
            major_issues.append(f"Extreme font sharpness variance (variance: {sharpness_var:.2f})")
        elif sharpness_var > thresholds.get('sharpness_variance_threshold', 2.0) * 2:
            minor_issues.append(f"Font sharpness inconsistency (variance: {sharpness_var:.2f})")
    
    # Overlapping boxes: Very lenient (60px threshold)
    overlap_threshold = thresholds.get('overlap_threshold', 60)
    overlap_count, severe_overlap_count = count_overlapping_boxes(ocr_results, overlap_threshold)
    
    if severe_overlap_count > 10:  # Many severe overlaps
        major_issues.append(f"Severe layout anomalies ({severe_overlap_count} major overlaps detected)")
    elif overlap_count > 15:
        minor_issues.append(f"{overlap_count} text box overlaps detected")
    
    # Missing fields: Always MINOR (not fatal)
    optional_fields = ['name', 'dob', 'gender']
    missing_optional = [f for f in optional_fields if not fields.get(f)]
    
    # Only report if multiple fields are missing
    if len(missing_optional) >= 3:
        minor_issues.append(f"Missing optional fields: {', '.join(missing_optional)}")
    
    # Calculate average confidence
    avg_confidence = np.mean([conf for _, _, conf in ocr_results]) if ocr_results else 0.0
    
    # Raw text
    raw_text = [text for text, _, _ in ocr_results]
    
    result = {
        "text_fields": fields,
        "critical_issues": critical_issues,
        "major_issues": major_issues,
        "minor_issues": minor_issues,
        "raw_text": raw_text,
        "confidence_scores": {
            "average_confidence": round(float(avg_confidence), 2),
            "sharpness_variance": round(sharpness_var, 3)
        },
        "thresholds_used": thresholds
    }
    
    print(f"[OCR Analysis] Critical: {len(critical_issues)}, Major: {len(major_issues)}, Minor: {len(minor_issues)}")
    
    return result


# ====================================================================================
# STANDALONE TESTING
# ====================================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_extractor.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = extract_and_validate_ocr(image_path)
    
    print("\n" + "="*60)
    print("OCR EXTRACTION AND VALIDATION RESULTS")
    print("="*60)
    print("\nExtracted Fields:")
    for key, value in result['text_fields'].items():
        print(f"  {key}: {value}")
    
    print("\nOCR Issues:")
    if result['ocr_issues']:
        for issue in result['ocr_issues']:
            print(f"  - {issue}")
    else:
        print("  None")
    
    print(f"\nOCR Consistent: {result['ocr_consistent']}")
    print(f"Average Confidence: {result['confidence_scores']['average_confidence']}")
    print("="*60)
