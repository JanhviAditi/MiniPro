"""
QR Code Checker (Best-Effort)
==============================
Extracts and validates QR code from Aadhaar images.

Features:
1. QR code detection and extraction using zxing-cpp
2. Secure QR decoding using pyaadhaar library
3. Cross-validation with OCR data
4. Graceful failure handling (no crashes if QR unavailable)

IMPORTANT: This module is OPTIONAL and best-effort.
If QR cannot be decoded, it returns qr_available=False without failing.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try importing zxing-cpp
try:
    import zxingcpp
    ZXING_AVAILABLE = True
except ImportError:
    ZXING_AVAILABLE = False
    print("[Warning] zxing-cpp not available. Install: pip install zxing-cpp")

# Try importing pyaadhaar
try:
    from pyaadhaar.decode import AadhaarSecureQr
    PYAADHAAR_AVAILABLE = True
except (ImportError, OSError) as e:
    PYAADHAAR_AVAILABLE = False
    print(f"[Warning] pyaadhaar not available or has dependency issues. QR decoding will be limited.")


# ====================================================================================
# QR CODE EXTRACTION
# ====================================================================================
def extract_qr_code_zxing(image_path: str) -> Optional[str]:
    """
    Extract QR code from image using zxing-cpp.
    
    Args:
        image_path: Path to image
    
    Returns:
        QR code raw data string, or None if not found
    """
    try:
        if not ZXING_AVAILABLE:
            return None
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to read QR code
        results = zxingcpp.read_barcodes(gray)
        
        if results:
            # Return first QR code found
            for result in results:
                if result.format.name == 'QRCode':
                    return result.text
        
        return None
    
    except Exception as e:
        print(f"[QR Extraction Error] {e}")
        return None


def extract_qr_code_opencv(image_path: str) -> Optional[str]:
    """
    Fallback: Extract QR code using OpenCV's QRCodeDetector.
    
    Args:
        image_path: Path to image
    
    Returns:
        QR code raw data string, or None if not found
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Initialize QR code detector
        qr_detector = cv2.QRCodeDetector()
        
        # Detect and decode
        data, bbox, straight_qr = qr_detector.detectAndDecode(img)
        
        if data:
            return data
        
        return None
    
    except Exception as e:
        print(f"[OpenCV QR Error] {e}")
        return None


def extract_qr_code(image_path: str) -> Optional[str]:
    """
    Extract QR code using available method.
    Priority: zxing-cpp > OpenCV
    
    Returns:
        QR code data or None
    """
    # Try zxing-cpp first (better for Aadhaar QR)
    if ZXING_AVAILABLE:
        qr_data = extract_qr_code_zxing(image_path)
        if qr_data:
            return qr_data
    
    # Fallback to OpenCV
    qr_data = extract_qr_code_opencv(image_path)
    return qr_data


# ====================================================================================
# AADHAAR SECURE QR DECODING
# ====================================================================================
def decode_aadhaar_qr(qr_data: str) -> Optional[Dict]:
    """
    Decode Aadhaar Secure QR using pyaadhaar library.
    
    Args:
        qr_data: Raw QR code data string
    
    Returns:
        Dictionary with decoded Aadhaar data, or None if decoding fails
    """
    try:
        if not PYAADHAAR_AVAILABLE:
            return None
        
        # Decode secure QR
        secure_qr = AadhaarSecureQr(qr_data)
        
        # Extract demographic data
        decoded_data = {
            'uid': secure_qr.aadhaar_number if hasattr(secure_qr, 'aadhaar_number') else None,
            'name': secure_qr.name if hasattr(secure_qr, 'name') else None,
            'dob': secure_qr.dob if hasattr(secure_qr, 'dob') else None,
            'gender': secure_qr.gender if hasattr(secure_qr, 'gender') else None,
            'address': secure_qr.address if hasattr(secure_qr, 'address') else None,
            'care_of': secure_qr.care_of if hasattr(secure_qr, 'care_of') else None,
            'house': secure_qr.house if hasattr(secure_qr, 'house') else None,
            'street': secure_qr.street if hasattr(secure_qr, 'street') else None,
            'landmark': secure_qr.landmark if hasattr(secure_qr, 'landmark') else None,
            'locality': secure_qr.locality if hasattr(secure_qr, 'locality') else None,
            'district': secure_qr.district if hasattr(secure_qr, 'district') else None,
            'state': secure_qr.state if hasattr(secure_qr, 'state') else None,
            'pincode': secure_qr.pincode if hasattr(secure_qr, 'pincode') else None,
            'post_office': secure_qr.post_office if hasattr(secure_qr, 'post_office') else None,
            'email': secure_qr.email_mobile_status if hasattr(secure_qr, 'email_mobile_status') else None,
            'photo_base64': secure_qr.image if hasattr(secure_qr, 'image') else None
        }
        
        # Check if signature is valid
        is_valid = secure_qr.verify_signature() if hasattr(secure_qr, 'verify_signature') else True
        decoded_data['signature_valid'] = is_valid
        
        return decoded_data
    
    except Exception as e:
        print(f"[QR Decode Error] {e}")
        return None


# ====================================================================================
# CROSS-VALIDATION WITH OCR
# ====================================================================================
def normalize_text(text: Optional[str]) -> str:
    """Normalize text for comparison"""
    if text is None:
        return ""
    return text.lower().strip().replace(" ", "")


def compare_fields(qr_data: Dict, ocr_data: Dict) -> Tuple[bool, list]:
    """
    Compare QR data with OCR data to detect mismatches.
    
    Args:
        qr_data: Decoded QR data
        ocr_data: OCR extracted data
    
    Returns:
        (matches: bool, mismatches: List[str])
    """
    mismatches = []
    
    # Compare Aadhaar number
    qr_uid = normalize_text(qr_data.get('uid'))
    ocr_uid = normalize_text(ocr_data.get('aadhaar_number'))
    
    if qr_uid and ocr_uid and qr_uid != ocr_uid:
        mismatches.append(f"Aadhaar number mismatch: QR={qr_uid[:4]}..., OCR={ocr_uid[:4]}...")
    
    # Compare name
    qr_name = normalize_text(qr_data.get('name'))
    ocr_name = normalize_text(ocr_data.get('name'))
    
    if qr_name and ocr_name:
        # Check if names have significant overlap (at least 70%)
        if qr_name not in ocr_name and ocr_name not in qr_name:
            # Calculate similarity
            common = sum(1 for a, b in zip(qr_name, ocr_name) if a == b)
            similarity = common / max(len(qr_name), len(ocr_name))
            
            if similarity < 0.7:
                mismatches.append(f"Name mismatch: QR={qr_data.get('name')}, OCR={ocr_data.get('name')}")
    
    # Compare DOB
    qr_dob = normalize_text(qr_data.get('dob'))
    ocr_dob = normalize_text(ocr_data.get('dob'))
    
    if qr_dob and ocr_dob and qr_dob != ocr_dob:
        mismatches.append(f"DOB mismatch: QR={qr_data.get('dob')}, OCR={ocr_data.get('dob')}")
    
    # Compare gender
    qr_gender = normalize_text(qr_data.get('gender'))
    ocr_gender = normalize_text(ocr_data.get('gender'))
    
    if qr_gender and ocr_gender:
        # Handle M/F vs Male/Female
        if 'm' in qr_gender:
            qr_gender_norm = 'male'
        elif 'f' in qr_gender:
            qr_gender_norm = 'female'
        else:
            qr_gender_norm = qr_gender
        
        if 'm' in ocr_gender:
            ocr_gender_norm = 'male'
        elif 'f' in ocr_gender:
            ocr_gender_norm = 'female'
        else:
            ocr_gender_norm = ocr_gender
        
        if qr_gender_norm != ocr_gender_norm:
            mismatches.append(f"Gender mismatch: QR={qr_data.get('gender')}, OCR={ocr_data.get('gender')}")
    
    matches = len(mismatches) == 0
    
    return matches, mismatches


# ====================================================================================
# MAIN QR CHECKING FUNCTION
# ====================================================================================
def check_qr_code(image_path: str, ocr_data: Optional[Dict] = None) -> Dict:
    """
    Extract and validate QR code from Aadhaar image (OPTIONAL - not required for valid documents).
    
    IMPORTANT: Missing QR does NOT indicate tampering. QR is optional.
    Only QR mismatch (if QR exists) or invalid signature is a CRITICAL issue.
    
    Args:
        image_path: Path to Aadhaar image
        ocr_data: Optional OCR extracted data for cross-validation
    
    Returns:
        Dictionary with weighted QR analysis (critical, major, minor issues)
    """
    print(f"\n[QR Checker] Processing: {image_path}")
    
    # Initialize result with weighted issues
    result = {
        "qr_available": False,
        "qr_data": None,
        "qr_matches_ocr": None,
        "critical_issues": [],  # weight = 5
        "major_issues": [],     # weight = 3
        "minor_issues": [],     # weight = 1
        "signature_valid": None
    }
    
    # Check if libraries are available
    if not ZXING_AVAILABLE and not cv2:
        result["minor_issues"].append("QR extraction library unavailable (non-critical)")
        print("[QR Checker] No QR extraction library available")
        return result
    
    # Step 1: Extract QR code
    qr_raw_data = extract_qr_code(image_path)
    
    if qr_raw_data is None:
        # IMPORTANT: Missing QR is NOT an issue - many genuine Aadhaar cards don't have QR
        print("[QR Checker] No QR code found (this is OK - QR is optional)")
        return result
    
    result["qr_available"] = True
    print("[QR Checker] QR code extracted successfully")
    
    # Step 2: Decode Aadhaar Secure QR
    if not PYAADHAAR_AVAILABLE:
        result["minor_issues"].append("QR decode library unavailable (non-critical)")
        print("[QR Checker] pyaadhaar not available, cannot decode")
        return result
    
    decoded_data = decode_aadhaar_qr(qr_raw_data)
    
    if decoded_data is None:
        result["minor_issues"].append("QR code present but decode failed")
        print("[QR Checker] QR decode failed")
        return result
    
    result["qr_data"] = decoded_data
    result["signature_valid"] = decoded_data.get('signature_valid', None)
    
    # Check signature validity - CRITICAL if invalid
    if result["signature_valid"] is False:
        result["critical_issues"].append("QR signature validation failed (UIDAI signature invalid)")
    
    print(f"[QR Checker] QR decoded successfully, Signature Valid: {result['signature_valid']}")
    
    # Step 3: Cross-validate with OCR data - CRITICAL if mismatch
    if ocr_data is not None and ocr_data.get('text_fields'):
        matches, mismatches = compare_fields(decoded_data, ocr_data['text_fields'])
        result["qr_matches_ocr"] = matches
        
        if not matches:
            # QR-OCR mismatch is CRITICAL (suggests data manipulation)
            for mismatch in mismatches:
                result["critical_issues"].append(f"QR-OCR mismatch: {mismatch}")
            print(f"[QR Checker] QR-OCR mismatch detected: {len(mismatches)} critical issues")
        else:
            print("[QR Checker] QR data matches OCR data")
    
    return result


# ====================================================================================
# STANDALONE TESTING
# ====================================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python qr_checker.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = check_qr_code(image_path)
    
    print("\n" + "="*60)
    print("QR CODE CHECK RESULTS")
    print("="*60)
    print(f"QR Available: {result['qr_available']}")
    print(f"Signature Valid: {result['signature_valid']}")
    
    if result['qr_data']:
        print("\nDecoded QR Data:")
        for key, value in result['qr_data'].items():
            if key != 'photo_base64':  # Skip photo for display
                print(f"  {key}: {value}")
    
    print("\nQR Issues:")
    if result['qr_issues']:
        for issue in result['qr_issues']:
            print(f"  - {issue}")
    else:
        print("  None")
    
    print("="*60)
