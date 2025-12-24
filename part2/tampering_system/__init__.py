"""
Aadhaar Tampering Detection System
===================================
Package initialization file.
"""

__version__ = "1.0.0"
__author__ = "Aadhaar Tampering Detection Team"

from .tamper_verifier import verify_tampering, quick_verify, verify_tampering_batch
from .visual_tamper_detector import detect_visual_tampering
from .ocr_extractor import extract_and_validate_ocr
from .qr_checker import check_qr_code

__all__ = [
    'verify_tampering',
    'quick_verify',
    'verify_tampering_batch',
    'detect_visual_tampering',
    'extract_and_validate_ocr',
    'check_qr_code'
]
