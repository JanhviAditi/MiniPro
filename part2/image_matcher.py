"""
STRICT Image-Based Aadhaar Verification System
===============================================
Compares uploaded Aadhaar card images with reference database using:
1. SSIM (Structural Similarity) - STRICT: ≥ 0.92
2. ORB Feature Matching - STRICT: ≥ 120 matches
3. HSV Histogram Correlation - STRICT: ≥ 0.90

ALL THREE metrics must pass for MATCH_FOUND, otherwise FORGED.
Returns match status: MATCH_FOUND or FORGED (NO leniency)
"""

import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


# ====================================================================================
# STRICT CONFIGURATION - NO LENIENCY
# ====================================================================================

# ULTRA-STRICT Thresholds - PIXEL-PERFECT MATCH REQUIRED
THRESHOLDS = {
    'ssim': 0.9999,      # ULTRA-STRICT: 99.99% similarity (near-pixel-perfect)
    'orb': 500,          # ULTRA-STRICT: 500+ perfect matches required
    'histogram': 0.9999, # ULTRA-STRICT: 99.99% color match required
    'pixel': 0.9999      # ULTRA-STRICT: 99.99% identical pixels required
}

# ORB Configuration - Maximum feature count for pixel-level precision
ORB_CONFIG = {
    'nfeatures': 10000,  # Maximum features for ultra-precision
    'scaleFactor': 1.05, # Very fine-grained scale detection
    'edgeThreshold': 3   # Ultra-strict edge detection
}

# Image preprocessing settings - MINIMAL processing for pixel-perfect comparison
RESIZE_WIDTH = None  # NO RESIZE - preserve exact dimensions for pixel-perfect matching
REFERENCE_DB_PATH = os.path.join(os.path.dirname(__file__), 'reference_db')


# ====================================================================================
# IMAGE PREPROCESSING - STRICT NORMALIZATION
# ====================================================================================

def preprocess_image(img_path: str) -> Optional[np.ndarray]:
    """
    ULTRA-STRICT: Load image with MINIMAL preprocessing.
    - NO RESIZE - preserve exact pixel dimensions
    - Convert to RGB only
    - NO blur, NO filtering - preserve every single pixel
    
    Args:
        img_path: Path to image file
        
    Returns:
        Image with minimal processing or None if error
    """
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Convert BGR to RGB ONLY - absolutely no other modifications
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image {img_path}: {e}")
        return None


# ====================================================================================
# METRIC 1: STRUCTURAL SIMILARITY (SSIM) - STRICT ≥ 0.92
# ====================================================================================

def compute_ssim(img1_path: str, img2_path: str) -> float:
    """
    STRICT Structural Similarity Index (SSIM) calculation.
    
    Requirements: SSIM ≥ 0.92 for MATCH_FOUND, else FORGED.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        
    Returns:
        SSIM score (float), 0.0 on error
    """
    try:
        # Preprocess both images
        img1 = preprocess_image(img1_path)
        img2 = preprocess_image(img2_path)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # ULTRA-STRICT: Dimensions must match EXACTLY
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape
        
        if h1 != h2 or w1 != w2:
            # Different dimensions = FORGED (NO resizing allowed)
            return 0.0
        
        # Calculate SSIM with full precision
        score = ssim(gray1, gray2, full=True)[0]
        
        return float(score)
        
    except Exception as e:
        print(f"[SSIM] Error: {e}")
        return 0.0


# ====================================================================================
# METRIC 2: ORB FEATURE MATCHING
# ====================================================================================

def compute_orb_similarity(img1_path: str, img2_path: str) -> int:
    """
    STRICT ORB (Oriented FAST and Rotated BRIEF) feature matching.
    
    Configuration:
    - nfeatures = 5000
    - scaleFactor = 1.1
    - edgeThreshold = 5
    
    Requirements: ≥ 120 good matches for MATCH_FOUND, else FORGED.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        
    Returns:
        Number of good matches (int), 0 on error
    """
    try:
        # Preprocess both images
        img1 = preprocess_image(img1_path)
        img2 = preprocess_image(img2_path)
        
        if img1 is None or img2 is None:
            return 0
        
        # Convert to grayscale for ORB
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Initialize ORB detector with STRICT parameters
        orb = cv2.ORB_create(
            nfeatures=ORB_CONFIG['nfeatures'],
            scaleFactor=ORB_CONFIG['scaleFactor'],
            edgeThreshold=ORB_CONFIG['edgeThreshold']
        )
        
        # Detect keypoints and compute descriptors
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return 0
        
        # Use BFMatcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort matches by distance (lower is better)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # ULTRA-STRICT: Filter near-perfect matches only (distance < 5)
        good_matches = [m for m in matches if m.distance < 5]
        
        return len(good_matches)
        
    except Exception as e:
        print(f"[ORB] Error: {e}")
        return 0


# ====================================================================================
# METRIC 3: COLOR HISTOGRAM SIMILARITY
# ====================================================================================

def compute_histogram_similarity(img1_path: str, img2_path: str) -> float:
    """
    STRICT HSV color histogram correlation.
    
    Compares color distribution using HSV color space with Gaussian blur.
    
    Requirements: Correlation ≥ 0.90 for MATCH_FOUND, else FORGED.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        
    Returns:
        Histogram correlation score (float), 0.0 on error
    """
    try:
        # Preprocess both images
        img1 = preprocess_image(img1_path)
        img2 = preprocess_image(img2_path)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Convert RGB to HSV
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
        
        # Calculate 3D histograms for H, S, V channels
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Calculate correlation using HISTCMP_CORREL method
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return float(correlation)
        
    except Exception as e:
        print(f"[Histogram] Error: {e}")
        return 0.0


# ====================================================================================
# METRIC 4: PIXEL-PERFECT COMPARISON
# ====================================================================================

def compute_pixel_similarity(img1_path: str, img2_path: str) -> float:
    """
    ULTRA-STRICT pixel-by-pixel exact comparison.
    
    Compares every single pixel - requires 99.99% identical pixels.
    Even a single changed pixel dramatically reduces the score.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        
    Returns:
        Pixel similarity ratio (0.0 to 1.0), 0.0 on error or dimension mismatch
    """
    try:
        # Load images without any preprocessing
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Dimensions must match EXACTLY
        if img1.shape != img2.shape:
            return 0.0
        
        # Pixel-by-pixel comparison
        total_pixels = img1.size
        diff = cv2.absdiff(img1, img2)
        identical_pixels = np.sum(diff == 0)
        
        # Calculate exact similarity ratio
        similarity = identical_pixels / total_pixels
        
        return float(similarity)
        
    except Exception as e:
        print(f"[PIXEL] Error: {e}")
        return 0.0


# ====================================================================================
# DECISION LOGIC
# ====================================================================================

def classify_match(ssim_score: float, orb_matches: int, hist_score: float, pixel_score: float) -> str:
    """
    ULTRA-STRICT match classification - PIXEL-PERFECT REQUIRED.
    
    ALL FOUR metrics must meet ultra-strict thresholds for MATCH_FOUND.
    If even ONE metric fails, result is FORGED.
    
    ULTRA-STRICT Requirements:
    - SSIM ≥ 0.9999 (99.99% structural similarity)
    - ORB ≥ 500 (500+ near-perfect feature matches)
    - Histogram ≥ 0.9999 (99.99% color match)
    - Pixel ≥ 0.9999 (99.99% identical pixels)
    
    Args:
        ssim_score: SSIM similarity score
        orb_matches: Number of ORB feature matches
        hist_score: Histogram correlation score
        pixel_score: Pixel-by-pixel similarity
        
    Returns:
        Match status: MATCH_FOUND or FORGED
    """
    # ALL FOUR must pass ultra-strict thresholds
    ssim_pass = ssim_score >= THRESHOLDS['ssim']
    orb_pass = orb_matches >= THRESHOLDS['orb']
    hist_pass = hist_score >= THRESHOLDS['histogram']
    pixel_pass = pixel_score >= THRESHOLDS['pixel']
    
    # ULTRA-STRICT: All four must pass
    if ssim_pass and orb_pass and hist_pass and pixel_pass:
        return "MATCH_FOUND"
    else:
        return "FORGED"


# ====================================================================================
# MAIN MATCHING FUNCTION
# ====================================================================================

def match_with_reference(image_path: str, verbose: bool = True) -> Dict:
    """
    STRICT matching against reference database.
    
    Returns best match with STRICT evaluation - ALL metrics must pass.
    
    Args:
        image_path: Path to uploaded Aadhaar card image
        verbose: Print detailed progress
        
    Returns:
        Dictionary with strict match results:
        {
            'match_status': 'MATCH_FOUND' | 'FORGED',
            'best_match_file': str,
            'scores': {
                'ssim': float,
                'orb': int,
                'hist': float
            }
        }
    """
    if verbose:
        print(f"\n{'='*70}")
        print("[STRICT IMAGE MATCHER] Starting reference database comparison")
        print(f"{'='*70}")
    
    # Check if uploaded image exists
    if not os.path.exists(image_path):
        return {
            "match_status": "FORGED",
            "best_match_file": None,
            "scores": {"ssim": 0.0, "orb": 0, "hist": 0.0}
        }
    
    # Check if reference database exists
    if not os.path.exists(REFERENCE_DB_PATH):
        return {
            "match_status": "FORGED",
            "best_match_file": None,
            "scores": {"ssim": 0.0, "orb": 0, "hist": 0.0}
        }
    
    # Get all reference images
    reference_files = [
        f for f in os.listdir(REFERENCE_DB_PATH)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if not reference_files:
        return {
            "match_status": "FORGED",
            "best_match_file": None,
            "scores": {"ssim": 0.0, "orb": 0, "hist": 0.0}
        }
    
    if verbose:
        print(f"[ULTRA-STRICT MATCHER] Found {len(reference_files)} reference images")
        print(f"[ULTRA-STRICT MATCHER] Uploaded image: {os.path.basename(image_path)}")
        print(f"[ULTRA-STRICT MATCHER] PIXEL-PERFECT THRESHOLDS:")
        print(f"  - SSIM ≥ {THRESHOLDS['ssim']} (99.99% structural match)")
        print(f"  - ORB ≥ {THRESHOLDS['orb']} (500+ perfect matches)")
        print(f"  - Histogram ≥ {THRESHOLDS['histogram']} (99.99% color match)")
        print(f"  - Pixel ≥ {THRESHOLDS['pixel']} (99.99% identical pixels)")
        print()
    
    # Compare with each reference image
    best_match = None
    best_score_sum = -1
    best_scores = None
    
    for idx, ref_file in enumerate(reference_files, 1):
        ref_path = os.path.join(REFERENCE_DB_PATH, ref_file)
        
        if verbose and idx % 10 == 0:
            print(f"[{idx}/{len(reference_files)}] Processing...")
        
        # Compute all FOUR metrics (including pixel-perfect)
        ssim_score = compute_ssim(image_path, ref_path)
        orb_matches = compute_orb_similarity(image_path, ref_path)
        hist_score = compute_histogram_similarity(image_path, ref_path)
        pixel_score = compute_pixel_similarity(image_path, ref_path)
        
        # Track best match (highest combined score)
        score_sum = ssim_score + (orb_matches / 500.0) + hist_score + pixel_score
        if score_sum > best_score_sum:
            best_score_sum = score_sum
            best_match = ref_file
            best_scores = {
                "ssim": round(ssim_score, 6),
                "orb": orb_matches,
                "hist": round(hist_score, 6),
                "pixel": round(pixel_score, 6)
            }
    
    # Determine overall match status using ULTRA-STRICT rules
    if best_match and best_scores:
        match_status = classify_match(
            best_scores["ssim"], 
            best_scores["orb"], 
            best_scores["hist"],
            best_scores["pixel"]
        )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"[ULTRA-STRICT MATCHER] Best match: {best_match}")
            print(f"[ULTRA-STRICT MATCHER] Scores:")
            print(f"  SSIM = {best_scores['ssim']} (need ≥{THRESHOLDS['ssim']})")
            print(f"  ORB = {best_scores['orb']} (need ≥{THRESHOLDS['orb']})")
            print(f"  Hist = {best_scores['hist']} (need ≥{THRESHOLDS['histogram']})")
            print(f"  Pixel = {best_scores['pixel']} (need ≥{THRESHOLDS['pixel']})")
            print(f"[ULTRA-STRICT MATCHER] Result: {match_status}")
            print(f"{'='*70}\n")
        
        return {
            "match_status": match_status,
            "best_match_file": best_match,
            "scores": best_scores
        }
    else:
        return {
            "match_status": "FORGED",
            "best_match_file": None,
            "scores": {"ssim": 0.0, "orb": 0, "hist": 0.0, "pixel": 0.0}
        }


# ====================================================================================
# STANDALONE TESTING
# ====================================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_matcher.py <image_path>")
        print("\nExample:")
        print("  python image_matcher.py path/to/aadhaar.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Run matching
    result = match_with_reference(image_path, verbose=True)
    
    # Print JSON result
    import json
    print("\n" + "="*70)
    print("JSON RESULT")
    print("="*70)
    print(json.dumps(result, indent=2))
    print("="*70)
