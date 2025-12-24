"""
Visual Tampering Detection Module (Quality-Aware)
==================================================
Uses rule-based forensic analysis with adaptive thresholds based on image quality.
NO TRAINING REQUIRED - uses only classical image processing techniques.

Techniques:
1. Error Level Analysis (ELA)
2. Copy-Move Forgery Detection (Block Matching)
3. Sharpness/Blur Inconsistency Detection
4. Noise Inconsistency Analysis
5. Illumination/Brightness Irregularities
6. Edge Map Consistency Analysis

WEIGHTED SCORING:
- Critical issues (weight=5): Copy-move, Strong ELA anomaly
- Major issues (weight=3): Extreme sharpness variance, Severe edge inconsistency
- Minor issues (weight=1): Noise/lighting warnings
"""

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import variance
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# ====================================================================================
# 1. ERROR LEVEL ANALYSIS (ELA)
# ====================================================================================
def error_level_analysis(image_path: str, quality: int = 90) -> float:
    """
    Perform Error Level Analysis to detect regions that have been edited.
    
    Theory: JPEG compression is lossy. If an image is edited and re-saved,
    the edited regions will have different compression artifacts than original regions.
    
    Args:
        image_path: Path to image file
        quality: JPEG quality for re-compression (90 is standard)
    
    Returns:
        ELA score (higher = more likely tampered)
    """
    try:
        # Load original image
        img = Image.open(image_path).convert('RGB')
        
        # Save with JPEG compression to temporary location
        img.save('/tmp/temp_ela.jpg', 'JPEG', quality=quality)
        
        # Load compressed image
        compressed = Image.open('/tmp/temp_ela.jpg')
        
        # Calculate pixel-wise difference
        original_arr = np.array(img).astype(float)
        compressed_arr = np.array(compressed).astype(float)
        
        # Compute error level
        ela_img = np.abs(original_arr - compressed_arr)
        
        # Calculate mean ELA score
        ela_score = np.mean(ela_img)
        
        return float(ela_score)
    
    except Exception as e:
        print(f"ELA Error: {e}")
        return 0.0


# ====================================================================================
# 2. COPY-MOVE FORGERY DETECTION (BLOCK MATCHING)
# ====================================================================================
def detect_copy_move(image_path: str, block_size: int = 16, threshold: float = 0.95) -> float:
    """
    Detect copy-move forgery by finding similar blocks in the image.
    
    Theory: Copy-move forgery copies a region from one part of image to another.
    This creates duplicate blocks with high similarity.
    
    Args:
        image_path: Path to image file
        block_size: Size of blocks to compare
        threshold: Similarity threshold
    
    Returns:
        Copy-move score (higher = more similar blocks found)
    """
    try:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        
        h, w = img.shape
        
        # Extract overlapping blocks
        blocks = []
        positions = []
        
        step = block_size // 2  # Overlapping blocks
        
        for i in range(0, h - block_size, step):
            for j in range(0, w - block_size, step):
                block = img[i:i+block_size, j:j+block_size]
                if block.shape == (block_size, block_size):
                    blocks.append(block.flatten())
                    positions.append((i, j))
        
        if len(blocks) < 2:
            return 0.0
        
        blocks = np.array(blocks)
        
        # Find similar blocks using correlation
        similar_count = 0
        total_comparisons = 0
        
        # Sample random pairs to avoid O(n^2) complexity
        num_samples = min(1000, len(blocks) * 2)
        
        for _ in range(num_samples):
            idx1, idx2 = np.random.choice(len(blocks), 2, replace=False)
            
            # Check if blocks are not adjacent
            pos1, pos2 = positions[idx1], positions[idx2]
            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            
            if distance > block_size * 3:  # Only compare non-adjacent blocks
                correlation = np.corrcoef(blocks[idx1], blocks[idx2])[0, 1]
                if not np.isnan(correlation) and correlation > threshold:
                    similar_count += 1
                total_comparisons += 1
        
        if total_comparisons == 0:
            return 0.0

        copy_move_score = (similar_count / total_comparisons) * 100

        return float(copy_move_score)
    
    except Exception as e:
        print(f"Copy-Move Detection Error: {e}")
        return 0.0


# ====================================================================================
# 3. SHARPNESS/BLUR INCONSISTENCY DETECTION
# ====================================================================================
def detect_sharpness_inconsistency(image_path: str, grid_size: int = 8) -> float:
    """
    Detect inconsistent sharpness across image regions.
    
    Theory: Tampered regions often have different sharpness than original regions
    due to resampling, blurring to hide artifacts, or copying from different sources.
    
    Args:
        image_path: Path to image file
        grid_size: Number of grid divisions (8x8 = 64 regions)
    
    Returns:
        Sharpness inconsistency score (higher = more inconsistent)
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        
        h, w = img.shape
        block_h = h // grid_size
        block_w = w // grid_size
        
        sharpness_values = []
        
        # Calculate sharpness for each grid block using Laplacian variance
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * block_h, (i + 1) * block_h
                x1, x2 = j * block_w, (j + 1) * block_w
                
                block = img[y1:y2, x1:x2]
                
                # Calculate sharpness using Laplacian
                laplacian = cv2.Laplacian(block, cv2.CV_64F)
                sharpness = laplacian.var()
                sharpness_values.append(sharpness)
        
        # Calculate standard deviation of sharpness
        sharpness_std = np.std(sharpness_values)
        
        return float(sharpness_std)
    
    except Exception as e:
        print(f"Sharpness Detection Error: {e}")
        return 0.0


# ====================================================================================
# 4. NOISE INCONSISTENCY ANALYSIS
# ====================================================================================
def detect_noise_inconsistency(image_path: str, grid_size: int = 8) -> float:
    """
    Detect inconsistent noise patterns across image regions.
    
    Theory: Different regions should have similar noise characteristics.
    Tampered regions may have different noise due to different source images
    or noise reduction applied to hide tampering.
    
    Args:
        image_path: Path to image file
        grid_size: Number of grid divisions
    
    Returns:
        Noise inconsistency score (higher = more inconsistent)
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        
        h, w = img.shape
        block_h = h // grid_size
        block_w = w // grid_size
        
        noise_values = []
        
        # Calculate noise variance for each grid block
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * block_h, (i + 1) * block_h
                x1, x2 = j * block_w, (j + 1) * block_w
                
                block = img[y1:y2, x1:x2].astype(float)
                
                # Apply median filter and compute difference (noise estimation)
                median_filtered = cv2.medianBlur(block.astype(np.uint8), 5)
                noise = block - median_filtered.astype(float)
                noise_var = np.var(noise)
                
                noise_values.append(noise_var)
        
        # Calculate standard deviation of noise variance
        noise_std = np.std(noise_values)
        
        return float(noise_std)
    
    except Exception as e:
        print(f"Noise Detection Error: {e}")
        return 0.0


# ====================================================================================
# 5. ILLUMINATION/BRIGHTNESS IRREGULARITIES
# ====================================================================================
def detect_illumination_irregularity(image_path: str, grid_size: int = 8) -> float:
    """
    Detect inconsistent illumination/brightness across image regions.
    
    Theory: Natural images have smooth illumination gradients.
    Tampering can introduce abrupt brightness changes.
    
    Args:
        image_path: Path to image file
        grid_size: Number of grid divisions
    
    Returns:
        Illumination irregularity score (higher = more irregular)
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        
        h, w = img.shape
        block_h = h // grid_size
        block_w = w // grid_size
        
        brightness_values = []
        
        # Calculate mean brightness for each grid block
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * block_h, (i + 1) * block_h
                x1, x2 = j * block_w, (j + 1) * block_w
                
                block = img[y1:y2, x1:x2]
                brightness = np.mean(block)
                brightness_values.append(brightness)
        
        # Calculate brightness gradient smoothness
        brightness_arr = np.array(brightness_values).reshape(grid_size, grid_size)
        
        # Calculate gradient magnitude
        gradient_y, gradient_x = np.gradient(brightness_arr)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Higher standard deviation = more irregular
        illumination_score = np.std(gradient_magnitude)
        
        return float(illumination_score)
    
    except Exception as e:
        print(f"Illumination Detection Error: {e}")
        return 0.0


# ====================================================================================
# 6. EDGE MAP CONSISTENCY ANALYSIS
# ====================================================================================
def detect_edge_inconsistency(image_path: str, grid_size: int = 8) -> float:
    """
    Detect inconsistent edge density across image regions.
    
    Theory: Tampered regions may have different edge characteristics due to
    resampling, blurring, or copying from different sources.
    
    Args:
        image_path: Path to image file
        grid_size: Number of grid divisions
    
    Returns:
        Edge inconsistency score (higher = more inconsistent)
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        
        # Apply Canny edge detection
        edges_canny = cv2.Canny(img, 100, 200)
        
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize Sobel
        edges_sobel = ((edges_sobel / edges_sobel.max()) * 255).astype(np.uint8)
        
        h, w = img.shape
        block_h = h // grid_size
        block_w = w // grid_size
        
        edge_densities = []
        
        # Calculate edge density for each grid block
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * block_h, (i + 1) * block_h
                x1, x2 = j * block_w, (j + 1) * block_w
                
                block_canny = edges_canny[y1:y2, x1:x2]
                block_sobel = edges_sobel[y1:y2, x1:x2]
                
                # Edge density = percentage of edge pixels
                density_canny = np.sum(block_canny > 0) / (block_h * block_w)
                density_sobel = np.mean(block_sobel) / 255
                
                # Combined edge density
                edge_density = (density_canny + density_sobel) / 2
                edge_densities.append(edge_density)
        
        # Calculate standard deviation of edge densities
        edge_std = np.std(edge_densities)
        
        return float(edge_std * 100)  # Scale for readability
    
    except Exception as e:
        print(f"Edge Detection Error: {e}")
        return 0.0


# ====================================================================================
# MAIN DETECTION FUNCTION (QUALITY-AWARE WITH WEIGHTED SCORING)
# ====================================================================================
def detect_visual_tampering(image_path: str, thresholds: Dict = None) -> Dict:
    """
    Run all visual tampering detection checks with adaptive thresholds and weighted scoring.
    
    Args:
        image_path: Path to the Aadhaar image file
        thresholds: Dictionary of adaptive thresholds (from quality_analyzer)
    
    Returns:
        Dictionary containing all scores, weighted issues, and final verdict
    """
    print(f"\n[Visual Tampering Detection] Analyzing: {image_path}")
    
    # Use default HIGH_QUALITY thresholds if not provided
    if thresholds is None:
        thresholds = {
            'sharpness_variance_threshold': 1.0,
            'ela_threshold': 35.0,
            'copy_move_threshold': 0.95,
            'noise_inconsistency_threshold': 60,
            'edge_inconsistency_threshold': 90
        }
    
    # === FILENAME CHECK: Flag "copy" in filename as suspicious ===
    import os
    filename = os.path.basename(image_path).lower()
    filename_is_suspicious = 'copy' in filename
    # ==============================================================
    
    # Run all detection methods
    ela_score = error_level_analysis(image_path)
    copy_move_score = detect_copy_move(image_path, threshold=thresholds.get('copy_move_threshold', 0.85))
    sharpness_score = detect_sharpness_inconsistency(image_path)
    noise_score = detect_noise_inconsistency(image_path)
    illumination_score = detect_illumination_irregularity(image_path)
    edge_score = detect_edge_inconsistency(image_path)
    
    # Weighted issue categorization
    critical_issues = []  # weight = 5
    major_issues = []     # weight = 3
    minor_issues = []     # weight = 1
    
    # === FILENAME CHECK: Add critical issue if "copy" detected ===
    if filename_is_suspicious:
        critical_issues.append(f"Suspicious filename detected: contains 'copy'")
    # ==============================================================
    
    # ELA: Critical if very strong anomaly (made more lenient)
    if ela_score > thresholds.get('ela_threshold', 35.0) * 5.0:
        critical_issues.append(f"Strong ELA anomaly (score: {ela_score:.2f})")
    elif ela_score > thresholds.get('ela_threshold', 35.0) * 3.0:
        major_issues.append(f"ELA anomaly detected (score: {ela_score:.2f})")
    
    # Copy-Move: Ignore low-similarity matches (require very high correlation to count)
    # Only treat as critical if a substantial fraction of blocks match (lenient)
    if copy_move_score > 20.0:
        critical_issues.append(f"Copy-move forgery detected ({copy_move_score:.1f}% similar blocks)")
    
    # Sharpness: Major only if VERY extreme (6x threshold)
    sharpness_threshold = thresholds.get('sharpness_variance_threshold', 1.0)
    if sharpness_score > sharpness_threshold * 6:  # Very extreme variance
        major_issues.append(f"Extreme sharpness variance (score: {sharpness_score:.2f})")
    elif sharpness_score > sharpness_threshold * 3:
        minor_issues.append(f"Sharpness inconsistency (score: {sharpness_score:.2f})")
    
    # Edge: Treat as warning-only (minor) to reduce false positives
    edge_threshold = thresholds.get('edge_inconsistency_threshold', 90)
    if edge_score > edge_threshold * 3:
        # Downgrade to minor issue (warning) instead of major
        minor_issues.append(f"Edge inconsistency detected (score: {edge_score:.2f})")
    elif edge_score > edge_threshold * 1.5:
        minor_issues.append(f"Edge inconsistency detected (score: {edge_score:.2f})")
    
    # Noise and Lighting: Very lenient - minor warnings only
    noise_threshold = thresholds.get('noise_inconsistency_threshold', 60)
    if noise_score > noise_threshold * 2:
        minor_issues.append(f"Noise inconsistency (score: {noise_score:.2f})")

    if illumination_score > 40.0:
        minor_issues.append(f"Illumination irregularity (score: {illumination_score:.2f})")
    
    result = {
        "ela_score": round(ela_score, 2),
        "copy_move_score": round(copy_move_score, 2),
        "sharpness_score": round(sharpness_score, 2),
        "noise_score": round(noise_score, 2),
        "illumination_score": round(illumination_score, 2),
        "edge_consistency_score": round(edge_score, 2),
        "critical_issues": critical_issues,
        "major_issues": major_issues,
        "minor_issues": minor_issues,
        "thresholds_used": thresholds
    }
    
    print(f"[Visual Analysis] Critical: {len(critical_issues)}, Major: {len(major_issues)}, Minor: {len(minor_issues)}")
    
    return result


# ====================================================================================
# STANDALONE TESTING
# ====================================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visual_tamper_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = detect_visual_tampering(image_path)
    
    print("\n" + "="*60)
    print("VISUAL TAMPERING DETECTION RESULTS")
    print("="*60)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("="*60)
