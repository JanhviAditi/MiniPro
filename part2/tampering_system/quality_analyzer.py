"""
Image Quality Analyzer
======================
Analyzes image quality based on blur, noise, and lighting to determine
adaptive thresholds for tampering detection.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


def variance_of_laplacian(image: np.ndarray) -> float:
    """
    Compute the Laplacian of the image and return the variance.
    Higher values indicate sharper images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def estimate_noise(image: np.ndarray) -> float:
    """
    Estimate noise level using high-frequency components.
    Uses standard deviation of the detail coefficients.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply high-pass filter to extract noise
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    high_freq = cv2.filter2D(gray, -1, kernel)
    
    return np.std(high_freq)


def analyze_lighting(image: np.ndarray) -> Dict[str, float]:
    """
    Analyze lighting quality based on brightness range and color uniformity.
    """
    # Convert to LAB color space for better luminance analysis
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Brightness metrics
    brightness_mean = np.mean(l_channel)
    brightness_std = np.std(l_channel)
    brightness_range = np.ptp(l_channel)  # peak-to-peak (max - min)
    
    # Color uniformity (lower std = more uniform)
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    color_std = (np.std(a_channel) + np.std(b_channel)) / 2
    
    return {
        'brightness_mean': float(brightness_mean),
        'brightness_std': float(brightness_std),
        'brightness_range': float(brightness_range),
        'color_uniformity': float(color_std)
    }


def classify_image_quality(image_path: str, verbose: bool = False) -> Tuple[str, Dict]:
    """
    Classify image quality as HIGH_QUALITY or LOW_QUALITY.
    
    Args:
        image_path: Path to the image file
        verbose: Print classification details
    
    Returns:
        Tuple of (quality_class, metrics_dict)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Compute quality metrics
    blur_score = variance_of_laplacian(image)
    noise_score = estimate_noise(image)
    lighting_metrics = analyze_lighting(image)
    
    metrics = {
        'blur_score': float(blur_score),
        'noise_score': float(noise_score),
        'lighting_metrics': lighting_metrics
    }
    
    # Classification thresholds
    # High blur_score (>100) = sharp image
    # Low noise_score (<30) = clean image
    # Moderate brightness_std (20-60) = good lighting
    
    is_sharp = blur_score > 100
    is_clean = noise_score < 30
    is_well_lit = 20 < lighting_metrics['brightness_std'] < 60
    
    # Determine quality class
    quality_indicators = sum([is_sharp, is_clean, is_well_lit])
    
    if quality_indicators >= 2:
        quality_class = "HIGH_QUALITY"
    else:
        quality_class = "LOW_QUALITY"
    
    if verbose:
        print(f"\n[Quality Analysis]")
        print(f"  Blur Score: {blur_score:.2f} {'✓' if is_sharp else '✗'} (Sharp: >100)")
        print(f"  Noise Score: {noise_score:.2f} {'✓' if is_clean else '✗'} (Clean: <30)")
        print(f"  Brightness Std: {lighting_metrics['brightness_std']:.2f} {'✓' if is_well_lit else '✗'} (Good: 20-60)")
        print(f"  Quality Class: {quality_class}")
    
    return quality_class, metrics


def get_adaptive_thresholds(quality_class: str) -> Dict[str, float]:
    """
    Get adaptive thresholds based on image quality classification.
    
    Args:
        quality_class: Either "HIGH_QUALITY" or "LOW_QUALITY"
    
    Returns:
        Dictionary of threshold values for tampering detection
    """
    if quality_class == "HIGH_QUALITY":
        return {
            'sharpness_variance_threshold': 1.0,
            'overlap_threshold': 40,  # pixels
            'ela_threshold': 35.0,  # lenient
            'copy_move_threshold': 0.95,
            'noise_inconsistency_threshold': 60,
            'edge_inconsistency_threshold': 90,
            'min_tamper_score': 2
        }
    else:  # LOW_QUALITY
        return {
            'sharpness_variance_threshold': 3.0,  # very lenient
            'overlap_threshold': 120,  # pixels - very lenient
            'ela_threshold': 55.0,  # very lenient
            'copy_move_threshold': 0.95,  # higher
            'noise_inconsistency_threshold': 150,  # warning only
            'edge_inconsistency_threshold': 210,  # warning only
            'min_tamper_score': 3  # need more evidence
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quality_analyzer.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    quality, metrics = classify_image_quality(image_path, verbose=True)
    thresholds = get_adaptive_thresholds(quality)
    
    print(f"\n[Adaptive Thresholds for {quality}]")
    for key, value in thresholds.items():
        print(f"  {key}: {value}")
