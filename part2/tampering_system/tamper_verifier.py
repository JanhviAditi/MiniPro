"""
Tamper Verification System (Quality-Aware with Weighted Scoring)
=================================================================
Main orchestrator combining all detection layers with adaptive thresholds.

WEIGHTED DECISION SYSTEM:
- Critical issues (weight=5): Invalid Aadhaar, QR mismatch, Copy-move, Strong ELA
- Major issues (weight=3): Extreme sharpness variance, Severe edge inconsistency, Layout anomalies
- Minor issues (weight=1): Noise warnings, Minor overlaps, Missing optional fields

DECISION LOGIC:
1. Calculate weighted score: critical*5 + major*3 + minor*1
2. If any critical issue present (score >= 5) → TAMPERED
3. Else if weighted sum >= 6 → POSSIBLY TAMPERED
4. Else → ORIGINAL

QUALITY-AWARE:
- HIGH_QUALITY images: Strict thresholds (sharpness_var=0.3, overlap=10px)
- LOW_QUALITY images: Lenient thresholds (sharpness_var=1.2, overlap=40px)

IMPORTANT: Missing QR does NOT count as tampered.
"""

import json
from typing import Dict, List
import os

# Import detection modules
from visual_tamper_detector import detect_visual_tampering
from ocr_extractor import extract_and_validate_ocr
from qr_checker import check_qr_code
from quality_analyzer import classify_image_quality, get_adaptive_thresholds


# ====================================================================================
# WEIGHTED DECISION FUSION LOGIC
# ====================================================================================
CRITICAL_WEIGHT = 5
MAJOR_WEIGHT = 3
MINOR_WEIGHT = 1

def fuse_detection_results(visual_result: Dict, ocr_result: Dict, qr_result: Dict) -> Dict:
    """
    Fuse results using weighted scoring system.
    
    Returns:
        Dictionary with final_status, confidence, weighted_score, and issue breakdown
    """
    # Collect all weighted issues
    critical_issues = []
    major_issues = []
    minor_issues = []
    
    # Visual analysis issues
    critical_issues.extend(visual_result.get('critical_issues', []))
    major_issues.extend(visual_result.get('major_issues', []))
    minor_issues.extend(visual_result.get('minor_issues', []))
    
    # OCR analysis issues
    critical_issues.extend(ocr_result.get('critical_issues', []))
    major_issues.extend(ocr_result.get('major_issues', []))
    minor_issues.extend(ocr_result.get('minor_issues', []))
    
    # QR analysis issues
    critical_issues.extend(qr_result.get('critical_issues', []))
    major_issues.extend(qr_result.get('major_issues', []))
    minor_issues.extend(qr_result.get('minor_issues', []))
    
    # Calculate weighted score
    weighted_score = (
        len(critical_issues) * CRITICAL_WEIGHT +
        len(major_issues) * MAJOR_WEIGHT +
        len(minor_issues) * MINOR_WEIGHT
    )
    
    # Decision logic
    if len(critical_issues) > 0:
        # Any critical issue → TAMPERED
        final_status = "TAMPERED"
        confidence = min(95, 70 + len(critical_issues) * 5)
    elif weighted_score >= 6:
        # Weighted sum >= 6 → POSSIBLY TAMPERED
        final_status = "POSSIBLY TAMPERED"
        confidence = min(75, 50 + weighted_score * 3)
    else:
        # Low score → ORIGINAL
        final_status = "ORIGINAL"
        confidence = max(80, 95 - weighted_score * 5)
    
    return {
        'final_status': final_status,
        'confidence': int(confidence),
        'weighted_score': weighted_score,
        'critical_issues': critical_issues,
        'major_issues': major_issues,
        'minor_issues': minor_issues,
        'total_issues': len(critical_issues) + len(major_issues) + len(minor_issues)
    }


# Note: Confidence calculation is now integrated into fuse_detection_results()


# ====================================================================================
# MAIN VERIFICATION FUNCTION
# ====================================================================================
def verify_tampering(image_path: str, verbose: bool = True) -> Dict:
    """
    Main function with quality-aware adaptive thresholds and weighted scoring.
    
    Args:
        image_path: Path to the Aadhaar image file
        verbose: If True, print progress messages
    
    Returns:
        Dictionary with quality-aware analysis and weighted scoring
    """
    if verbose:
        print("\n" + "="*70)
        print("QUALITY-AWARE AADHAAR TAMPERING VERIFICATION")
        print("="*70)
        print(f"Analyzing: {image_path}")
        print("="*70)
    
    # Validate image path
    if not os.path.exists(image_path):
        return {
            "error": "Image file not found",
            "image_path": image_path,
            "final_status": "ERROR"
        }
    
    # Step 0: Analyze Image Quality and Get Adaptive Thresholds
    if verbose:
        print("\n[0/3] Analyzing Image Quality...")
    
    quality_class, quality_metrics = classify_image_quality(image_path, verbose=verbose)
    thresholds = get_adaptive_thresholds(quality_class)
    
    if verbose:
        print(f"Quality Classification: {quality_class}")
    
    # Layer 1: Visual Tampering Detection (with adaptive thresholds)
    if verbose:
        print("\n[1/3] Running Visual Tampering Detection...")
    
    visual_result = detect_visual_tampering(image_path, thresholds=thresholds)
    
    # Layer 2: OCR Extraction and Validation (with adaptive thresholds)
    if verbose:
        print("\n[2/3] Running OCR Extraction and Validation...")
    
    ocr_result = extract_and_validate_ocr(image_path, thresholds=thresholds)
    
    # Layer 3: QR Code Checking (OPTIONAL - Missing QR is OK)
    if verbose:
        print("\n[3/3] Running QR Code Validation (OPTIONAL)...")
    
    qr_result = check_qr_code(image_path, ocr_result)
    
    # Fuse results with weighted scoring
    if verbose:
        print("\n" + "-"*70)
        print("Fusing detection results with weighted scoring...")
    
    fusion_result = fuse_detection_results(visual_result, ocr_result, qr_result)
    
    # Extract decision from fusion result
    final_status = fusion_result['final_status']
    confidence = fusion_result['confidence']
    weighted_score = fusion_result['weighted_score']
    
    # Generate recommendation
    if final_status == "TAMPERED":
        recommendation = "REJECT: Document shows strong signs of tampering. Manual verification required."
    elif final_status == "POSSIBLY TAMPERED":
        recommendation = "REVIEW: Document has suspicious characteristics. Additional verification recommended."
    else:
        recommendation = "ACCEPT: Document appears to be original. No significant tampering detected."
    
    # Build final result with new format
    result = {
        "image_path": image_path,
        "quality": quality_class,
        "quality_metrics": quality_metrics,
        "thresholds_used": thresholds,
        "visual_analysis": visual_result,
        "ocr_analysis": ocr_result,
        "qr_analysis": qr_result,
        "final_status": final_status,
        "confidence": confidence,
        "weighted_score": weighted_score,
        "scores": {
            "critical_count": len(fusion_result['critical_issues']),
            "major_count": len(fusion_result['major_issues']),
            "minor_count": len(fusion_result['minor_issues']),
            "total_count": fusion_result['total_issues']
        },
        "critical_issues": fusion_result['critical_issues'],
        "major_issues": fusion_result['major_issues'],
        "minor_issues": fusion_result['minor_issues'],
        "summary": {
            "recommendation": recommendation
        }
    }
    
    if verbose:
        print("-"*70)
        print(f"\nFINAL VERDICT: {final_status}")
        print(f"Confidence: {confidence}%")
        print(f"Weighted Score: {weighted_score} (Critical: {len(fusion_result['critical_issues'])} × 5, Major: {len(fusion_result['major_issues'])} × 3, Minor: {len(fusion_result['minor_issues'])} × 1)")
        
        if fusion_result['critical_issues']:
            print("\n⚠ CRITICAL Issues (weight=5):")
            for issue in fusion_result['critical_issues']:
                print(f"  • {issue}")
        
        if fusion_result['major_issues']:
            print("\n⚠ MAJOR Issues (weight=3):")
            for issue in fusion_result['major_issues']:
                print(f"  • {issue}")
        
        if fusion_result['minor_issues']:
            print(f"\n⚠ MINOR Issues (weight=1): {len(fusion_result['minor_issues'])} warnings")
        
        print(f"\nRecommendation: {recommendation}")
        print("="*70)
    
    return result


def verify_tampering_batch(image_paths, verbose: bool = False):
    """
    Verify multiple images in batch.
    
    Args:
        image_paths: List of image file paths
        verbose: If True, print progress for each image
    
    Returns:
        List of verification results
    """
    results = []
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {image_path}")
        result = verify_tampering(image_path, verbose=verbose)
        results.append(result)
    
    return results


def export_result_json(result: Dict, output_path: str):
    """
    Export verification result to JSON file.
    
    Args:
        result: Verification result dictionary
        output_path: Path to save JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResult exported to: {output_path}")


# ====================================================================================
# INTEGRATION HELPER
# ====================================================================================
def quick_verify(image_path: str) -> bool:
    """
    Quick verification for simple integration.
    
    Args:
        image_path: Path to Aadhaar image
    
    Returns:
        True if ORIGINAL, False if TAMPERED or POSSIBLY TAMPERED
    
    Usage:
        if quick_verify("aadhaar.jpg"):
            print("Document is authentic")
        else:
            print("Document is suspicious or tampered")
    """
    result = verify_tampering(image_path, verbose=False)
    return result['final_status'] == "ORIGINAL"


# ====================================================================================
# STANDALONE TESTING
# ====================================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tamper_verifier.py <image_path> [output_json]")
        print("\nExample:")
        print("  python tamper_verifier.py aadhaar_sample.jpg")
        print("  python tamper_verifier.py aadhaar_sample.jpg result.json")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Run verification
    result = verify_tampering(image_path, verbose=True)
    
    # Export to JSON if output path provided
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        export_result_json(result, output_path)
    
    # Print summary
    print("\n" + "="*70)
    print("QUICK SUMMARY")
    print("="*70)
    print(f"Status: {result['final_status']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Quality: {result['quality']}")
    print(f"Weighted Score: {result['weighted_score']}")
    print(f"Critical Issues: {result['scores']['critical_count']}")
    print(f"Major Issues: {result['scores']['major_count']}")
    print(f"Minor Issues: {result['scores']['minor_count']}")
    print("="*70)
