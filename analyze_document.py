"""
Unified Aadhaar Document Analysis Pipeline
===========================================
This script integrates two systems:
1. Part 1: Aadhaar Classification (checks if document is an Aadhaar card)
2. Part 2: Tampering Detection (analyzes Aadhaar cards for tampering)

Usage:
    python analyze_document.py path/to/image.jpg

Output:
    JSON result with classification and tampering status
"""

import os
import sys
import json
import cv2
import numpy as np
import threading


# ====================================================================================
# THREAD-SAFE MODEL SINGLETON
# ====================================================================================

_model_lock = threading.Lock()
_model = None
_model_path = None


def load_model_once():
    """
    Thread-safe singleton pattern for loading the Keras model.
    Model is loaded only once at import time and reused across all requests.
    
    Returns:
        Loaded Keras model
    """
    global _model, _model_path
    
    if _model is None:
        with _model_lock:
            # Double-check pattern
            if _model is None:
                from tensorflow.keras.models import load_model
                
                _model_path = os.path.join(
                    os.path.dirname(__file__), 
                    "part1", "model", "aadhar_model.keras"
                )
                
                if not os.path.exists(_model_path):
                    raise FileNotFoundError(
                        f"Model not found at {_model_path}. "
                        "Please train the model first by running: python part1/src/train_model.py"
                    )
                
                try:
                    # Load without compiling to save memory
                    _model = load_model(_model_path, compile=False)
                    print(f"✓ Classification model loaded from: {_model_path}")
                except Exception as e:
                    # Clear the global variable so it can retry on next call
                    _model = None
                    raise RuntimeError(f"Failed to load classification model: {str(e)}")
    
    return _model


# ====================================================================================
# STEP 1: AADHAAR CLASSIFICATION (from part1)
# ====================================================================================

class AadhaarClassifier:
    """
    Wrapper for Part 1 classification model.
    Uses singleton pattern to load model once and reuse across requests.
    """
    
    def __init__(self):
        """Initialize classifier configuration."""
        self.img_size = (224, 224)  # Standard size from config
        self.prediction_threshold = 0.5
    
    def predict(self, image_path):
        """
        Predict if the image is an Aadhaar card.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (is_aadhaar: bool, confidence: float, raw_score: float)
        """
        # Get singleton model instance (loads once, reuses thereafter)
        model = load_model_once()
        
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image file: {image_path}")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            img_resized = cv2.resize(img_rgb, self.img_size)
            
            # Normalize and add batch dimension
            img_array = np.expand_dims(img_resized / 255.0, axis=0)
            
            # Predict using singleton model
            pred_prob = model.predict(img_array, verbose=0)[0][0]
            
            # Determine class based on threshold
            # Sigmoid output: < 0.5 = Aadhaar (class 0), > 0.5 = Not Aadhaar (class 1)
            is_aadhaar = pred_prob <= self.prediction_threshold
            confidence = (1 - pred_prob) if is_aadhaar else pred_prob
            
            return is_aadhaar, float(confidence), float(pred_prob)
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")


# ====================================================================================
# STEP 2: IMAGE-BASED VERIFICATION (NEW SYSTEM)
# ====================================================================================

def verify_aadhaar_image(image_path):
    """
    STRICT SYSTEM: Verify Aadhaar by comparing with reference database images.
    
    Uses STRICT three metrics (ALL must pass):
    1. SSIM ≥ 0.92 (Structural Similarity)
    2. ORB ≥ 120 (Feature Matching)
    3. Histogram ≥ 0.90 (Color Correlation)
    
    Args:
        image_path: Path to the Aadhaar image
        
    Returns:
        dict: Strict match result with format:
        {
            'match_status': 'MATCH_FOUND' | 'FORGED',
            'best_match_file': str,
            'scores': {'ssim': float, 'orb': int, 'hist': float}
        }
    """
    try:
        # Import image matcher
        part2_path = os.path.join(os.path.dirname(__file__), "part2")
        if part2_path not in sys.path:
            sys.path.insert(0, part2_path)
        
        from image_matcher import match_with_reference
        
    except Exception as e:
        print(f"[verify_aadhaar_image] Failed to import image_matcher: {e}")
        return {
            "match_status": "FORGED",
            "best_match_file": None,
            "scores": {"ssim": 0.0, "orb": 0, "hist": 0.0}
        }
    
    try:
        # Run STRICT image matching
        match_result = match_with_reference(image_path, verbose=False)
        return match_result
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[verify_aadhaar_image] Exception occurred:\n{tb}")
        return {
            "match_status": "FORGED",
            "best_match_file": None,
            "scores": {"ssim": 0.0, "orb": 0, "hist": 0.0}
        }


# ====================================================================================
# UNIFIED PIPELINE
# ====================================================================================

def analyze_document(image_path, verbose=True):
    """
    Complete unified pipeline for Aadhaar document analysis.
    
    Pipeline:
    1. Classify if document is an Aadhaar card (Part 1)
    2. If Aadhaar: Analyze for tampering (Part 2)
    3. Return unified JSON result
    
    Args:
        image_path: Path to the document image
        verbose: If True, print progress messages
        
    Returns:
        dict: Unified analysis result in JSON format (always returns a dict, never None)
    """
    if verbose:
        print("\n" + "="*70)
        print("UNIFIED AADHAAR DOCUMENT ANALYSIS")
        print("="*70)
        print(f"Image: {image_path}")
        print("="*70)
    
    # Validate image path
    if not os.path.exists(image_path):
        return {
            "error": "Image file not found",
            "image_path": image_path,
            "is_aadhaar": None,
            "tampering_status": "ERROR",
            "tampering_details": None
        }
    
    try:
        # -----------------------------------------------------------------------
        # STEP 1: AADHAAR CLASSIFICATION
        # -----------------------------------------------------------------------
        if verbose:
            print("\n[STEP 1/2] Classifying document type...")
        
        # Initialize classifier (loads model once)
        classifier = AadhaarClassifier()
        
        # Predict with defensive error handling
        try:
            prediction_result = classifier.predict(image_path)
            
            # Handle different return types
            if prediction_result is None:
                raise ValueError("Classifier returned None - prediction failed")
            
            if isinstance(prediction_result, tuple) and len(prediction_result) >= 2:
                is_aadhaar = prediction_result[0]
                confidence = prediction_result[1]
                raw_score = prediction_result[2] if len(prediction_result) > 2 else confidence
            else:
                raise ValueError(f"Unexpected classifier return type: {type(prediction_result)}")
            
            # Ensure boolean type
            is_aadhaar = bool(is_aadhaar)
            confidence = float(confidence)
            raw_score = float(raw_score)
            
        except Exception as classifier_error:
            print(f"✗ Classifier error: {str(classifier_error)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Classification failed: {str(classifier_error)}",
                "image_path": image_path,
                "is_aadhaar": None,
                "tampering_status": "ERROR",
                "tampering_details": None
            }
        
        if verbose:
            classification = "✓ AADHAAR CARD" if is_aadhaar else "✗ NOT AADHAAR"
            print(f"  Result: {classification}")
            print(f"  Confidence: {confidence * 100:.2f}%")
            print(f"  Raw Score: {raw_score:.4f} (< 0.5 = Aadhaar, > 0.5 = Not Aadhaar)")
        
        # If NOT Aadhaar, stop here
        if not is_aadhaar:
            if verbose:
                print("\n" + "="*70)
                print("FINAL RESULT: NOT AN AADHAAR CARD")
                print("Tampering detection skipped (not applicable)")
                print("="*70)
            else:
                print(f"\n✗ NOT an Aadhaar card (confidence: {confidence*100:.2f}%)")
            
            return {
                "is_aadhaar": False,
                "classification_confidence": confidence,
                "classification_raw_score": raw_score,
                "match_status": "NOT_APPLICABLE",
                "best_match_file": None,
                "match_scores": None,
                "message": f"Document is not an Aadhaar card (confidence: {confidence*100:.2f}%). Image matching not applicable."
            }
        
        # -----------------------------------------------------------------------
        # STEP 2: IMAGE-BASED VERIFICATION (NEW SYSTEM)
        # -----------------------------------------------------------------------
        if verbose:
            print("\n[STEP 2/2] Comparing with reference database images...")
        
        match_result = verify_aadhaar_image(image_path)
        
        # Defensive: ensure match_result is a dict
        if match_result is None or not isinstance(match_result, dict):
            print("[analyze_document] Warning: match_result is None or invalid; replacing with safe default.")
            match_result = {
                "match_status": "UNKNOWN",
                "best_match_file": None,
                "best_match_scores": None,
                "error": "Verification unavailable"
            }
        
        # Extract key information from STRICT matcher
        match_status = match_result.get("match_status", "FORGED")
        best_match_file = match_result.get("best_match_file")
        scores = match_result.get("scores", {"ssim": 0.0, "orb": 0, "hist": 0.0})
        
        if verbose:
            print(f"\n  Match Status: {match_status}")
            if best_match_file:
                print(f"  Best Match: {best_match_file}")
            if scores:
                print(f"  STRICT Scores:")
                print(f"    - SSIM: {scores.get('ssim', 'N/A')} (threshold: ≥0.92)")
                print(f"    - ORB: {scores.get('orb', 'N/A')} (threshold: ≥120)")
                print(f"    - Histogram: {scores.get('hist', 'N/A')} (threshold: ≥0.90)")
        
        # -----------------------------------------------------------------------
        # CONSTRUCT UNIFIED RESULT WITH STRICT VERDICT
        # -----------------------------------------------------------------------
        unified_result = {
            "is_aadhaar": True,
            "classification_confidence": confidence,
            "classification_raw_score": raw_score,
            "match_status": match_status,
            "best_match_file": best_match_file,
            "match_scores": scores,
            "match_details": match_result
        }
        
        if verbose:
            print("\n" + "="*70)
            print("FINAL UNIFIED RESULT - STRICT MODE")
            print("="*70)
            print(f"Document Type: AADHAAR CARD")
            print(f"Verification: {match_status}")
            if match_status == "MATCH_FOUND":
                print(f"✓ VERIFIED - All 3 metrics passed strict thresholds")
            else:
                print(f"✗ FORGED/TAMPERED - One or more metrics failed")
            if best_match_file:
                print(f"Best Match Reference: {best_match_file}")
            print("="*70)
        
        return unified_result
        
    except Exception as e:
        # Comprehensive error handling with traceback
        import traceback
        print(f"\n✗ ERROR: {str(e)}")
        traceback.print_exc()
        
        error_result = {
            "error": str(e),
            "error_traceback": traceback.format_exc(),
            "image_path": image_path,
            "is_aadhaar": None,
            "match_status": "ERROR",
            "best_match_file": None,
            "match_scores": None
        }
        
        return error_result


# ====================================================================================
# CLI INTERFACE
# ====================================================================================

def main():
    """Command-line interface for the unified pipeline."""
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_document.py <image_path>")
        print("\nExample:")
        print("  python analyze_document.py path/to/aadhaar.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Run analysis
    result = analyze_document(image_path, verbose=True)
    
    # Print JSON result
    print("\n" + "="*70)
    print("JSON OUTPUT")
    print("="*70)
    print(json.dumps(result, indent=2))
    print("="*70)
    
    # Return appropriate exit code
    if "error" in result:
        sys.exit(1)
    elif result.get("match_status") == "NO_MATCH":
        sys.exit(2)  # Exit code 2 for non-matching documents
    else:
        sys.exit(0)


# ====================================================================================
# ENTRY POINT
# ====================================================================================

if __name__ == "__main__":
    main()
