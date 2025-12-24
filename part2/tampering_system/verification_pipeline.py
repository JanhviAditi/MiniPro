"""
Complete Aadhaar Document Verification Pipeline
================================================
Combines Aadhaar classification with tampering detection

Pipeline Flow:
1. Upload document image
2. Classify: Aadhaar vs Non-Aadhaar
3. If Aadhaar: Check for tampering
4. Make final decision: ACCEPT / REJECT / MANUAL_REVIEW
"""

import os
import sys
from typing import Dict, Tuple

# Import tampering detection from this project
from tamper_verifier import verify_tampering


class AadhaarClassifierStub:
    """
    STUB/PLACEHOLDER for your existing Aadhaar classifier.
    Replace this with your actual classifier implementation.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize your classifier model.
        
        Args:
            model_path: Path to your trained model file
        """
        self.model_path = model_path
        print("[Classifier] Using STUB classifier - Replace with your actual model!")
        
        # TODO: Load your actual model here
        # Example:
        # import joblib
        # self.model = joblib.load(model_path)
        # OR
        # from tensorflow import keras
        # self.model = keras.models.load_model(model_path)
    
    def predict(self, image_path: str) -> Tuple[bool, float]:
        """
        Classify if image is Aadhaar or not.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (is_aadhaar: bool, confidence: float 0-1)
        
        TODO: Replace this with your actual classification logic
        """
        # STUB IMPLEMENTATION - Always returns True with 95% confidence
        # Replace with your actual prediction code:
        
        # Example for sklearn:
        # features = self.extract_features(image_path)
        # prediction = self.model.predict([features])[0]
        # confidence = self.model.predict_proba([features])[0].max()
        # return prediction == 1, confidence
        
        # Example for TensorFlow/Keras:
        # from tensorflow.keras.preprocessing import image
        # img = image.load_img(image_path, target_size=(224, 224))
        # img_array = image.img_to_array(img) / 255.0
        # img_array = np.expand_dims(img_array, axis=0)
        # prediction = self.model.predict(img_array)[0][0]
        # is_aadhaar = prediction > 0.5
        # confidence = prediction if is_aadhaar else (1 - prediction)
        # return is_aadhaar, confidence
        
        print(f"[Classifier] Analyzing: {image_path}")
        print("[Classifier] ⚠️ STUB MODE - Replace with your classifier!")
        
        # For testing: Return True (Aadhaar) with 95% confidence
        return True, 0.95


class AadhaarVerificationPipeline:
    """
    End-to-end Aadhaar document verification pipeline.
    Combines classification and tampering detection.
    """
    
    def __init__(self, classifier_model_path=None):
        """
        Initialize the complete verification pipeline.
        
        Args:
            classifier_model_path: Path to your Aadhaar classifier model
        """
        # Initialize classifier (replace stub with your actual classifier)
        self.classifier = AadhaarClassifierStub(classifier_model_path)
    
    def verify_document(self, image_path: str, verbose: bool = True) -> Dict:
        """
        Run complete verification pipeline on a document.
        
        Args:
            image_path: Path to uploaded document image
            verbose: Print progress messages
        
        Returns:
            Dictionary containing:
            - classification: Aadhaar vs Non-Aadhaar result
            - tampering_analysis: Detailed tampering detection (if Aadhaar)
            - final_decision: ACCEPTED / REJECTED / MANUAL_REVIEW
            - recommendation: Human-readable recommendation
        """
        if verbose:
            print("\n" + "="*80)
            print("AADHAAR DOCUMENT VERIFICATION PIPELINE")
            print("="*80)
            print(f"Document: {image_path}")
        
        # Validate file exists
        if not os.path.exists(image_path):
            return {
                "error": "File not found",
                "image_path": image_path,
                "final_decision": "ERROR"
            }
        
        # Initialize result structure
        result = {
            "image_path": image_path,
            "classification": None,
            "tampering_analysis": None,
            "final_decision": None,
            "recommendation": None
        }
        
        # ====================================================================
        # STAGE 1: Document Classification (Aadhaar vs Non-Aadhaar)
        # ====================================================================
        if verbose:
            print("\n" + "-"*80)
            print("[STAGE 1/2] DOCUMENT CLASSIFICATION")
            print("-"*80)
        
        try:
            is_aadhaar, confidence = self.classifier.predict(image_path)
            
            result["classification"] = {
                "is_aadhaar": is_aadhaar,
                "confidence": float(confidence),
                "confidence_percent": f"{confidence*100:.1f}%"
            }
            
            if verbose:
                status_symbol = "✓" if is_aadhaar else "✗"
                doc_type = "Aadhaar Card" if is_aadhaar else "Non-Aadhaar Document"
                print(f"\n{status_symbol} Classification: {doc_type}")
                print(f"  Confidence: {confidence*100:.1f}%")
            
            # If not Aadhaar, reject immediately
            if not is_aadhaar:
                result["final_decision"] = "REJECTED"
                result["recommendation"] = "REJECT: Document is not an Aadhaar card."
                
                if verbose:
                    print("\n" + "="*80)
                    print("FINAL DECISION: REJECTED ✗")
                    print("Reason: Not an Aadhaar card")
                    print("="*80)
                
                return result
        
        except Exception as e:
            result["error"] = f"Classification failed: {str(e)}"
            result["final_decision"] = "ERROR"
            
            if verbose:
                print(f"\n✗ Classification Error: {e}")
            
            return result
        
        # ====================================================================
        # STAGE 2: Tampering Detection (Quality-Aware)
        # ====================================================================
        if verbose:
            print("\n" + "-"*80)
            print("[STAGE 2/2] TAMPERING DETECTION")
            print("-"*80)
        
        try:
            # Run quality-aware tampering detection
            tampering_result = verify_tampering(image_path, verbose=verbose)
            result["tampering_analysis"] = tampering_result
            
            # Extract tampering status
            tamper_status = tampering_result["final_status"]
            tamper_confidence = tampering_result["confidence"]
            
            # ================================================================
            # FINAL DECISION LOGIC
            # ================================================================
            if tamper_status == "TAMPERED":
                # Critical issues detected - reject document
                result["final_decision"] = "REJECTED"
                result["recommendation"] = (
                    "REJECT: Document shows strong evidence of tampering. "
                    f"Confidence: {tamper_confidence}%"
                )
            
            elif tamper_status == "POSSIBLY TAMPERED":
                # Suspicious characteristics - flag for manual review
                result["final_decision"] = "MANUAL_REVIEW"
                result["recommendation"] = (
                    "FLAG FOR MANUAL REVIEW: Document has suspicious characteristics. "
                    f"Weighted score: {tampering_result['weighted_score']}"
                )
            
            else:  # ORIGINAL
                # No significant tampering detected - accept
                result["final_decision"] = "ACCEPTED"
                result["recommendation"] = (
                    "ACCEPT: Document appears authentic. "
                    f"Quality: {tampering_result['quality']}, "
                    f"Confidence: {tamper_confidence}%"
                )
        
        except Exception as e:
            result["error"] = f"Tampering detection failed: {str(e)}"
            result["final_decision"] = "ERROR"
            
            if verbose:
                print(f"\n✗ Tampering Detection Error: {e}")
            
            return result
        
        # ====================================================================
        # FINAL OUTPUT
        # ====================================================================
        if verbose:
            print("\n" + "="*80)
            print("PIPELINE COMPLETE")
            print("="*80)
            
            # Decision symbol
            if result["final_decision"] == "ACCEPTED":
                symbol = "✓✓✓"
                color = "GREEN"
            elif result["final_decision"] == "MANUAL_REVIEW":
                symbol = "⚠⚠⚠"
                color = "YELLOW"
            else:  # REJECTED
                symbol = "✗✗✗"
                color = "RED"
            
            print(f"\n{symbol} FINAL DECISION: {result['final_decision']} {symbol}")
            print(f"\nRecommendation:")
            print(f"  {result['recommendation']}")
            print("\n" + "="*80)
        
        return result
    
    def verify_batch(self, image_paths: list, verbose: bool = False) -> list:
        """
        Verify multiple documents in batch.
        
        Args:
            image_paths: List of image file paths
            verbose: Print detailed progress for each document
        
        Returns:
            List of verification results (one per document)
        """
        results = []
        
        print(f"\n{'='*80}")
        print(f"BATCH VERIFICATION: {len(image_paths)} documents")
        print(f"{'='*80}\n")
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
            
            result = self.verify_document(img_path, verbose=verbose)
            results.append(result)
            
            # Print quick summary
            decision = result.get('final_decision', 'ERROR')
            
            if decision == "ACCEPTED":
                symbol = "✓"
            elif decision == "MANUAL_REVIEW":
                symbol = "⚠"
            elif decision == "REJECTED":
                symbol = "✗"
            else:
                symbol = "?"
            
            print(f"  {symbol} Decision: {decision}\n")
        
        # Print batch summary
        print(f"{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        
        accepted = sum(1 for r in results if r.get('final_decision') == 'ACCEPTED')
        rejected = sum(1 for r in results if r.get('final_decision') == 'REJECTED')
        review = sum(1 for r in results if r.get('final_decision') == 'MANUAL_REVIEW')
        errors = sum(1 for r in results if r.get('final_decision') == 'ERROR')
        
        print(f"Total Documents: {len(results)}")
        print(f"  ✓ Accepted: {accepted}")
        print(f"  ⚠ Need Review: {review}")
        print(f"  ✗ Rejected: {rejected}")
        print(f"  ? Errors: {errors}")
        print(f"{'='*80}\n")
        
        return results


# ============================================================================
# STANDALONE USAGE
# ============================================================================
if __name__ == "__main__":
    import json
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single document: python verification_pipeline.py <image_path> [output.json]")
        print("  Batch: python verification_pipeline.py <image1> <image2> <image3> ...")
        sys.exit(1)
    
    # Initialize pipeline
    print("Initializing Aadhaar Verification Pipeline...")
    pipeline = AadhaarVerificationPipeline()
    
    # Single document or batch?
    if len(sys.argv) == 2 or (len(sys.argv) == 3 and sys.argv[2].endswith('.json')):
        # Single document
        image_path = sys.argv[1]
        result = pipeline.verify_document(image_path)
        
        # Export to JSON if output path provided
        if len(sys.argv) == 3:
            output_path = sys.argv[2]
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Result exported to: {output_path}")
    
    else:
        # Batch processing
        image_paths = sys.argv[1:]
        results = pipeline.verify_batch(image_paths)
        
        # Export batch results
        output_path = "batch_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Batch results exported to: {output_path}")
