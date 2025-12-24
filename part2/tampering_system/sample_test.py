"""
Sample Test Script
==================
Demonstrates how to use the Aadhaar Tampering Detection System.

This script shows various ways to integrate the tampering detection
into your existing pipeline.
"""

import os
from typing import List
from tamper_verifier import verify_tampering, quick_verify, verify_tampering_batch, export_result_json


def example_1_basic_usage():
    """
    Example 1: Basic usage - verify a single image
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Replace with your actual image path
    image_path = r"C:\Users\91966\OneDrive\Desktop\Alvina\testimg.jpg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        print("Please provide a valid Aadhaar image path.")
        return
    
    # Run tampering verification
    result = verify_tampering(image_path, verbose=True)
    
    # Access results
    print(f"\nFinal Status: {result['final_status']}")
    print(f"Confidence: {result['confidence']}%")


def example_2_quick_check():
    """
    Example 2: Quick boolean check for simple integration
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Quick Check")
    print("="*70)
    
    image_path = "sample_aadhaar.jpg"
    
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return
    
    # Simple true/false check
    is_original = quick_verify(image_path)
    
    if is_original:
        print("✓ Document is ORIGINAL")
    else:
        print("✗ Document is SUSPICIOUS or TAMPERED")


def example_3_integration_with_classifier():
    """
    Example 3: Integration with existing Aadhaar classifier
    
    This shows how to plug the tampering detection into a pipeline
    where Aadhaar vs Non-Aadhaar classification is already done.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Integration with Classifier")
    print("="*70)
    
    image_path = "sample_aadhaar.jpg"
    
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return
    
    # Step 1: Assume Aadhaar classification is already done
    # (Replace with your actual classifier)
    is_aadhaar = classify_aadhaar(image_path)
    
    if not is_aadhaar:
        print("[Classification] Not an Aadhaar document")
        return
    
    print("[Classification] Aadhaar document detected ✓")
    
    # Step 2: Run tampering verification
    print("\n[Tampering Detection] Starting verification...")
    result = verify_tampering(image_path, verbose=False)
    
    # Step 3: Make decision
    status = result['final_status']
    
    if status == "ORIGINAL":
        print("\n[FINAL DECISION] ✓ ACCEPT - Document is authentic")
    elif status == "POSSIBLY TAMPERED":
        print("\n[FINAL DECISION] ⚠ REVIEW - Manual verification recommended")
    else:  # TAMPERED
        print("\n[FINAL DECISION] ✗ REJECT - Document is tampered")
    
    # Print detailed issues
    if result['summary']['critical_issues']:
        print("\nCritical Issues Found:")
        for issue in result['summary']['critical_issues']:
            print(f"  • {issue}")


def example_4_batch_processing():
    """
    Example 4: Process multiple images in batch
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing")
    print("="*70)
    
    # List of images to process
    image_paths = [
        "sample_aadhaar_1.jpg",
        "sample_aadhaar_2.jpg",
        "sample_aadhaar_3.jpg"
    ]
    
    # Filter existing files
    existing_paths = [path for path in image_paths if os.path.exists(path)]
    
    if not existing_paths:
        print("[ERROR] No valid image files found")
        print("Please provide valid Aadhaar image paths in the list")
        return
    
    # Process batch
    results = verify_tampering_batch(existing_paths, verbose=False)
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['image_path']}")
        print(f"    Status: {result['final_status']}")
        print(f"    Confidence: {result['confidence']}%")
        print(f"    Issues: {result['summary']['total_issues']}")


def example_5_detailed_analysis():
    """
    Example 5: Access detailed analysis results
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Detailed Analysis")
    print("="*70)
    
    image_path = "sample_aadhaar.jpg"
    
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return
    
    result = verify_tampering(image_path, verbose=False)
    
    # Access visual analysis
    print("\n--- VISUAL ANALYSIS ---")
    print(f"ELA Score: {result['visual_analysis']['ela_score']}")
    print(f"Copy-Move Score: {result['visual_analysis']['copy_move_score']}")
    print(f"Failed Checks: {result['visual_analysis']['failed_checks']}")
    
    # Access OCR analysis
    print("\n--- OCR ANALYSIS ---")
    fields = result['ocr_analysis']['text_fields']
    print(f"Aadhaar Number: {fields.get('aadhaar_number', 'Not found')}")
    print(f"Name: {fields.get('name', 'Not found')}")
    print(f"DOB: {fields.get('dob', 'Not found')}")
    print(f"Gender: {fields.get('gender', 'Not found')}")
    print(f"OCR Issues: {result['ocr_analysis']['ocr_issues']}")
    
    # Access QR analysis
    print("\n--- QR ANALYSIS ---")
    print(f"QR Available: {result['qr_analysis']['qr_available']}")
    print(f"Signature Valid: {result['qr_analysis']['signature_valid']}")
    print(f"QR Issues: {result['qr_analysis']['qr_issues']}")
    
    # Export to JSON
    export_result_json(result, "detailed_analysis.json")


def example_6_export_results():
    """
    Example 6: Export results to JSON file
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Export Results")
    print("="*70)
    
    image_path = "sample_aadhaar.jpg"
    
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return
    
    result = verify_tampering(image_path, verbose=False)
    
    # Export to JSON
    output_path = "tampering_analysis_result.json"
    export_result_json(result, output_path)
    
    print(f"\n✓ Results exported to: {output_path}")
    print("You can now share or process this JSON file further.")


# ====================================================================================
# MOCK CLASSIFIER (Replace with your actual classifier)
# ====================================================================================
def classify_aadhaar(image_path: str) -> bool:
    """
    Mock Aadhaar classifier.
    
    In your actual implementation, replace this with your trained
    Aadhaar vs Non-Aadhaar classifier.
    
    Args:
        image_path: Path to image
    
    Returns:
        True if Aadhaar, False otherwise
    """
    # Mock implementation - replace with your actual classifier
    # Example:
    # from your_classifier import predict_aadhaar
    # return predict_aadhaar(image_path)
    
    print("[Mock Classifier] Assuming image is Aadhaar (replace with real classifier)")
    return True


# ====================================================================================
# MAIN MENU
# ====================================================================================
def main():
    """
    Main function with interactive menu
    """
    print("\n" + "="*70)
    print("AADHAAR TAMPERING DETECTION - SAMPLE TESTS")
    print("="*70)
    print("\nAvailable Examples:")
    print("  1. Basic Usage")
    print("  2. Quick Check")
    print("  3. Integration with Classifier")
    print("  4. Batch Processing")
    print("  5. Detailed Analysis")
    print("  6. Export Results")
    print("  0. Run All Examples")
    print("\nNote: Make sure to update image paths in the examples!")
    
    choice = input("\nSelect example (0-6): ").strip()
    
    if choice == "1":
        example_1_basic_usage()
    elif choice == "2":
        example_2_quick_check()
    elif choice == "3":
        example_3_integration_with_classifier()
    elif choice == "4":
        example_4_batch_processing()
    elif choice == "5":
        example_5_detailed_analysis()
    elif choice == "6":
        example_6_export_results()
    elif choice == "0":
        example_1_basic_usage()
        example_2_quick_check()
        example_3_integration_with_classifier()
        example_4_batch_processing()
        example_5_detailed_analysis()
        example_6_export_results()
    else:
        print("\nInvalid choice!")


# ====================================================================================
# STANDALONE TESTING
# ====================================================================================
if __name__ == "__main__":
    # You can either run the interactive menu or test directly
    
    # Option 1: Interactive menu
    # main()
    
    # Option 2: Direct test (uncomment and update path)
    print("\n" + "="*70)
    print("DIRECT TEST MODE")
    print("="*70)
    print("\nTo run examples, update image paths in this file and uncomment the examples.")
    print("\nAvailable functions:")
    print("  - verify_tampering(image_path)")
    print("  - quick_verify(image_path)")
    print("  - verify_tampering_batch([image_paths])")
    print("\nExample usage:")
    print('  result = verify_tampering("path/to/aadhaar.jpg")')
    print('  print(result["final_status"])')
    print("\n" + "="*70)
    
    # Test with testimg.jpg
    test_image = r"C:\Users\91966\OneDrive\Desktop\Alvina\testimg.jpg"
    if os.path.exists(test_image):
        result = verify_tampering(test_image, verbose=True)
        print(f"\nFinal Status: {result['final_status']}")
    else:
        print(f"\nPlease update 'test_image' path in sample_test.py")
