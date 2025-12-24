"""
Simple Flask UI for Unified Aadhaar Fraud Detection System
===========================================================
Provides a clean web interface for document analysis without modifying existing code.

Usage:
    python app_ui.py
    
Access at:
    http://localhost:5000/
"""

import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import json

# Import the unified analysis function
# Model will be loaded once on first request and cached thereafter
from analyze_document import analyze_document

# ====================================================================================
# FLASK APP CONFIGURATION
# ====================================================================================

app = Flask(__name__)
app.secret_key = 'aadhaar_fraud_detection_secret_key_2025'

# Upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ====================================================================================
# HELPER FUNCTIONS
# ====================================================================================

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_verdict(result):
    """
    Generate STRICT user-friendly verdict from the analysis result.
    
    STRICT MODE:
    - NOT Aadhaar → "Not Aadhaar"
    - Aadhaar + MATCH_FOUND → "Verified"
    - Aadhaar + FORGED → "Document is FORGED / TAMPERED"
    
    Args:
        result: Dictionary from analyze_document()
        
    Returns:
        tuple: (verdict_text, verdict_class, icon)
    """
    # Defensive check
    if result is None:
        return "Error: No result returned", "error", "⚠️"
    
    if not isinstance(result, dict):
        return f"Error: Invalid result type {type(result)}", "error", "⚠️"
    
    # Check for errors
    if 'error' in result:
        error_msg = result.get('error', 'Unknown error')
        return f"Error: {error_msg}", "error", "⚠️"
    
    # Not an Aadhaar card
    if not result.get('is_aadhaar', False):
        confidence = result.get('classification_confidence', 0)
        return "Not an Aadhaar Card", "not-aadhaar", "❌"
    
    # Aadhaar card - check STRICT match status
    match_status = result.get('match_status', 'FORGED')
    
    if match_status == 'MATCH_FOUND':
        return "✓ VERIFIED — Pixel-Perfect Match (99.99%+)", "original", "✅"
    elif match_status == 'FORGED':
        return "✗ FORGED / TAMPERED — Failed Ultra-Strict Verification", "tampered", "🚫"
    else:
        return "Verification Error", "error", "⚠️"


def format_confidence(confidence):
    """Format confidence as percentage."""
    if confidence is None:
        return "N/A"
    return f"{confidence * 100:.1f}%" if confidence < 1 else f"{confidence:.1f}%"


# ====================================================================================
# ROUTES
# ====================================================================================

@app.route('/')
def index():
    """Display the upload form."""
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis."""
    
    # Check if file is present in the request
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    # Validate file type
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run the unified analysis (verbose=False for cleaner logs)
        print(f"\n{'='*60}")
        print(f"Analyzing uploaded file: {filename}")
        print(f"{'='*60}")
        
        result = analyze_document(filepath, verbose=False)
        
        # Defensive check: ensure result is a dictionary
        if result is None:
            result = {
                'is_aadhaar': False,
                'error': 'Analysis returned no result',
                'tampering_status': 'ERROR',
                'classification_confidence': None,
                'tampering_details': None
            }
        
        if not isinstance(result, dict):
            result = {
                'is_aadhaar': False,
                'error': f'Invalid result type: {type(result)}',
                'tampering_status': 'ERROR',
                'classification_confidence': None,
                'tampering_details': None
            }
        
        # Check if there's an error - show as "Not Aadhaar" with error details
        if 'error' in result and result.get('is_aadhaar') is None:
            # Treat errors as "Not Aadhaar" to avoid confusion
            result['is_aadhaar'] = False
            result['tampering_status'] = 'NOT_APPLICABLE'
            result['message'] = f"Could not analyze document: {result['error'][:100]}"
        
        # Generate verdict
        try:
            verdict, verdict_class, icon = get_verdict(result)
        except Exception as verdict_error:
            print(f"Error generating verdict: {verdict_error}")
            verdict = "Analysis Error"
            verdict_class = "error"
            icon = "⚠️"
        
        # Extract key information for display with safe dictionary access (NEW IMAGE-BASED SYSTEM)
        match_scores = result.get('match_scores')
        if match_scores is None or not isinstance(match_scores, dict):
            match_scores = {}
        
        match_details = result.get('match_details')
        if match_details is None or not isinstance(match_details, dict):
            match_details = {}
        
        # Build display data with ULTRA-STRICT mode results
        display_data = {
            'filename': filename,
            'is_aadhaar': result.get('is_aadhaar', False),
            'classification_confidence': format_confidence(result.get('classification_confidence')),
            'match_status': result.get('match_status', 'FORGED'),
            'verdict': verdict,
            'verdict_class': verdict_class,
            'icon': icon,
            'error': result.get('error', None),
            'message': result.get('message', None),
            # ULTRA-STRICT Image matching results (4 metrics)
            'best_match_file': result.get('best_match_file', 'N/A'),
            'ssim_score': match_scores.get('ssim', 'N/A'),
            'orb_matches': match_scores.get('orb', 'N/A'),
            'histogram_score': match_scores.get('hist', 'N/A'),
            'pixel_score': match_scores.get('pixel', 'N/A'),
            'strict_thresholds': {
                'ssim': '≥ 0.9999',
                'orb': '≥ 500',
                'hist': '≥ 0.9999',
                'pixel': '≥ 0.9999'
            }
        }
        
        print(f"Analysis complete: {verdict}")
        print(f"{'='*60}\n")
        
        # Clean up uploaded file (optional - uncomment to auto-delete)
        # os.remove(filepath)
        
        return render_template('result.html', data=display_data)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        
        # Make memory errors user-friendly
        if 'Unable to allocate' in error_msg or 'not enough memory' in error_msg:
            user_msg = 'Insufficient memory to process the image. Please try restarting the application or use a smaller image.'
        elif 'NoneType' in error_msg:
            user_msg = 'Analysis failed. Please ensure the image is a valid document and try again.'
        else:
            user_msg = f'Error processing file: {error_msg[:200]}'
        
        flash(user_msg, 'error')
        print(f"\n{'='*60}")
        print(f"ERROR: {error_msg}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        return redirect(url_for('index'))


@app.route('/about')
def about():
    """Display information about the system."""
    return """
    <h1>Aadhaar Fraud Detection System</h1>
    <p>This system combines two detection layers:</p>
    <ul>
        <li><strong>Classification:</strong> Identifies if the document is an Aadhaar card</li>
        <li><strong>Tampering Detection:</strong> Analyzes Aadhaar cards for signs of tampering</li>
    </ul>
    <a href="/">← Back to Upload</a>
    """


# ====================================================================================
# ERROR HANDLERS
# ====================================================================================

@app.errorhandler(413)
def too_large(e):
    """Handle file too large errors."""
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))


# ====================================================================================
# MAIN ENTRY POINT
# ====================================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("AADHAAR FRAUD DETECTION - WEB INTERFACE")
    print("="*70)
    print("\nStarting Flask server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"Max file size: {MAX_FILE_SIZE / (1024*1024):.0f}MB")
    print("\n" + "="*70)
    print("Access the application at:")
    print("    http://localhost:5000/")
    print("="*70 + "\n")
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)
