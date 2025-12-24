"""
Web Application for Aadhaar Tampering Detection
================================================
A Flask-based web interface for uploading and verifying Aadhaar cards.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from tamper_verifier import verify_tampering
import json
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home page with upload interface"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and tampering verification"""
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Run tampering detection
        result = verify_tampering(filepath, verbose=False)
        
        # Format result for web display
        response = {
            'success': True,
            'filename': filename,
            'uploaded_file': unique_filename,
            'result': {
                'status': result['final_status'],
                'confidence': result['confidence'],
                'quality': result.get('quality', 'UNKNOWN'),
                'weighted_score': result.get('weighted_score', 0),
                'scores': result.get('scores', {}),
                'critical_issues': result.get('critical_issues', []),
                'major_issues': result.get('major_issues', []),
                'minor_issues': result.get('minor_issues', []),
                'recommendation': result.get('recommendation', '')
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/history')
def history():
    """Show verification history"""
    try:
        files = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                files.append({
                    'filename': filename,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sort by modified time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({'files': files})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 AADHAAR TAMPERING DETECTION - WEB INTERFACE")
    print("="*70)
    print("\n✓ Server starting...")
    print("✓ Upload folder ready")
    print("\n📍 Open your browser and go to:")
    print("   👉 http://localhost:5000")
    print("\n💡 Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
