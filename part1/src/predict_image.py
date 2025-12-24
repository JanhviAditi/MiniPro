"""
Aadhar Card Detection - Image Prediction Script
Loads a trained model and predicts whether an image contains an Aadhar card
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import config

# Import select_image function
from select_image import select_image

print("=" * 60)
print("AADHAR CARD DETECTION - PREDICTION")
print("=" * 60)

# ---------------------------------
# 1. LOAD MODEL
# ---------------------------------
print("\n[1/3] Loading Model...")

if not os.path.exists(config.MODEL_SAVE_PATH):
    print(f"ERROR: Model not found at {config.MODEL_SAVE_PATH}")
    print("\nPlease train the model first by running:")
    print("  python train_model.py")
    sys.exit(1)

try:
    model = load_model(config.MODEL_SAVE_PATH)
    print(f"  ✓ Model loaded successfully from: {config.MODEL_SAVE_PATH}")
    print(f"  ✓ Model size: {os.path.getsize(config.MODEL_SAVE_PATH) / (1024*1024):.2f} MB")
except Exception as e:
    print(f"ERROR: Failed to load model - {str(e)}")
    sys.exit(1)

# ---------------------------------
# 2. SELECT IMAGE
# ---------------------------------
print("\n[2/3] Selecting Image...")
print("  Please select an image using the file dialog...")

img_path = select_image()

if not img_path or not os.path.exists(img_path):
    print("ERROR: No valid image selected!")
    sys.exit(1)

print(f"  ✓ Image selected: {os.path.basename(img_path)}")

# ---------------------------------
# 3. LOAD & PREPROCESS IMAGE
# ---------------------------------
print("\n[3/3] Processing Image...")

try:
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not read image file: {img_path}")
        sys.exit(1)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img_rgb, config.IMG_SIZE)
    
    # Normalize and add batch dimension
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    
    print(f"  ✓ Image loaded and preprocessed")
    print(f"  ✓ Original size: {img_rgb.shape}")
    print(f"  ✓ Model input size: {img_array.shape}")
    
except Exception as e:
    print(f"ERROR: Failed to process image - {str(e)}")
    sys.exit(1)

# ---------------------------------
# 4. PREDICT
# ---------------------------------
print("\n" + "=" * 60)
print("RUNNING PREDICTION...")
print("=" * 60)

try:
    pred_prob = model.predict(img_array, verbose=0)[0][0]
    
    # Determine class based on threshold
    # Note: class_indices are usually {'aadhar': 0, 'not_aadhar': 1}
    # Sigmoid output > 0.5 means class 1 (not_aadhar)
    pred_label = "Not Aadhaar" if pred_prob > config.PREDICTION_THRESHOLD else "Aadhaar"
    confidence = pred_prob if pred_prob > 0.5 else (1 - pred_prob)
    
    print(f"\nPrediction: {pred_label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Raw Score: {pred_prob:.4f} (< 0.5 = Aadhaar, > 0.5 = Not Aadhaar)")
    
except Exception as e:
    print(f"ERROR: Prediction failed - {str(e)}")
    sys.exit(1)

# ---------------------------------
# 5. DISPLAY RESULT
# ---------------------------------
print("\nDisplaying result...")
print("=" * 60)

# Create figure with result
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img_rgb)
ax.axis("off")

# Add title with prediction and confidence
title_color = "green" if pred_label == "Aadhaar" else "red"
ax.set_title(
    f"{pred_label}\nConfidence: {confidence * 100:.2f}%",
    fontsize=16,
    fontweight='bold',
    color=title_color,
    pad=20
)

# Add filename at bottom
plt.figtext(0.5, 0.02, f"File: {os.path.basename(img_path)}", 
            ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.show()

print("\n✓ Prediction complete!")
