"""
Configuration file for Aadhar Card Detection System
Centralized settings for paths, model parameters, and training configuration
"""
import os

# Base directory - project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_PATH = os.path.join(DATASET_DIR, "training")
TEST_PATH = os.path.join(DATASET_DIR, "testing")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "aadhar_model.keras")

# Image parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# Training parameters
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10  # Initial training with frozen base
EPOCHS_PHASE2 = 3   # Fine-tuning phase
LEARNING_RATE_PHASE1 = 0.0005
LEARNING_RATE_PHASE2 = 0.0001

# Data augmentation parameters
ROTATION_RANGE = 5
ZOOM_RANGE = 0.1
SHEAR_RANGE = 0.1
HORIZONTAL_FLIP = False

# Model architecture parameters
DENSE_UNITS = 128
DROPOUT_RATE = 0.3
FINE_TUNE_LAYERS = 20  # Number of layers to unfreeze for fine-tuning

# Class names
CLASS_NAMES = ['aadhar', 'not_aadhar']

# Prediction threshold
PREDICTION_THRESHOLD = 0.5

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        MODEL_DIR,
        TRAIN_PATH,
        TEST_PATH,
        os.path.join(TRAIN_PATH, "aadhar"),
        os.path.join(TRAIN_PATH, "not_aadhar"),
        os.path.join(TEST_PATH, "aadhar"),
        os.path.join(TEST_PATH, "not_aadhar")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print("Directory structure created successfully!")

if __name__ == "__main__":
    # Create directories when config is run directly
    create_directories()
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Training Path: {TRAIN_PATH}")
    print(f"Testing Path: {TEST_PATH}")
    print(f"Model Save Path: {MODEL_SAVE_PATH}")
