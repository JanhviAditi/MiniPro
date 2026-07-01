# Aadhar Card Detection System

A deep learning-based system to detect and classify whether an image contains an Aadhar card or not using MobileNetV2 and TensorFlow.

## 📋 Project Overview

This project implements a binary image classifier that can distinguish between Aadhar card images and non-Aadhar card images. It uses transfer learning with MobileNetV2 as the base model and custom layers for classification.

## ✅ Project Status: **FULLY TRAINED & OPERATIONAL**

- **Model:** Successfully trained on 886 images
- **Training Data:** 360 Aadhar + 334 Non-Aadhar images
- **Testing Data:** 83 Aadhar + 109 Non-Aadhar images
- **Model Size:** 20.3 MB
- **Last Trained:** November 21, 2025
- **Status:** Ready for predictions ✓

## 🚀 Features

- **Transfer Learning**: Uses pre-trained MobileNetV2 for efficient feature extraction
- **Data Augmentation**: Improves model generalization with rotation, zoom, and shear
- **Fine-tuning**: Two-phase training for optimal performance
- **Easy Prediction**: Simple GUI-based image selection for testing
- **Model Persistence**: Saves trained model for future use

## 📁 Project Structure

```
Aadhar-Part1/
├── README.md                 # Project documentation
├── TRAINING_GUIDE.md        # Detailed training instructions
├── requirements.txt         # Python dependencies
├── dataset/                 # Training and testing data
│   ├── training/
│   │   ├── aadhar/         # Aadhar card images
│   │   └── not_aadhar/     # Non-Aadhar images
│   └── testing/
│       ├── aadhar/         # Test Aadhar images
│       └── not_aadhar/     # Test non-Aadhar images
├── model/                   # Saved trained models
│   └── aadhar_model.keras  # Trained model (created after training)
└── src/
    ├── config.py           # Configuration settings
    ├── train_model.py      # Model training script
    ├── predict_image.py    # Prediction script
    └── select_image.py     # Image selection utility
```

## 🛠️ Installation

✅ **Environment Already Set Up!**

### Current Setup
- **Python:** 3.13.3 (Virtual Environment)
- **TensorFlow:** 2.20.0 ✓
- **All Dependencies:** Installed ✓
- **GPU:** Not available (CPU training used)

### For New Setup

1. **Clone or download this repository**
   ```powershell
   cd c:\Users\91966\OneDrive\Desktop\part1\Aadhar-Part1
   ```

2. **Install required packages**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```powershell
   python -c "import tensorflow; print(tensorflow.__version__)"
   ```

**Note:** This project uses a virtual environment located at `.venv/`

## 📊 Dataset Preparation

✅ **Dataset Already Prepared!** The dataset is organized and ready.

**Current Dataset:**
- **Training Aadhar:** 360 images ✓
- **Training Non-Aadhar:** 334 images ✓
- **Testing Aadhar:** 83 images ✓
- **Testing Non-Aadhar:** 109 images ✓
- **Total:** 886 images
- **Class Balance:** Good ✓

**Dataset Structure:**
```
dataset/
├── training/
│   ├── aadhar/         # 360 Aadhar card images
│   └── not_aadhar/     # 334 Non-Aadhar images
└── testing/
    ├── aadhar/         # 83 test Aadhar images
    └── not_aadhar/     # 109 test non-Aadhar images
```

**To add more data:** Simply copy additional images into the respective folders and retrain the model.

## 🎯 Training the Model

✅ **Model Already Trained!** The model has been successfully trained and is ready to use.

If you need to retrain, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions.

**To retrain:**
```powershell
cd src
python train_model.py
```

The training process includes:
- Phase 1: Training top layers (10 epochs)
- Phase 2: Fine-tuning last 20 layers (3 epochs)
- Model saved to `model/aadhar_model.keras`

**Current Model Stats:**
- Training completed successfully
- Dataset: 886 total images (694 training, 192 testing)
- Model file: `model/aadhar_model.keras` (20.3 MB)

## 🔮 Making Predictions

✅ **Ready to Use!** The model is trained and ready for predictions.

**Run predictions:**
```powershell
cd src
python predict_image.py
```

This will:
1. Open a file dialog to select any image from your computer
2. Load the trained model automatically
3. Process and predict the image (Aadhar or Not Aadhar)
4. Display the result with confidence score in a window

**You can test with:**
- Images from your test dataset
- Any Aadhar card image from your computer
- Random photos to verify it correctly identifies non-Aadhar images

## 📈 Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Shape**: 224×224×3 (RGB images)
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense(128, activation='relu')
  - Dropout(0.3)
  - Dense(1, activation='sigmoid')
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

## ⚙️ Configuration

Edit `src/config.py` to customize:
- Dataset paths
- Model save location
- Training parameters (epochs, batch size, learning rate)
- Image dimensions

## 📝 Requirements

- tensorflow >= 2.10.0
- opencv-python >= 4.5.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0
- pillow >= 8.3.0
- tk (for GUI dialogs)

## 🎓 Training Tips

1. **More Data = Better Results**: Current dataset (886 images) is good; 1000+ would be excellent
2. **Balanced Dataset**: ✅ Current balance is good (360 vs 334 training images)
3. **Image Quality**: Use clear, well-lit images
4. **Variety**: Include different angles, backgrounds, and conditions
5. **GPU Recommended**: Training will be much faster with a CUDA-enabled GPU (CPU used: ~25-35 min)

## 🚀 Quick Start Guide

**To make predictions right now:**
```powershell
cd src
python predict_image.py
```
Then select any image from your computer and see the results!

**Project is fully operational and ready to use!** ✓

## 🐛 Troubleshooting

**Model not found error:**
- ✅ Model exists at `model/aadhar_model.keras`
- If retaining, ensure training completes without errors

**To test the model:**
- Run `python src/predict_image.py`
- Select any image when the file dialog opens
- View prediction results in the display window

**To retrain:**
- Run `python src/train_model.py` from the src directory
- Training takes 25-35 minutes on CPU
- Model automatically saves to `model/aadhar_model.keras`

**Out of memory:**
- Reduce batch size in `config.py`
- Use a smaller image size

## 📄 License

This project is open source and available for educational purposes.
