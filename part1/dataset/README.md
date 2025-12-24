# Dataset Directory

This directory contains the training and testing datasets for the Aadhar Card Detection model.

## Directory Structure

```
dataset/
├── training/
│   ├── aadhar/         # Training images of Aadhar cards
│   └── not_aadhar/     # Training images without Aadhar cards
└── testing/
    ├── aadhar/         # Testing images of Aadhar cards
    └── not_aadhar/     # Testing images without Aadhar cards
```

## How to Prepare Your Dataset

### 1. Collect Images

**Aadhar Card Images:**
- Real Aadhar card photos
- Sample/dummy Aadhar cards (for practice)
- Various angles and lighting conditions
- Different backgrounds

**Non-Aadhar Images:**
- Other ID cards (PAN, Driving License, etc.)
- Random documents
- Regular photos without documents
- Similar looking cards that are not Aadhar

### 2. Organize Images

Place your collected images in the appropriate folders:

- `training/aadhar/` - 80% of your Aadhar card images
- `training/not_aadhar/` - 80% of your non-Aadhar images
- `testing/aadhar/` - 20% of your Aadhar card images
- `testing/not_aadhar/` - 20% of your non-Aadhar images

### 3. Dataset Size Recommendations

**Minimum (for learning):**
- 100 images per class for training
- 25 images per class for testing

**Recommended (for better accuracy):**
- 500+ images per class for training
- 100+ images per class for testing

**Professional:**
- 1000+ images per class for training
- 200+ images per class for testing

### 4. Image Quality Guidelines

- **Format:** JPG, JPEG, or PNG
- **Resolution:** Any (will be resized to 224x224)
- **Quality:** Clear and readable
- **Variety:** Different angles, lighting, backgrounds

### 5. Important Notes

- **Balance:** Keep roughly equal numbers of aadhar and not_aadhar images
- **Privacy:** If using real Aadhar cards, ensure you have permission and mask sensitive information
- **Augmentation:** The training script automatically applies data augmentation (rotation, zoom, shear)
- **Test Set:** Never train on test images - keep them separate for unbiased evaluation

## Sample Dataset Sources

For practice/learning purposes, you can:
1. Create dummy Aadhar cards using image editing tools
2. Use publicly available sample documents (with watermarks)
3. Download similar ID card datasets from platforms like Kaggle
4. Collect images from the internet (respecting copyright)

## Verify Your Dataset

Before training, you can verify your dataset structure by running:

```powershell
python src/config.py
```

This will show you the dataset paths and confirm directories exist.
