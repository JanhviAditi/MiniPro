import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import config

# Verify dataset exists
if not os.path.exists(config.TRAIN_PATH):
    print(f"ERROR: Training path not found: {config.TRAIN_PATH}")
    print("\nPlease organize your dataset as follows:")
    print(f"  {config.TRAIN_PATH}/aadhar/     - Place Aadhar card images here")
    print(f"  {config.TRAIN_PATH}/not_aadhar/ - Place non-Aadhar images here")
    print(f"  {config.TEST_PATH}/aadhar/      - Place test Aadhar images here")
    print(f"  {config.TEST_PATH}/not_aadhar/  - Place test non-Aadhar images here")
    print("\nRun 'python config.py' to create the directory structure.")
    sys.exit(1)

if not os.path.exists(config.TEST_PATH):
    print(f"ERROR: Testing path not found: {config.TEST_PATH}")
    print("\nRun 'python config.py' to create the directory structure.")
    sys.exit(1)

print("=" * 60)
print("AADHAR CARD DETECTION - MODEL TRAINING")
print("=" * 60)
print(f"\nConfiguration:")
print(f"  Training Path: {config.TRAIN_PATH}")
print(f"  Testing Path: {config.TEST_PATH}")
print(f"  Image Size: {config.IMG_SIZE}")
print(f"  Batch Size: {config.BATCH_SIZE}")
print(f"  Phase 1 Epochs: {config.EPOCHS_PHASE1}")
print(f"  Phase 2 Epochs: {config.EPOCHS_PHASE2}")
print("=" * 60)

# ---------------------------------
# 1. DATA PREPARATION
# ---------------------------------
print("\n[1/5] Preparing Data Generators...")

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=config.ROTATION_RANGE,
    zoom_range=config.ZOOM_RANGE,
    shear_range=config.SHEAR_RANGE,
    horizontal_flip=config.HORIZONTAL_FLIP
)

test_datagen = ImageDataGenerator(rescale=1/255)

try:
    train_dataset = train_datagen.flow_from_directory(
        config.TRAIN_PATH,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    test_dataset = test_datagen.flow_from_directory(
        config.TEST_PATH,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"  ✓ Training samples: {train_dataset.samples}")
    print(f"  ✓ Testing samples: {test_dataset.samples}")
    print(f"  ✓ Classes found: {train_dataset.class_indices}")
    
except Exception as e:
    print(f"\nERROR: Failed to load dataset - {str(e)}")
    print("\nMake sure your dataset is organized correctly:")
    print("  dataset/training/aadhar/ - with images")
    print("  dataset/training/not_aadhar/ - with images")
    print("  dataset/testing/aadhar/ - with images")
    print("  dataset/testing/not_aadhar/ - with images")
    sys.exit(1)

# ---------------------------------
# 2. BASE MODEL
# ---------------------------------
print("\n[2/5] Building Model Architecture...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(config.IMG_WIDTH, config.IMG_HEIGHT, config.IMG_CHANNELS),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Freeze initial layers
print(f"  ✓ Base model loaded: MobileNetV2")
print(f"  ✓ Total base layers: {len(base_model.layers)}")

# ---------------------------------
# 3. CUSTOM TOP LAYERS
# ---------------------------------
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(config.DENSE_UNITS, activation='relu'),
    tf.keras.layers.Dropout(config.DROPOUT_RATE),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE_PHASE1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"  ✓ Custom layers added")
print(f"  ✓ Total trainable parameters: {model.count_params():,}")

# Model summary
print("\nModel Architecture:")
model.summary()

# Create callbacks
os.makedirs(config.MODEL_DIR, exist_ok=True)
checkpoint = ModelCheckpoint(
    config.MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# ---------------------------------
# 4. TRAIN PHASE 1
# ---------------------------------
print("\n" + "=" * 60)
print("[3/5] TRAINING PHASE 1 - Base Layers Frozen")
print("=" * 60)
print(f"Training for {config.EPOCHS_PHASE1} epochs...")

history_phase1 = model.fit(
    train_dataset,
    epochs=config.EPOCHS_PHASE1,
    validation_data=test_dataset,
    callbacks=[checkpoint]
)

print(f"\n✓ Phase 1 Complete!")
print(f"  Final Training Accuracy: {history_phase1.history['accuracy'][-1]:.4f}")
print(f"  Final Validation Accuracy: {history_phase1.history['val_accuracy'][-1]:.4f}")

# ---------------------------------
# 5. FINE-TUNING
# ---------------------------------
print("\n" + "=" * 60)
print(f"[4/5] TRAINING PHASE 2 - Fine-Tuning Last {config.FINE_TUNE_LAYERS} Layers")
print("=" * 60)

# Unfreeze the last layers
for layer in base_model.layers[-config.FINE_TUNE_LAYERS:]:
    layer.trainable = True

print(f"  ✓ Unfrozen last {config.FINE_TUNE_LAYERS} layers")
print(f"  ✓ New trainable parameters: {sum([tf.size(var).numpy() for var in model.trainable_variables]):,}")

model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE_PHASE2),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\nFine-tuning for {config.EPOCHS_PHASE2} epochs...")

history_phase2 = model.fit(
    train_dataset,
    epochs=config.EPOCHS_PHASE2,
    validation_data=test_dataset,
    callbacks=[checkpoint]
)

print(f"\n✓ Phase 2 Complete!")
print(f"  Final Training Accuracy: {history_phase2.history['accuracy'][-1]:.4f}")
print(f"  Final Validation Accuracy: {history_phase2.history['val_accuracy'][-1]:.4f}")

# ---------------------------------
# 6. SAVE FINAL MODEL
# ---------------------------------
print("\n" + "=" * 60)
print("[5/5] SAVING MODEL")
print("=" * 60)

# Ensure directory exists
os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

# Save the final model
model.save(config.MODEL_SAVE_PATH)

print(f"✓ Model saved successfully!")
print(f"  Location: {config.MODEL_SAVE_PATH}")
print(f"  File size: {os.path.getsize(config.MODEL_SAVE_PATH) / (1024*1024):.2f} MB")

# ---------------------------------
# 7. TRAINING SUMMARY
# ---------------------------------
print("\n" + "=" * 60)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 60)
print(f"Phase 1 (Frozen Base):")
print(f"  Starting Accuracy: {history_phase1.history['accuracy'][0]:.4f}")
print(f"  Final Accuracy: {history_phase1.history['accuracy'][-1]:.4f}")
print(f"  Final Val Accuracy: {history_phase1.history['val_accuracy'][-1]:.4f}")
print(f"\nPhase 2 (Fine-Tuning):")
print(f"  Starting Accuracy: {history_phase2.history['accuracy'][0]:.4f}")
print(f"  Final Accuracy: {history_phase2.history['accuracy'][-1]:.4f}")
print(f"  Final Val Accuracy: {history_phase2.history['val_accuracy'][-1]:.4f}")
print(f"\nModel Location: {config.MODEL_SAVE_PATH}")
print("\n✓ You can now run 'python predict_image.py' to test your model!")
print("=" * 60)


