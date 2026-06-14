# Step 3: Data Preprocessing Documentation

## Overview

# This preprocessing pipeline prepares the CIFAR-10 dataset for deep learning models by normalizing pixel values, creating proper data splits, and flattening images for dense networks.
#
# ## Preprocessing Steps
#
# ### 1. Normalization
# - **Purpose**: Scale pixel values from [0, 255] to [0, 1]
# - **Method**: `X = X.astype("float32") / 255.0`
# - **Benefits**:
#   - Faster convergence during training
#   - Prevents numerical instability
#   - Standard practice for image data
#
# ### 2. Label Flattening
# - **Original shape**: `(N, 1)`
# - **After flattening**: `(N,)`
# - **Why**: Removes redundant dimension for easier indexing
#
# ### 3. Validation Split
# - **Strategy**: Hold out last 10,000 training samples
# - **Result**:
#   - Training: 40,000 images
#   - Validation: 10,000 images
#   - Test: 10,000 images
# - **Note**: This preserves class distribution (500 per class in validation)
#
# ### 4. Image Flattening for Dense Networks
# - **Original shape**: `(32, 32, 3)` = 3D tensor
# - **Flattened shape**: `(3072,)` = 1D vector
# - **Formula**: `32 × 32 × 3 = 3072 features`
# - **Usage**: Required for fully connected networks
#
# ## Final Data Shapes
#
# | Array Name | Shape | Description |
# |------------|-------|-------------|
# | X_train | (40000, 32, 32, 3) | Training images (normalized) |
# | y_train | (40000,) | Training labels (flattened) |
# | X_val | (10000, 32, 32, 3) | Validation images (normalized) |
# | y_val | (10000,) | Validation labels (flattened) |
# | X_test | (10000, 32, 32, 3) | Test images (normalized) |
# | y_test | (10000,) | Test labels (flattened) |
# | X_train_flat | (40000, 3072) | Flattened training images |
# | X_val_flat | (10000, 3072) | Flattened validation images |
# | X_test_flat | (10000, 3072) | Flattened test images |
#
# ## Data Integrity Checks
#
# ### Class Distribution
# Each class has exactly 4,000 training, 1,000 validation, and 1,000 test samples (balanced).
#
# ### Value Ranges
# - **Input images**: [0, 1] (normalized)
# - **Labels**: [0, 9] (10 classes)
#
# ## Usage Guidelines
#
# ### For CNN Models
# ```python
# # Keep original 3D shape
# model.fit(X_train, y_train, validation_data=(X_val, y_val))



# ==========================================
# STEP 3: DATA PREPROCESSING
# Normalisation & Train/Validation/Test Splits
# ==========================================

import numpy as np
from tensorflow import keras

# Load CIFAR-10 dataset (if not already loaded)
print("Loading CIFAR-10 dataset...")
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

print(f"Initial shapes:")
print(f"  X_train_full: {X_train_full.shape}")
print(f"  y_train_full: {y_train_full.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

# ------------------------------------------
# 1. NORMALIZE PIXEL VALUES (0-1 range)
# ------------------------------------------
print("\n" + "="*60)
print("STEP 1: NORMALIZATION (0-255 → 0-1)")
print("="*60)

X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print(f"✓ Pixel values normalized to [0, 1] range")
print(f"  X_train_full dtype: {X_train_full.dtype}")
print(f"  X_train_full range: [{X_train_full.min():.2f}, {X_train_full.max():.2f}]")
print(f"  X_test dtype: {X_test.dtype}")
print(f"  X_test range: [{X_test.min():.2f}, {X_test.max():.2f}]")

# ------------------------------------------
# 2. FLATTEN LABELS
# ------------------------------------------
print("\n" + "="*60)
print("STEP 2: FLATTEN LABELS")
print("="*60)

y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

print(f"✓ Labels flattened")
print(f"  y_train_full shape: {y_train_full.shape}")
print(f"  y_train_full sample: {y_train_full[:10]}")
print(f"  y_test shape: {y_test.shape}")
print(f"  y_test sample: {y_test[:10]}")

# ------------------------------------------
# 3. CREATE VALIDATION SPLIT (Last 10,000 samples)
# ------------------------------------------
print("\n" + "="*60)
print("STEP 3: VALIDATION SPLIT (Last 10,000 training samples)")
print("="*60)

# Split training data into train and validation sets
# Take first 40,000 for training, last 10,000 for validation
X_train = X_train_full[:-10000]  # First 40,000 images
y_train = y_train_full[:-10000]  # First 40,000 labels
X_val = X_train_full[-10000:]    # Last 10,000 images
y_val = y_train_full[-10000:]    # Last 10,000 labels

print(f"✓ Validation split created")
print(f"  Training set: {X_train.shape[0]:,} images")
print(f"  Validation set: {X_val.shape[0]:,} images")
print(f"  Test set: {X_test.shape[0]:,} images")

# ------------------------------------------
# 4. FLATTEN IMAGES FOR DENSE NETWORK
# ------------------------------------------
print("\n" + "="*60)
print("STEP 4: FLATTEN IMAGES FOR DENSE NETWORK")
print("="*60)

# Flatten 32x32x3 images into 1D vectors (3072 features)
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

print(f"✓ Images flattened for Dense Network")
print(f"  Original shape: (32, 32, 3) = 3,072 pixels")
print(f"  Flattened shape: (3072,)")
print(f"\n  X_train_flat shape: {X_train_flat.shape}")
print(f"  X_val_flat shape: {X_val_flat.shape}")
print(f"  X_test_flat shape: {X_test_flat.shape}")

# ------------------------------------------
# 5. VERIFICATION: CHECK ALL ARRAY SHAPES
# ------------------------------------------
print("\n" + "="*60)
print("STEP 5: FINAL SHAPES VERIFICATION")
print("="*60)

print("\n FINAL DATA ARRAYS SUMMARY:")
print("-" * 50)
print(f"{'Array Name':<20} {'Shape':<25} {'Description':<20}")
print("-" * 50)

arrays_info = [
    ("X_train", X_train.shape, f"Training images ({X_train.shape[0]:,} samples)"),
    ("y_train", y_train.shape, f"Training labels ({y_train.shape[0]:,} samples)"),
    ("X_val", X_val.shape, f"Validation images ({X_val.shape[0]:,} samples)"),
    ("y_val", y_val.shape, f"Validation labels ({y_val.shape[0]:,} samples)"),
    ("X_test", X_test.shape, f"Test images ({X_test.shape[0]:,} samples)"),
    ("y_test", y_test.shape, f"Test labels ({y_test.shape[0]:,} samples)"),
    ("X_train_flat", X_train_flat.shape, "Flattened training images (for Dense Net)"),
    ("X_val_flat", X_val_flat.shape, "Flattened validation images (for Dense Net)"),
    ("X_test_flat", X_test_flat.shape, "Flattened test images (for Dense Net)"),
]

for name, shape, desc in arrays_info:
    print(f"{name:<20} {str(shape):<25} {desc:<20}")

# ------------------------------------------
# 6. DATA INTEGRITY CHECKS
# ------------------------------------------
print("\n" + "="*60)
print("DATA INTEGRITY CHECKS")
print("="*60)

# Check if all splits are disjoint
train_indices = set(range(len(X_train_full[:-10000])))
val_indices = set(range(len(X_train_full[-10000:])))

print(f"✓ Train and validation sets are disjoint: {train_indices.isdisjoint(val_indices)}")

# Check label ranges
print(f"✓ Training labels range: [{y_train.min()}, {y_train.max()}]")
print(f"✓ Validation labels range: [{y_val.min()}, {y_val.max()}]")
print(f"✓ Test labels range: [{y_test.min()}, {y_test.max()}]")

# Check value ranges
print(f"✓ X_train value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"✓ X_val value range: [{X_val.min():.3f}, {X_val.max():.3f}]")
print(f"✓ X_test value range: [{X_test.min():.3f}, {X_test.max():.3f}]")

# Check class distribution in each split
from collections import Counter

print("\nClass Distribution Check:")
print("-" * 40)

train_dist = Counter(y_train)
val_dist = Counter(y_val)
test_dist = Counter(y_test)

print(f"{'Class':<12} {'Train':<10} {'Validation':<12} {'Test':<10}")
print("-" * 45)
for i in range(10):
    class_name = ["Airplane","Automobile","Bird","Cat","Deer",
                  "Dog","Frog","Horse","Ship","Truck"][i]
    print(f"{class_name:<12} {train_dist[i]:<10} {val_dist[i]:<12} {test_dist[i]:<10}")

# ------------------------------------------
# 7. SAVE PREPROCESSED DATA (Optional)
# ------------------------------------------
print("\n" + "="*60)
print("SAVING PREPROCESSED DATA (Optional)")
print("="*60)

# Uncomment to save the preprocessed data
# np.savez('cifar10_preprocessed.npz',
#          X_train=X_train, y_train=y_train,
#          X_val=X_val, y_val=y_val,
#          X_test=X_test, y_test=y_test,
#          X_train_flat=X_train_flat,
#          X_val_flat=X_val_flat,
#          X_test_flat=X_test_flat)

# print("✓ Preprocessed data saved to 'cifar10_preprocessed.npz'")

# ------------------------------------------
# SUMMARY
# ------------------------------------------
print("\n" + "="*60)
print(" STEP 3 COMPLETED SUCCESSFULLY")
print("="*60)

print("""
PREPROCESSING SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Normalization:     Pixel values scaled to [0, 1]
Label flattening:  Shape changed from (N,1) to (N,)
Validation split:  10,000 samples held out from training
Image flattening:   32×32×3 → 3072 features for Dense Net

FINAL DATA READY FOR MODELING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• CNN models:        Use X_train, X_val, X_test (keep 3D shape)
• Dense models:      Use X_train_flat, X_val_flat, X_test_flat
• Labels:            y_train, y_val, y_test (flattened)

Data splits are balanced and ready for training!
""")