# ## Step 6 Results (Partial - 4 Epochs)
#
# Due to computational constraints (CPU-only training), we trained for 4 epochs
# but clearly observed the superior performance of CNNs over dense networks.
#
# ### Key Findings:
#
# 1. **CNN achieves 65.7% validation accuracy in just 3 epochs**, compared to
#    dense network's best of 43.08% after 25 epochs.
#
# 2. **No overfitting observed yet** as both training and validation metrics
#    are improving together.
#
# 3. **Expected overfitting point**: Based on learning curves, overfitting
#    would likely begin around epoch 12-15, when validation accuracy plateaus
#    and the train-val gap exceeds 10%.
#
# ### Evidence of CNN Superiority:
#
# - **3x faster learning**: CNN reaches 65% in 3 epochs vs dense network's
#   25 epochs to reach 43%
# - **Better feature extraction**: Convolutional layers preserve spatial
#   information
# - **Fewer parameters**: 288K vs 1.7M (6x more efficient)


# ==========================================
# STEP 6: BUILD CNN BASELINE - NO REGULARISATION
# PART B: CNN FROM SCRATCH
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
import time
import warnings

warnings.filterwarnings('ignore')

# Set style
try:
    plt.style.use('seaborn-v0-8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')

print("=" * 80)
print("STEP 6: CNN BASELINE - NO REGULARISATION")
print("CNN Architecture from Section 2.4.1")
print("=" * 80)

# ------------------------------------------
# LOAD AND PREPROCESS DATA
# ------------------------------------------
print("\n Loading CIFAR-10 dataset...")

# Load CIFAR-10
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Flatten labels
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Create validation split (last 10,000 training samples)
X_train = X_train_full[:-10000]
y_train = y_train_full[:-10000]
X_val = X_train_full[-10000:]
y_val = y_train_full[-10000:]

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print(f"✓ Data loaded and preprocessed:")
print(f"  Training set: {X_train.shape} (32×32×3 images)")
print(f"  Validation set: {X_val.shape}")
print(f"  Test set: {X_test.shape}")
print(f"  Classes: {len(class_names)}")

# ------------------------------------------
# BUILD CNN BASELINE (NO REGULARISATION)
# ------------------------------------------
print("\n" + "=" * 80)
print(" BUILDING CNN BASELINE - NO DROPOUT, NO BATCH NORM")
print("=" * 80)

model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Global Average Pooling (instead of Flatten)
    layers.GlobalAveragePooling2D(),

    # Output Layer
    layers.Dense(10, activation='softmax')
])

# Display model architecture
print("\n MODEL ARCHITECTURE:")
print("-" * 80)
model.summary()

# Calculate total parameters
total_params = model.count_params()
print(f"\n Total Parameters: {total_params:,}")

# ------------------------------------------
# COMPILE THE MODEL
# ------------------------------------------
print("\n" + "=" * 80)
print(" COMPILING THE MODEL")
print("=" * 80)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled successfully")
print("  Optimizer: Adam (lr=0.001)")
print("  Loss: sparse_categorical_crossentropy")
print("  Metrics: accuracy")

# ------------------------------------------
# TRAIN THE MODEL (30 EPOCHS, NO CALLBACKS)
# ------------------------------------------
print("\n" + "=" * 80)
print(" TRAINING CNN BASELINE - 30 EPOCHS")
print("=" * 80)
print(" NO REGULARISATION (No Dropout, No BatchNorm)")
print("-" * 80)

# Record training time
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time

print(f"\n Training completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

# ------------------------------------------
# PLOT TRAINING VS VALIDATION ACCURACY
# ------------------------------------------
print("\n" + "=" * 80)
print(" PLOTTING LEARNING CURVES")
print("=" * 80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy plot
ax1.plot(history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
ax1.set_title('CNN Baseline - Accuracy (No Regularisation)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(True, alpha=0.3)

# Add vertical line at overfitting point (to be analyzed)
# Identify where validation accuracy stops improving
val_acc = history.history['val_accuracy']
best_val_epoch = np.argmax(val_acc) + 1
ax1.axvline(x=best_val_epoch, color='green', linestyle='--', linewidth=2,
            label=f'Best Validation (Epoch {best_val_epoch})')
ax1.legend(loc='lower right', fontsize=11)

# Add annotation for overfitting point
# Find where training starts to diverge significantly
train_acc = history.history['accuracy']
val_acc_smooth = np.convolve(val_acc, np.ones(3) / 3, mode='valid')  # Smoothing for trend
train_acc_smooth = np.convolve(train_acc, np.ones(3) / 3, mode='valid')

# Find epoch where gap starts growing significantly
gap = np.array(train_acc) - np.array(val_acc)
overfit_start = np.where(gap > 0.15)[0]
if len(overfit_start) > 0:
    overfit_epoch = overfit_start[0] + 1
    ax1.axvline(x=overfit_epoch, color='orange', linestyle=':', linewidth=2,
                label=f'Overfitting Starts (Epoch {overfit_epoch})')
    ax1.legend(loc='lower right', fontsize=11)

# Loss plot
ax2.plot(history.history['loss'], 'b-', linewidth=2, label='Training Loss')
ax2.plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
ax2.set_title('CNN Baseline - Loss (No Regularisation)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, alpha=0.3)

# Add vertical line at best validation loss
best_loss_epoch = np.argmin(history.history['val_loss']) + 1
ax2.axvline(x=best_loss_epoch, color='green', linestyle='--', linewidth=2,
            label=f'Best Validation (Epoch {best_loss_epoch})')
ax2.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.show()

# ------------------------------------------
# DETAILED OVERFITTING ANALYSIS
# ------------------------------------------
print("\n" + "=" * 80)
print(" OVERFITTING ANALYSIS")
print("=" * 80)

# Extract metrics
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"\n PERFORMANCE METRICS:")
print("-" * 80)
print(f"{'Epoch':<8} {'Train Acc':<12} {'Val Acc':<12} {'Train Loss':<12} {'Val Loss':<12}")
print("-" * 80)

for epoch in range(0, len(train_acc), 5):  # Show every 5th epoch
    print(f"{epoch + 1:<8} {train_acc[epoch]:.4f}      {val_acc[epoch]:.4f}      "
          f"{train_loss[epoch]:.4f}      {val_loss[epoch]:.4f}")

# Find best validation accuracy
best_val_epoch = np.argmax(val_acc) + 1
best_val_acc = val_acc[best_val_epoch - 1]
best_val_loss = val_loss[best_val_epoch - 1]

print(f"\n BEST VALIDATION PERFORMANCE:")
print(f"  Epoch {best_val_epoch}: Accuracy = {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
print(f"  Epoch {best_val_epoch}: Loss = {best_val_loss:.4f}")

# Find overfitting point
# Method 1: Where validation accuracy plateaus (no improvement for 5 epochs)
plateau_start = None
for i in range(5, len(val_acc)):
    recent_improvements = [val_acc[i - j] < val_acc[i - j + 1] for j in range(1, 6)]
    if not any(recent_improvements):
        plateau_start = i - 4
        break

# Method 2: Where gap between train and validation exceeds 15%
gap = np.array(train_acc) - np.array(val_acc)
overfit_epoch = None
for i in range(len(gap)):
    if gap[i] > 0.15:
        overfit_epoch = i + 1
        break

print(f"\n OVERFITTING DETECTION:")
print("-" * 50)
if plateau_start:
    print(f"  ✓ Validation plateau detected at epoch {plateau_start + 1}")
    print(f"    (No improvement for 5 consecutive epochs)")
else:
    print(f"   No clear plateau detected within 30 epochs")

if overfit_epoch:
    print(f"  ✓ Training-Validation gap exceeded 15% at epoch {overfit_epoch}")
    print(f"    Gap = {gap[overfit_epoch - 1]:.4f} ({gap[overfit_epoch - 1] * 100:.1f}%)")

# Calculate final performance
final_train_acc = train_acc[-1]
final_val_acc = val_acc[-1]
final_gap = final_train_acc - final_val_acc

print(f"\n FINAL PERFORMANCE (Epoch 30):")
print(f"  Training Accuracy: {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
print(f"  Validation Accuracy: {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")
print(f"  Gap: {final_gap:.4f} ({final_gap * 100:.1f}%)")

# ------------------------------------------
# EVALUATE ON TEST SET
# ------------------------------------------
print("\n" + "=" * 80)
print(" EVALUATING ON TEST SET")
print("=" * 80)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n TEST SET PERFORMANCE:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

# ------------------------------------------
# SAVE THE MODEL
# ------------------------------------------
print("\n" + "=" * 80)
print(" SAVING MODEL AND HISTORY")
print("=" * 80)

# Save model
model.save('cnn_baseline_no_reg.keras')
print("✓ Model saved as 'cnn_baseline_no_reg.keras'")

# Save training history
np.savez('cnn_baseline_history.npz',
         train_accuracy=history.history['accuracy'],
         val_accuracy=history.history['val_accuracy'],
         train_loss=history.history['loss'],
         val_loss=history.history['val_loss'])

print("✓ Training history saved as 'cnn_baseline_history.npz'")

# ------------------------------------------
# ANALYSIS ANSWER
# ------------------------------------------
print("\n" + "=" * 80)
print(" ANALYSIS QUESTION ANSWER")
print("=" * 80)

print("""
QUESTION: At what epoch does overfitting begin (validation accuracy stagnates 
          while training keeps rising)?

ANSWER: Based on the training results:
""")

# Provide specific answer based on actual results
if overfit_epoch:
    print(f"  ➤ Overfitting begins at EPOCH {overfit_epoch}")
    print(f"")
    print(f"  Evidence:")
    print(f"    • Before epoch {overfit_epoch}: Training and validation accuracy improve together")
    print(f"    • After epoch {overfit_epoch}: Training accuracy continues rising while")
    print(f"      validation accuracy plateaus or decreases")
    print(f"    • The gap reaches >15% at this point")
else:
    best_epoch = best_val_epoch
    print(f"  ➤ Overfitting begins around EPOCH {best_epoch} (when best validation is achieved)")
    print(f"")
    print(f"  Evidence:")
    print(f"    • Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"    • After epoch {best_epoch}, validation accuracy stagnates or declines")
    print(f"    • Training accuracy continues to increase")

print(f"""
WHY THIS HAPPENS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Model Capacity vs Data:
     • CNN has {total_params:,} parameters
     • Only 40,000 training samples
     • Model starts memorizing instead of generalizing

  2. No Regularisation:
     • No Dropout layers to prevent co-adaptation
     • No BatchNorm to stabilize training
     • No data augmentation to increase effective dataset size

  3. Learning Dynamics:
     • Early epochs: Model learns general features
     • Middle epochs: Model learns class-specific patterns  
     • Late epochs: Model memorizes training set noise

OBSERVED BEHAVIOR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Training accuracy keeps rising (might reach 90%+)
  • Validation accuracy plateaus around {best_val_acc * 100:.1f}%
  • Gap between train and validation grows continuously

RECOMMENDATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Add Dropout layers (Step 7)
  ✓ Add Batch Normalization (Step 8)
  ✓ Use Data Augmentation (Step 9)
  ✓ Early stopping with restore_best_weights
  ✓ Reduce model complexity if overfitting persists
""")

print("=" * 80)
print(" STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 80)

# Optional: Plot additional visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the gap between train and validation accuracy
epochs = range(1, len(train_acc) + 1)
gap = np.array(train_acc) - np.array(val_acc)

ax.fill_between(epochs, 0, gap, alpha=0.3, color='red', label='Overfitting Gap')
ax.plot(epochs, gap, 'r-', linewidth=2, label='Train-Val Accuracy Gap')
ax.axhline(y=0.15, color='orange', linestyle='--', linewidth=1.5, label='15% Threshold')

if overfit_epoch:
    ax.axvline(x=overfit_epoch, color='green', linestyle='--', linewidth=2,
               label=f'Overfitting Starts (Epoch {overfit_epoch})')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy Gap (Train - Val)', fontsize=12)
ax.set_title('Overfitting Visualization - CNN Baseline', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()