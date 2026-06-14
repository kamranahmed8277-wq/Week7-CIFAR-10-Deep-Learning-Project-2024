
# Step 8: Add Dropout — Full Regularised CNN

## Objective
# Build a fully regularised CNN combining BatchNormalization and Dropout to prevent overfitting and achieve optimal generalization performance on CIFAR-10 dataset.

## Architecture Overview

### Complete Network Structure

# Predicted Progression:
# Epoch 8-10:  71-72% validation accuracy
# Epoch 11-13: 72-73% validation accuracy (peak)
# Epoch 14-15: 72-73% (plateau)
#
# Expected Best Validation Accuracy: 72-74%
# Expected Test Accuracy: 71-73%
# Optimal Stopping Point: Epoch 12-14





### Performance Comparison Table

# ==========================================
# STEP 8: ADD DROPOUT — FULL REGULARISED CNN
# OPTIMIZED FOR SPEED (4x FASTER TRAINING)
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
print("STEP 8: FULL REGULARISED CNN (OPTIMIZED - 4x FASTER)")
print("BatchNorm + Dropout (0.25 after MaxPool, 0.5 before Dense)")
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

print(f"✓ Data loaded:")
print(f"  Training: {X_train.shape}")
print(f"  Validation: {X_val.shape}")
print(f"  Test: {X_test.shape}")

# ------------------------------------------
# BUILD OPTIMIZED REGULARISED CNN (Smaller, Faster)
# ------------------------------------------
print("\n" + "=" * 80)
print("🏗️ BUILDING OPTIMIZED CNN (Smaller for Speed)")
print("=" * 80)

model_full = keras.Sequential([
    # First Convolutional Block - Reduced filters
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Second Convolutional Block - Reduced filters
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Third Convolutional Block - Reduced filters
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Global Average Pooling
    layers.GlobalAveragePooling2D(),

    # Dropout before final layer
    layers.Dropout(0.5),

    # Output Layer
    layers.Dense(10, activation='softmax')
])

# Display model architecture
print("\n OPTIMIZED CNN ARCHITECTURE:")
print("-" * 80)
model_full.summary()

# Calculate total parameters
total_params_full = model_full.count_params()
print(f"\nTotal Parameters: {total_params_full:,}")

# ------------------------------------------
# COMPILE WITH OPTIMIZED SETTINGS
# ------------------------------------------
print("\n" + "=" * 80)
print(" COMPILING WITH OPTIMIZED SETTINGS")
print("=" * 80)

# Use slightly higher learning rate for faster convergence
optimizer = keras.optimizers.Adam(learning_rate=0.0015)  # Increased from 0.001

model_full.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled with optimized settings:")
print("  Optimizer: Adam (lr=0.0015) - 50% higher for faster convergence")
print("  Loss: sparse_categorical_crossentropy")

# ------------------------------------------
# OPTIMIZED CALLBACKS (Reduced patience for faster stopping)
# ------------------------------------------
print("\n" + "=" * 80)
print(" OPTIMIZED CALLBACKS (Faster Early Stopping)")
print("=" * 80)

# More aggressive early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,  # Reduced from 15 to 8
    restore_best_weights=True,
    verbose=1
)

# ModelCheckpoint (save only best)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    'cnn_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# More aggressive LR reduction
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,  # Reduced from 7 to 4
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

print("✓ Optimized callbacks configured:")
print(f"  • EarlyStopping: patience=8 (was 15) - Saves 35 min")
print(f"  • ReduceLROnPlateau: patience=4 (was 7) - More responsive")
print(f"  • Batch Size: 128 (2x larger = 2x faster)")

# ------------------------------------------
# TRAIN WITH OPTIMIZED PARAMETERS (UP TO 40 EPOCHS)
# ------------------------------------------
print("\n" + "=" * 80)
print(" TRAINING FULLY REGULARISED CNN (OPTIMIZED)")
print("=" * 80)
print(f"  Max Epochs: 40 (reduced from 100)")
print(f"  Batch Size: 128 (increased from 64 - 2x faster)")
print(f"  Expected time per epoch: ~2-2.5 minutes (was 5 minutes)")
print(f"  Training samples: {X_train.shape[0]:,}")
print(f"  Validation samples: {X_val.shape[0]:,}")
print("-" * 80)

# Record start time
start_time_full = time.time()

history_full = model_full.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,  # Reduced from 100 to 40
    batch_size=128,  # Increased from 64 to 128 (2x faster)
    callbacks=callbacks,
    verbose=1
)

# Calculate training time
end_time_full = time.time()
training_time_full = end_time_full - start_time_full
training_time_minutes = training_time_full / 60

print("\n" + "=" * 80)
print(" TRAINING COMPLETE!")
print("=" * 80)
print(f"  Total training time: {training_time_full:.2f} seconds ({training_time_minutes:.2f} minutes)")
print(f"  Total epochs completed: {len(history_full.history['loss'])}")
print(f"  Early stopping triggered: {'Yes' if len(history_full.history['loss']) < 40 else 'No'}")

# ------------------------------------------
# QUICK ANALYSIS
# ------------------------------------------
print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

# Extract metrics
train_acc = history_full.history['accuracy']
val_acc = history_full.history['val_accuracy']
train_loss = history_full.history['loss']
val_loss = history_full.history['val_loss']

# Best validation accuracy
best_val_acc = max(val_acc)
best_val_epoch = np.argmax(val_acc) + 1
best_val_loss = min(val_loss)

# Final metrics
final_train_acc = train_acc[-1]
final_val_acc = val_acc[-1]
final_epochs = len(train_acc)

print(f"\n BEST PERFORMANCE:")
print(f"  Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
print(f"  Best Validation Loss: {best_val_loss:.4f}")
print(f"  Achieved at Epoch: {best_val_epoch}")

print(f"\n FINAL PERFORMANCE (Epoch {final_epochs}):")
print(f"  Training Accuracy: {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
print(f"  Validation Accuracy: {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")
print(f"  Gap: {(final_train_acc - final_val_acc) * 100:.2f}%")

# Learning rate analysis
lr_history = history_full.history.get('lr', [0.0015] * len(train_acc))
print(f"\nLearning Rate Schedule:")
print(f"  Initial LR: 0.0015")
print(f"  Final LR: {lr_history[-1]:.6f}")

# ------------------------------------------
# QUICK EVALUATION
# ------------------------------------------
print("\n" + "=" * 80)
print("TEST SET EVALUATION")
print("=" * 80)

test_loss_full, test_accuracy_full = model_full.evaluate(X_test, y_test, verbose=0)

print(f"\nTEST SET PERFORMANCE:")
print(f"  Test Loss: {test_loss_full:.4f}")
print(f"  Test Accuracy: {test_accuracy_full:.4f} ({test_accuracy_full * 100:.2f}%)")

# ------------------------------------------
# SIMPLIFIED PLOTS (Faster rendering)
# ------------------------------------------
print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1 = axes[0]
ax1.plot(train_acc, 'b-', linewidth=2, label='Training Accuracy')
ax1.plot(val_acc, 'r-', linewidth=2, label='Validation Accuracy')
ax1.scatter(best_val_epoch - 1, best_val_acc, color='green', s=100, zorder=5,
            label=f'Best: {best_val_acc:.4f}')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Full Regularised CNN - Accuracy', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Loss plot
ax2 = axes[1]
ax2.plot(train_loss, 'b-', linewidth=2, label='Training Loss')
ax2.plot(val_loss, 'r-', linewidth=2, label='Validation Loss')
ax2.scatter(best_val_epoch - 1, best_val_loss, color='green', s=100, zorder=5,
            label=f'Best: {best_val_loss:.4f}')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Full Regularised CNN - Loss', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.suptitle('Step 8: Optimized Regularised CNN (40 Epochs Max)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ------------------------------------------
# QUICK COMPARISON
# ------------------------------------------
print("\n" + "=" * 80)
print(" COMPARISON WITH PREVIOUS STEPS")
print("=" * 80)

comparison = {
    'Model': ['Dense Network', 'CNN Baseline', 'CNN + BN', 'CNN + BN + Dropout'],
    'Best Val Acc': [43.08, 68.00, 72.43, best_val_acc * 100],
    'Time (min)': ['N/A', '~50', '~40', f'{training_time_minutes:.1f}']
}

print(f"\n{'Model':<22} {'Best Val Acc':<15} {'Training Time':<15}")
print("-" * 55)
for i in range(len(comparison['Model'])):
    print(f"{comparison['Model'][i]:<22} {comparison['Best Val Acc'][i]:>6.2f}%{'':<7} "
          f"{comparison['Time (min)'][i]:<15}")

improvement = best_val_acc * 100 - 68.00
print(f"\n IMPROVEMENT from CNN Baseline: +{improvement:.2f}%")
print(f"⚡ SPEED: {training_time_minutes:.1f} minutes total (vs 4+ hours original)")

# ------------------------------------------
# SAVE RESULTS
# ------------------------------------------
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save model
model_full.save('cnn_full_regularised_optimized.keras')
print("✓ Model saved as 'cnn_full_regularised_optimized.keras'")

# Save history
np.savez('cnn_full_regularised_history_optimized.npz',
         train_accuracy=train_acc,
         val_accuracy=val_acc,
         train_loss=train_loss,
         val_loss=val_loss,
         learning_rates=lr_history,
         best_val_accuracy=best_val_acc,
         best_val_epoch=best_val_epoch,
         training_time=training_time_full)

print("✓ Training history saved")

# ------------------------------------------
# SPEED OPTIMIZATION SUMMARY
# ------------------------------------------
print("\n" + "=" * 80)
print("⚡ SPEED OPTIMIZATION SUMMARY")
print("=" * 80)

print("""
OPTIMIZATIONS APPLIED (4x FASTER):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. BATCH SIZE: 64 → 128 (2x faster)
   • Fewer batches per epoch: 625 → 313 steps
   • Time per epoch: 5 min → 2.5 min

2. MAX EPOCHS: 100 → 40 (2.5x faster)
   • Reduced from 100 to 40 maximum
   • Still enough to see convergence

3. EARLY STOPPING: patience 15 → 8
   • Stops 7 epochs earlier when plateau detected
   • Saves ~35 minutes

4. LEARNING RATE: 0.001 → 0.0015
   • 50% higher initial LR
   • Faster convergence in early epochs

5. REDUCE LR PATIENCE: 7 → 4
   • More responsive LR reduction
   • Adapts faster to plateaus

EXPECTED IMPROVEMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Original time: ~4-5 hours
• Optimized time: ~1-1.5 hours
• Speedup: 3-4x FASTER!

ACCURACY TRADE-OFF:
• Expected accuracy: 73-75% (vs 75-77% original)
• Small 1-2% drop for 4x speed improvement
• Well worth the trade-off for experimentation
""")

print("=" * 80)
print("STEP 8 COMPLETED SUCCESSFULLY (OPTIMIZED VERSION)")
print("=" * 80)

# Optional: Print what to expect next
print("\n" + "=" * 80)
print(" WHAT TO EXPECT FROM OPTIMIZED TRAINING")
print("=" * 80)
print("""
TYPICAL PROGRESSION (Optimized):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 1-5:   40-50% accuracy (rapid learning)
Epoch 6-10:  55-65% accuracy (steady improvement)
Epoch 11-15: 65-72% accuracy (slowing down)
Epoch 16-20: 72-74% accuracy (peak performance)
Epoch 20-25: 73-75% accuracy (plateau, early stopping triggers)

Total training time: 60-90 minutes (vs 4-5 hours original)
Best accuracy: 73-75% (vs 75-77% original)

✓ OPTIMIZATION SUCCESSFUL - Ready to train!
""")