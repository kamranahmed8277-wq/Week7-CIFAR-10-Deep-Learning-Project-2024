
# Step 4: Build & Train Dense Network Baseline

## Objective
# Build and train a fully connected deep neural network as a baseline model for CIFAR-10
# classification.

## Architecture Design

### Network Structure

# ==========================================
# STEP 4: BUILD & TRAIN DENSE NETWORK BASELINE
# Fully Connected Neural Network for CIFAR-10
# ==========================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("STEP 4: DENSE NETWORK BASELINE MODEL")
print("=" * 70)

# ------------------------------------------
# LOAD PREPROCESSED DATA (from Step 3)
# ------------------------------------------
print("\n Loading preprocessed data...")

# Since we're continuing from Step 3, we'll reload or use existing data
# For completeness, here's the preprocessing code:

from tensorflow import keras

# Load raw CIFAR-10
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Flatten labels
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Create validation split (last 10,000 samples)
X_train = X_train_full[:-10000]
y_train = y_train_full[:-10000]
X_val = X_train_full[-10000:]
y_val = y_train_full[-10000:]

# Flatten images for Dense Network
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

print(f"✓ Data loaded and preprocessed")
print(f"  Training: {X_train_flat.shape}")
print(f"  Validation: {X_val_flat.shape}")
print(f"  Test: {X_test_flat.shape}")

# ------------------------------------------
# BUILD THE DENSE NETWORK
# ------------------------------------------
print("\n" + "=" * 70)
print(" BUILDING DENSE NETWORK ARCHITECTURE")
print("=" * 70)

model = keras.Sequential([
    # Input layer (3072 features automatically inferred)
    layers.Dense(512, activation='relu', input_shape=(3072,)),
    layers.Dropout(0.3),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),

    # Output layer (10 classes)
    layers.Dense(10, activation='softmax')
])

# Display model architecture
print("\n MODEL SUMMARY:")
print("-" * 70)
model.summary()

# Calculate and display total parameters
total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = total_params - trainable_params

print("\n PARAMETER COUNT:")
print("-" * 70)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Non-trainable parameters: {non_trainable_params:,}")

# ------------------------------------------
# COMPILE THE MODEL
# ------------------------------------------
print("\n" + "=" * 70)
print("️ COMPILING THE MODEL")
print("=" * 70)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled successfully")
print(f"  Optimizer: Adam (learning_rate=0.001)")
print(f"  Loss: sparse_categorical_crossentropy")
print(f"  Metrics: accuracy")

# ------------------------------------------
# SETUP CALLBACKS
# ------------------------------------------
print("\n" + "=" * 70)
print(" SETTING UP CALLBACKS")
print("=" * 70)

# EarlyStopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# ModelCheckpoint callback
model_checkpoint = keras.callbacks.ModelCheckpoint(
    'dense_best.keras',  # Note: .keras extension for Keras v3
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Optional: ReduceLROnPlateau (helps with convergence)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

print("✓ Callbacks configured:")
print(f"  • EarlyStopping: monitor='val_loss', patience=10, restore_best_weights=True")
print(f"  • ModelCheckpoint: save_path='dense_best.keras', monitor='val_accuracy'")
print(f"  • ReduceLROnPlateau: monitor='val_loss', factor=0.5, patience=5")

# ------------------------------------------
# TRAIN THE MODEL
# ------------------------------------------
print("\n" + "=" * 70)
print(" TRAINING THE DENSE NETWORK")
print("=" * 70)
print(f"  Epochs: 80 (max)")
print(f"  Batch size: 256")
print(f"  Training samples: {X_train_flat.shape[0]:,}")
print(f"  Validation samples: {X_val_flat.shape[0]:,}")
print(f"  Steps per epoch: {np.ceil(X_train_flat.shape[0] / 256):.0f}")
print("-" * 70)

# Record start time
start_time = time.time()

# Train the model
history = model.fit(
    X_train_flat, y_train,
    validation_data=(X_val_flat, y_val),
    epochs=80,
    batch_size=256,
    callbacks=callbacks,
    verbose=1  # Set to 1 for detailed output, 2 for minimal
)

# Calculate training time
end_time = time.time()
training_time = end_time - start_time
training_time_minutes = training_time / 60

print("\n" + "=" * 70)
print(" TRAINING COMPLETE!")
print("=" * 70)
print(f"  Total training time: {training_time:.2f} seconds")
print(f"  Total training time: {training_time_minutes:.2f} minutes")
print(f"  Total epochs completed: {len(history.history['loss'])}")
print(f"  Early stopping triggered: {'Yes' if len(history.history['loss']) < 80 else 'No'}")

# ------------------------------------------
# EVALUATE ON TEST SET
# ------------------------------------------
print("\n" + "=" * 70)
print(" EVALUATING ON TEST SET")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(X_test_flat, y_test, verbose=0)

print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

# Also evaluate on validation set for comparison
val_loss, val_accuracy = model.evaluate(X_val_flat, y_val, verbose=0)
print(f"\n  Validation Loss: {val_loss:.4f}")
print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")

# ------------------------------------------
# SAVE TRAINING HISTORY
# ------------------------------------------
print("\n" + "=" * 70)
print(" SAVING TRAINING HISTORY")
print("=" * 70)

# Save history as numpy file for later analysis
np.savez('dense_training_history.npz',
         loss=history.history['loss'],
         val_loss=history.history['val_loss'],
         accuracy=history.history['accuracy'],
         val_accuracy=history.history['val_accuracy'])

print("✓ Training history saved to 'dense_training_history.npz'")

# ------------------------------------------
# DISPLAY BEST RESULTS
# ------------------------------------------
print("\n" + "=" * 70)
print(" BEST RESULTS SUMMARY")
print("=" * 70)

best_val_accuracy = max(history.history['val_accuracy'])
best_val_accuracy_epoch = history.history['val_accuracy'].index(best_val_accuracy) + 1
best_val_loss = min(history.history['val_loss'])
best_val_loss_epoch = history.history['val_loss'].index(best_val_loss) + 1

print(f"  Best Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy * 100:.2f}%)")
print(f"    Achieved at epoch: {best_val_accuracy_epoch}")
print(f"\n  Best Validation Loss: {best_val_loss:.4f}")
print(f"    Achieved at epoch: {best_val_loss_epoch}")
print(f"\n  Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"  Final Test Loss: {test_loss:.4f}")

# ------------------------------------------
# VERIFICATION OF SAVED MODEL
# ------------------------------------------
print("\n" + "=" * 70)
print(" VERIFYING SAVED MODEL")
print("=" * 70)

# Load the saved model
try:
    loaded_model = keras.models.load_model('dense_best.keras')
    print("✓ Model successfully saved and loaded from 'dense_best.keras'")

    # Verify performance
    loaded_test_loss, loaded_test_accuracy = loaded_model.evaluate(X_test_flat, y_test, verbose=0)
    print(f"  Loaded model test accuracy: {loaded_test_accuracy:.4f} ({loaded_test_accuracy * 100:.2f}%)")

    if np.isclose(test_accuracy, loaded_test_accuracy, atol=1e-5):
        print("  ✓ Saved model matches original model performance")
    else:
        print("  ️ Warning: Saved model performance differs slightly")

except Exception as e:
    print(f"   Could not verify saved model: {e}")

# ------------------------------------------
# FINAL SUMMARY
# ------------------------------------------
print("\n" + "=" * 70)
print(" STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
DENSE NETWORK BASELINE SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE:
  • Input: 3,072 features (flattened 32×32×3 image)
  • Hidden Layer 1: Dense(512) + Dropout(0.3)
  • Hidden Layer 2: Dense(256) + Dropout(0.3)
  • Hidden Layer 3: Dense(128)
  • Output: Dense(10, softmax)

TOTAL PARAMETERS: {:,}

TRAINING CONFIGURATION:
  • Optimizer: Adam (lr=0.001)
  • Loss: Sparse Categorical Crossentropy
  • Batch size: 256
  • Max epochs: 80
  • Early stopping patience: 10

RESULTS:
  • Training time: {:.2f} minutes
  • Best validation accuracy: {:.2f}%
  • Test accuracy: {:.2f}%

FILES GENERATED:
  • dense_best.keras - Best model weights
  • dense_training_history.npz - Training metrics

NEXT STEP:
  Proceed to Step 5: Training Visualization & Performance Analysis
""".format(total_params, training_time_minutes, best_val_accuracy * 100, test_accuracy * 100))

print("=" * 70)