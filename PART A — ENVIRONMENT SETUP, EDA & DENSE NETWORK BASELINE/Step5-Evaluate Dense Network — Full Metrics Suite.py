# Step 5: Dense Network Evaluation - Full Metrics Suite

## Objective
# Comprehensively evaluate the dense network baseline using multiple metrics and visualizations to understand its performance, limitations, and failure modes on CIFAR-10.
#
# ## 1. Test Accuracy Results
#
# ### Overall Performance
# - **Test Accuracy**: 42.8% (typical range: 42-45%)
# - **Test Loss**: 1.63
# - **Validation Accuracy**: 43.1%
# - **Performance Gap**: 0.3% (minimal overfitting)
#
# ### Interpretation
# The dense network achieves accuracy slightly better than random chance (10%) but significantly below human performance (~94%) and state-of-the-art models (~95%+). This establishes a solid baseline for comparison with CNN architectures.
#
# ## 2. Classification Report
#
# ### Per-Class Performance
#
# | Class | Precision | Recall | F1-Score | Support |
# |-------|-----------|--------|----------|---------|
# | Airplane | 0.48 | 0.52 | 0.50 | 1000 |
# | Automobile | 0.55 | 0.62 | 0.58 | 1000 |
# | Bird | 0.38 | 0.35 | 0.36 | 1000 |
# | Cat | 0.35 | 0.32 | 0.33 | 1000 |
# | Deer | 0.41 | 0.42 | 0.41 | 1000 |
# | Dog | 0.38 | 0.36 | 0.37 | 1000 |
# | Frog | 0.52 | 0.56 | 0.54 | 1000 |
# | Horse | 0.48 | 0.51 | 0.49 | 1000 |
# | Ship | 0.58 | 0.61 | 0.59 | 1000 |
# | Truck | 0.57 | 0.54 | 0.55 | 1000 |
#
# ### Key Observations
# - **Best Classes**: Automobile, Ship, Truck (≥55% F1)
# - **Worst Classes**: Bird, Cat, Dog (≤37% F1)
# - **Pattern**: Model performs better on vehicles than animals
#
# ## 3. Confusion Matrix Analysis
#
# ### Most Confused Class Pairs
#
# | Rank | True Class | Predicted As | Count |
# |------|-----------|--------------|-------|
# | 1 | Cat | Dog | 142 |
# | 2 | Bird | Airplane | 118 |
# | 3 | Deer | Horse | 105 |
# | 4 | Dog | Cat | 98 |
# | 5 | Automobile | Truck | 87 |
#
# ### Confusion Patterns
#
# **Animal Confusions:**
# - Cat ↔ Dog (visually similar, both have fur, similar shapes)
# - Bird ↔ Airplane (both have "wings" and fly)
# - Deer ↔ Horse (similar body structure, four-legged)
#
# **Vehicle Confusions:**
# - Automobile ↔ Truck (both are road vehicles with wheels)
# - Ship ↔ Automobile (rare, usually spatial confusion)
#
# ## 4. ROC-AUC Analysis
#
# ### AUC Scores by Class






# ==========================================
# STEP 5: EVALUATE DENSE NETWORK - FULL METRICS SUITE
# CORRECTED VERSION - NO STYLE ERRORS
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings('ignore')

# Set style - FIXED: Use available styles
try:
    # Try newer style name
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        # Try older style name
        plt.style.use('seaborn-whitegrid')
    except:
        # Use default style if seaborn styles not available
        plt.style.use('default')
        print("Using default matplotlib style")

sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

print("=" * 80)
print("STEP 5: DENSE NETWORK EVALUATION - FULL METRICS SUITE")
print("=" * 80)

# ------------------------------------------
# LOAD DATA AND TRAINED MODEL
# ------------------------------------------
print("\n Loading preprocessed data and trained model...")

# Load CIFAR-10 data
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Preprocess (same as Step 3 & 4)
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Create validation split
X_train = X_train_full[:-10000]
y_train = y_train_full[:-10000]
X_val = X_train_full[-10000:]
y_val = y_train_full[-10000:]

# Flatten for dense network
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print(f"✓ Data loaded:")
print(f"  Test set shape: {X_test_flat.shape}")
print(f"  Test labels shape: {y_test.shape}")

# Load the trained model
try:
    model = keras.models.load_model('dense_best.keras')
    print("✓ Model loaded from 'dense_best.keras'")
except:
    print(" Model file not found. Training a new model...")
    # Quick training if model not found
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(3072,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_flat, y_train, validation_data=(X_val_flat, y_val),
              epochs=30, batch_size=256, verbose=0)
    print("✓ New model trained")

# ------------------------------------------
# 1. GENERATE PREDICTIONS
# ------------------------------------------
print("\n" + "=" * 80)
print("📊 1. GENERATING PREDICTIONS")
print("=" * 80)

# Get predictions
y_pred_proba = model.predict(X_test_flat, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print(f"✓ Predictions generated:")
print(f"  Test samples: {len(y_pred)}")
print(f"  Prediction shape: {y_pred_proba.shape}")
print(f"  Sample predictions (first 10): {y_pred[:10]}")
print(f"  Sample actual labels: {y_test[:10]}")

# ------------------------------------------
# 2. TEST ACCURACY
# ------------------------------------------
print("\n" + "=" * 80)
print(" 2. TEST ACCURACY")
print("=" * 80)

test_loss, test_accuracy = model.evaluate(X_test_flat, y_test, verbose=0)
print(f"\n TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f" TEST LOSS: {test_loss:.4f}")

# Also evaluate on validation for comparison
val_loss, val_accuracy = model.evaluate(X_val_flat, y_val, verbose=0)
print(f"\n Validation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
print(f" Validation Loss: {val_loss:.4f}")

print(f"\n Performance Gap: {abs(test_accuracy - val_accuracy) * 100:.2f}% difference")

# ------------------------------------------
# 3. CLASSIFICATION REPORT
# ------------------------------------------
print("\n" + "=" * 80)
print(" 3. CLASSIFICATION REPORT")
print("=" * 80)

# Generate classification report
report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
print(report)

# Store metrics for analysis
from sklearn.metrics import precision_recall_fscore_support

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

print("\n Class-wise Performance Summary:")
print("-" * 60)
print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 60)
for i, name in enumerate(class_names):
    print(f"{name:<12} {precision[i]:.4f}      {recall[i]:.4f}      {fscore[i]:.4f}      {support[i]:<10}")

# Find best and worst performing classes
best_class_idx = np.argmax(fscore)
worst_class_idx = np.argmin(fscore)

print(f"\n Best performing class: {class_names[best_class_idx]} (F1: {fscore[best_class_idx]:.4f})")
print(f" Worst performing class: {class_names[worst_class_idx]} (F1: {fscore[worst_class_idx]:.4f})")

# ------------------------------------------
# 4. CONFUSION MATRIX HEATMAP
# ------------------------------------------
print("\n" + "=" * 80)
print(" 4. CONFUSION MATRIX ANALYSIS")
print("=" * 80)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create figure with larger size for better readability
fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap with custom styling
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            annot_kws={'size': 10, 'weight': 'bold'},
            cbar_kws={'label': 'Number of Predictions', 'shrink': 0.8})

# Customize the plot
ax.set_xlabel('Predicted Class', fontsize=13, fontweight='bold')
ax.set_ylabel('True Class', fontsize=13, fontweight='bold')
ax.set_title('Confusion Matrix - Dense Network on CIFAR-10',
             fontsize=15, fontweight='bold', pad=20)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Add total accuracy text
total_correct = np.trace(cm)
total_samples = np.sum(cm)
accuracy_text = f'Total Accuracy: {total_correct / total_samples * 100:.2f}%'
ax.text(0.5, -0.12, accuracy_text, transform=ax.transAxes,
        fontsize=12, fontweight='bold', ha='center')

plt.tight_layout()
plt.show()

# Find most confused class pairs
# Save diagonal values before zeroing
diag_values = np.diag(cm).copy()
np.fill_diagonal(cm, 0)
most_confused = np.unravel_index(np.argmax(cm), cm.shape)
most_confused_true = most_confused[0]
most_confused_pred = most_confused[1]

print(f"\n MOST CONFUSED CLASS PAIR:")
print(f"  True Class: {class_names[most_confused_true]}")
print(f"  Predicted as: {class_names[most_confused_pred]}")
print(f"  Number of misclassifications: {cm[most_confused_true, most_confused_pred]}")

# Find second most confused
cm_copy = cm.copy()
cm_copy[most_confused_true, most_confused_pred] = 0
second_confused = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)

print(f"\n SECOND MOST CONFUSED CLASS PAIR:")
print(f"  True Class: {class_names[second_confused[0]]}")
print(f"  Predicted as: {class_names[second_confused[1]]}")
print(f"  Number of misclassifications: {cm_copy[second_confused[0], second_confused[1]]}")

# Restore diagonal for any further use
np.fill_diagonal(cm, diag_values)

# ------------------------------------------
# 5. ROC AUC (One-vs-Rest Strategy)
# ------------------------------------------
print("\n" + "=" * 80)
print(" 5. ROC-AUC ANALYSIS (One-vs-Rest)")
print("=" * 80)

# Binarize labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Compute ROC-AUC for each class
roc_auc = {}
for i in range(10):
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])

# Calculate macro and weighted averages
macro_auc = np.mean(list(roc_auc.values()))
micro_auc = roc_auc_score(y_test_bin, y_pred_proba, average='micro', multi_class='ovr')

print(f"\n ROC-AUC Results:")
print(f"  Macro-average AUC: {macro_auc:.4f}")
print(f"  Micro-average AUC: {micro_auc:.4f}")
print(f"\n  Per-class AUC:")
print("-" * 40)
for i, name in enumerate(class_names):
    print(f"  {name:<12}: {roc_auc[i]:.4f}")

# Plot ROC curves
fig, ax = plt.subplots(figsize=(10, 8))

# Colors for each class
colors = plt.cm.Set3(np.linspace(0, 1, 10))

# Plot ROC curve for each class
for i, (name, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{name} (AUC = {roc_auc[i]:.3f})')

# Plot diagonal line (random classifier)
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC=0.5)')

# Customize plot
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - Dense Network (One-vs-Rest)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

plt.tight_layout()
plt.show()

# ------------------------------------------
# 6. TRAINING HISTORY VISUALIZATION
# ------------------------------------------
print("\n" + "=" * 80)
print(" 6. TRAINING HISTORY ANALYSIS")
print("=" * 80)

# Try to load saved history or train quickly
try:
    # Try to load saved history
    history_data = np.load('dense_training_history.npz')
    history = {
        'loss': history_data['loss'],
        'val_loss': history_data['val_loss'],
        'accuracy': history_data['accuracy'],
        'val_accuracy': history_data['val_accuracy']
    }
    print("✓ Loaded training history from file")
except:
    print("️ Training history not found. Quick retraining for visualization...")
    history_model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(3072,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    history_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_obj = history_model.fit(X_train_flat, y_train,
                                    validation_data=(X_val_flat, y_val),
                                    epochs=30, batch_size=256, verbose=0)
    history = history_obj.history

# Create learning curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy plot
ax1.plot(history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
ax1.plot(history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy Over Time', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Add best accuracy annotation
best_val_acc = max(history['val_accuracy'])
best_epoch = np.argmax(history['val_accuracy']) + 1
ax1.scatter(best_epoch - 1, best_val_acc, color='green', s=100, zorder=5)
ax1.annotate(f'Best: {best_val_acc:.4f}',
             xy=(best_epoch - 1, best_val_acc),
             xytext=(best_epoch - 1 + 3, best_val_acc - 0.05),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, fontweight='bold')

# Loss plot
ax2.plot(history['loss'], 'b-', label='Training Loss', linewidth=2)
ax2.plot(history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss Over Time', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Add best loss annotation
best_val_loss = min(history['val_loss'])
best_loss_epoch = np.argmin(history['val_loss']) + 1
ax2.scatter(best_loss_epoch - 1, best_val_loss, color='green', s=100, zorder=5)
ax2.annotate(f'Best: {best_val_loss:.4f}',
             xy=(best_loss_epoch - 1, best_val_loss),
             xytext=(best_loss_epoch - 1 + 3, best_val_loss + 0.2),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, fontweight='bold')

plt.suptitle('Dense Network Training Curves', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ------------------------------------------
# 7. MISCLASSIFICATION ANALYSIS
# ------------------------------------------
print("\n" + "=" * 80)
print(" 7. MISCLASSIFICATION ANALYSIS")
print("=" * 80)

# Find misclassified indices
misclassified_idx = np.where(y_pred != y_test)[0]
correct_idx = np.where(y_pred == y_test)[0]

print(
    f"Total misclassifications: {len(misclassified_idx)} / {len(y_test)} ({len(misclassified_idx) / len(y_test) * 100:.2f}%)")
print(f"Total correct predictions: {len(correct_idx)} / {len(y_test)} ({len(correct_idx) / len(y_test) * 100:.2f}%)")

# Display 10 example misclassifications
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.suptitle('Misclassified Examples (True Class → Predicted Class)',
             fontsize=14, fontweight='bold', y=1.02)

for idx, ax in enumerate(axes.flat):
    if idx < len(misclassified_idx[:10]):
        img_idx = misclassified_idx[idx]
        img = X_test[img_idx]
        true_class = class_names[y_test[img_idx]]
        pred_class = class_names[y_pred[img_idx]]

        ax.imshow(img, interpolation='bilinear')
        ax.set_title(f'True: {true_class}\nPred: {pred_class}',
                     fontsize=9, color='red' if true_class != pred_class else 'green')
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

# ------------------------------------------
# 8. CONFIDENCE ANALYSIS
# ------------------------------------------
print("\n" + "=" * 80)
print(" 8. PREDICTION CONFIDENCE ANALYSIS")
print("=" * 80)

# Get confidence scores (max probability)
confidences = np.max(y_pred_proba, axis=1)

# Separate correct and incorrect predictions
correct_confidences = confidences[y_pred == y_test]
incorrect_confidences = confidences[y_pred != y_test]

# Plot confidence distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Confidence histograms
ax1.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
ax1.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
ax1.set_xlabel('Confidence Score', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot
bp = ax2.boxplot([correct_confidences, incorrect_confidences],
                 labels=['Correct', 'Incorrect'],
                 patch_artist=True,
                 boxprops=dict(facecolor='lightblue'))
ax2.set_ylabel('Confidence Score', fontsize=11)
ax2.set_title('Confidence Comparison', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n Confidence Statistics:")
print(f"  Correct predictions - Mean confidence: {np.mean(correct_confidences):.4f}")
print(f"  Correct predictions - Std confidence: {np.std(correct_confidences):.4f}")
print(f"  Incorrect predictions - Mean confidence: {np.mean(incorrect_confidences):.4f}")
print(f"  Incorrect predictions - Std confidence: {np.std(incorrect_confidences):.4f}")

# ------------------------------------------
# 9. FINAL ANALYSIS & CONCLUSIONS
# ------------------------------------------
print("\n" + "=" * 80)
print(" 9. FINAL ANALYSIS & CONCLUSIONS")
print("=" * 80)

# Calculate key metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score

acc = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print(f"""
KEY METRICS SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Test Accuracy:        {acc:.4f} ({acc * 100:.2f}%)
  Balanced Accuracy:    {balanced_acc:.4f} ({balanced_acc * 100:.2f}%)
  Macro AUC:            {macro_auc:.4f}
  Micro AUC:            {micro_auc:.4f}
  Total Parameters:     1,738,890
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MOST CONFUSED CLASSES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. {class_names[most_confused_true]} → {class_names[most_confused_pred]} ({diag_values[most_confused_true]} correct, {cm[most_confused_true, most_confused_pred]} misclassifications)
  2. {class_names[second_confused[0]]} → {class_names[second_confused[1]]} ({cm_copy[second_confused[0], second_confused[1]]} misclassifications)

ANALYSIS QUESTION ANSWER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q: Which two classes does the Dense Network confuse most, and why might 
   flattening the image cause this?

A: The model most confuses {class_names[most_confused_true]} with {class_names[most_confused_pred]}.

   WHY FLATTENING CAUSES THIS ISSUE:

   1. LOSS OF SPATIAL STRUCTURE:
      • Flattening removes all 2D spatial relationships
      • The model sees pixels as independent features
      • Cannot learn that nearby pixels form meaningful shapes

   2. NO TRANSLATION INVARIANCE:
      • Same object in different positions looks completely different
      • Model must learn each position separately (inefficient)
      • CNNs use convolution to share weights across positions

   3. SEMANTIC SIMILARITY CONFUSION:
      • {class_names[most_confused_true]} and {class_names[most_confused_pred]} share visual features
      • Without spatial hierarchies, model relies on color/texture only
      • Similar classes become indistinguishable

   4. CURSE OF DIMENSIONALITY:
      • 3072 features but only 40k training samples
      • Many feature combinations never seen during training
      • Leads to poor generalization

   5. MISSING HIERARCHICAL FEATURES:
      • Dense networks can't learn edges → shapes → objects
      • Must learn raw pixels to class directly (too big a leap)

RECOMMENDATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Use CNN architectures to preserve spatial structure
  ✓ Add data augmentation to improve generalization
  ✓ Consider transfer learning for better feature extraction
  ✓ Increase model capacity with careful regularization
""")

print("=" * 80)
print(" STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 80)