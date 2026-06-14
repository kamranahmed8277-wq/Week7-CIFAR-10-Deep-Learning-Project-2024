
# # Step 2: Dataset Exploration & Visualization
#
# In this step, we explore the CIFAR-10 dataset to understand its structure and distribution.
#
# We will:
# - Visualize sample images (one per class) using a 2×5 grid
# - Confirm pixel value range (0–255)
# - Analyze class distribution in training data
# - Compute mean and standard deviation for each color channel (R, G, B)
# - Visualize pixel intensity distribution using histograms
#
# This helps us understand how to properly normalize and preprocess the dataset before training deep learning models.


# ==========================================
# CIFAR-10 COMPLETE EDA PIPELINE
# Professional Visualization & Analysis
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ------------------------------------------
# LOAD AND PREPARE DATA
# ------------------------------------------
print("Loading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Clean up labels
y_train = y_train.flatten()
y_test = y_test.flatten()

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Color mapping for visualization
class_colors = {
    'Airplane': '#1f77b4', 'Automobile': '#ff7f0e', 'Bird': '#2ca02c',
    'Cat': '#d62728', 'Deer': '#9467bd', 'Dog': '#8c564b',
    'Frog': '#e377c2', 'Horse': '#7f7f7f', 'Ship': '#bcbd22',
    'Truck': '#17becf'
}

print(f"✓ Dataset loaded successfully!")
print(f"  Training: {X_train.shape[0]} images")
print(f"  Testing: {X_test.shape[0]} images")
print(f"  Image size: {X_train.shape[1]}x{X_train.shape[2]} pixels")
print(f"  Classes: {len(class_names)}\n")

# ==========================================
# 1. SAMPLE IMAGES GRID (Perfect Alignment)
# ==========================================
print("=" * 70)
print("SECTION 1: SAMPLE IMAGES VISUALIZATION")
print("=" * 70)

# Create figure with perfect grid layout
fig = plt.figure(figsize=(14, 8))
fig.suptitle('CIFAR-10 Dataset: Sample Images by Class',
             fontsize=18, fontweight='bold', y=0.98)

# Create 2x5 grid using GridSpec for perfect alignment
gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3,
                      top=0.92, bottom=0.05, left=0.05, right=0.95)

# Get one representative image per class
samples = {}
for class_idx in range(10):
    class_mask = y_train == class_idx
    class_indices = np.where(class_mask)[0]
    samples[class_idx] = X_train[class_indices[0]]

# Display images with perfect alignment
for class_idx in range(10):
    row = class_idx // 5
    col = class_idx % 5
    ax = fig.add_subplot(gs[row, col])

    # Display image with enhancement
    img = samples[class_idx]
    ax.imshow(img, interpolation='bilinear')

    # Add class label with background
    class_name = class_names[class_idx]
    ax.set_title(class_name, fontsize=11, fontweight='bold',
                 pad=8, backgroundcolor='white', alpha=0.9)
    ax.axis('off')

    # Add subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
        spine.set_linewidth(1)

plt.tight_layout()
plt.show()

# ==========================================
# 2. CLASS DISTRIBUTION (Bar Chart)
# ==========================================
print("\n" + "=" * 70)
print("SECTION 2: CLASS DISTRIBUTION ANALYSIS")
print("=" * 70)

unique, counts = np.unique(y_train, return_counts=True)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(class_names, counts, color=[class_colors[name] for name in class_names],
              edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on top of bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 50,
            f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Class', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Images', fontsize=13, fontweight='bold')
ax.set_title('CIFAR-10 Class Distribution (Training Set)',
             fontsize=15, fontweight='bold', pad=20)
ax.set_ylim(0, 5500)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_xticklabels(class_names, rotation=45, ha='right')

# Add horizontal line at expected value
ax.axhline(y=5000, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Expected (5000)')
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("\n✓ Dataset is perfectly balanced: 5,000 images per class")
print(f"  Total training images: {sum(counts)}")

# ==========================================
# 3. PIXEL INTENSITY STATISTICS
# ==========================================
print("\n" + "=" * 70)
print("SECTION 3: PIXEL INTENSITY ANALYSIS")
print("=" * 70)

# Original pixel values analysis
print("\nOriginal Pixel Values (0-255):")
print(f"  • Minimum value: {X_train.min()}")
print(f"  • Maximum value: {X_train.max()}")
print(f"  • Data type: {X_train.dtype}")
print(f"  • Value range: 0-255 (8-bit unsigned integer)")

# Normalize data
X_train_norm = X_train.astype('float32') / 255.0

print("\nNormalized Pixel Values (0-1):")
print(f"  • Minimum value: {X_train_norm.min():.4f}")
print(f"  • Maximum value: {X_train_norm.max():.4f}")
print(f"  • Data type: {X_train_norm.dtype}")

# Channel statistics
mean_channel = np.mean(X_train_norm, axis=(0, 1))
std_channel = np.std(X_train_norm, axis=(0, 1))

print("\nChannel Statistics (RGB):")
print(f"{'Channel':<12} {'Mean':<12} {'Std Dev':<12}")
print("-" * 36)
channels = ['Red (R)', 'Green (G)', 'Blue (B)']
for i, channel in enumerate(channels):
    print(f"{channel:<12} {mean_channel[i]:.4f}       {std_channel[i]:.4f}")

# ==========================================
# 4. RGB CHANNEL DISTRIBUTIONS
# ==========================================
print("\n" + "=" * 70)
print("SECTION 4: RGB CHANNEL VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Pixel Intensity Distribution by RGB Channel',
             fontsize=16, fontweight='bold', y=1.02)

channel_data = [
    (0, 'Red', '#ff4444'),
    (1, 'Green', '#44ff44'),
    (2, 'Blue', '#4444ff')
]

for idx, (channel, name, color) in enumerate(channel_data):
    data = X_train[:, :, :, channel].ravel()

    # Create histogram
    n, bins, patches = axes[idx].hist(data, bins=50, color=color,
                                      alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add statistics lines
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)

    axes[idx].axvline(mean_val, color='black', linestyle='-',
                      linewidth=2, label=f'Mean: {mean_val:.1f}')
    axes[idx].axvline(median_val, color='gray', linestyle='--',
                      linewidth=2, label=f'Median: {median_val:.1f}')

    # Add shaded area for ±1 std
    axes[idx].axvspan(mean_val - std_val, mean_val + std_val,
                      alpha=0.2, color=color, label=f'±1 Std: {std_val:.1f}')

    axes[idx].set_xlabel('Pixel Intensity (0-255)', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{name} Channel Distribution', fontsize=12, fontweight='bold')
    axes[idx].legend(loc='upper right', fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 5. PER-CLASS CHANNEL MEANS (Heatmap)
# ==========================================
print("\n" + "=" * 70)
print("SECTION 5: PER-CLASS COLOR ANALYSIS")
print("=" * 70)

# Calculate per-class channel means
class_means = np.zeros((10, 3))
for i in range(10):
    class_mask = y_train == i
    class_images = X_train_norm[class_mask]
    class_means[i] = np.mean(class_images, axis=(0, 1))

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(class_means, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Mean Pixel Intensity', fontsize=11, fontweight='bold')

# Set labels
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Red', 'Green', 'Blue'], fontsize=11, fontweight='bold')
ax.set_yticks(range(10))
ax.set_yticklabels(class_names, fontsize=11, fontweight='bold')
ax.set_xlabel('RGB Channels', fontsize=12, fontweight='bold')
ax.set_ylabel('Class', fontsize=12, fontweight='bold')
ax.set_title('Average Color Profile by Class', fontsize=14, fontweight='bold', pad=20)

# Add text annotations
for i in range(10):
    for j in range(3):
        text = ax.text(j, i, f'{class_means[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Print color profile insights
print("\nColor Profile Insights:")
print("-" * 40)
for i, class_name in enumerate(class_names):
    rgb = class_means[i]
    dominant = channels[np.argmax(rgb)]
    print(f"  • {class_name}: Dominant color is {dominant} (R:{rgb[0]:.2f}, G:{rgb[1]:.2f}, B:{rgb[2]:.2f})")

# ==========================================
# 6. SUMMARY REPORT
# ==========================================
print("\n" + "=" * 70)
print("EDA SUMMARY REPORT")
print("=" * 70)

print(f"""
DATASET OVERVIEW:
• Total training images: {X_train.shape[0]:,}
• Total test images: {X_test.shape[0]:,}
• Image dimensions: {X_train.shape[1]} × {X_train.shape[2]} pixels
• Color channels: 3 (RGB)
• Number of classes: {len(class_names)}
• Class balance: Perfectly balanced (5,000 images/class)

PIXEL STATISTICS (Original):
• Value range: 0-255
• Mean pixel value: {np.mean(X_train):.1f}
• Std pixel value: {np.std(X_train):.1f}

NORMALIZATION RECOMMENDATION:
• Recommended scaling: / 255.0
• Resulting range: 0.0 - 1.0
• Format: float32 for memory efficiency

KEY OBSERVATIONS:
1. Dataset is well-balanced - no class imbalance issues
2. Pixel values range from 0-255 (standard 8-bit RGB)
3. All channels show similar distribution patterns
4. Some classes have distinctive color profiles (e.g., 'Ship' has stronger blue)
5. Image resolution (32×32) is relatively low - consider data augmentation
""")

print("=" * 70)
print("✓ EDA COMPLETED SUCCESSFULLY")
print("=" * 70)