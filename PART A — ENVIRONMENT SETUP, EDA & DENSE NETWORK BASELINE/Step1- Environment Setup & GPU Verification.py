# # Step 1: Environment Setup & GPU Verification
#
# In this step, we import all required libraries for deep learning, data analysis, visualization, and model evaluation.
#
# We also verify whether TensorFlow can detect a GPU, which can significantly speed up model training.
#
# Next, we load the CIFAR-10 dataset, inspect its shape, and define the class names that will be used throughout the project for visualization and prediction analysis.



# ==========================================
# Step 1: Environment Setup & GPU Verification
# ==========================================

# Import Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ------------------------------------------
# Check TensorFlow Version
# ------------------------------------------

print("TensorFlow Version:", tf.__version__)

# ------------------------------------------
# Verify GPU Availability
# ------------------------------------------

gpus = tf.config.list_physical_devices("GPU")

print("\nAvailable GPUs:")
print(gpus)

if len(gpus) > 0:
    print(f"\nGPU Detected: {len(gpus)} GPU(s) available")
else:
    print("\nNo GPU detected. Training will run on CPU.")

# ------------------------------------------
# Load CIFAR-10 Dataset
# ------------------------------------------

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# ------------------------------------------
# Display Dataset Shapes
# ------------------------------------------

print("\nDataset Shapes")
print("-" * 30)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

# ------------------------------------------
# Define CIFAR-10 Class Names
# ------------------------------------------

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print("\nClass Names:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")