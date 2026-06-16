# CIFAR-10 Deep Learning Project: From Dense Networks to Transfer Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.13+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Project Overview

This comprehensive deep learning project implements and compares three distinct neural network architectures for CIFAR-10 image classification, progressing from a simple dense network baseline to state-of-the-art transfer learning with MobileNetV2.

### 🎯 Key Achievements

| Model | Test Accuracy | Training Time | Parameters |
|-------|--------------|---------------|------------|
| Dense Network | 43.08% | 15 min | 1.74M |
| CNN + Augmentation | 73.5% | 90 min | 0.29M |
| **MobileNetV2 Transfer Learning** | **86.85%** | **18 min** | **2.59M** |

### 📊 Dataset
- **CIFAR-10**: 60,000 32×32 RGB images (50,000 train, 10,000 test)
- **10 Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Perfectly Balanced**: 6,000 images per class

## 🏗️ Project Structure
CIFAR-10-Deep-Learning-Project/
├── Part A - Environment Setup, EDA & Dense Network Baseline/
│ ├── Step1-Environment_Setup.py
│ ├── Step2-EDA_Visualization.py
│ ├── Step3-Data_Preprocessing.py
│ ├── Step4-Dense_Network_Baseline.py
│ └── Step5-Dense_Network_Evaluation.py
│
├── Part B - CNN from Scratch with Regularisation & Augmentation/
│ ├── Step6-CNN_Baseline_No_Reg.py
│ ├── Step7-CNN_with_BatchNorm.py
│ ├── Step8-CNN_Full_Regularised.py
│ ├── Step9-Regularisation_Ablation_Study.py
│ ├── Step10-Data_Augmentation.py
│ └── Step11-CNN_Full_Evaluation.py
│
├── Part C - Transfer Learning, Model Comparison & Diagnostics/
│ ├── Step12-Preprocess_for_MobileNetV2.py
│ ├── Step13-Transfer_Learning_Phase1.py
│ ├── Step14-Transfer_Learning_Phase2.py
│ └── Step15-Final_Model_Comparison.py
│
├── Part D - Dashboard, Model Saving & Written Report/
│ ├── Step16-Complete_Dashboard.py
│ ├── Step17-Save_Load_Inference_Demo.py
│ └── Step18-Written_Analysis_Report.md
│
├── models/
│ ├── week7_best_model.keras
│ ├── mobilenetv2_finetuned.keras
│ └── cnn_best.keras
│
├── outputs/
│ ├── week7_dashboard.png
│ ├── inference_demo.png
│ └── confusion_matrices.npz
│
├── requirements.txt
├── .gitignore
└── README.md

text

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
Installation
Clone the repository

bash
git clone https://github.com/yourusername/CIFAR-10-Deep-Learning-Project.git
cd CIFAR-10-Deep-Learning-Project
Create virtual environment

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Run the complete pipeline

bash
python run_all_steps.py
📚 18-Step Learning Journey
Phase 1: Foundations (Steps 1-5)
Step 1-2: Environment setup & EDA visualization

Step 3: Data preprocessing & normalization

Step 4-5: Dense network baseline & evaluation

Phase 2: CNN Development (Steps 6-11)
Step 6: CNN baseline (no regularization) - demonstrates overfitting

Step 7: BatchNormalization addition - stabilizes training

Step 8: Dropout integration - full regularization

Step 9: Ablation study - quantifies regularization impact

Step 10: Data augmentation - improves generalization

Step 11: Full CNN evaluation - comprehensive metrics

Phase 3: Transfer Learning (Steps 12-15)
Step 12: Preprocess images for MobileNetV2 (96×96, [-1,1] range)

Step 13: Feature extraction (frozen base) - Phase 1

Step 14: Fine-tuning (unfreeze last 30 layers) - Phase 2

Step 15: Three-model final comparison

Phase 4: Deployment (Steps 16-18)
Step 16: 6-chart comprehensive dashboard

Step 17: Model saving, loading & inference demo

Step 18: Complete written analysis report

📊 Key Results
Performance Comparison
text
Accuracy Progression:
Dense Network:      ████████████████████████████████████████░░░░  43.1%
CNN from Scratch:   ████████████████████████████████████████████████████████████  73.5%
MobileNetV2 (TL):   ████████████████████████████████████████████████████████████████████  86.9%
Regularisation Impact
Configuration	Test Accuracy	Overfitting Gap
No Regularisation	65.2%	15.2%
BatchNorm Only	72.8%	5.8%
BN + Dropout	73.5%	3.2%
Transfer Learning Impact
Phase	Strategy	Best Val Acc	Training Time
Phase 1	Feature Extraction	84.0%	10 min
Phase 2	Fine-tuning	86.85%	8 min
🎨 Visualizations
Dashboard (6 Charts)
Training History - Learning curves of best model

Confusion Matrix - Classification error patterns

Per-Class Accuracy - 3-model comparison by class

Model Performance - Accuracy & F1 scores

Misclassified Images - 10 example errors

ROC Curves - Discriminative ability comparison

Sample Output
text
Inference Demo (5 random images):
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│  🟢 Cat │  🟢 Ship│  🔴 Truck│ 🟢 Frog │  🔴 Bird│
│  Conf:  │  Conf:  │  Conf:  │  Conf:  │  Conf:  │
│   89%   │   95%   │   52%   │   88%   │   61%   │
└─────────┴─────────┴─────────┴─────────┴─────────┘
🛠️ Technologies Used
Technology	Version	Purpose
Python	3.8+	Core programming language
TensorFlow	2.13+	Deep learning framework
Keras	2.13+	High-level neural networks API
NumPy	1.24+	Numerical computations
Pandas	1.5+	Data manipulation
Matplotlib	3.7+	Visualization
Seaborn	0.12+	Statistical visualizations
Scikit-learn	1.2+	Metrics and evaluation
📈 Learning Outcomes
After completing this project, you will understand:

✅ Neural Network Fundamentals

Forward/backward propagation mathematics

Activation functions (ReLU, Softmax)

Loss functions (Categorical Crossentropy)

✅ CNN Architecture

Convolutional layers, pooling, feature maps

Spatial hierarchy learning

Filter evolution (edges → shapes → objects)

✅ Regularisation Techniques

BatchNormalization: stabilizes training, enables higher LR

Dropout: prevents co-adaptation, improves generalization

Data Augmentation: increases effective dataset size

✅ Transfer Learning

Feature extraction vs fine-tuning

Learning rate scheduling (100× lower for fine-tuning)

Knowledge transfer from ImageNet

✅ Model Deployment

Model saving/loading (.keras format)

Inference pipeline development

Production-ready API functions

🚀 Deployment
Production Inference Pipeline
python
from step17 import production_inference_pipeline

# Load and predict
result = production_inference_pipeline(image_array)
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top-3: {result['top_3_classes']}")
API Endpoint Ready
json
POST /predict
{
    "image": "base64_encoded_image",
    "top_k": 3
}

Response:
{
    "status": "success",
    "prediction": "Cat",
    "confidence": 0.892,
    "top_3_predictions": [
        {"class": "Cat", "probability": 0.892},
        {"class": "Dog", "probability": 0.088},
        {"class": "Deer", "probability": 0.012}
    ]
}
📝 Key Insights
1. Transfer Learning > Training from Scratch
+13.4% accuracy improvement

5× faster training (18 vs 90 minutes)

Leverages 1.4M ImageNet images

2. BatchNorm is Critical
Contributes 80% of regularization benefit

Only 0.6% parameter overhead

Enables 50% higher learning rates

3. Fine-tuning Requires Low LR
100× lower learning rate (1e-5 vs 1e-3)

Prevents catastrophic forgetting

Preserves general features while adapting

4. Data Augmentation Works
+3-5% accuracy improvement

Reduces overfitting by 2-3%

Effectively multiplies dataset size

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Kamran Ahmed
LinkedIn: linkedin.com/in/kamran-ahmed-ai44

🙏 Acknowledgments
CIFAR-10 dataset creators

TensorFlow/Keras team for excellent frameworks

ImageNet for pre-training models

Internship instructor for guidance
