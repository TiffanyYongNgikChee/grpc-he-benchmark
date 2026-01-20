# ML Pipeline for Privacy-Preserving Medical AI

This directory contains the machine learning pipeline for training medical prediction models that will be used with homomorphic encryption.

## 🎯 Purpose

Train PyTorch models on medical datasets, export to ONNX format, and integrate with the Rust HE infrastructure for encrypted inference.

## 📁 Structure

```
ml_pipeline/
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── scripts/             # Python scripts
│   ├── load_datasets.py # Load and preprocess datasets
│   ├── train_model.py   # Train PyTorch models
│   └── export_onnx.py   # Export to ONNX format
├── notebooks/           # Jupyter notebooks
│   └── 01_exploratory_analysis.ipynb
├── data/                # Processed datasets
│   ├── breast_cancer/
│   └── heart_disease/
└── models/              # Trained models
    ├── *.pt             # PyTorch checkpoints
    └── *.onnx           # ONNX exported models
```

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
cd ml_pipeline

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import onnx; print(f'ONNX: {onnx.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
```

### 3. Load Datasets

```bash
python scripts/load_datasets.py
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook notebooks/
```

## 📊 Datasets

| Dataset | Samples | Features | Target |
|---------|---------|----------|--------|
| Breast Cancer Wisconsin | 569 | 30 | Malignant/Benign |
| Cleveland Heart Disease | 303 | 13 | Heart Disease Risk |

## 🔗 Integration with HE

After training and exporting models:
1. ONNX models will be loaded in Rust
2. Patient data will be encrypted using SEAL/HELib/OpenFHE
3. Inference runs on encrypted data
4. Results are decrypted at the hospital

## 📝 Related Issues

- Issue #34: Python Environment Setup
- Issue #37: Complete ML Pipeline
