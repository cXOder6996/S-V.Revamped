# SkinVision

AI-powered skin lesion image classification using deep learning.

> **⚠️ DISCLAIMER**: This application is a technical demonstration of AI-based image classification. It is **NOT** a medical diagnostic tool and should not be used for medical decision-making. Always consult a qualified healthcare professional for any health concerns.

## Overview

SkinVision classifies dermoscopic skin lesion images into 7 categories using an EfficientNetB0 model trained on the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T):

| Class | Full Name |
|-------|-----------|
| `akiec` | Actinic Keratoses |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevi |
| `vasc` | Vascular Lesions |

## Important Limitations

- **Trained on dermoscopic images only**. Results on non-dermoscopic photographs (e.g., smartphone photos) are unreliable.
- This is a **student portfolio project** demonstrating ML engineering practices, not a clinical tool.
- Model performance varies significantly across classes due to dataset imbalance.

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended for training)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd SkinVision

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset

This project uses the HAM10000 dataset. Download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) or the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

Place the dataset files so your config points to:
- `HAM10000_metadata.csv`
- Image directory containing all `.jpg` files

### Model Weights

Model weights are not included in the repository. To obtain them:

1. **Train from scratch** using `training/train.py` (recommended for reproducibility)
2. Place the resulting `best_model.keras` in the `models/` directory

### Running the Application

```bash
streamlit run app.py
```

## Project Structure

```
SkinVision/
├── app.py                  # Streamlit entry point
├── config.py               # Central configuration
├── requirements.txt        # Pinned dependencies
├── src/
│   ├── preprocessing.py    # Image preprocessing
│   ├── inference.py        # Model loading & prediction
│   ├── explainability.py   # Grad-CAM visualization
│   └── postprocessing.py   # Uncertainty & result formatting
├── training/
│   ├── train.py            # Reproducible training script
│   ├── evaluate.py         # Evaluation & metrics
│   └── config.yaml         # Training hyperparameters
├── models/                 # Model artifacts (gitignored)
├── tests/                  # Unit tests
└── pages/                  # Streamlit pages
```

## Training

See `training/config.yaml` for hyperparameter configuration.

```bash
python training/train.py --data-dir /path/to/HAM10000 --config training/config.yaml
```

## Technology Stack

- **Framework**: TensorFlow 2.20.0 / Keras 3.11.3
- **Architecture**: EfficientNetB0 (ImageNet pretrained)
- **Frontend**: Streamlit
- **Explainability**: Grad-CAM
- **Dataset**: HAM10000 (10,015 dermoscopic images)

## Acknowledgments

- HAM10000 dataset: Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.