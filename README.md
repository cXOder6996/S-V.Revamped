# SkinVision

[![CI](https://github.com/cXOder6996/S-V.Revamped/actions/workflows/ci.yml/badge.svg)](https://github.com/cXOder6996/S-V.Revamped/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.20](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.11-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

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
git clone https://github.com/cXOder6996/S-V.Revamped.git
cd S-V.Revamped

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

## Testing

Run unit tests locally with pytest:

```bash
pytest tests/ -v
```

## Contributing

Contributions, bug reports, and suggestions are welcome! Please check [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, development guidelines, and pull request procedures.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- HAM10000 dataset: Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.