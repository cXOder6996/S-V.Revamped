"""
Configuration module for the SkinVision project.
"""
import os

CLASS_INFO = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

CLASS_KEYS = sorted(list(CLASS_INFO.keys()))
NUM_CLASSES = 7
IMG_SIZE = (224, 224)
DEFAULT_MODEL_PATH = os.path.join('models', 'best_model.keras')
PREPROCESSING_MODE = 'raw_0_255'  # TF 2.20 EfficientNetB0 expects raw [0,255] input
MIN_IMAGE_SIZE = (64, 64)
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

DISCLAIMER = 'This application is a technical demonstration of AI-based image classification. It is NOT a medical diagnostic tool and should not be used for medical decision-making. Always consult a qualified healthcare professional.'
DOMAIN_NOTICE = 'This model was trained exclusively on dermoscopic images. Results on non-dermoscopic photographs (e.g., smartphone photos) are unreliable and should not be interpreted.'
