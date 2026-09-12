import streamlit as st
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MEDICAL_DISCLAIMER

st.set_page_config(page_title="About - SkinVision", layout="wide")
st.title("About SkinVision")

st.markdown("""
### Project Description
SkinVision is an AI-powered dermatological analysis tool designed to classify skin lesions into 7 classes based on the HAM10000 dataset:
- Actinic Keratoses (akiec)
- Basal Cell Carcinoma (bcc)
- Benign Keratosis (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic Nevi (nv)
- Vascular Lesions (vasc)

### Tech Stack
- **Framework**: TensorFlow 2.20.0 / Keras 3.11.3
- **Model**: EfficientNetB0 (ImageNet pretrained)
- **Frontend**: Streamlit
- **Augmentation**: Albumentations
""")

st.error(f"### Medical Disclaimer\n{MEDICAL_DISCLAIMER}")
