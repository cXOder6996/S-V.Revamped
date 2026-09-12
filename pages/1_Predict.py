import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DOMAIN_NOTICE
from src.preprocessing import validate_image, check_image_quality, preprocess_for_inference
from src.inference import load_model, predict
from src.postprocessing import format_prediction_result
from src.explainability import generate_gradcam, validate_gradcam, overlay_gradcam

st.set_page_config(page_title="Predict - SkinVision", layout="wide")
st.title("Skin Lesion Analysis")

st.warning(DOMAIN_NOTICE)

@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.keras")
    try:
        return load_model(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        st.stop()

model = get_model()

uploaded_file = st.file_uploader("Upload a dermoscopic image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing image..."):
        is_valid, validation_msg = validate_image(image_np, return_message=True)
        if not is_valid:
            st.error(f"Image validation failed: {validation_msg}")
            st.stop()
            
        quality_warnings = check_image_quality(image_np)
        for warning in quality_warnings:
            st.warning(f"Image Quality Warning: {warning}")
            
        preprocessed_img = preprocess_for_inference(image_np)
        
        probs = predict(model, preprocessed_img)
        predicted_class = int(np.argmax(probs))
        
        formatted_result = format_prediction_result(probs)
        top_pred = formatted_result["top_prediction"]
        
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Top Prediction", value=f"{top_pred['class_name']} ({top_pred['class_key']})")
            if formatted_result["abstain_recommended"]:
                st.error(f"⚠️ {formatted_result.get('abstain_reason', 'The model is highly uncertain about this prediction. Abstention recommended.')}")
                
        with col2:
            st.metric(label="Confidence", value=f"{top_pred['probability'] * 100:.2f}%")
            st.metric(label="Uncertainty (Entropy)", value=f"{formatted_result['uncertainty']['entropy']:.4f}")
            
        st.write("#### Top Predictions")
        for item in formatted_result["top_k"]:
            prob_pct = item["probability"] * 100
            st.write(f"**{item['class_name']}** (`{item['class_key']}`): {prob_pct:.2f}%")
            st.progress(float(item["probability"]))

        st.subheader("Explainability (Grad-CAM)")
        try:
            heatmap = generate_gradcam(model, preprocessed_img, class_index=predicted_class)
            
            # Resize heatmap to match original image dimensions
            try:
                import cv2
                heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
            except ImportError:
                from PIL import Image
                pil_hm = Image.fromarray(heatmap).resize((image_np.shape[1], image_np.shape[0]), Image.Resampling.BILINEAR)
                heatmap = np.array(pil_hm)
            
            is_valid, gradcam_warnings = validate_gradcam(heatmap, image_np)
            for gw in gradcam_warnings:
                st.warning(f"Grad-CAM Note: {gw}")
            overlay = overlay_gradcam(image_np, heatmap)
            col_hm1, col_hm2 = st.columns(2)
            with col_hm1:
                st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)
            with col_hm2:
                st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate Grad-CAM: {str(e)}")
