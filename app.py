import streamlit as st
import os
import sys

# Add project root to path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DISCLAIMER

st.set_page_config(page_title="SkinVision", layout="wide", page_icon="🔍")

st.title("SkinVision")
st.write("Welcome to SkinVision, an AI-powered dermatological analysis tool.")

st.info("Please navigate to the Predict page using the sidebar to analyze an image.")

st.warning(DISCLAIMER)
