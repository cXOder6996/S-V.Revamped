import streamlit as st
import json
import os
import pandas as pd

st.set_page_config(page_title="Model Info - SkinVision", layout="wide")
st.title("Model Information & Evaluation Metrics")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_path = os.path.join(base_dir, "models", "evaluation_results.json")
history_path = os.path.join(base_dir, "models", "training_history.json")

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

eval_results = load_json(eval_path)
history = load_json(history_path)

if not eval_results and not history:
    st.info("Model has not been evaluated yet. Please train and evaluate the model to see metrics here.")
else:
    if history:
        st.subheader("Training History")
        try:
            df_history = pd.DataFrame(history)
            if 'loss' in df_history.columns and 'val_loss' in df_history.columns:
                st.line_chart(df_history[['loss', 'val_loss']])
            if 'accuracy' in df_history.columns and 'val_accuracy' in df_history.columns:
                st.line_chart(df_history[['accuracy', 'val_accuracy']])
        except Exception as e:
            st.error(f"Could not parse training history: {str(e)}")
            
    if eval_results:
        st.subheader("Evaluation Results")
        st.json(eval_results)
