import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Model Info - SkinVision", layout="wide")
st.title("Model Information & Evaluation Metrics")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(base_dir, "experiments", "results_table.csv")

st.markdown("""
This page displays the historical progression of model performance across experiments.
The application currently uses the **E8** configuration (E7 checkpoint with 4-view Test-Time Augmentation).
""")

if os.path.exists(results_path):
    try:
        df_results = pd.read_csv(results_path)
        # Display just the key metrics for readability
        display_cols = [
            'experiment_id', 'architecture', 'input_size', 'val_macro_f1', 
            'mel_recall', 'inference_latency_p50_ms', 'notes'
        ]
        
        # Format the columns if they exist
        existing_cols = [c for c in display_cols if c in df_results.columns]
        
        st.subheader("Experiment Progression")
        st.dataframe(df_results[existing_cols], use_container_width=True)
        
        st.subheader("Detailed Full Results")
        with st.expander("Show all configuration and metric columns"):
            st.dataframe(df_results, use_container_width=True)
            
    except Exception as e:
        st.error(f"Could not load results table: {str(e)}")
else:
    st.info("No experiment results found. Run evaluation to populate results.")

