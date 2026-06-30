import streamlit as st
import os
import joblib

st.set_page_config(
    page_title="ATLAS (Automated Text & Language Assessment System)",
    page_icon=":material/psychology:",
    layout="wide"
)

@st.cache_resource
def _load_model():
    model_path = 'model_autism_syntax_rf.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

if "model" not in st.session_state:
    m = _load_model()
    st.session_state.model = m
    st.session_state.model_ready = m is not None

page = st.navigation([
    st.Page("app_pages/home.py", title="Analisis", icon=":material/quick_reference:"),
    st.Page("app_pages/dataset.py", title="Dataset", icon=":material/table:"),
    st.Page("app_pages/evaluation.py", title="Evaluasi Model", icon=":material/analytics:"),
], position="top")

page.run()
