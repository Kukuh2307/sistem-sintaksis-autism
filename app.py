import streamlit as st
import os
import joblib

st.set_page_config(
    page_title="ATLAS (Automated Text & Language Assessment System)",
    page_icon=":material/psychology:",
    layout="wide"
)

MODEL_PATH = 'model_autism_syntax_rf.pkl'

@st.cache_resource
def _load_model(_mtime):
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
if "model" not in st.session_state or st.session_state.get("_mtime") != mtime:
    st.session_state._mtime = mtime
    m = _load_model(mtime)
    st.session_state.model = m
    st.session_state.model_ready = m is not None

page = st.navigation([
    st.Page("app_pages/home.py", title="Analisis", icon=":material/quick_reference:"),
    st.Page("app_pages/dataset.py", title="Dataset", icon=":material/table:"),
    st.Page("app_pages/evaluation.py", title="Evaluasi Model", icon=":material/analytics:"),
], position="top")

page.run()
