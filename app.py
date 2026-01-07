import streamlit as st
import nltk
import sys
import os
import traceback

# Ensure the local directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from paraphraser import ParaphraserEngine, SemanticValidator, NeuralEngine
from ai_avoider import AIDetectionAvoider

# Set page config
st.set_page_config(
    page_title="Paraphraser & Humanizer",
    page_icon="🪄",
    layout="wide"
)

# Initialize Session State
if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
if 'output' not in st.session_state: st.session_state.output = ""

# Simple Theme Toggle
def toggle():
    st.session_state.theme = 'Light' if st.session_state.theme == 'Dark' else 'Dark'

# UI Colors
is_dark = st.session_state.theme == 'Dark'
bg = "#0f172a" if is_dark else "#f8fafc"
text = "#f8fafc" if is_dark else "#0f172a"
card = "#1e293b" if is_dark else "#ffffff"

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    .stTextArea textarea {{ background-color: {card} !important; color: {text} !important; border-radius: 12px; }}
    .stButton button {{ background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; border-radius: 20px; }}
</style>
""", unsafe_allow_html=True)

# Resource Loading
@st.cache_resource
def get_resources():
    nltk_path = os.path.join(os.getcwd(), "nltk_data")
    if nltk_path not in nltk.data.path: nltk.data.path.append(nltk_path)
    
    # Critical NLTK downloads
    for pkg in ['punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'stopwords']:
        try:
            nltk.download(pkg, download_dir=nltk_path, quiet=True)
        except: pass
        
    return ParaphraserEngine(), AIDetectionAvoider(), SemanticValidator()

@st.cache_resource
def get_neural():
    try:
        return NeuralEngine(model_name="prajjwal1/bert-tiny")
    except:
        return None

def main():
    st.title("🪄 Paraphraser & Humanizer")
    
    # Engines
    engine, avoider, validator = get_resources()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["Basic", "Neural (AI)"])
        intensity = st.slider("Intensity", 0.1, 1.0, 0.6)
        do_humanize = st.checkbox("Humanize", value=True)
        
        if st.button("🌓 Toggle Theme"): toggle()
        
        st.info("🔒 100% Local & Private processing.")

    # Neural Engine logic
    neural = None
    if mode == "Neural (AI)":
        neural = get_neural()
        if not neural: st.warning("Neural model not found. Using Basic mode.")

    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input")
        inp = st.text_area("Paste text...", height=300, label_visibility="collapsed")
        if st.button("Transform Text ✨"):
            if inp:
                with st.spinner("Processing..."):
                    try:
                        # 1. Paraphrase
                        res = engine.paraphrase(inp, intensity, neural_engine=neural)
                        
                        # 2. Humanize
                        if do_humanize and res:
                            res = avoider.humanize(res, intensity)
                        
                        # 3. Validate & Improve
                        if res:
                            res = validator.improve_paraphrase(inp, res, engine)
                        
                        st.session_state.output = res or "Error: Processing produced no output."
                    except Exception as e:
                        st.error(f"Processing Error: {e}")
                        st.code(traceback.format_exc())
            else:
                st.warning("Please enter some text.")

    with col2:
        st.subheader("Output")
        st.text_area("Result", value=st.session_state.output, height=300, label_visibility="collapsed")
        if st.session_state.output:
            st.success("Complete!")

if __name__ == "__main__":
    main()
