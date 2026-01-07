import streamlit as st
import nltk
import sys
import os

# Ensure the local directory is in sys.path for correct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from paraphraser import ParaphraserEngine, SemanticValidator, NeuralEngine
from ai_avoider import AIDetectionAvoider

# Set page config
st.set_page_config(
    page_title="Paraphraser & Humanizer",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Theme
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'Dark'

def toggle_theme():
    if st.session_state.theme_mode == 'Dark':
        st.session_state.theme_mode = 'Light'
    else:
        st.session_state.theme_mode = 'Dark'

# Define Theme Colors
themes = {
    'Dark': {
        'bg_gradient': 'linear-gradient(-45deg, #0b0f19, #1a1f2e, #111827, #0f172a)',
        'text': '#e2e8f0',
        'card_bg': '#1e293b',
        'card_border': '#334155',
        'input_bg': '#0f172a',
        'header': '#f8fafc',
        'shadow': '0 10px 15px -3px rgba(0, 0, 0, 0.3)'
    },
    'Light': {
        'bg_gradient': 'linear-gradient(-45deg, #eff6ff, #f8fafc, #e0f2fe, #f1f5f9)',
        'text': '#1e293b',
        'card_bg': '#ffffff',
        'card_border': '#e2e8f0',
        'input_bg': '#f8fafc',
        'header': '#0f172a',
        'shadow': '0 10px 15px -3px rgba(0, 0, 0, 0.05)'
    }
}

current_theme = themes[st.session_state.theme_mode]

# Custom CSS with Dynamic Theme
st.markdown(f"""
<style>
    /* Breathing Gradient Animation */
    @keyframes gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .stApp {{
        background: {current_theme['bg_gradient']};
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: {current_theme['text']};
    }}
    
    /* Card-like Text Areas (Quillbot Style) */
    .stTextArea textarea {{
        background-color: {current_theme['input_bg']} !important;
        color: {current_theme['text']} !important;
        border: 1px solid {current_theme['card_border']} !important;
        border-radius: 16px;
        font-size: 17px;
        line-height: 1.6;
        padding: 20px;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }}
    
    .stTextArea textarea:focus {{
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        transform: translateY(-2px);
    }}
    
    /* Headers */
    h1, h2, h3, .stMarkdownContainer h1, .stMarkdownContainer h2 {{
        color: {current_theme['header']} !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    
    /* Buttons */
    .stButton button {{
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 99px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4);
    }}
    .stButton button:hover {{
        filter: brightness(1.1);
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.5);
    }}
    .stButton button:active {{
        transform: translateY(0);
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: {current_theme['card_bg']};
        border-right: 1px solid {current_theme['card_border']};
    }}
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {current_theme['text']};
    }}
    
</style>
""", unsafe_allow_html=True)

# Initialize NLTK
@st.cache_resource
def setup_nltk():
    """Download required NLTK data"""
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)
    
    # List of required packages
    required_packages = [
        'punkt', 
        'punkt_tab', 
        'wordnet', 
        'omw-1.4', 
        'averaged_perceptron_tagger', 
        'averaged_perceptron_tagger_eng', 
        'stopwords'
    ]
    
    for package in required_packages:
        try:
            if 'punkt' in package:
                nltk.data.find(f'tokenizers/{package}')
            elif 'wordnet' in package or 'omw' in package or 'stopwords' in package:
                nltk.data.find(f'corpora/{package}')
            elif 'tagger' in package:
                nltk.data.find(f'taggers/{package}')
        except LookupError:
            nltk.download(package, download_dir=nltk_data_path, quiet=True)
            nltk.download(package, quiet=True)
            
    return True

# Initialize Engines
@st.cache_resource
def load_engines():
    try:
        engine = ParaphraserEngine()
        avoider = AIDetectionAvoider()
        validator = SemanticValidator()
        return engine, avoider, validator
    except Exception as e:
        st.error(f"Error initializing engines: {e}")
        return None, None, None

@st.cache_resource
def load_neural_engine():
    try:
        # Using distilroberta-base as requested (standard Python version of Xenova/distilroberta-base)
        # cache_dir ensures it doesn't download every time
        return NeuralEngine(model_name="distilroberta-base", cache_dir="./model_cache")
    except Exception as e:
        st.error(f"Error loading neural engine: {e}")
        return None

# Main App logic
def main():
    # Header with title and toggle
    col_header, col_toggle = st.columns([4, 1])
    with col_header:
        st.title("Paraphraser & Humanizer")
        st.markdown("Transform AI-generated text into human-like content.")
    with col_toggle:
        st.write("") # Spacer
        st.write("") 
        if st.session_state.theme_mode == 'Dark':
            st.button("☀️ Light Mode", on_click=toggle_theme, key="theme_toggle")
        else:
            st.button("🌙 Dark Mode", on_click=toggle_theme, key="theme_toggle")
    
    # Setup resources
    setup_nltk()
    engine, avoider, validator = load_engines()
    
    if not engine:
        st.error("Failed to load application engines. Please check logs.")
        return

    # Sidebar Controls
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Paraphrase Mode", ["Basic (NLTK)", "Neural (AI Model)"], help="Basic is faster, Neural is more context-aware.")
        intensity = st.slider("Intensity", 0.1, 1.0, 0.6, 0.1)
        humanize = st.checkbox("Humanize (AI Avoidance)", value=True)
        
        # Load neural engine if needed
        neural_engine = None
        if mode == "Neural (AI Model)":
            neural_engine = load_neural_engine()
            if not neural_engine:
                st.warning("Neural engine failed to load. Falling back to Basic mode.")
        
        st.markdown(f"""
        <div style='background-color: {current_theme['input_bg']}; padding: 15px; border-radius: 12px; margin-top: 20px; border: 1px solid {current_theme['card_border']}'>
            <h4 style='margin:0; color:{current_theme['text']}'>How it works</h4>
            <ul style='color:{current_theme['text']}; font-size: 14px; padding-left: 20px; margin-top: 10px'>
                <li><b>Paraphrase</b>: Context-aware synonym replacement</li>
                <li><b>Humanize</b>: Structural variations for natural flow</li>
                <li><b>Validate</b>: Logic & meaning checks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("Runs locally with Python")

    # Main Area
    st.write("") # Spacer
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Text")
        input_text = st.text_area("Input", height=400, placeholder="Paste your text here...", label_visibility="collapsed")
        
        st.write("")
        process_btn = st.button("Paraphrase Text 🪄")

    # Processing State
    if 'output_text' not in st.session_state:
        st.session_state.output_text = ""

    if process_btn and input_text:
        with st.spinner("Processing..."):
            try:
                # Step 1: Paraphrase
                result = engine.paraphrase(input_text, intensity, neural_engine=neural_engine)
                
                # Step 2: Humanize
                if humanize:
                    result = avoider.humanize(result, intensity)
                
                # Step 3: Improve
                result = validator.improve_paraphrase(input_text, result, engine)
                
                st.session_state.output_text = result
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    with col2:
        st.subheader("Result")
        st.text_area("Output", value=st.session_state.output_text, height=400, label_visibility="collapsed")
        
        if st.session_state.output_text:
            st.success("Processing complete!")

if __name__ == "__main__":
    main()
