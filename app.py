import streamlit as st
import nltk
from paraphraser import ParaphraserEngine, SemanticValidator
from ai_avoider import AIDetectionAvoider
import os

# Set page config
st.set_page_config(
    page_title="Paraphraser & Humanizer",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Apple-like Dark Theme
st.markdown("""
<style>
    /* Global Theme Overrides */
    .stApp {
        background-color: #0b0f19;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Text Inputs */
    .stTextArea textarea {
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
        font-size: 16px;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 30px;
        font-weight: 600;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton button:hover {
        filter: brightness(1.1);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Sliders */
    div[data-baseweb="slider"] div {
        background-color: #3b82f6 !important;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .success {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid #10b981;
        color: #ecfdf5;
    }
    .error {
        background-color: rgba(239, 68, 68, 0.2);
        border: 1px solid #ef4444;
        color: #fef2f2;
    }
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
    
    status_text = st.empty()
    
    for package in required_packages:
        try:
            # Check if exists depending on the package type
            if 'punkt' in package:
                nltk.data.find(f'tokenizers/{package}')
            elif 'wordnet' in package or 'omw' in package or 'stopwords' in package:
                nltk.data.find(f'corpora/{package}')
            elif 'tagger' in package:
                nltk.data.find(f'taggers/{package}')
        except LookupError:
            # If not found, download it
            # status_text.info(f"Downloading {package}...")
            nltk.download(package, download_dir=nltk_data_path, quiet=True)
            # Also download to default for safety
            nltk.download(package, quiet=True)
            
    # status_text.empty()
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

# Main App logic
def main():
    st.title("Paraphraser & Humanizer")
    st.markdown("Transform AI-generated text into human-like content.")
    
    # Setup resources
    setup_nltk()
    engine, avoider, validator = load_engines()
    
    if not engine:
        st.error("Failed to load application engines. Please check logs.")
        return

    # Sidebar Controls
    with st.sidebar:
        st.header("Settings")
        intensity = st.slider("Intensity", 0.1, 1.0, 0.6, 0.1, help="Higher intensity means more changes to the text.")
        humanize = st.checkbox("Humanize (AI Avoidance)", value=True, help="Adds variations to bypass AI detection.")
        
        st.markdown("---")
        st.markdown("### How it works")
        st.info("""
        1. **Paraphrase**: Replaces words with context-appropriate synonyms.
        2. **Humanize**: Adds natural imperfections and structure variations.
        3. **Validate**: Checks if meaning is preserved.
        """)
        
        st.markdown("---")
        st.caption("Runs locally with Python")

    # Main Area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Text")
        input_text = st.text_area("Input", height=350, placeholder="Paste your text here...", label_visibility="collapsed")
        
        process_btn = st.button("Paraphrase Text 🪄")

    # Processing State
    if 'output_text' not in st.session_state:
        st.session_state.output_text = ""

    if process_btn and input_text:
        with st.spinner("Processing..."):
            try:
                # Step 1: Paraphrase
                result = engine.paraphrase(input_text, intensity)
                
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
        st.text_area("Output", value=st.session_state.output_text, height=350, label_visibility="collapsed")
        
        if st.session_state.output_text:
            st.success("Processing complete!")

if __name__ == "__main__":
    main()
