"""
LM-Lab: Glass Box Interpretability Suite
Main router for model visualization modules - optimized for iframe embedding
"""

import streamlit as st
import importlib

# Import model registry metadata
from models.model_registry import MODEL_INFO, get_model_info

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    layout="wide", 
    page_title="LM-Lab",
    initial_sidebar_state="collapsed"
)

# ============ CSS INJECTION FOR IFRAME EMBEDDING ============
# Hides Streamlit branding and optimizes layout for iframe embedding
st.markdown("""
    <style>
    /* Hide Streamlit header and footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .streamlit-container {padding: 0rem !important;}
    
    /* Remove default margins and padding */
    [data-testid="stVerticalBlock"] {padding-left: 0rem !important; padding-right: 0rem !important; gap: 0rem !important;}
    
    /* Glass background for theme blending with Blowfish Tailwind theme */
    .stApp {background: transparent !important;}
    .main {background: transparent !important;}
    [data-testid="stAppViewContainer"] {background: transparent !important;}
    
    /* Ensure full width responsiveness in iframe */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important; 
        padding-right: 1rem !important; 
        max-width: 100% !important;
    }
    
    /* Hide Made with Streamlit badge */
    footer:nth-child(n) {display: none !important;}
    .viewerBadge_container__r5tak {display: none !important;}
    
    /* Remove scrollbars for cleaner embedding */
    /* html, body {overflow-x: hidden !important;} */
    
    /* Dashboard Card Styles */
    .model-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
        height: 100%;
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        background-color: rgba(255, 255, 255, 0.1);
        border-color: #FF6C6C;
    }
    .model-title {
        color: #FF6C6C;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .model-desc {
        color: #ddd;
        font-size: 0.9rem;
    }
    
    /* Custom buttons */
    .stButton>button {
        background-color: #FF6C6C;
        color: white;
        border: none;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #ff4d4d;
        color: white;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)


# ============ ROUTER LOGIC ============
def get_query_param(key):
    """Get query parameter with compatibility for older Streamlit versions"""
    try:
        query_params = st.query_params
        val = query_params.get(key)
        return val if val else None
    except AttributeError:
        # Fallback
        try:
            import streamlit as st_compat
            query_params = st_compat.experimental_get_query_params()
            val = query_params.get(key)
            return val[0] if val and isinstance(val, list) else val
        except:
            return None

def main():
    """
    Main router function.
    Usage:
    - Root: Dashboard showing available models
    - ?model=name: Load specific model visualization
    """
    
    # Get model from URL query params
    model_name = get_query_param("model")
    
    if model_name:
        model_name = model_name.lower().strip()
        
        # Validate model exists
        if model_name not in MODEL_INFO:
            st.error(f"❌ Model '{model_name}' not found.")
            st.info(f"**Available models:** {', '.join(MODEL_INFO.keys())}")
            if st.button("⬅️ Back to Dashboard"):
                st.query_params.clear()
                st.rerun()
            return
        
        # Load and render the selected model
        try:
            # Dynamic import based on model name
            if model_name == "bigram":
                from models.bigram_viz import render_bigram
                render_bigram()
            elif model_name == "mlp":
                from models.mlp_viz import render_mlp
                render_mlp()
            # Add future models here
            else:
                st.warning(f"⚠️ Visualization for '{model_name}' is not yet implemented.")
                
        except Exception as e:
            st.error(f"❌ Error rendering {model_name} visualization")
            st.error(f"Details: {str(e)}")
            with st.expander("📋 See full error trace"):
                st.code(str(e), language="text")
                
    else:
        # ============ DASHBOARD VIEW ============
        st.markdown("<h1 style='text-align: center; color: #FF6C6C; margin-bottom: 2rem;'>🧠 LM-Lab Model Registry</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; margin-bottom: 3rem;'>Select a model below to explore its architecture and capabilities.</p>", unsafe_allow_html=True)
        
        # Create a grid layout for model cards
        cols = st.columns(3)
        
        for idx, (key, info) in enumerate(MODEL_INFO.items()):
            col_idx = idx % 3
            
            with cols[col_idx]:
                # Card HTML structure
                card_html = f"""
                <a href="?model={key}" target="_self" style="text-decoration: none;">
                    <div class="model-card">
                        <div class="model-title">{info['name']}</div>
                        <div class="model-desc">{info['description']}</div>
                        <div style="margin-top: 15px; font-size: 0.8rem; color: #aaa;">
                            Complexity: <span style="color: #FF6C6C;">{info.get('complexity', 'Unknown')}</span>
                        </div>
                    </div>
                </a>
                """
                st.markdown(card_html, unsafe_allow_html=True)

# ============ THEME SYNCHRONIZATION ============
# This script ensures dark mode by default
@st.cache_resource
def init_theme():
    """Initialize theme - dark mode by default for glass aesthetic"""
    return {
        "primaryColor": "#FF6C6C",
        "backgroundColor": "#0E1117",
        "secondaryBackgroundColor": "#161B22",
        "textColor": "#FAFAFA",
        "font": "sans serif"
    }

# Initialize theme
init_theme()

# ============ RUN ============
if __name__ == "__main__":
    main()

