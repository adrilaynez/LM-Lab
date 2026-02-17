"""
LM-Lab: Glass Box Interpretability Suite
Main router for model visualization modules - optimized for iframe embedding
"""

import streamlit as st
from models.bigram_viz import render_bigram
from models.mlp_viz import render_mlp

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
    header {display: none !important;}
    footer {display: none !important;}
    .streamlit-container {padding: 0rem !important;}
    
    /* Remove default margins and padding */
    [data-testid="stVerticalBlock"] {padding-left: 0rem !important; padding-right: 0rem !important;}
    
    /* Glass background for theme blending with Blowfish Tailwind theme */
    .main {background: transparent !important;}
    [data-testid="stAppViewContainer"] {background: transparent !important;}
    
    /* Ensure full width responsiveness in iframe */
    .block-container {padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important;}
    
    /* Hide Made with Streamlit badge */
    footer:nth-child(n) {display: none !important;}
    .viewerBadge_container__r5tak {display: none !important;}
    
    /* Remove scrollbars for cleaner embedding */
    html, body {overflow-x: hidden !important;}
    </style>
""", unsafe_allow_html=True)

# ============ MODEL REGISTRY & ROUTING ============
# Scalable registry for all model visualizations
MODELS = {
    "bigram": {
        "name": "Bigram Model",
        "description": "Character-level bigram language model trained on Paul Graham essays",
        "render": render_bigram
    },
    "mlp": {
        "name": "MLP Model",
        "description": "Multi-layer perceptron with character embeddings - learns non-linear patterns",
        "render": render_mlp
    }
    # Future models - just add and register here:
    # "rnn": {
    #     "name": "LSTM/RNN Model",
    #     "description": "Recurrent neural network for sequence modeling",
    #     "render": render_rnn
    # },
    # "transformer": {
    #     "name": "Transformer Model",
    #     "description": "Multi-head attention-based language model",
    #     "render": render_transformer
    # }
}

# ============ ROUTER LOGIC ============
def main():
    """
    Main router function with query_params support
    Usage: ?model=bigram (default), ?model=mlp, ?model=rnn, etc.
    """
    # Get model from URL query params - compatible with older Streamlit versions
    try:
        query_params = st.query_params
    except AttributeError:
        # Fallback for older Streamlit versions
        try:
            import streamlit as st_compat
            query_params = st_compat.experimental_get_query_params()
        except:
            query_params = {}
    
    # Convert query_params to dict if needed
    if isinstance(query_params, dict):
        model_name = query_params.get("model", ["bigram"])[0] if "model" in query_params else "bigram"
    else:
        model_name = query_params.get("model", "bigram") if hasattr(query_params, 'get') else "bigram"
    
    model_name = model_name.lower()
    
    # Validate model exists
    if model_name not in MODELS:
        st.error(f"❌ Model '{model_name}' not found.")
        st.info(f"**Available models:** {', '.join(MODELS.keys())}")
        st.markdown("**Use query params to select models:**")
        for model_key in MODELS.keys():
            st.code(f"?model={model_key}", language="text")
        return
    
    model_config = MODELS[model_name]
    
    # Display model title and description
    st.title(f"🔬 {model_config['name']}")
    st.markdown(f"*{model_config['description']}*")
    st.divider()
    
    # Render the selected model's visualization
    try:
        model_config["render"]()
    except Exception as e:
        st.error(f"❌ Error rendering {model_name} visualization")
        st.error(f"Details: {str(e)}")
        with st.expander("📋 See full error trace"):
            st.code(str(e), language="text")

# ============ THEME SYNCHRONIZATION ============
# This script ensures dark mode by default
# When embedded in Blowfish (Tailwind), it will inherit parent styles
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

