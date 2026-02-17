"""
TEMPLATE: How to Add MLP Model Visualization
Copy this structure and modify as needed for any new model
"""

# ============ STEP 1: Create models/mlp_viz.py ============
"""
import streamlit as st
import torch
import plotly.graph_objects as go
import pandas as pd
from models import get_model_class
from utils.tokenizer import CharTokenizer


@st.cache_resource
def load_mlp_model():
    '''Load and cache the trained MLP model'''
    checkpoint_path = "checkpoints/mlp_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    training_info = checkpoint.get('training_info', {})
    
    # Reconstruct tokenizer
    tokenizer = CharTokenizer()
    tokenizer.chars = config['chars']
    tokenizer.stoi = {ch:i for i,ch in enumerate(tokenizer.chars)}
    tokenizer.itos = {i:ch for i,ch in enumerate(tokenizer.chars)}
    
    # Instantiate model
    ModelClass = get_model_class(config['model_type'])
    model = ModelClass(config['vocab_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer, config, training_info


def render_mlp():
    '''Main visualization function for MLP model'''
    
    try:
        model, tokenizer, config, training_info = load_mlp_model()
    except FileNotFoundError:
        st.error("❌ MLP checkpoint not found. Train the model first.")
        st.info("Run: python train.py with config.MODEL_TYPE = 'mlp'")
        st.stop()
    
    # ============ MODEL INFORMATION ============
    st.subheader("Model Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", "MLP")
    with col2:
        st.metric("Vocabulary Size", config['vocab_size'])
    with col3:
        st.metric("Hidden Layers", config.get('hidden_size', 'N/A'))
    with col4:
        st.metric("Total Parameters", f"{training_info.get('total_parameters', 0):,}")
    
    # ============ TRAINING METRICS ============
    st.subheader("Training Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Loss", f"{training_info.get('final_loss', 0):.4f}")
    with col2:
        st.metric("Training Steps", training_info.get('training_steps', 0))
    with col3:
        st.metric("Learning Rate", config.get('learning_rate', 'N/A'))
    with col4:
        st.metric("Batch Size", config.get('batch_size', 'N/A'))
    
    # ============ LOSS CURVE ============
    if 'loss_history' in training_info:
        st.subheader("Training Loss Over Time")
        
        loss_data = pd.DataFrame({
            'Step': range(len(training_info['loss_history'])),
            'Loss': training_info['loss_history']
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=loss_data['Step'],
            y=loss_data['Loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='#FF6C6C', width=2)
        ))
        
        fig.update_layout(
            title="Loss Progression",
            xaxis_title="Step",
            yaxis_title="Loss",
            height=400
        )
        st.plotly_chart(fig, width='stretch')  # IMPORTANT: width='stretch'
    
    # ============ ADD MLP-SPECIFIC VISUALIZATIONS ============
    # Example: Embedding space visualization, activation patterns, etc.
    
    st.info("🚧 Additional MLP-specific visualizations coming soon!")
"""

# ============ STEP 2: Update app.py ============
"""
In app.py, add the import at the top:

from models.mlp_viz import render_mlp

Then update the MODELS dictionary:

MODELS = {
    "bigram": {
        "name": "Bigram Model",
        "description": "Character-level bigram language model",
        "render": render_bigram
    },
    "mlp": {
        "name": "MLP Model",
        "description": "Multi-layer perceptron with character embeddings",
        "render": render_mlp
    }
}
"""

# ============ STEP 3: Create models/mlp.py ============
"""
Create the actual MLP model class in models/mlp.py:

from .base import LMEngine
import torch
import torch.nn as nn


class MLPModel(LMEngine):
    def __init__(self, vocab_size, hidden_size=256):
        super().__init__(vocab_size)
        self.hidden_size = hidden_size
        
        # Your MLP architecture
        self.embed = nn.Embedding(vocab_size, 64)
        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self.relu = nn.ReLU()
        
    def forward(self, idx, targets=None):
        # idx shape: (batch_size, seq_len)
        x = self.embed(idx[:, -1])  # Take last token
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits, targets)
        
        return logits, loss
    
    def get_internals(self, idx):
        # Return intermediate representations for visualization
        with torch.no_grad():
            embeddings = self.embed(idx[:, -1])
        
        return {
            "embeddings": embeddings,
            "hidden_features": embeddings.shape[-1]
        }
"""

# ============ STEP 4: Register in models/__init__.py ============
"""
Add to models/__init__.py:

from .mlp import MLPModel

def get_model_class(model_type):
    if model_type == 'bigram':
        return BigramModel
    elif model_type == 'mlp':
        return MLPModel
    else:
        raise ValueError(f"Unknown model type: {model_type}")
"""

# ============ STEP 5: Update config.py ============
"""
Add MLP configuration to config.py:

# Model selection
MODEL_TYPE = 'mlp'  # Change to 'mlp' to train MLP

# MLP-specific hyperparameters (if different from bigram)
HIDDEN_SIZE = 256
"""

# ============ STEP 6: Train and Deploy ============
"""
# Train the model locally
python train.py

# Test locally
streamlit run app.py

# Push to GitHub
git add -A
git commit -m "feat: Add MLP model visualization"
git push origin main

# Streamlit Cloud will auto-redeploy
# Access at: https://lm-lab.streamlit.app?model=mlp
"""

# ============ KEY POINTS FOR NEW MODELS ============
"""
✅ DO:
- Use @st.cache_resource for model loading
- Use width='stretch' for all charts
- Follow the render_model() function pattern
- Keep visualizations modular
- Add error handling with try/except
- Document with docstrings
- Use st.divider() for sections
- Keep CSS in app.py consistent

❌ DON'T:
- Import model-specific code in app.py
- Hardcode file paths (use config)
- Use deprecated Streamlit parameters
- Add CSS in individual modules
- Mix router logic with visualization
- Make modules co-dependent
"""

# ============ TESTING CHECKLIST ============
"""
For each new model:
□ Train locally and verify checkpoint saves
□ Test visualization renders without errors
□ Check all charts use width='stretch'
□ Verify caching with @st.cache_resource
□ Test query params: ?model=your_model
□ Test iframe embedding: ?embedded=true
□ Check for console errors (F12)
□ Verify responsive layout on mobile
□ Review CSS hiding for Streamlit branding
"""
