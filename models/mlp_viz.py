"""
MLP Model Visualization Module
Interactive charts and predictions for MLP language model
"""
import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from utils.tokenizer import CharTokenizer
from utils.data import load_data
from models import get_model_class


@st.cache_resource
def load_mlp_model():
    """Load and cache MLP model from checkpoint"""
    import os
    checkpoint_path = os.path.join('checkpoints', 'mlp_checkpoint.pt')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Recreate tokenizer
    raw_text = load_data('data/paul_graham.txt')
    tokenizer = CharTokenizer()
    tokenizer.train(raw_text)
    
    # Reconstruct model
    config = checkpoint['config']
    model_class = get_model_class('mlp')
    model = model_class(vocab_size=config['vocab_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint, tokenizer


def render_mlp():
    """Main MLP visualization function"""
    model, checkpoint, tokenizer = load_mlp_model()
    
    # ============ MODEL INFO ============
    st.markdown("### 📊 Model Architecture")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Parameters", f"{checkpoint['training_info']['total_parameters']:,}")
    with col2:
        st.metric("Vocab Size", checkpoint['config']['vocab_size'])
    with col3:
        st.metric("Block Size", checkpoint['config']['block_size'])
    with col4:
        st.metric("Final Loss", f"{checkpoint['training_info']['final_loss']:.4f}")
    
    st.divider()
    
    # ============ TRAINING LOSS CURVE ============
    st.markdown("### 📈 Training Loss Curve")
    loss_history = checkpoint['training_info']['loss_history']
    
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        y=loss_history,
        mode='lines',
        name='Training Loss',
        line=dict(color='#FF6C6C', width=2)
    ))
    fig_loss.update_layout(
        xaxis_title="Training Step",
        yaxis_title="Cross-Entropy Loss",
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_loss, width='stretch')
    
    st.divider()
    
    # ============ NETWORK ARCHITECTURE VISUALIZATION ============
    st.markdown("### 🧠 Network Architecture")
    st.write("""
    **MLP Architecture:**
    - **Input:** Character sequence (batch_size × block_size)
    - **Embedding:** 96 vocab → 10-dim vectors (96 × 10 = 960 params)
    - **Hidden1:** 80 → 64 neurons (5,120 params + batch norm)
    - **Hidden2:** 64 → 32 neurons (2,048 params + batch norm)
    - **Output:** 32 → 96 (vocab) predictions (3,072 params + bias)
    
    **Total: 11,584 parameters** (vs 9,216 for Bigram)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hidden layer 1 weights heatmap
        st.markdown("**Hidden Layer 1 Weights Heatmap**")
        hidden1_weights = checkpoint['model_state_dict']['hidden1.weight'].numpy()
        
        fig_h1 = go.Figure(data=go.Heatmap(
            z=hidden1_weights,
            colorscale='Viridis',
            name='Weights'
        ))
        fig_h1.update_layout(
            xaxis_title="Input Features",
            yaxis_title="Hidden Neurons",
            height=400,
            width=450
        )
        st.plotly_chart(fig_h1, use_container_width=True)
    
    with col2:
        # Hidden layer 2 weights heatmap
        st.markdown("**Hidden Layer 2 Weights Heatmap**")
        hidden2_weights = checkpoint['model_state_dict']['hidden2.weight'].numpy()
        
        fig_h2 = go.Figure(data=go.Heatmap(
            z=hidden2_weights,
            colorscale='Plasma',
            name='Weights'
        ))
        fig_h2.update_layout(
            xaxis_title="Input Features (H1 Activations)",
            yaxis_title="Hidden Neurons",
            height=400,
            width=450
        )
        st.plotly_chart(fig_h2, use_container_width=True)
    
    st.divider()
    
    # ============ TEXT GENERATION ============
    st.markdown("### 🎲 Text Generation")
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_input("Starting text:", value="The future of", max_chars=50)
    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8)
    
    max_new_tokens = st.slider("Tokens to generate", 10, 200, 50)
    
    if st.button("🚀 Generate Text", type="primary"):
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt)
        if not prompt_tokens:
            st.warning("⚠️ Invalid prompt text")
        else:
            generated = []
            context = torch.tensor([prompt_tokens], dtype=torch.long)
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # Get last block_size tokens
                    x = context[:, -checkpoint['config']['block_size']:]
                    
                    # Get logits
                    logits, _ = model(x)
                    
                    # Apply temperature
                    logits = logits[:, -1, :] / temperature
                    
                    # Sample
                    probs = torch.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    
                    generated.append(idx_next.item())
                    context = torch.cat([context, idx_next.unsqueeze(0)], dim=1)
            
            # Decode
            generated_text = prompt + tokenizer.decode(generated)
            st.text_area("Generated Text:", value=generated_text, height=200, disabled=True)
    
    st.divider()
    
    # ============ CHARACTER PREDICTION ============
    st.markdown("### 🔮 Character Prediction")
    st.write("See what the model predicts for the next character given a context:")
    
    pred_text = st.text_input("Enter context:", value="The meaning of life", max_chars=100)
    
    if pred_text:
        tokens = tokenizer.encode(pred_text)
        if tokens:
            # Pad or truncate to block_size
            block_size = checkpoint['config']['block_size']
            if len(tokens) < block_size:
                tokens = [tokenizer.stoi.get('\n', 0)] * (block_size - len(tokens)) + tokens
            else:
                tokens = tokens[-block_size:]
            
            x = torch.tensor([tokens], dtype=torch.long)
            
            with torch.no_grad():
                logits, _ = model(x)
                logits = logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_k = 10
            top_probs, top_indices = torch.topk(probs[0], top_k)
            top_probs = top_probs.numpy()
            top_chars = [tokenizer.itos[idx] for idx in top_indices.numpy()]
            
            # Display as bar chart
            fig_pred = go.Figure(data=[
                go.Bar(
                    x=top_probs,
                    y=[f"'{c}'" if c != '\n' else "'\\n'" for c in top_chars],
                    orientation='h',
                    marker=dict(color=top_probs, colorscale='Viridis')
                )
            ])
            fig_pred.update_layout(
                xaxis_title="Probability",
                yaxis_title="Character",
                height=400,
                showlegend=False,
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_pred, width='stretch')
            
            # Detailed prediction table
            st.markdown("**Top Predictions:**")
            pred_data = {
                'Character': [f"'{c}'" if c != '\n' else "'\\n'" for c in top_chars],
                'Probability': [f"{p:.4f}" for p in top_probs],
                'Rank': list(range(1, top_k + 1))
            }
            st.dataframe(pred_data, use_container_width=True)
    
    st.divider()
    
    # ============ MODEL INSIGHTS ============
    st.markdown("### 💡 Model Interpretability")
    st.markdown("""
    **MLP Advantages:**
    - ✅ **More capacity** than Bigram (11.5K vs 9.2K parameters)
    - ✅ **Character embeddings** learn useful representations
    - ✅ **Multiple layers** capture non-linear patterns
    - ✅ **Hidden activations** reveal feature learning
    - ✅ **Better contextual understanding** with full sequence processing
    
    **How it works:**
    1. Each character in the context window gets a 10-dimensional embedding
    2. These embeddings are concatenated (80 dimensions total for block_size=8)
    3. Two hidden layers with ReLU activations process the context
    4. Batch normalization stabilizes training
    5. Final layer produces probabilities for all 96 characters
    
    **Key Insights:**
    - The model learns that common characters like spaces and 'e' have high probability
    - Context influences predictions (MLPs use full sequence context)
    - Hidden layer activations show which features are most discriminative
    - Temperature parameter controls randomness vs determinism in generation
    
    **Compared to Bigram:**
    - MLP processes entire sequence context at once
    - Bigram only looks at single previous character
    - MLP can learn multi-character patterns
    - Both models are fast enough for real-time interaction
    """)
    
    st.divider()
    
    # ============ TRAINING INFO ============
    st.markdown("### 📋 Training Details")
    info = checkpoint['training_info']
    train_info_data = {
        'Metric': [
            'Training Steps',
            'Batch Size',
            'Learning Rate',
            'Final Loss',
            'Training Data Size',
            'Validation Data Size',
            'Unique Characters',
            'Training Data (chars)',
        ],
        'Value': [
            f"{checkpoint['config']['batch_size']} steps per epoch",
            '32',
            str(checkpoint['config']['learning_rate']),
            f"{info['final_loss']:.4f}",
            f"{info['train_data_size']:,} tokens",
            f"{info['val_data_size']:,} tokens",
            str(info['unique_characters']),
            f"{info['raw_text_size']:,} characters",
        ]
    }
    st.dataframe(train_info_data, use_container_width=True)
