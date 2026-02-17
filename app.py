import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from models import get_model_class
from models.model_registry import get_model_info, format_model_info
from utils.tokenizer import CharTokenizer

# Page configuration
st.set_page_config(layout="wide", page_title="LM-Lab")
st.title("LM-Lab: Language Model Visualizer")

# 1. Load trained model
checkpoint_path = "checkpoints/bigram_checkpoint.pt"

if not os.path.exists(checkpoint_path):
    st.warning("Checkpoint not found. Train the model first.")
    st.stop()

# Load checkpoint
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

# ============ MODEL INFORMATION SECTION ============
st.divider()
st.subheader("Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Type", config['model_type'].upper())
with col2:
    st.metric("Vocabulary Size", config['vocab_size'])
with col3:
    st.metric("Total Parameters", f"{training_info.get('total_parameters', 0):,}")
with col4:
    st.metric("Trainable Parameters", f"{training_info.get('trainable_parameters', 0):,}")

# Data statistics
st.subheader("Training Data Statistics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Raw Text Size", f"{training_info.get('raw_text_size', 0):,} chars")
with col2:
    st.metric("Training Tokens", f"{training_info.get('train_data_size', 0):,}")
with col3:
    st.metric("Validation Tokens", f"{training_info.get('val_data_size', 0):,}")
with col4:
    st.metric("Unique Characters", training_info.get('unique_characters', 0))
with col5:
    st.metric("Block Size", config.get('block_size', 'N/A'))

# Training configuration
st.subheader("Training Configuration")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info(f"Learning Rate: {config.get('learning_rate', 'N/A')}")
with col2:
    st.info(f"Batch Size: {config.get('batch_size', 'N/A')}")
with col3:
    st.info(f"Training Steps: {training_info.get('training_steps', 'N/A')}")
with col4:
    st.success(f"Final Loss: {training_info.get('final_loss', 'N/A'):.4f}")

# Training loss curve
if 'loss_history' in training_info and training_info['loss_history']:
    st.subheader("Training Loss Over Time")
    
    loss_data = pd.DataFrame({
        'Step': range(0, len(training_info['loss_history'])),
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
        title="Loss Progression During Training",
        xaxis_title="Training Step",
        yaxis_title="Loss",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, width='stretch')

# ============ WEIGHT MATRIX VISUALIZATION ============
st.divider()

# 2. Extract model internals
dummy = torch.zeros((1,1), dtype=torch.long)
internals = model.get_internals(dummy)

# 3. Visualize
if 'matrix' in internals:
    st.subheader(internals.get('description', 'Matrix'))
    
    weights = internals['matrix']
    probs = torch.nn.functional.softmax(weights, dim=1).detach().numpy()
    
    df = pd.DataFrame(probs, index=tokenizer.chars, columns=tokenizer.chars)
    
    fig = px.imshow(
        df, 
        labels=dict(x="Next Character", y="Current Character", color="Probability"),
        color_continuous_scale="Blues",
        height=800
    )
    st.plotly_chart(fig, width='stretch', key="bigram_heatmap")

# 4. Text generation demo
st.divider()
st.subheader("Generate Text")
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    start_char = st.text_input("Start with:", value="A", max_chars=1)
    num_tokens = st.slider("Number of characters:", 10, 500, 100)
with col2:
    temperature = st.slider("Diversity:", 0.1, 2.0, 1.0, 0.1)

with col3:
    if st.button("Generate", key="generate_btn"):
        try:
            g = torch.Generator().manual_seed(42)
            
            if start_char not in tokenizer.stoi:
                st.error(f"Character '{start_char}' not found in vocabulary!")
            else:
                # Start with the initial character
                output_indices = [tokenizer.stoi[start_char]]
                current_idx = torch.tensor([[tokenizer.stoi[start_char]]], dtype=torch.long)
                
                for i in range(num_tokens - 1):
                    # Get logits from model (bigram returns 2D: batch_size x vocab_size)
                    logits, _ = model(current_idx)
                    logits = logits[0, :] / temperature  # Apply temperature, shape: (vocab_size,)
                    
                    # Get probabilities and sample next character
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1, generator=g)
                    
                    next_char_idx = idx_next.item()
                    output_indices.append(next_char_idx)
                    
                    # Update current_idx for next iteration (bigram only needs last char)
                    current_idx = torch.tensor([[next_char_idx]], dtype=torch.long)
                
                generated_text = tokenizer.decode(output_indices)
                st.write("**Generated Text:**")
                st.text(generated_text)
                st.caption(f"Length: {len(generated_text)} characters | Temperature: {temperature:.1f}")
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")

# 5. Next character/word prediction visualizer
st.divider()
st.subheader("Predict Next Characters")

pred_tab1, pred_tab2 = st.tabs(["Character-by-Character", "Word Prediction"])

with pred_tab1:
    st.write("Enter text and see the model's predictions for the next character")
    input_text = st.text_input("Input text:", value="Hello")
    num_predictions = st.slider("Show top predictions:", 1, 20, 5, key="char_preds")
    
    if input_text:
        try:
            if input_text[-1] not in tokenizer.stoi:
                st.error(f"Last character '{input_text[-1]}' not in vocabulary!")
            else:
                # Get the last character index
                last_char_idx = tokenizer.stoi[input_text[-1]]
                current_idx = torch.tensor([[last_char_idx]], dtype=torch.long)
                
                # Get predictions from model
                with torch.no_grad():
                    logits, _ = model(current_idx)
                    logits = logits[0, :]
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probs, k=min(num_predictions, len(probs)))
                
                # Display predictions
                pred_data = []
                for prob, idx in zip(top_probs, top_indices):
                    char = tokenizer.itos[idx.item()]
                    pred_data.append({
                        "Character": char if char != '\n' else '↵',
                        "Probability": f"{prob.item()*100:.2f}%"
                    })
                
                df = pd.DataFrame(pred_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualization
                fig = go.Figure(data=[
                    go.Bar(
                        x=[tokenizer.itos[idx.item()] if tokenizer.itos[idx.item()] != '\n' else '↵' 
                           for idx in top_indices],
                        y=top_probs.cpu().numpy() * 100,
                        marker_color='#FF6C6C'
                    )
                ])
                fig.update_layout(
                    title=f"Predictions after '{input_text[-1]}'",
                    xaxis_title="Next Character",
                    yaxis_title="Probability (%)",
                    height=400
                )
                st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.error(f"Error: {str(e)}")

with pred_tab2:
    st.write("Enter a sentence and see predictions for the next character(s)")
    phrase_input = st.text_area("Input phrase:", value="The quick brown", height=100)
    next_word_length = st.slider("How many characters for next word:", 1, 10, 3, key="word_length")
    
    if phrase_input:
        try:
            if phrase_input[-1] not in tokenizer.stoi:
                st.error(f"Last character '{phrase_input[-1]}' not in vocabulary!")
            else:
                st.write(f"**Input:** {phrase_input}")
                
                # Generate next word character by character
                generated_word = []
                current_char = phrase_input[-1]
                
                for step in range(next_word_length):
                    char_idx = tokenizer.stoi[current_char]
                    current_idx = torch.tensor([[char_idx]], dtype=torch.long)
                    
                    with torch.no_grad():
                        logits, _ = model(current_idx)
                        logits = logits[0, :]
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Sample next character
                    idx_next = torch.multinomial(probs, num_samples=1)
                    next_char = tokenizer.itos[idx_next.item()]
                    generated_word.append(next_char)
                    current_char = next_char
                    
                    # Show step-by-step
                    st.write(f"Step {step+1}: **{current_char}** (probability: {probs[idx_next].item()*100:.2f}%)")
                
                full_prediction = ''.join(generated_word)
                st.write(f"**Predicted next word characters:** {full_prediction}")
                st.write(f"**Full text would be:** {phrase_input} {full_prediction}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============ DETAILED MODEL INFORMATION ============
st.divider()
st.subheader("About This Model")

model_info = format_model_info(config['model_type'])

if model_info:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**{model_info['name']}**")
        st.write(model_info['description'])
    
    with col2:
        st.metric("Model Type", model_info['type'])
        st.metric("Complexity", model_info['complexity'])
    
    # How it works
    st.subheader("How It Works")
    for step in model_info['how_it_works']:
        st.write(f"• {step}")
    
    # Strengths and Limitations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Strengths")
        for strength in model_info['strengths']:
            st.write(f"• {strength}")
    
    with col2:
        st.subheader("⚠️ Limitations")
        for limitation in model_info['limitations']:
            st.write(f"• {limitation}")
    
    # Use Cases
    st.subheader("Use Cases")
    for use_case in model_info['use_cases']:
        st.write(f"• {use_case}")

