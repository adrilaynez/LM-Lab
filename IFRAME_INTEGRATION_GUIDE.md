# LM-Lab: Glass Box Interpretability Suite

## Overview
LM-Lab is a modular, interpretable language model visualization framework designed for iframe embedding into portfolio sites. Built with Streamlit, optimized for the Blowfish Hugo theme.

## Architecture

### Modular Structure
```
app.py                          # Router/Entry point (pure)
models/
  ├── bigram_viz.py            # Bigram visualization module
  ├── mlp_viz.py               # (Future) MLP visualization
  ├── rnn_viz.py               # (Future) RNN visualization
  └── transformer_viz.py       # (Future) Transformer visualization
```

### Key Design Patterns
- **Query Param Routing**: `?model=bigram` | `?model=mlp` | `?model=rnn`
- **Registry Pattern**: MODELS dict for scalable model registration
- **Modular Visualization**: Each model has its own `render_*()` function
- **Caching**: `@st.cache_resource` for expensive computations

---

## Deployment

### 1. Deploy to Streamlit Cloud

```bash
# Push to GitHub (already done)
git push origin main

# Visit https://share.streamlit.io
# Click "New app"
# Repository: adrilaynez/LM-Lab
# Main file: app.py
# Click Deploy
```

Your app will get a URL: `https://lm-lab.streamlit.app`

### 2. Embed in Hugo Blowfish Theme

Add to your portfolio project markdown:

```markdown
---
title: "LM-Lab: Glass Box Interpretability Suite"
description: "Interactive visualization of language models"
draft: false
---

## Features

- Character-level bigram modeling
- Interactive text generation
- Probability predictions
- Weight matrix visualization
- Extensible framework for new models

**[Launch in Full Window →](https://lm-lab.streamlit.app)**
```

Or embed as iframe in your Hugo template:

```html
<div class="frame-container">
  <iframe 
    src="https://lm-lab.streamlit.app?model=bigram&embedded=true" 
    height="900" 
    style="width:100%;border:none;border-radius:8px;">
  </iframe>
</div>
```

### 3. CSS for Blowfish Theme Integration

In your Hugo theme CSS (or post frontmatter):

```css
.frame-container {
  width: 100%;
  max-width: 1200px;
  margin: 2rem auto;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  overflow: hidden;
}

@media (prefers-color-scheme: dark) {
  .frame-container {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
  }
}
```

---

## URL Query Parameters

### Model Selection
```
?model=bigram        # Character-level bigram (default)
?model=mlp          # Multi-layer perceptron (future)
?model=rnn          # LSTM/RNN (future)
?model=transformer  # Transformer/GPT (future)
```

### Examples
- `https://lm-lab.streamlit.app`
- `https://lm-lab.streamlit.app?model=bigram`
- `https://lm-lab.streamlit.app?model=mlp` (when available)

---

## Adding New Models

### Step 1: Create Visualization Module
Create `models/your_model_viz.py`:

```python
import streamlit as st
import torch

@st.cache_resource
def load_your_model():
    """Load and cache your trained model"""
    # Your loading logic
    return model, config, data

def render_your_model():
    """Main visualization function"""
    model, config, data = load_your_model()
    
    st.subheader("Your Model Visualization")
    # Add your visualizations here
    # Remember to use width='stretch' for responsiveness!
    st.plotly_chart(fig, width='stretch')
```

### Step 2: Register in app.py
Update `app.py` MODELS dict:

```python
from models.your_model_viz import render_your_model

MODELS = {
    "bigram": {...},
    "your_model": {
        "name": "Your Model Name",
        "description": "Description of your model",
        "render": render_your_model
    }
}
```

### Step 3: That's it!
- Your model is now available at `?model=your_model`
- No other changes needed

---

## Performance Optimizations

### 1. Caching
All expensive operations are cached with `@st.cache_resource`:

```python
@st.cache_resource
def load_bigram_model():
    # Loads once, reused across reruns
    ...
```

### 2. Responsive Layouts
All charts use `width='stretch'` for iframe responsiveness:

```python
st.plotly_chart(fig, width='stretch')
```

### 3. CSS Optimization
- Zero padding for iframe embedding
- Transparent background for theme blending
- Hidden Streamlit branding

---

## Current Model: Bigram

- **Type**: Statistical (no neural network)
- **Parameters**: 9,216 (96×96 weight matrix)
- **Training Data**: Paul Graham essays (637K characters)
- **Training Time**: ~5 seconds
- **Final Loss**: 2.57

### Visualizations
- ✅ Training loss curve
- ✅ Character transition matrix heatmap
- ✅ Text generation with temperature control
- ✅ Character prediction with probabilities
- ✅ Word prediction (step-by-step)
- ✅ Model information and interpretability guide

---

## Future Models (Planned)

### MLP (Multi-Layer Perceptron)
- Character embeddings
- Hidden layer visualization
- Activation heatmaps

### RNN/LSTM
- Hidden state evolution
- Gate visualizations
- Sequence attention

### Transformer
- Multi-head attention patterns
- Token embeddings
- Attention matrices

---

## Troubleshooting

### Issue: App fails to load
1. Check Streamlit Cloud logs
2. Verify requirements.txt versions
3. Test locally: `streamlit run app.py`

### Issue: Charts don't show in iframe
- Ensure all charts have `width='stretch'`
- Check browser console for errors
- Verify iframe has sufficient height

### Issue: Slow performance
- Check that @st.cache_resource is being used
- Monitor model loading time
- Consider lazy loading for heavy visualizations

---

## Development

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Train model (optional - checkpoint included)
python train.py

# Run app
streamlit run app.py

# Test with query params
streamlit run app.py -- --query-params="model=bigram"
```

### Browser Dev Tools
- F12 → Console for errors
- Network tab to monitor iframe communication
- CSS inspector for styling issues

---

## Links

- **Live App**: https://lm-lab.streamlit.app
- **GitHub Repo**: https://github.com/adrilaynez/LM-Lab
- **Portfolio**: https://adrianlaynez.dev
- **Blowfish Theme**: https://blowfish.page

---

## Architecture Benefits

1. **Scalability**: Add models without touching core code
2. **Maintainability**: Each model isolated in its own module
3. **Performance**: Caching prevents unnecessary recomputation
4. **UX**: Clean, branded-free interface for embedding
5. **Extensibility**: Query params allow future enhancements

---

## Example Integration

Add this to your Hugo portfolio's projects section:

```html
<section class="project-card">
  <h3>LM-Lab: Glass Box Interpretability</h3>
  <p>Interactive visualization of language models with neural network interpretability.</p>
  
  <iframe 
    src="https://lm-lab.streamlit.app?model=bigram&embedded=true"
    height="800"
    style="width:100%;border:none;border-radius:8px;margin:1rem 0;">
  </iframe>
  
  <a href="https://lm-lab.streamlit.app" target="_blank">
    Open in Full Window →
  </a>
</section>
```

---

## Contact & Support

- GitHub Issues: https://github.com/adrilaynez/LM-Lab/issues
- Email: adrilaynezortiz@gmail.com
- Portfolio: https://adrianlaynez.dev
