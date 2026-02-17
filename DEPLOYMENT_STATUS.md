# 🚀 LM-Lab Deployment Status & Quick Start

## ✅ Completed

### Architecture
- ✅ Modular visualization system with query param routing
- ✅ Registry pattern for scalable model registration
- ✅ CSS injection for iframe embedding
- ✅ Caching system for performance
- ✅ Theme synchronization for Blowfish compatibility
- ✅ Responsive layout with full-width charts

### Current Model: Bigram
- ✅ Text generation with temperature control
- ✅ Character prediction visualizer
- ✅ Word prediction (step-by-step)
- ✅ Training loss visualization
- ✅ Weight matrix heatmap
- ✅ Model information registry
- ✅ Data statistics display

### Documentation
- ✅ IFRAME_INTEGRATION_GUIDE.md - Complete integration instructions
- ✅ TEMPLATE_ADD_NEW_MODEL.py - Template for adding new models
- ✅ Code comments throughout

---

## 🚀 Quick Start for Portfolio Integration

### 1. Verify Deployment (You're Here!)
```bash
git log --oneline
# Should see:
# 46aa2fc docs: Add template guide for adding new model visualizations
# 0b347d0 docs: Add comprehensive iframe integration guide for Blowfish theme
# e93824f refactor: Modular Glass Box architecture with query param routing
```

### 2. Check Streamlit Cloud Status
Visit: https://share.streamlit.io
- Look for your "lm-lab" app
- Click "Manage app" → Check logs if needed
- Status should be "running" (green indicator)

### 3. Test Direct URLs
- Default: https://lm-lab.streamlit.app
- Bigram model: https://lm-lab.streamlit.app?model=bigram
- (Future) MLP: https://lm-lab.streamlit.app?model=mlp

### 4. Embed in Hugo Blowfish

#### Option A: Link to Full App
In your project markdown:
```markdown
[Launch LM-Lab →](https://lm-lab.streamlit.app?model=bigram)
```

#### Option B: Embed as iframe
In your Hugo template or markdown:
```html
<iframe 
  src="https://lm-lab.streamlit.app?model=bigram&embedded=true" 
  height="900" 
  style="width:100%;border:none;border-radius:8px;margin:1rem 0;">
</iframe>
```

#### Option C: Add CSS Styling
In your theme's CSS:
```css
.lm-lab-frame {
  width: 100%;
  max-width: 1200px;
  height: 900px;
  border: none;
  border-radius: 8px;
  margin: 2rem auto;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
```

Then use:
```html
<iframe 
  src="https://lm-lab.streamlit.app?model=bigram&embedded=true"
  class="lm-lab-frame">
</iframe>
```

---

## 📋 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | Streamlit | Web UI |
| **ML** | PyTorch | Model training & inference |
| **Visualizations** | Plotly | Interactive charts |
| **Data** | Pandas | Data manipulation |
| **Deployment** | Streamlit Cloud | Hosting |

---

## 🏗️ Project Structure

```
LM-Lab/
├── app.py                          # Main router (pure)
├── train.py                        # Training script
├── config.py                       # Configuration
├── requirements.txt                # Dependencies
│
├── models/
│   ├── __init__.py                 # Factory pattern
│   ├── base.py                     # Abstract base class
│   ├── bigram.py                   # Bigram model
│   ├── bigram_viz.py               # Bigram visualization (NEW)
│   ├── model_registry.py           # Model info registry
│   └── [mlp_viz.py]                # (Future) MLP visualization
│
├── utils/
│   ├── tokenizer.py                # Character tokenizer
│   └── data.py                     # Data utilities
│
├── data/
│   └── paul_graham.txt             # Training data
│
├── checkpoints/
│   └── bigram_checkpoint.pt        # Trained model
│
└── docs/
    ├── README.md                   # Main documentation
    ├── IFRAME_INTEGRATION_GUIDE.md # Integration with Hugo
    ├── TEMPLATE_ADD_NEW_MODEL.py   # Template for new models
    └── (This file)
```

---

## 🎯 Features Implemented

### App.py (Router)
- Query param routing: `?model=bigram`
- CSS injection for iframe embedding
- Streamlit branding removal
- Transparent background for theme blending
- Error handling with fallbacks
- Theme initialization

### Bigram Visualization
- Model loading with caching
- 6 metric displays (type, vocab size, parameters, etc.)
- 5 data statistics displays
- Training configuration display
- Training loss curve (Plotly)
- Character transition matrix heatmap (Plotly)
- Text generation with temperature control
- Character prediction with probability rankings
- Word prediction (step-by-step generation)
- Model information and interpretability guide

### Performance Optimizations
- `@st.cache_resource` for model loading
- `width='stretch'` for all charts (responsive)
- Zero padding for iframe efficiency
- Lazy loading ready

---

## 🔄 Workflow for Adding New Models

### Quick Version
1. Create `models/your_model_viz.py` with `render_your_model()` function
2. Import in `app.py`
3. Add to MODELS dictionary
4. That's it! Your model is live at `?model=your_model`

### See TEMPLATE_ADD_NEW_MODEL.py for:
- Complete MLP example with code
- Step-by-step instructions
- All registration points
- Testing checklist
- Common pitfalls

---

## 🌐 Integration Examples

### For Blog Post
```markdown
# My Language Model Visualizer

I built an interactive visualization of language models. Try it here:

<iframe 
  src="https://lm-lab.streamlit.app?model=bigram&embedded=true" 
  height="800" 
  style="width:100%;border:none;border-radius:8px;">
</iframe>

This app demonstrates:
- Character-level language modeling
- Probability prediction
- Interactive text generation

[View full app →](https://lm-lab.streamlit.app)
```

### For Portfolio Project Card
```markdown
---
title: "LM-Lab: Glass Box Interpretability Suite"
description: "Interactive visualization of language models"
date: 2026-02-17
tags: ["Machine Learning", "PyTorch", "Streamlit", "NLP"]
link: "https://lm-lab.streamlit.app"
---

## Description
An interpretable language model visualizer built from scratch in PyTorch.

## Try It
[Launch LM-Lab →](https://lm-lab.streamlit.app)

## Features
- Bigram character-level model
- Interactive text generation
- Probability predictions
- Weight matrix visualization
- Extensible for new models
```

---

## 🔗 Important Links

| Resource | URL |
|----------|-----|
| **Live App** | https://lm-lab.streamlit.app |
| **GitHub Repo** | https://github.com/adrilaynez/LM-Lab |
| **Portfolio** | https://adrianlaynez.dev |
| **Blowfish Theme** | https://blowfish.page |

---

## ✨ Key Features That Make This "Glass Box"

1. **Interpretability First**: All decisions visualized, nothing hidden
2. **Modular Design**: Easy to add new models without complexity
3. **Performance**: Caching and optimization built-in
4. **Embeddable**: Seamless integration into portfolio sites
5. **Scalable**: Designed to grow from Bigram → MLP → RNN → Transformer
6. **Educational**: Comments and documentation throughout

---

## 🚦 Deployment Status

| Component | Status | Details |
|-----------|--------|---------|
| **Streamlit Cloud** | ✅ Deployed | Live at https://lm-lab.streamlit.app |
| **GitHub Integration** | ✅ Connected | Auto-deploys on push |
| **Requirements.txt** | ✅ Optimized | PyTorch CPU build included |
| **CSS Injection** | ✅ Working | Hides Streamlit branding |
| **Query Params** | ✅ Working | ?model=bigram routing |
| **Caching** | ✅ Implemented | Model loads once |
| **Responsive Layout** | ✅ Verified | Works in iframe |
| **Documentation** | ✅ Complete | Three guide files included |

---

## 🎓 Next Steps

### Immediate
- [ ] Test embedding in your Hugo site
- [ ] Verify iframe displays correctly
- [ ] Check responsive behavior on mobile

### Soon (Optional)
- [ ] Add MLP model following TEMPLATE_ADD_NEW_MODEL.py
- [ ] Enhance visualizations (e.g., embedding space)
- [ ] Add training dashboard

### Future Roadmap
- [ ] RNN/LSTM visualization
- [ ] Transformer with attention maps
- [ ] Model comparison view
- [ ] Custom training interface

---

## 💡 Pro Tips

1. **Always use `width='stretch'`** on Plotly charts for iframe responsiveness
2. **Cache expensive operations** with `@st.cache_resource`
3. **Keep modules isolated** - each model in its own file
4. **Test locally first**: `streamlit run app.py`
5. **Push to GitHub to deploy**: Streamlit Cloud auto-redeploys

---

## 🆘 Troubleshooting

### App won't load
1. Check Streamlit Cloud logs: https://share.streamlit.io
2. Test locally: `streamlit run app.py`
3. Verify requirements.txt has no conflicts

### Charts not showing in iframe
- Add `width='stretch'` to all charts
- Increase iframe height (try 900px+)
- Check browser console (F12) for errors

### Performance issues
- Ensure @st.cache_resource is used
- Monitor model loading time
- Reduce cache size if needed

---

## 📞 Support

- **Issues**: GitHub Issues in adrilaynez/LM-Lab
- **Questions**: adrilaynezortiz@gmail.com
- **Portfolio**: https://adrianlaynez.dev

---

## 📅 Version History

- **v1.0** - Initial release with modular architecture
  - Bigram visualization complete
  - Query param routing working
  - CSS injection for iframe embedding
  - Documentation complete
  - Ready for portfolio integration

---

**Last Updated**: February 17, 2026  
**Status**: ✅ Production Ready
