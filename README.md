# 🧪 LM Lab: Interactive Mechanics of Language Models

**LM Lab** is an educational observability suite designed to visualize the mathematical evolution of Large Language Models. 

Unlike standard "AI Wrappers," this project implements the entire history of Sequence Modeling—from simple Bigram statistics to the complex self-attention mechanisms of modern Transformers—**from scratch in PyTorch**.

The goal is not just to generate text, but to create a "Glass Box" where we can inspect:
* **Attention Maps:** How tokens "look" at each other.
* **Logit Lens:** How probability distributions evolve layer-by-layer.
* **Hidden States:** The geometry of the residual stream.

## 🚀 Roadmap & Architecture

| Level | Model | Concept Visualized | Status |
| :--- | :--- | :--- | :--- |
| **0** | **Bigram** | Statistical Co-occurrence | ✅ Done |
| **1** | **MLP** | Fixed Context Window (Bengio 2003) | 🚧 Planned |
| **2** | **RNN/GRU** | Recurrence & Vanishing Gradients | 🚧 Planned |
| **3** | **Transformer** | Self-Attention & Positional Encodings | 🚧 Planned |

## 🛠️ Tech Stack
* **Core:** Python 3.10+, PyTorch `nn.Module`
* **Visualization:** Streamlit, Plotly
* **Observability:** Custom hooks for internal state extraction.

## 📦 Usage
```bash
pip install -r requirements.txt
streamlit run app.py
