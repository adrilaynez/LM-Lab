"""
Dataset Explorer
----------------
Provides dataset interpretability by searching for specific N-gram occurrences.
Used to answer "Why did the model predict X?".
"""

import sys
from functools import lru_cache
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.data import load_data
from api.config import DATA_DIR

# Cache dataset in memory
@lru_cache(maxsize=1)
def get_dataset():
    data_path = DATA_DIR / "paul_graham.txt"
    try:
        return load_data(data_path)
    except Exception:
        return load_data(str(data_path))

def search_dataset(context_tokens: list[str], next_token: str = None, limit: int = 10):
    """
    Search for occurrences of `context_tokens` (and optional `next_token`) in the training corpus.
    Returns snippets with surrounding context.
    """
    text = get_dataset()
    
    # Construct query string
    # Context tokens are characters.
    normalized_ctx = "".join(context_tokens)
    full_query = normalized_ctx
    if next_token:
        full_query += next_token
        
    query_len = len(full_query)
    
    if query_len == 0:
        return {"query": "", "count": 0, "examples": []}
    
    # Simple linear scan (fast enough for 1MB text)
    # For larger datasets, we would need a suffix array or index.
    
    indices = []
    start_idx = 0
    while True:
        try:
            idx = text.index(full_query, start_idx)
            indices.append(idx)
            start_idx = idx + 1
        except ValueError:
            break
            
    # Format results
    examples = []
    snippet_padding = 20
    
    for idx in indices[:limit]:
        # Extract snippet with padding
        start = max(0, idx - snippet_padding)
        end = min(len(text), idx + query_len + snippet_padding)
        
        pre = text[start:idx]
        match = text[idx : idx + query_len]
        post = text[idx + query_len : end]
        
        # Escape newlines for display
        snippet = f"{pre.replace('\n', ' ')}[[{match.replace('\n', ' ')}]]{post.replace('\n', ' ')}"
        examples.append(snippet)
        
    return {
        "query": f"Context: '{normalized_ctx}' -> Next: '{next_token}'",
        "count": len(indices),
        "examples": examples,
        "source": "Paul Graham Essays"
    }
