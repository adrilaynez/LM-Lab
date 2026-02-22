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

CORPUS_NAME = "Paul Graham Essays"
SNIPPET_PADDING = 25  # characters of context shown around each match


# Cache dataset in memory — loaded once per process
@lru_cache(maxsize=1)
def get_dataset() -> str:
    data_path = DATA_DIR / "paul_graham.txt"
    try:
        return load_data(data_path)
    except Exception:
        return load_data(str(data_path))


def _collect_all_indices(text: str, query: str) -> list[int]:
    """
    Linear scan to collect every position where `query` appears in `text`.
    O(n) — fast enough for a ~1 MB corpus.
    """
    indices: list[int] = []
    start = 0
    query_len = len(query)
    while True:
        pos = text.find(query, start)
        if pos == -1:
            break
        indices.append(pos)
        start = pos + 1  # allow overlapping matches
    return indices


def _sample_evenly(indices: list[int], limit: int) -> list[int]:
    """
    Pick `limit` positions spread evenly across the full list so that
    returned snippets represent the whole corpus, not just the beginning.
    """
    if len(indices) <= limit:
        return indices
    step = len(indices) / limit
    return [indices[int(i * step)] for i in range(limit)]


def _make_snippet(text: str, pos: int, query_len: int) -> str:
    """Build a display snippet with [[match]] markers and cleaned whitespace."""
    start = max(0, pos - SNIPPET_PADDING)
    end = min(len(text), pos + query_len + SNIPPET_PADDING)

    pre   = text[start:pos].replace('\n', ' ')
    match = text[pos:pos + query_len].replace('\n', ' ')
    post  = text[pos + query_len:end].replace('\n', ' ')
    return f"{pre}[[{match}]]{post}"


def search_dataset(
    context_tokens: list[str],
    next_token: str | None = None,
    limit: int = 10,
) -> dict:
    """
    Search for occurrences of `context_tokens` (and optional `next_token`) in
    the training corpus.

    Strategy (in order):
      1. Search for the full ``context + next_token`` string.
      2. If that yields zero results, fall back to searching for ``context``
         alone (the transition exists in training, just not in the sampled
         snippet window).  The fallback snippets are clearly labelled.
      3. Results are sampled evenly across the corpus so that the frontend
         always shows diverse examples rather than the same opening paragraph.

    Returns a dict matching DatasetLookupResponse schema.
    """
    text = get_dataset()

    normalized_ctx = "".join(context_tokens)
    full_query = normalized_ctx + (next_token or "")
    query_len = len(full_query)

    if query_len == 0:
        return {
            "query": "",
            "count": 0,
            "examples": [],
            "source": CORPUS_NAME,
        }

    # ── 1. Primary search: full context + next_token ──────────────────────────
    all_indices = _collect_all_indices(text, full_query)
    used_query = full_query
    is_fallback = False

    # ── 2. Fallback: context only (when full query has no matches) ────────────
    if not all_indices and normalized_ctx:
        ctx_indices = _collect_all_indices(text, normalized_ctx)
        if ctx_indices:
            all_indices = ctx_indices
            used_query = normalized_ctx
            is_fallback = True

    total_count = len(all_indices)

    # ── 3. Sample evenly across corpus, then build snippets ───────────────────
    sampled = _sample_evenly(all_indices, limit)
    examples: list[str] = []
    for pos in sampled:
        snippet = _make_snippet(text, pos, len(used_query))
        if is_fallback:
            # Prefix so the frontend can distinguish fallback results
            snippet = f"[context only] {snippet}"
        examples.append(snippet)

    query_label = (
        f"Context: '{normalized_ctx}' -> Next: '{next_token}'"
        if next_token
        else f"Context: '{normalized_ctx}'"
    )

    return {
        "query": query_label,
        "count": total_count,
        "examples": examples,
        "source": CORPUS_NAME,
    }
