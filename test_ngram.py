"""
Verification Script for N-Gram API
----------------------------------
Tests:
1. N=1, N=2, N=5 visualization (success)
2. N=6 visualization (failure - combinatorial explosion)
3. Dataset lookup
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from api.main import app

client = TestClient(app)

def test_ngram_visualize():
    print("\n🧪 Testing N-Gram Visualization...")
    
    # Test N=1 (Bigram)
    payload = {"text": "the", "context_size": 1, "top_k": 5}
    response = client.post("/api/v1/models/ngram/visualize", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ N=1 Success: {data['model_id']}")
        assert data["context_size"] == 1
        assert "active_slice" in data
        assert "matrix" in data["active_slice"]
    else:
        print(f"  ❌ N=1 Failed: {response.text}")

    # Test N=2
    payload = {"text": "the", "context_size": 2, "top_k": 5}
    response = client.post("/api/v1/models/ngram/visualize", json=payload)
    if response.status_code == 200:
        print(f"  ✅ N=2 Success")
    else:
        print(f"  ❌ N=2 Failed: {response.text}")

    # Test N=5
    payload = {"text": "hello world", "context_size": 5, "top_k": 5}
    response = client.post("/api/v1/models/ngram/visualize", json=payload)
    if response.status_code == 200:
        print(f"  ✅ N=5 Success")
    else:
        print(f"  ❌ N=5 Failed: {response.text}")

    # Test N=6 (Safeguard)
    payload = {"text": "hello world", "context_size": 6, "top_k": 5}
    response = client.post("/api/v1/models/ngram/visualize", json=payload)
    if response.status_code == 400:
        err = response.json()
        if "context_too_large" in str(err):
             print(f"  ✅ N=6 Safeguard Triggered (Expected 400)")
        else:
             print(f"  ⚠️ N=6 Returned 400 but unexpected content: {err}")
    else:
        print(f"  ❌ N=6 Failed: Expected 400, got {response.status_code}")

def test_dataset_lookup():
    print("\n🧪 Testing Dataset Lookup...")
    payload = {"context": ["t", "h"], "next_token": "e"}
    response = client.post("/api/v1/models/ngram/dataset_lookup", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ Lookup Success: Found {data['count']} examples")
        assert "examples" in data
        assert "query" in data
        assert "source" in data
    else:
        print(f"  ❌ Lookup Failed: {response.text}")

if __name__ == "__main__":
    try:
        test_ngram_visualize()
        test_dataset_lookup()
        print("\n✨ All Verification Tests Completed")
    except Exception as e:
        print(f"\n❌ Verification Error: {e}")
        import traceback
        traceback.print_exc()
