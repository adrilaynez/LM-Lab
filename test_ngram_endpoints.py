"""
Verification Script for N-Gram Stepwise Prediction & Generation Endpoints
--------------------------------------------------------------------------
Tests:
1. ngram/predict_stepwise with N=1, N=2, N=5
2. ngram/generate with N=1, N=2
3. Edge cases: short text, unseen contexts
4. N=6 safeguard
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from api.main import app

client = TestClient(app)

def test_ngram_predict_stepwise():
    print("\n🧪 Testing N-Gram Stepwise Prediction...")
    
    # Test N=1
    payload = {"text": "the", "context_size": 1, "steps": 3, "top_k": 5}
    response = client.post("/api/v1/models/ngram/predict_stepwise", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ N=1 Stepwise: {len(data['steps'])} steps, prediction='{data['final_prediction']}'")
        assert data["context_size"] == 1
        assert len(data["steps"]) == 3
        assert all("context_window" in s for s in data["steps"])
        assert all("top_k" in s for s in data["steps"])
        assert data["model_id"] == "ngram_n1"
    else:
        print(f"  ❌ N=1 Stepwise Failed: {response.status_code} - {response.text}")
        return False

    # Test N=2
    payload = {"text": "hello", "context_size": 2, "steps": 5, "top_k": 5}
    response = client.post("/api/v1/models/ngram/predict_stepwise", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ N=2 Stepwise: {len(data['steps'])} steps, prediction='{data['final_prediction']}'")
        assert data["context_size"] == 2
        assert all(len(s["context_window"]) <= 2 for s in data["steps"])
    else:
        print(f"  ❌ N=2 Stepwise Failed: {response.status_code} - {response.text}")
        return False

    # Test N=5
    payload = {"text": "hello world", "context_size": 5, "steps": 3, "top_k": 5}
    response = client.post("/api/v1/models/ngram/predict_stepwise", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ N=5 Stepwise: {len(data['steps'])} steps, prediction='{data['final_prediction']}'")
    else:
        print(f"  ❌ N=5 Stepwise Failed: {response.status_code} - {response.text}")
        return False

    # Test short text (text shorter than context_size)
    payload = {"text": "a", "context_size": 3, "steps": 2, "top_k": 5}
    response = client.post("/api/v1/models/ngram/predict_stepwise", json=payload)
    if response.status_code == 200:
        print(f"  ✅ Short text edge case handled")
    else:
        print(f"  ❌ Short text Failed: {response.status_code} - {response.text}")
        return False

    return True

def test_ngram_generate():
    print("\n🧪 Testing N-Gram Generation...")
    
    # Test N=1
    payload = {"start_text": "T", "context_size": 1, "num_tokens": 50, "temperature": 1.0}
    response = client.post("/api/v1/models/ngram/generate", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ N=1 Generate: {data['length']} chars, starts with '{data['generated_text'][:20]}...'")
        assert data["context_size"] == 1
        assert data["model_id"] == "ngram_n1"
        assert data["temperature"] == 1.0
        assert data["start_text"] == "T"
    else:
        print(f"  ❌ N=1 Generate Failed: {response.status_code} - {response.text}")
        return False

    # Test N=2
    payload = {"start_text": "The ", "context_size": 2, "num_tokens": 50, "temperature": 0.8}
    response = client.post("/api/v1/models/ngram/generate", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ N=2 Generate: {data['length']} chars, temp={data['temperature']}")
    else:
        print(f"  ❌ N=2 Generate Failed: {response.status_code} - {response.text}")
        return False

    # Test N=5 with longer seed
    payload = {"start_text": "hello", "context_size": 5, "num_tokens": 30, "temperature": 1.0}
    response = client.post("/api/v1/models/ngram/generate", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ N=5 Generate: {data['length']} chars")
    else:
        print(f"  ❌ N=5 Generate Failed: {response.status_code} - {response.text}")
        return False

    return True

def test_safeguards():
    print("\n🧪 Testing Safeguards...")
    
    # N=6 should fail for stepwise
    payload = {"text": "hello", "context_size": 6, "steps": 3}
    response = client.post("/api/v1/models/ngram/predict_stepwise", json=payload)
    if response.status_code == 400 or response.status_code == 422:
        print(f"  ✅ N=6 Stepwise Safeguard (Expected 400/422, got {response.status_code})")
    else:
        print(f"  ❌ N=6 Stepwise: Expected 400/422, got {response.status_code}")
        return False

    # N=6 should fail for generate
    payload = {"start_text": "hello", "context_size": 6, "num_tokens": 10}
    response = client.post("/api/v1/models/ngram/generate", json=payload)
    if response.status_code == 400 or response.status_code == 422:
        print(f"  ✅ N=6 Generate Safeguard (Expected 400/422, got {response.status_code})")
    else:
        print(f"  ❌ N=6 Generate: Expected 400/422, got {response.status_code}")
        return False

    return True

def test_transition_matrix_diagnostic():
    """Verify backend populates active_slice correctly for N>1."""
    print("\n🧪 Transition Matrix Diagnostic...")
    
    payload = {"text": "hello", "context_size": 2, "top_k": 5}
    response = client.post("/api/v1/models/ngram/visualize", json=payload)
    if response.status_code == 200:
        data = response.json()
        vis = data.get("visualization", {})
        active_slice = vis.get("active_slice")
        
        if active_slice is None:
            print(f"  ❌ active_slice is None!")
            return False
            
        matrix = active_slice.get("matrix")
        if matrix is None:
            print(f"  ❌ active_slice.matrix is None!")
            return False
            
        print(f"  ✅ active_slice.matrix present")
        print(f"     shape: {matrix['shape']}")
        print(f"     row_labels: {matrix['row_labels']}")
        print(f"     col_labels count: {len(matrix['col_labels'])}")
        print(f"     data rows: {len(matrix['data'])}, cols per row: {len(matrix['data'][0])}")
        
        # The bug: row_labels=["Ctx"] (1 item), col_labels has V items
        # TransitionMatrix.tsx uses n=row_labels.length for BOTH dimensions
        if len(matrix['row_labels']) == 1 and len(matrix['col_labels']) > 1:
            print(f"  ⚠️  CONFIRMED: Matrix is {matrix['shape'][0]}×{matrix['shape'][1]} (rectangular)")
            print(f"     TransitionMatrix.tsx will render 1×1 grid (Frontend bug)")
            print(f"     Backend data is CORRECT. Issue is in frontend rendering.")
        
        # Check next_token_probs
        ntp = active_slice.get("next_token_probs", {})
        nonzero = sum(1 for v in ntp.values() if v > 0.001)
        print(f"  ✅ next_token_probs: {nonzero} non-trivial entries out of {len(ntp)}")
        
        # Check context tokens
        ctx = active_slice.get("context_tokens", [])
        print(f"  ✅ context_tokens: {ctx}")
        
        return True
    else:
        print(f"  ❌ Visualize Failed: {response.status_code}")
        return False


if __name__ == "__main__":
    results = []
    try:
        results.append(("Stepwise Prediction", test_ngram_predict_stepwise()))
        results.append(("Text Generation", test_ngram_generate()))
        results.append(("Safeguards", test_safeguards()))
        results.append(("Transition Matrix Diagnostic", test_transition_matrix_diagnostic()))
        
        print("\n" + "=" * 60)
        all_pass = all(r[1] for r in results)
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {name}")
        
        if all_pass:
            print("\n✨ All Verification Tests Passed!")
        else:
            print("\n❌ Some tests failed.")
    except Exception as e:
        print(f"\n❌ Verification Error: {e}")
        import traceback
        traceback.print_exc()
