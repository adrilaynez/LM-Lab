
import requests
import json
import sys

BASE_URL = "http://localhost:8000/api/v1"

def test_visualize():
    print("Testing /models/bigram/visualize...")
    try:
        resp = requests.post(
            f"{BASE_URL}/models/bigram/visualize",
            json={"text": "test", "top_k": 5}
        )
        if resp.status_code != 200:
            print(f"FAILED: Status {resp.status_code}")
            print(resp.text)
            return False
            
        data = resp.json()
        vis = data.get("visualization", {})
        
        if "historical_context" not in vis:
            print("FAILED: 'historical_context' missing from response.")
            print("Keys found:", list(vis.keys()))
            return False
            
        print("SUCCESS: 'historical_context' present.")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_dataset_lookup():
    print("\nTesting /models/bigram/dataset_lookup...")
    try:
        resp = requests.post(
            f"{BASE_URL}/models/bigram/dataset_lookup",
            json={"context": ["t"], "next_token": "e"}
        )
        if resp.status_code != 200:
            print(f"FAILED: Status {resp.status_code}")
            print(resp.text)
            return False
            
        print("SUCCESS: Dataset lookup works.")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    v_ok = test_visualize()
    d_ok = test_dataset_lookup()
    
    if v_ok and d_ok:
        print("\nALL TESTS PASSED. Backend is updated.")
        sys.exit(0)
    else:
        print("\nTESTS FAILED. Backend may need restart.")
        sys.exit(1)
