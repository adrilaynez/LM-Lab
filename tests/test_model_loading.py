
import sys
import unittest
from pathlib import Path
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock config to avoid loading env vars if not needed, but we need CHECKPOINT_DIR
from api.config import CHECKPOINT_DIR, DEVICE
from api.services.inference import _load_model, _load_ngram_model

class TestModelLoading(unittest.TestCase):
    def test_load_bigram(self):
        print("\nTesting Bigram loading from v1...")
        try:
            model, tokenizer, config, info = _load_model("bigram")
            self.assertIsNotNone(model)
            print("  Success!")
        except Exception as e:
            self.fail(f"Failed to load bigram: {e}")

    def test_load_mlp(self):
        print("\nTesting MLP loading from v1...")
        try:
            model, tokenizer, config, info = _load_model("mlp")
            self.assertIsNotNone(model)
            print("  Success!")
        except Exception as e:
            self.fail(f"Failed to load MLP: {e}")

    def test_load_ngram_via_inference(self):
        print("\nTesting N-Gram loading via inference service...")
        try:
            # Try N=2
            model, tokenizer = _load_ngram_model(2)
            self.assertIsNotNone(model)
            print("  Success!")
        except Exception as e:
            self.fail(f"Failed to load N-Gram (N=2): {e}")

if __name__ == '__main__':
    unittest.main()
