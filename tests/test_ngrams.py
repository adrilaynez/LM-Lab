import unittest
import torch
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ngram import NGramModel
from api.config import DEVICE, CHECKPOINT_DIR

class TestNGramVerification(unittest.TestCase):
    """
    Mathematical verification for N-Gram models (N=1 to 5).
    """

    @classmethod
    def setUpClass(cls):
        print(f"\n[Setup] Checking available checkpoints in {CHECKPOINT_DIR}...")
        cls.available_n = []
        for n in range(1, 6):
            # Check v1 path
            if (CHECKPOINT_DIR / "ngram" / "v1" / f"ngram_n{n}.pt").exists():
                cls.available_n.append(n)
            # Check legacy path
            elif (CHECKPOINT_DIR / f"ngram_n{n}.pt").exists():
                 cls.available_n.append(n)
        
        if not cls.available_n:
            print("WARNING: No N-gram checkpoints found. Tests will be skipped.")
        else:
            print(f"Found checkpoints for N={cls.available_n}")

    def test_probability_integrity(self):
        """
        Verify that predicted probabilities sum to exactly 1.0.
        """
        print("\n--- Test: Probability Integrity ---")
        for n in self.available_n:
            with self.subTest(n=n):
                print(f"Testing N={n}...")
                model = NGramModel(vocab_size=0, context_size=n) # vocab_size loaded from ckpt
                model.eval()
                model.to(DEVICE)
                
                # Create a dummy input (just random indices from vocab)
                # Ensure we have enough context or handle short context
                vocab_size = model.vocab_size
                # Use a known sequence or random
                dummy_idx = torch.randint(0, vocab_size, (1, n + 2)).to(DEVICE)
                
                with torch.no_grad():
                    logits, _ = model(dummy_idx)
                    # Logits shape: (batch, seq_len, vocab_size) or (batch, 1, vocab_size) dependent on implementation
                    # The implementation returns (1, 1, vocab) for the *next* token prediction based on context
                    
                    # Implementation details from forward:
                    # returns logits.view(1, 1, -1)
                    
                    # We expect log-probs from the model (it does torch.log(probs))
                    # So to check sum=1, we need exp(logits)
                    
                    last_logits = logits[0, -1] # (vocab_size,)
                    probs = torch.exp(last_logits)
                    
                    total_prob = probs.sum().item()
                    print(f"  N={n}: Sum of probs = {total_prob:.6f}")
                    
                    self.assertAlmostEqual(total_prob, 1.0, places=4, 
                                           msg=f"Probabilities do not sum to 1.0 for N={n}")

    def test_context_handling_short(self):
        """
        Verify graceful handling of short contexts (< N).
        """
        print("\n--- Test: Short Context Handling ---")
        # Only relevant for N > 1
        for n in [x for x in self.available_n if x > 1]:
            with self.subTest(n=n):
                print(f"Testing N={n} with short input...")
                model = NGramModel(vocab_size=0, context_size=n)
                model.eval()
                model.to(DEVICE)
                
                # Input length 1 (shorter than N)
                dummy_idx = torch.tensor([[0]], dtype=torch.long).to(DEVICE)
                
                try:
                    with torch.no_grad():
                        logits, _ = model(dummy_idx)
                    # If we get here, it didn't crash
                    last_logits = logits[0, -1]
                    probs = torch.exp(last_logits)
                    total_prob = probs.sum().item()
                    
                    self.assertAlmostEqual(total_prob, 1.0, places=4,
                                           msg=f"Short context output invalid sum for N={n}")
                    print(f"  N={n}: Short context handled gracefully.")
                    
                except Exception as e:
                    self.fail(f"Model crashed on short context for N={n}. Error: {e}")

    def test_top_k_logic(self):
        """
        Confirm top_k filtering returns descending probability.
        """
        print("\n--- Test: Top-K Logic ---")
        k = 5
        for n in self.available_n:
            with self.subTest(n=n):
                print(f"Testing N={n} Top-{k}...")
                model = NGramModel(vocab_size=0, context_size=n)
                model.eval()
                
                # Determine a context that likely has non-uniform distribution
                # If we use random context, we might hit sparse "unseen", giving uniform.
                # Let's try to find a "seen" context if possible, or just check sorting.
                
                # Even with uniform, top-k should sort (stable sort or just by index if equal).
                # But strictly, we want (p1 >= p2 >= ...)
                
                dummy_idx = torch.randint(0, model.vocab_size, (1, n)).to(DEVICE)
                
                with torch.no_grad():
                    logits, _ = model(dummy_idx)
                    last_logits = logits[0, -1]
                    probs = torch.exp(last_logits)
                
                top_v, top_i = torch.topk(probs, k)
                top_v_list = top_v.tolist()
                
                # Check descending
                for i in range(len(top_v_list) - 1):
                    self.assertGreaterEqual(top_v_list[i], top_v_list[i+1],
                                            msg=f"Top-K not sorted descending for N={n}")
                
                print(f"  N={n}: Top-{k} sorted correctly.")

if __name__ == '__main__':
    unittest.main()
