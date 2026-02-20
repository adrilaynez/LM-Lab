import torch
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp import MLPModel

def test_mlp_shapes():
    print("Testing MLP shapes...")
    vocab_size = 65
    context_size = 3
    emb_dim = 10
    hidden_size = 200
    model = MLPModel(vocab_size, context_size, emb_dim, hidden_size)
    
    batch_size = 32
    x = torch.randint(0, vocab_size, (batch_size, context_size))
    logits, _ = model(x)
    
    assert logits.shape == (batch_size, vocab_size), f"Expected {(batch_size, vocab_size)}, got {logits.shape}"
    print("✅ Shapes passed.")

def test_mlp_initial_loss():
    print("Testing initial loss baseline...")
    vocab_size = 65
    context_size = 3
    model = MLPModel(vocab_size, context_size)
    
    batch_size = 1000 # Larger batch for better estimation
    x = torch.randint(0, vocab_size, (batch_size, context_size))
    y = torch.randint(0, vocab_size, (batch_size,))
    
    logits, _ = model(x)
    loss = F.cross_entropy(logits, y).item()
    
    expected_loss = -torch.log(torch.tensor(1.0 / vocab_size)).item()
    print(f"Initial loss: {loss:.4f}, Expected: {expected_loss:.4f}")
    
    # We expect it to be close if initialization is scaled properly
    assert abs(loss - expected_loss) < 0.7, f"Initial loss {loss} too far from {expected_loss}"
    print("✅ Initial loss passed.")

def test_reproducibility():
    print("Testing reproducibility...")
    vocab_size = 65
    model1 = MLPModel(vocab_size, seed=1337)
    model2 = MLPModel(vocab_size, seed=1337)
    
    x = torch.randint(0, vocab_size, (1, 3))
    l1, _ = model1(x)
    l2, _ = model2(x)
    
    assert torch.allclose(l1, l2), "Models with same seed produced different results"
    
    # Check weights
    assert torch.allclose(model1.W1, model2.W1), "W1 mismatch"
    assert torch.allclose(model1.W2, model2.W2), "W2 mismatch"
    print("✅ Reproducibility passed.")

def test_dead_neuron_calculation():
    print("Testing dead neuron calculation...")
    vocab_size = 65
    hidden_size = 100
    model = MLPModel(vocab_size, hidden_size=hidden_size)
    
    # Force saturated activations
    with torch.no_grad():
        model.W1.fill_(100.0)
        model.b1.fill_(100.0)
        
    x = torch.randint(0, vocab_size, (32, 3))
    model(x)
    dead_rate = model.calculate_dead_neurons(threshold=0.99)
    print(f"Dead rate (forced high): {dead_rate}")
    assert dead_rate > 0.9, "Dead rate should be high for saturated neurons"
    
    # Force zero activations (tanh(0) = 0)
    with torch.no_grad():
        model.W1.zero_()
        model.b1.zero_()
        
    model(x)
    dead_rate = model.calculate_dead_neurons(threshold=0.99)
    print(f"Dead rate (forced zero): {dead_rate}")
    assert dead_rate == 0.0, "Dead rate should be zero for non-saturated neurons"
    print("✅ Dead neuron calculation passed.")

if __name__ == "__main__":
    try:
        test_mlp_shapes()
        test_mlp_initial_loss()
        test_reproducibility()
        test_dead_neuron_calculation()
        print("\n🎉 All MLP unit tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n💥 An error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
