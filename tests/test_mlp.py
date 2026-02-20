
import unittest
import torch
import torch.nn.functional as F
from models.mlp import MLPModel

class TestMLP(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 27 # 26 letters + .
        self.context_size = 3
        self.emb_dim = 10
        self.hidden_size = 64
        self.model = MLPModel(self.vocab_size, self.context_size, self.emb_dim, self.hidden_size)


    def test_output_shape(self):
        batch_size = 8
        # Create dummy input: batch of indices [0..vocab_size-1]
        x = torch.randint(0, self.vocab_size, (batch_size, self.context_size))
        
        logits, _ = self.model(x)
        
        self.assertEqual(logits.shape, (batch_size, self.vocab_size))

    def test_initial_loss_is_uniform(self):
        """
        Test that initial loss is roughly -log(1/vocab_size).
        """
        batch_size = 32
        x = torch.randint(0, self.vocab_size, (batch_size, self.context_size))
        targets = torch.randint(0, self.vocab_size, (batch_size,))
        
        logits, _ = self.model(x)
        loss = F.cross_entropy(logits, targets)
        
        expected_loss = -torch.log(torch.tensor(1.0 / self.vocab_size))
        
        # Should be close (within 10%)
        # Because we scaled weights down, logits should be near zero, so uniform probs.
        print(f"Initial Loss: {loss.item():.4f}, Expected: {expected_loss.item():.4f}")
        self.assertTrue(torch.abs(loss - expected_loss) < 0.2, "Initial loss should be close to uniform entropy")

    def test_reproducibility(self):
        """
        Ensure manual_seed ensures identical weights.
        """
        model1 = MLPModel(self.vocab_size, seed=42)
        model2 = MLPModel(self.vocab_size, seed=42)
        
        # Check specific weights
        self.assertTrue(torch.equal(model1.W1, model2.W1))
        self.assertTrue(torch.equal(model1.C.weight, model2.C.weight))
        
        # Check forward pass
        x = torch.randint(0, self.vocab_size, (5, 3))
        logits1, _ = model1(x)
        logits2, _ = model2(x)
        self.assertTrue(torch.equal(logits1, logits2))

    def test_stochasticity(self):
        """
        Ensure different seeds produce different weights.
        """
        model1 = MLPModel(self.vocab_size, seed=42)
        model2 = MLPModel(self.vocab_size, seed=1337)
        
        self.assertFalse(torch.equal(model1.W1, model2.W1))

    def test_dead_neuron_calculation(self):
        x = torch.randint(0, self.vocab_size, (10, 3))
        self.model(x)
        rate = self.model.calculate_dead_neurons()
        self.assertTrue(0.0 <= rate <= 1.0)

if __name__ == '__main__':
    unittest.main()
