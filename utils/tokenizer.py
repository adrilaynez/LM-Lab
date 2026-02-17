# utils/tokenizer.py
import torch

class CharTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.chars = []
        self.vocab_size = 0

    def train(self, text):
        """Build vocabulary from text"""
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}
        
    def encode(self, s):
        """String -> List of integers"""
        return [self.stoi[c] for c in s]

    def decode(self, l):
        """List of integers -> String"""
        if isinstance(l, torch.Tensor):
            l = l.tolist()
        return ''.join([self.itos[i] for i in l])

    def save(self, path):
        """Save vocabulary to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'stoi': self.stoi, 
                'itos': self.itos, 
                'chars': self.chars
            }, f)
    
    def load(self, path):
        """Load vocabulary from disk"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.stoi = data['stoi']
            self.itos = data['itos']
            self.chars = data['chars']
            self.vocab_size = len(self.chars)

