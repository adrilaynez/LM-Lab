import torch.nn as nn
from abc import ABC, abstractmethod

class LMEngine(nn.Module, ABC):
    """
    Base class for all LM-Lab models.
    Ensures compatibility with the UI.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    @abstractmethod
    def get_internals(self, idx, targets=None):
        """
        Returns visualization data for the model.
        Must return a dictionary with data to visualize.
        """
        pass
