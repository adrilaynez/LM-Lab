import torch.nn as nn
from abc import ABC, abstractmethod

class LMEngine(nn.Module, ABC):
    """
    Clase base para todos los modelos de LM-Lab.
    Asegura que cualquier modelo que traigas de titan-engine sea compatible con la UI.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    @abstractmethod
    def get_internals(self, idx, targets=None):
        """
        La función mágica para el Glass Box.
        Debe devolver un diccionario con todo lo que quieras visualizar.
        """
        pass
