from .bigram import BigramModel
from .mlp import MLPModel
from .ngram import NGramModel
from .gpt import GPTModel

def get_model_class(model_type):
    """
    Factory function: Returns the model class based on config string.
    """
    if model_type == 'bigram':
        return BigramModel
    elif model_type == 'ngram':
        return NGramModel
    elif model_type == 'mlp':
        return MLPModel
    elif model_type == 'gpt':
        return GPTModel
    else:
        raise ValueError(f"Unknown model architecture: {model_type}")