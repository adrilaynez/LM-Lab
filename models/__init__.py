from .bigram import BigramModel

def get_model_class(model_type):
    """
    Factory function: Returns the model class based on config string.
    """
    if model_type == 'bigram':
        return BigramModel
    elif model_type == 'gpt':
        raise NotImplementedError("GPT not implemented yet")
    else:
        raise ValueError(f"Unknown model architecture: {model_type}")