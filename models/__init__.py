from .bigram import BigramModel
# Eventually: from .gpt import GPT
# Eventually: from .rnn import RNN

def get_model_class(model_type):
    """
    Factory function: Returns the class (not the instance) 
    based on the configuration string.
    """
    if model_type == 'bigram':
        return BigramModel
    elif model_type == 'gpt':
        # return GPT
        raise NotImplementedError("GPT not implemented yet")
    else:
        raise ValueError(f"Unknown architecture: {model_type}")