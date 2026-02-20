"""
Model Registry and Information
Stores metadata and detailed information about available language models
"""

MODEL_INFO = {
    'bigram': {
        'name': 'Bigram Model',
        'description': 'Character-level bigram language model that learns transition probabilities between consecutive characters.',
        'type': 'Statistical',
        'complexity': 'Low',
        'parameters': 'vocab_size^2',
        'training_time': 'Fast (seconds)',
        'memory_efficient': True,
        'how_it_works': [
            '1. Creates a matrix W of shape (vocab_size, vocab_size)',
            '2. Each row represents the current character',
            '3. Each column represents the probability of the next character',
            '4. Given input character idx, predicts next character via softmax(W[idx])',
            '5. Learns by minimizing cross-entropy loss during training'
        ],
        'strengths': [
            'Simple and interpretable',
            'Fast to train and generate',
            'Low memory footprint',
            'Good for understanding character transitions'
        ],
        'limitations': [
            'Only considers one previous character',
            'Cannot learn long-range dependencies',
            'Limited context for complex patterns',
            'No hidden state or memory'
        ],
        'use_cases': [
            'Character-level text generation',
            'Baseline model for comparison',
            'Educational purposes',
            'Quick prototyping'
        ],
        'visualization': 'Weight matrix shows learned character transition probabilities'
    },
    'mlp': {
        'name': 'Multi-Layer Perceptron',
        'description': 'Feed-forward neural network with multiple hidden layers for improved expressiveness.',
        'type': 'Neural Network (Feed-forward)',
        'complexity': 'Medium',
        'parameters': 'Depends on architecture',
        'training_time': 'Moderate (minutes)',
        'memory_efficient': False,
        'how_it_works': [
            '1. Embeds input characters into dense vectors',
            '2. Concatenates embeddings from context window',
            '3. Passes through multiple fully-connected layers',
            '4. Each layer applies non-linearity (ReLU)',
            '5. Output layer predicts next character probabilities'
        ],
        'strengths': [
            'More expressive than bigram',
            'Can learn complex patterns',
            'Reasonable training time',
            'Interpretable embeddings'
        ],
        'limitations': [
            'Fixed context window size',
            'Requires more parameters than bigram',
            'Longer training time',
            'Still cannot capture long-range dependencies'
        ],
        'use_cases': [
            'Medium-length dependency modeling',
            'Character embedding visualization',
            'Comparison with transformers'
        ],
        'visualization': 'Character embeddings and weight distribution in hidden layers'
    },
    'rnn': {
        'name': 'Recurrent Neural Network',
        'description': 'Sequential model with recurrent connections enabling processing of variable-length sequences.',
        'type': 'Neural Network (Recurrent)',
        'complexity': 'High',
        'parameters': 'Varies with hidden size',
        'training_time': 'Slow (hours)',
        'memory_efficient': False,
        'how_it_works': [
            '1. Embeds input characters',
            '2. Maintains hidden state through recurrence',
            '3. At each step: h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)',
            '4. Predicts next character from current hidden state',
            '5. Hidden state propagates information across sequences'
        ],
        'strengths': [
            'Can model variable-length sequences',
            'Captures long-range dependencies better than MLP',
            'Processes sequences sequentially',
            'Suitable for language modeling'
        ],
        'limitations': [
            'Prone to vanishing gradient problem',
            'Slower training and generation',
            'More hyperparameters to tune',
            'Difficult to parallelize'
        ],
        'use_cases': [
            'Language modeling',
            'Machine translation',
            'Speech recognition'
        ],
        'visualization': 'Hidden state evolution over sequence'
    },
    'gpt': {
        'name': 'Generative Pre-trained Transformer',
        'description': 'Transformer-based autoregressive model using self-attention for parallel processing.',
        'type': 'Transformer',
        'complexity': 'Very High',
        'parameters': 'Millions',
        'training_time': 'Very Slow (days)',
        'memory_efficient': False,
        'how_it_works': [
            '1. Embeds input characters and adds positional encodings',
            '2. Multi-head self-attention: computes relevance of all positions',
            '3. Feed-forward networks process attended features',
            '4. Multiple transformer layers stack',
            '5. Output layer predicts next character distribution'
        ],
        'strengths': [
            'Captures long-range dependencies',
            'Parallel processing enables fast training',
            'State-of-the-art performance',
            'Highly scalable architecture'
        ],
        'limitations': [
            'High computational cost',
            'Requires large amounts of data',
            'Many hyperparameters',
            'Difficult to interpret'
        ],
        'use_cases': [
            'High-quality text generation',
            'Pre-training for downstream tasks',
            'Large-scale language models'
        ],
        'visualization': 'Attention patterns between tokens'
    },
    'ngram': {
        'name': 'N-Gram Model',
        'description': 'Generalization of Bigram to N-token context windows (N=1..4).',
        'type': 'Statistical',
        'complexity': 'Low-Medium',
        'parameters': 'vocab_size^(N+1)',
        'training_time': 'Fast (seconds)',
        'memory_efficient': 'Decreases with N',
        'how_it_works': [
            '1. Counts occurrences of N-token sequences',
            '2. Stores probabilities in (N+1)-dimensional tensor',
            '3. Predicts next token based on previous N tokens',
            '4. "Active Slice" visualizes the relevant transition matrix for current context'
        ],
        'strengths': [
            'Simple and interpretable',
            'Captures local dependencies better than Bigram',
            'Exact probabilities for seen contexts'
        ],
        'limitations': [
            'Exponential parameter growth with N',
            'Sparse data problem for large N',
            'No semantic understanding'
        ],
        'use_cases': [
            'Baselines',
            'Spell checking',
            'Simple text generation'
        ],
        'visualization': 'Active Slice of Transition Tensor'
    }
}


def get_model_info(model_type):
    """Get detailed information about a specific model."""
    return MODEL_INFO.get(model_type, {})


def get_all_models():
    """Get all available models."""
    return list(MODEL_INFO.keys())


def format_model_info(model_type):
    """Format model information for display."""
    info = get_model_info(model_type)
    if not info:
        return None
    
    formatted = {
        'name': info.get('name', 'Unknown'),
        'description': info.get('description', ''),
        'type': info.get('type', ''),
        'complexity': info.get('complexity', ''),
        'how_it_works': info.get('how_it_works', []),
        'strengths': info.get('strengths', []),
        'limitations': info.get('limitations', []),
        'use_cases': info.get('use_cases', [])
    }
    
    return formatted

def get_model_url_param(model_type):
    """Get the URL parameter for a specific model."""
    return f"?model={model_type}"
