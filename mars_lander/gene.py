import numpy as np
from parameter import *

def action(input, gene):
    assert len(input) == input_size

    params = np.asarray(gene, dtype=float)
    cursor = 0

    # Decode flattened parameters into a two-layer perceptron
    w1 = params[cursor:cursor + input_size * hidden_size].reshape(input_size, hidden_size)
    cursor += input_size * hidden_size
    b1 = params[cursor:cursor + hidden_size]
    cursor += hidden_size
    w2 = params[cursor:cursor + hidden_size * output_size].reshape(hidden_size, output_size)
    cursor += hidden_size * output_size
    b2 = params[cursor:cursor + output_size]

    x = np.asarray(input, dtype=float)
    hidden = x @ w1 + b1
    output = hidden @ w2 + b2

    if output_size < 6:
        raise ValueError("output_size must be at least 6 to split into two softmax groups")

    def softmax_choice(logits):
        # Sample discrete command from logits using softmax probabilities
        shifted = logits - np.max(logits)
        exp_scores = np.exp(shifted)
        probs = exp_scores / exp_scores.sum()
        mapping = np.array([1, 0, -1])
        return mapping[np.random.choice(len(mapping), p=probs)]

    first = softmax_choice(output[:3])
    second = softmax_choice(output[3:6])
    return np.array([first, second])


x = generate_init_gene(10)
print(x)