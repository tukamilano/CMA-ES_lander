import numpy as np
from parameter import *


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))



def action(inputs, gene):
    assert len(inputs) == input_size

    params = np.asarray(gene, dtype=float)
    cursor = 0

    w1 = params[cursor:cursor + input_size * hidden_size].reshape(input_size, hidden_size)
    cursor += input_size * hidden_size
    b1 = params[cursor:cursor + hidden_size]
    cursor += hidden_size
    w2 = params[cursor:cursor + hidden_size * output_size].reshape(hidden_size, output_size)
    cursor += hidden_size * output_size
    b2 = params[cursor:cursor + output_size]

    x = np.asarray(inputs, dtype=float)
    hidden = x @ w1 + b1
    hidden = np.tanh(hidden)
    output = hidden @ w2 + b2

    angle_norm = np.tanh(output[0])
    power_norm = 2 * np.tanh(output[1]) - 1

    return np.array([angle_norm, power_norm])