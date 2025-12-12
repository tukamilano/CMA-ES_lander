import numpy as np
from parameter import gene_num, input_size, layer_sizes


def _slice_gene(params: np.ndarray):
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    cursor = 0

    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        w_end = cursor + in_dim * out_dim
        w = params[cursor:w_end].reshape(in_dim, out_dim)
        cursor = w_end

        b_end = cursor + out_dim
        b = params[cursor:b_end]
        cursor = b_end

        weights.append(w)
        biases.append(b)

    if cursor != params.size:
        raise ValueError("gene length does not match layer configuration")

    return weights, biases


def action(inputs, gene):
    assert len(inputs) == input_size

    params = np.asarray(gene, dtype=float)
    if params.size != gene_num:
        raise ValueError("unexpected gene size")

    weights, biases = _slice_gene(params)

    activation = np.asarray(inputs, dtype=float)
    last_layer = len(weights) - 1

    for idx, (w, b) in enumerate(zip(weights, biases)):
        activation = activation @ w + b
        if idx != last_layer:
            activation = np.tanh(activation)

    angle_norm = np.tanh(activation[0])
    power_norm = 2 * np.tanh(activation[1]) - 1

    return np.array([angle_norm, power_norm])