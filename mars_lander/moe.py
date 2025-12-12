import json
from pathlib import Path
from typing import Iterable

import numpy as np

from gene import action as expert_action
from parameter import (
    input_size,
    moe_gene_num,
    moe_hidden_size,
    moe_output_size,
)


_EXPERT_FILENAMES = [f"best_gene{i}.json" for i in range(1, moe_output_size + 1)]


def _load_gene(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="ascii"))
    array = np.asarray(data, dtype=float)
    return array


def load_gate_gene(path: Path | None = None) -> np.ndarray:
    """Load a trained gating vector from JSON."""
    if path is None:
        path = Path(__file__).resolve().parent / "best_gate.json"
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"gate gene file not found: {path}")
    return _load_gene(path)


def load_expert_genes(root: Path | None = None) -> list[np.ndarray]:
    """Load the pre-trained expert policies from JSON files.

    Args:
        root: Directory containing the expert JSON files. Defaults to the project
            root (same directory as this module).

    Returns:
        A list of gene vectors for each expert, ordered by case index.

    Raises:
        FileNotFoundError: When any expected expert file is missing.
    """
    if root is None:
        root = Path(__file__).resolve().parent

    genes: list[np.ndarray] = []
    for filename in _EXPERT_FILENAMES:
        path = (root / filename).resolve()
        if not path.exists():
            raise FileNotFoundError(f"expert gene file not found: {path}")
        genes.append(_load_gene(path))
    return genes


def _unpack_gate_params(params: np.ndarray):
    if params.size != moe_gene_num:
        raise ValueError(f"expected gate gene size {moe_gene_num}, got {params.size}")

    cursor = 0
    if moe_hidden_size > 0:
        w1 = params[cursor : cursor + input_size * moe_hidden_size].reshape(input_size, moe_hidden_size)
        cursor += input_size * moe_hidden_size
        b1 = params[cursor : cursor + moe_hidden_size]
        cursor += moe_hidden_size
        w2 = params[cursor : cursor + moe_hidden_size * moe_output_size].reshape(moe_hidden_size, moe_output_size)
        cursor += moe_hidden_size * moe_output_size
        b2 = params[cursor : cursor + moe_output_size]
        return w1, b1, w2, b2

    w = params[cursor : cursor + input_size * moe_output_size].reshape(input_size, moe_output_size)
    cursor += input_size * moe_output_size
    b = params[cursor : cursor + moe_output_size]
    return w, b


def gating_weights(features: Iterable[float], gate_gene: np.ndarray) -> np.ndarray:
    """Compute softmax mixture weights for the experts."""
    params = np.asarray(gate_gene, dtype=float)
    unpacked = _unpack_gate_params(params)

    x = np.asarray(features, dtype=float)
    if x.shape[0] != input_size:
        raise ValueError(f"expected {input_size} input features, got {x.shape[0]}")

    if moe_hidden_size > 0:
        w1, b1, w2, b2 = unpacked
        hidden = np.tanh(x @ w1 + b1)
        logits = hidden @ w2 + b2
    else:
        w, b = unpacked
        logits = x @ w + b

    logits = logits - logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    weights = exp_logits / exp_logits.sum()
    return weights


def mixture_action(
    features: Iterable[float],
    gate_gene: np.ndarray,
    expert_genes: list[np.ndarray] | None = None,
    mode: str = "soft",
) -> np.ndarray:
    """Route through experts using the gating network.

    Args:
        features: Normalised sensor inputs.
        gate_gene: Parameter vector for the gating network.
        expert_genes: Optional override of expert weights.
        mode: ``"soft"`` (default) performs a weighted sum of expert actions,
            while ``"hard"`` selects the highest-weight expert.
    """
    if expert_genes is None:
        expert_genes = load_expert_genes(Path(__file__).resolve().parent)

    if len(expert_genes) != moe_output_size:
        raise ValueError(f"expected {moe_output_size} expert genes, got {len(expert_genes)}")

    weights = gating_weights(features, gate_gene)
    if mode == "hard":
        best_idx = int(np.argmax(weights))
        return expert_action(features, expert_genes[best_idx])
    if mode != "soft":
        raise ValueError("mode must be 'soft' or 'hard'")

    expert_outputs = np.vstack([expert_action(features, gene) for gene in expert_genes])
    return weights @ expert_outputs
