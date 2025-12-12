import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

from cma import CMA
from moe import load_expert_genes, mixture_action
from parameter import max_generation, moe_gene_num
from main import CASE_DATA_MAP, _create_board


_CASE_NAMES: tuple[str, ...] = tuple(f"case{i}" for i in range(1, 6))
_EXPERT_GENES: list[np.ndarray] | None = None
_SELECTION_MODE: str = "soft"


def _init_worker(expert_dir: str, selection_mode: str):
    global _EXPERT_GENES
    global _SELECTION_MODE
    expert_root = Path(expert_dir)
    _EXPERT_GENES = load_expert_genes(expert_root)
    _SELECTION_MODE = selection_mode


def _simulation(gate_gene: Iterable[float], board):
    max_steps = 1500
    for _ in range(max_steps):
        terminated, score = board.is_terminate()
        if terminated:
            return score

        features = board.sensor()
        angle_norm, power_norm = mixture_action(
            features,
            gate_gene,
            _EXPERT_GENES,
            mode=_SELECTION_MODE,
        )

        target_angle = max(-90.0, min(90.0, round(max(-1.0, min(1.0, angle_norm)) * 90.0 / 15.0) * 15.0))
        target_power = int(round(max(0.0, min(1.0, power_norm)) * 4.0))

        board.update(target_power, target_angle)

    return -1000.0


def evaluate(gate_gene: np.ndarray):
    assert _EXPERT_GENES is not None, "Expert genes not initialised"

    total_score = 0.0
    success_count = 0.0

    for case_name in _CASE_NAMES:
        board = _create_board(CASE_DATA_MAP[case_name])
        score = _simulation(gate_gene, board)
        if score >= 200.0:
            success_count += 1.0
        total_score += score

    success_rate = success_count / len(_CASE_NAMES)
    return -total_score, success_rate


def run_optimization(max_workers: int | None = None, selection_mode: str = "soft"):
    optimizer = CMA(mean=np.zeros(moe_gene_num), sigma=0.5)
    best_gate_path = Path(__file__).resolve().parent / "best_gate.json"

    best_overall_score = -np.inf
    best_overall_success_rate = 0.0

    expert_dir = Path(__file__).resolve().parent
    _init_worker(str(expert_dir), selection_mode)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(str(expert_dir), selection_mode),
    ) as executor:
        for generation in range(max_generation):
            genes = [optimizer.ask() for _ in range(optimizer.population_size)]
            results = list(executor.map(evaluate, genes))

            losses = np.fromiter((loss for loss, _ in results), dtype=float, count=len(results))
            success_rates = np.fromiter((sr for _, sr in results), dtype=float, count=len(results))

            solutions = [(gene, float(loss)) for gene, loss in zip(genes, losses, strict=True)]
            optimizer.tell(solutions)

            scores = -losses
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            mean_score = float(scores.mean())
            best_success_rate = float(success_rates.max())
            current_best_gene = genes[best_idx]
            current_best_success_rate = float(success_rates[best_idx])

            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_success_rate = current_best_success_rate
                best_gate_path.write_text(
                    json.dumps(current_best_gene.tolist()),
                    encoding="ascii",
                )
                print(
                    f"    new best saved: score={best_overall_score:.2f}, "
                    f"success_rate={best_overall_success_rate:.2f} -> {best_gate_path.name}"
                )

            print(
                f"generation {generation:03d}: best_score={best_score:.2f}, "
                f"mean_score={mean_score:.2f}, best_success_rate={best_success_rate:.2f}"
            )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train MoE gate across cases 1-5")
    parser.add_argument(
        "--max-workers",
        type=int,
        help="number of worker processes (default: use CPU count)",
    )
    parser.add_argument(
        "--mode",
        choices=("soft", "hard"),
        default="soft",
        help="gating mode: soft (weighted sum) or hard (pick best expert)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_optimization(max_workers=args.max_workers, selection_mode=args.mode)


if __name__ == "__main__":
    main()
