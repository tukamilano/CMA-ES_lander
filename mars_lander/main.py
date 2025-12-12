import argparse
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from parameter import *
from cma import CMA

from simulate.simulate import Board, _build_ground  # type: ignore
from gene import action


CASE_DIR = Path(__file__).resolve().parent / "simulate"


def _load_case(path: Path):
    with path.open("r", encoding="ascii") as f:
        lines = [line.strip() for line in f if line.strip()]

    iterator = iter(lines)
    surfaceN = int(next(iterator))
    points = [tuple(map(int, next(iterator).split())) for _ in range(surfaceN)]
    ground_height_list, flat_ground_pair = _build_ground(points)
    x, y, h_speed, v_speed, fuel, rotate, power = map(int, next(iterator).split())
    return ground_height_list, flat_ground_pair, (x, y, h_speed, v_speed, fuel, rotate, power)

CASE_DATA_MAP = {
    f"case{i}": _load_case(CASE_DIR / f"case{i}.txt")
    for i in range(1, 6)
}

_ACTIVE_CASE_NAME: str = "case1"


def _init_worker(case_name: str):
    global _ACTIVE_CASE_NAME
    _ACTIVE_CASE_NAME = case_name

def _create_board(case_data):
    ground_height_list, flat_ground_pair, initial = case_data
    x, y, h_speed, v_speed, fuel, rotate, power = initial
    return Board(
        ground_height_list=list(ground_height_list),
        lander_pos=[x, y],
        lander_speed=[h_speed, v_speed],
        flat_ground_pair=flat_ground_pair,
        init_rotate=rotate,
        init_power=power,
        init_fuel=fuel,
    )


def simulation(gene, board: Board):
    max_steps = 1500
    for _ in range(max_steps):
        terminated, score = board.is_terminate()
        if terminated:
            return score

        features = board.sensor()
        angle_norm, power_norm = action(features, gene)

        target_angle = max(-90.0, min(90.0, round(max(-1.0, min(1.0, angle_norm)) * 90.0 / 15.0) * 15.0))
        target_power = int(round(max(0.0, min(1.0, power_norm)) * 4.0))

        board.update(target_power, target_angle)

    return -1000.0


def evaluate(gene):
    total_score = 0.0
    case_data = CASE_DATA_MAP[_ACTIVE_CASE_NAME]
    board = _create_board(case_data)
    score = simulation(gene, board)
    success = float(score >= 200.0)
    total_score += score

    return -total_score, success


def _best_gene_filename(case_name: str) -> str:
    if not case_name:
        return "best_gene.npy"
    if case_name.startswith("case"):
        suffix = case_name[len("case") :]
        suffix = suffix or case_name
        return f"best_gene{suffix}.npy"
    return f"best_gene_{case_name}.npy"


def run_optimization(case_name: str, max_workers: int | None = None):
    optimizer = CMA(mean=np.zeros(gene_num), sigma=0.5)
    base_path = Path(__file__).resolve().parent / _best_gene_filename(case_name)
    best_gene_path = base_path.with_suffix(".json")
    best_overall_score = -np.inf
    best_overall_success_rate = 0.0

    _init_worker(case_name)

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(case_name,)) as executor:
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
                best_overall_score = float(best_score)
                best_overall_success_rate = float(current_best_success_rate)
                best_gene_path.write_text(
                    json.dumps(current_best_gene.tolist()),
                    encoding="ascii",
                )
                print(
                    f"    new best saved: score={best_overall_score:.2f}, success_rate={best_overall_success_rate:.2f} -> {best_gene_path.name}"
                )
            print(
                f"generation {generation:03d}: best_score={best_score:.2f}, mean_score={mean_score:.2f}, "
                f"best_success_rate={best_success_rate:.2f}"
            )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train CMA-ES controller on a specific Mars Lander case")
    parser.set_defaults(case="case1")

    case_group = parser.add_mutually_exclusive_group()
    case_group.add_argument("--case", choices=sorted(CASE_DATA_MAP.keys()), help="case name (default: case1)")
    for name in CASE_DATA_MAP:
        case_group.add_argument(
            f"--{name}",
            dest="case",
            action="store_const",
            const=name,
            help=f"shortcut for --case {name}",
        )

    parser.add_argument("--max-workers", type=int, help="number of worker processes (default: use CPU count)")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_optimization(args.case, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
    

