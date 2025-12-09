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


CASE_DATA = [
    _load_case(CASE_DIR / f"case{i}.txt")
    for i in range(1, 6)
]


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
    success_count = 0
    for case_data in CASE_DATA:
        board = _create_board(case_data)
        score = simulation(gene, board)
        if score >= 200.0:
            success_count += 1
        total_score += score

    success_rate = success_count / 5.0
    return -total_score, success_rate



def run_optimization(max_workers: int | None = None):
    optimizer = CMA(mean=np.zeros(gene_num), sigma=0.5)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for generation in range(max_generation):
            genes = [optimizer.ask() for _ in range(optimizer.population_size)]
            results = list(executor.map(evaluate, genes))

            solutions = [(gene, loss) for gene, (loss, _) in zip(genes, results)]
            optimizer.tell(solutions)

            losses = np.array([loss for loss, _ in results], dtype=float)
            scores = -losses
            best_score = scores.max()
            mean_score = scores.mean()
            best_success_rate = max(success_rate for _, success_rate in results)
            print(
                f"generation {generation:03d}: best_score={best_score:.2f}, mean_score={mean_score:.2f}, "
                f"best_success_rate={best_success_rate:.2f}"
            )


if __name__ == "__main__":
    run_optimization()
    

