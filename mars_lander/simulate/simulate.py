import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


CASE_DIR = Path(__file__).resolve().parent


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

class Board():
    '''
    物理システムと終了条件の記述
    '''
    def __init__(self, ground_height_list, lander_pos, lander_speed, flat_ground_pair, init_rotate, init_power, init_fuel):
        self.width = 7000
        self.height = 3000
        self.gravity = 3.711
        self.ground_height_list = ground_height_list
        self.lander_pos = [float(lander_pos[0]), float(lander_pos[1])]
        self.lander_speed = [float(lander_speed[0]), float(lander_speed[1])]
        self.flat_ground_pair = flat_ground_pair
        self.rotate = init_rotate
        self.power = init_power
        self.fuel = init_fuel
        self.init_fuel = init_fuel
        self._landing_x_min = min(self.flat_ground_pair[0][0], self.flat_ground_pair[1][0])
        self._landing_x_max = max(self.flat_ground_pair[0][0], self.flat_ground_pair[1][0])
        self._max_distance = self._compute_max_distance()
    
    def is_terminate(self):
        x, y = self.lander_pos
        horizontal_speed, vertical_speed = self.lander_speed

        if (x < 0) or (self.width <= x) or (y < 0) or (self.height <= y):
            return True, self.score(
                hit_landing_area=False,
                success=False,
                horizontal_speed=horizontal_speed,
                vertical_speed=vertical_speed,
            )

        ground_index = int(round(x))
        if 0 <= ground_index < len(self.ground_height_list):
            ground_height = self.ground_height_list[ground_index]
            if y <= ground_height:  # 着地
                hit_landing_area = self._landing_x_min <= x <= self._landing_x_max
                safe_speed = abs(horizontal_speed) <= 20 and abs(vertical_speed) <= 40
                upright = self.rotate == 0
                success = hit_landing_area and safe_speed and upright
                return True, self.score(
                    hit_landing_area=hit_landing_area,
                    success=success,
                    horizontal_speed=horizontal_speed,
                    vertical_speed=vertical_speed,
                )

        return False, None
        
    def score(self, hit_landing_area, success, horizontal_speed, vertical_speed):
        current_speed = math.hypot(horizontal_speed, vertical_speed)

        if not hit_landing_area:
            distance = self._distance_to_landing_area(*self.lander_pos)
            score = 100 - (100 * distance / self._max_distance)
            speed_penalty = 0.1 * max(current_speed - 100, 0)
            return score - speed_penalty

        if not success:
            x_penalty = 0.0
            if abs(horizontal_speed) > 20:
                x_penalty = (abs(horizontal_speed) - 20) / 2

            y_penalty = 0.0
            if abs(vertical_speed) > 40:
                if vertical_speed < -40:
                    y_penalty = (-40 - vertical_speed) / 2
                else:
                    y_penalty = (vertical_speed - 40) / 2

            rotate_penalty = abs(self.rotate) * 0.5

            return 200 - x_penalty - y_penalty - rotate_penalty

        fuel_bonus = 0
        if self.init_fuel:
            fuel_bonus = 100 * self.fuel / self.init_fuel

        return 200 + fuel_bonus

    def _distance_to_landing_area(self, x, y):
        x1, y1 = self.flat_ground_pair[0]
        x2, y2 = self.flat_ground_pair[1]

        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        landing_y = y1

        if x1 <= x <= x2:
            return abs(y - landing_y)

        if x < x1:
            return math.hypot(x - x1, y - landing_y)

        return math.hypot(x - x2, y - landing_y)

    def _compute_max_distance(self):
        corners = [(0, 0), (0, self.height), (self.width, 0), (self.width, self.height)]
        distances = [self._distance_to_landing_area(cx, cy) for cx, cy in corners]
        max_distance = max(distances)
        return max_distance if max_distance else 1
    
    def update(self, desired_power, desired_angle):
        desired_power = _clamp(desired_power, 0, 4)
        desired_angle = _clamp(desired_angle, -90, 90)

        power_delta = _clamp(desired_power - self.power, -1, 1)
        angle_delta = _clamp(desired_angle - self.rotate, -15, 15)

        self.power = _clamp(self.power + power_delta, 0, 4)
        self.rotate = _clamp(self.rotate + angle_delta, -90, 90)

        if self.fuel <= 0:
            self.power = 0

        thrust = min(self.power, self.fuel)
        self.fuel = max(self.fuel - thrust, 0)
        if self.power > thrust:
            self.power = thrust

        acc_x = -thrust * math.sin(math.radians(self.rotate))
        acc_y = thrust * math.cos(math.radians(self.rotate)) - self.gravity

        self.lander_pos[0] += self.lander_speed[0] + 0.5 * acc_x
        self.lander_pos[1] += self.lander_speed[1] + 0.5 * acc_y

        self.lander_speed[0] += acc_x
        self.lander_speed[1] += acc_y

    def sensor(self, normalize: bool = True):
        '''
        Return 13 sensor features describing the lander state.

        Contents:
        - Distances from the lander to terrain or board boundary when
          casting rays at angles [-90, -60, -30, -15, 0, 15, 30, 60, 90] degrees
          relative to the lander (positive angles tilt to the right).
        - Relative x/y position of the lander from the landing-zone midpoint.
        - Horizontal and vertical speed components.
        - Current rotation angle and engine power setting.

        When ``normalize`` is True (default) each feature is scaled into [-1, 1]
        using expected bounds so that the downstream policy sees balanced inputs.
        '''
        angles = [-90, -60, -30, -15, 0, 15, 30, 60, 90]
        distance_features = [self._distance_in_direction(angle) for angle in angles]

        midpoint_x = (self.flat_ground_pair[0][0] + self.flat_ground_pair[1][0]) / 2
        midpoint_y = self.flat_ground_pair[0][1]
        relative_x = self.lander_pos[0] - midpoint_x
        relative_y = self.lander_pos[1] - midpoint_y

        horizontal_speed = self.lander_speed[0]
        vertical_speed = self.lander_speed[1]

        features = distance_features + [
            relative_x,
            relative_y,
            horizontal_speed,
            vertical_speed,
            self.rotate,
            self.power,
        ]

        if not normalize:
            return features

        return self._normalize_features(
            distance_features,
            relative_x,
            relative_y,
            horizontal_speed,
            vertical_speed,
            self.rotate,
            self.power,
            midpoint_x,
            midpoint_y,
        )

    def _normalize_features(self, distances, relative_x, relative_y,
                             horizontal_speed, vertical_speed, rotate, power,
                             midpoint_x, midpoint_y):
        normalized = []

        max_distance = max(self._max_distance, 1e-6)
        for distance in distances:
            clamped = _clamp(distance, 0.0, max_distance)
            normalized.append(2 * (clamped / max_distance) - 1)

        half_span_x = max(midpoint_x, self.width - midpoint_x, 1e-6)
        rel_x = _clamp(relative_x, -half_span_x, half_span_x)
        normalized.append(rel_x / half_span_x)

        half_span_y = max(midpoint_y, self.height - midpoint_y, 1e-6)
        rel_y = _clamp(relative_y, -half_span_y, half_span_y)
        normalized.append(rel_y / half_span_y)

        max_h_speed = 20.0
        h_speed = _clamp(horizontal_speed, -max_h_speed, max_h_speed)
        normalized.append(h_speed / max_h_speed)

        max_v_speed = 40.0
        v_speed = _clamp(vertical_speed, -max_v_speed, max_v_speed)
        normalized.append(v_speed / max_v_speed)

        normalized.append(_clamp(rotate, -90.0, 90.0) / 90.0)
        normalized.append(2 * (_clamp(power, 0.0, 4.0) / 4.0) - 1)

        return normalized

    def _ground_height_at(self, x):
        if x <= 0:
            return self.ground_height_list[0]
        if x >= len(self.ground_height_list) - 1:
            return self.ground_height_list[-1]

        left_index = int(math.floor(x))
        right_index = min(left_index + 1, len(self.ground_height_list) - 1)
        fraction = x - left_index
        left_height = self.ground_height_list[left_index]
        right_height = self.ground_height_list[right_index]
        return left_height * (1 - fraction) + right_height * fraction

    def _is_below_ground(self, x, y):
        if not (0 <= x < self.width):
            return False
        return y <= self._ground_height_at(x)

    def _distance_in_direction(self, angle_deg):
        radians = math.radians(angle_deg)
        direction_x = math.sin(radians)
        direction_y = -math.cos(radians)

        if direction_x == 0 and direction_y == 0:
            return 0

        current_x = float(self.lander_pos[0])
        current_y = float(self.lander_pos[1])
        max_distance = self._distance_to_boundary(current_x, current_y, direction_x, direction_y)

        travelled = 0.0
        step_size = 5.0
        while travelled < max_distance:
            remaining = max_distance - travelled
            step = step_size if remaining > step_size else remaining
            next_x = current_x + direction_x * step
            next_y = current_y + direction_y * step

            if self._is_below_ground(next_x, next_y):
                low = 0.0
                high = step
                for _ in range(10):  # refine collision point
                    mid = (low + high) / 2
                    test_x = current_x + direction_x * mid
                    test_y = current_y + direction_y * mid
                    if self._is_below_ground(test_x, test_y):
                        high = mid
                    else:
                        low = mid
                return travelled + high

            current_x = next_x
            current_y = next_y
            travelled += step

        return max_distance

    def _distance_to_boundary(self, x, y, direction_x, direction_y):
        distances = []

        if direction_x > 0:
            distances.append((self.width - x) / direction_x)
        elif direction_x < 0:
            distances.append((0 - x) / direction_x)

        if direction_y > 0:
            distances.append((self.height - y) / direction_y)
        elif direction_y < 0:
            distances.append((0 - y) / direction_y)

        positive_distances = [d for d in distances if d > 0]
        return min(positive_distances) if positive_distances else 0

def _build_ground(points: List[Tuple[int, int]]) -> Tuple[List[int], Tuple[Tuple[int, int], Tuple[int, int]]]:
    heights = [0] * 7000
    landing_segment = None

    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        if dy == 0:
            landing_segment = ((x1, y1), (x2, y2))
        for x in range(x1, x2 + 1):
            t = (x - x1) / dx
            y = y1 + t * dy
            heights[x] = int(round(y))

    if landing_segment is None:
        raise ValueError("Flat landing segment not found in surface definition")

    return heights, landing_segment

def _load_case_file(path: Path):
    with path.open("r", encoding="ascii") as f:
        lines = [line.strip() for line in f if line.strip()]

    iterator = iter(lines)
    surface_count = int(next(iterator))
    points = [tuple(map(int, next(iterator).split())) for _ in range(surface_count)]
    heights, landing_segment = _build_ground(points)
    x, y, h_speed, v_speed, fuel, rotate, power = map(int, next(iterator).split())
    initial_state = (x, y, h_speed, v_speed, fuel, rotate, power)
    return heights, landing_segment, initial_state


def _create_board_from_case(case_name: str) -> Board:
    case_path = CASE_DIR / f"{case_name}.txt"
    if not case_path.exists():
        available = sorted(p.stem for p in CASE_DIR.glob("case*.txt"))
        message = f"case '{case_name}' not found. available cases: {', '.join(available)}"
        raise FileNotFoundError(message)

    ground_height_list, flat_ground_pair, initial = _load_case_file(case_path)
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


def simulate_case(case_name: str, gene: np.ndarray, step_limit: int = 1500, verbose: bool = True):
    from gene import action  # local import avoids circular dependency during module load
    from parameter import input_size

    board = _create_board_from_case(case_name)

    for step in range(step_limit):
        terminated, score = board.is_terminate()
        if terminated:
            if verbose:
                print(f"terminated at step {step}: score={score:.2f}")
            return score, step

        features = board.sensor()
        if len(features) != input_size:
            raise ValueError(f"expected {input_size} features, got {len(features)}")

        angle_norm, power_norm = action(features, gene)

        clamped_angle = max(-1.0, min(1.0, angle_norm))
        target_angle = round((clamped_angle * 90.0) / 15.0) * 15.0
        target_angle = max(-90.0, min(90.0, target_angle))

        clamped_power = max(0.0, min(1.0, power_norm))
        target_power = int(round(clamped_power * 4.0))

        if verbose:
            print(
                f"step {step:03d} | pos=({board.lander_pos[0]:7.2f}, {board.lander_pos[1]:7.2f}) "
                f"speed=({board.lander_speed[0]:6.2f}, {board.lander_speed[1]:6.2f}) "
                f"target=(P {target_power}, ang {target_angle:+.0f})"
            )

        board.update(target_power, target_angle)

    if verbose:
        print("terminated: step limit reached without landing")
    return None, step_limit


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Simulate lander behavior for a single case")
    parser.add_argument("--case", default="case1", help="case name without extension (default: case1)")
    parser.add_argument(
        "--gene-file",
        type=Path,
        help="path to .npy file containing a gene vector (default: best_gene.npy if present)",
    )
    parser.add_argument("--seed", type=int, help="random seed when no gene file is available")
    parser.add_argument("--steps", type=int, default=1500, help="maximum number of simulation steps")
    parser.add_argument("--quiet", action="store_true", help="suppress per-step logs")
    return parser.parse_args(argv)


def _default_gene_path(case_name: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    if case_name.startswith("case"):
        suffix = case_name[len("case") :]
        suffix = suffix or case_name
        return root / f"best_gene{suffix}.json"
    return root / f"best_gene_{case_name}.json"


def main(argv=None):
    from parameter import gene_num

    args = _parse_args(argv)
    default_gene_path = _default_gene_path(args.case)

    gene_path: Path | None = None
    if args.gene_file:
        gene_path = args.gene_file
    elif default_gene_path.exists():
        gene_path = default_gene_path

    if gene_path is not None:
        if not gene_path.exists():
            raise FileNotFoundError(f"gene file not found: {gene_path}")
        data = json.loads(gene_path.read_text(encoding="ascii"))
        gene = np.asarray(data, dtype=float)
        if not args.quiet and gene_path != args.gene_file:
            print(f"loaded gene from {gene_path.name}")
    else:
        rng = np.random.default_rng(args.seed)
        gene = rng.normal(size=gene_num)
        if not args.quiet:
            print("no gene file found; simulating with a random gene")

    simulate_case(args.case, gene, step_limit=args.steps, verbose=not args.quiet)


if __name__ == "__main__":
    main()