"""
Self-contained Mars Lander bot for CodinGame.

Copy the contents of this file into the CodinGame editor.  The bot can run
purely with the built-in heuristic controller, but if you paste the parameters
from ``best_gene1.json`` into ``BEST_GENE`` it will use the trained neural
network policy that matches the local simulator.
"""

import math
import sys
from typing import List, Tuple

# Network layout shared with simulate.py
INPUT_SIZE = 15
HIDDEN_SIZE = 16
OUTPUT_SIZE = 2
EXPECTED_GENE_LEN = (INPUT_SIZE * HIDDEN_SIZE) + HIDDEN_SIZE + (HIDDEN_SIZE * OUTPUT_SIZE) + OUTPUT_SIZE

# Paste the numeric array from best_gene1.json here (should contain 290 floats).
# The fallback heuristic controller is used when this list is empty or has the
# wrong length.
BEST_GENE: List[float] = []


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _build_ground(points: List[Tuple[int, int]]):
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
        raise ValueError("Flat landing segment not found")

    return heights, landing_segment


def _ground_height_at(heights: List[int], x: float) -> float:
    if x <= 0:
        return heights[0]
    if x >= len(heights) - 1:
        return heights[-1]
    left_index = int(math.floor(x))
    right_index = min(left_index + 1, len(heights) - 1)
    fraction = x - left_index
    return heights[left_index] * (1 - fraction) + heights[right_index] * fraction


def _distance_to_boundary(x: float, y: float, direction_x: float, direction_y: float) -> float:
    distances = []
    if direction_x > 0:
        distances.append((7000 - x) / direction_x)
    elif direction_x < 0:
        distances.append((0 - x) / direction_x)
    if direction_y > 0:
        distances.append((3000 - y) / direction_y)
    elif direction_y < 0:
        distances.append((0 - y) / direction_y)
    positive = [d for d in distances if d > 0]
    return min(positive) if positive else 0


def _is_below_ground(heights: List[int], x: float, y: float) -> bool:
    if not (0 <= x < 7000):
        return False
    return y <= _ground_height_at(heights, x)


def _distance_in_direction(heights: List[int], lander_pos: Tuple[float, float], angle_deg: float) -> float:
    radians = math.radians(angle_deg)
    direction_x = math.sin(radians)
    direction_y = -math.cos(radians)
    if direction_x == 0 and direction_y == 0:
        return 0.0

    current_x, current_y = lander_pos
    max_distance = _distance_to_boundary(current_x, current_y, direction_x, direction_y)
    travelled = 0.0
    step_size = 5.0
    while travelled < max_distance:
        remaining = max_distance - travelled
        step = step_size if remaining > step_size else remaining
        next_x = current_x + direction_x * step
        next_y = current_y + direction_y * step
        if _is_below_ground(heights, next_x, next_y):
            low, high = 0.0, step
            for _ in range(10):
                mid = (low + high) / 2
                test_x = current_x + direction_x * mid
                test_y = current_y + direction_y * mid
                if _is_below_ground(heights, test_x, test_y):
                    high = mid
                else:
                    low = mid
            return travelled + high
        current_x, current_y = next_x, next_y
        travelled += step
    return max_distance


def _normalize_features(distances: List[float], relative_x: float, relative_y: float,
                        horizontal_speed: float, vertical_speed: float,
                        rotate: float, power: float,
                        midpoint_x: float, midpoint_y: float) -> List[float]:
    max_distance = max(3000.0, max(distances) if distances else 0.0, 1e-6)
    normalized: List[float] = []
    for distance in distances:
        clamped = _clamp(distance, 0.0, max_distance)
        normalized.append(2 * (clamped / max_distance) - 1)

    half_span_x = max(midpoint_x, 7000 - midpoint_x, 1e-6)
    normalized.append(_clamp(relative_x, -half_span_x, half_span_x) / half_span_x)

    half_span_y = max(midpoint_y, 3000 - midpoint_y, 1e-6)
    normalized.append(_clamp(relative_y, -half_span_y, half_span_y) / half_span_y)

    normalized.append(_clamp(horizontal_speed, -20.0, 20.0) / 20.0)
    normalized.append(_clamp(vertical_speed, -40.0, 40.0) / 40.0)
    normalized.append(_clamp(rotate, -90.0, 90.0) / 90.0)
    normalized.append(2 * (_clamp(power, 0.0, 4.0) / 4.0) - 1)
    return normalized


def _sensor_features(heights: List[int], lander_pos: Tuple[float, float], lander_speed: Tuple[float, float],
                     angle: float, power: float, landing_segment: Tuple[Tuple[int, int], Tuple[int, int]]):
    angles = [-90, -60, -30, -15, 0, 15, 30, 60, 90]
    distances = [_distance_in_direction(heights, lander_pos, a) for a in angles]
    midpoint_x = (landing_segment[0][0] + landing_segment[1][0]) / 2
    midpoint_y = landing_segment[0][1]
    relative_x = lander_pos[0] - midpoint_x
    relative_y = lander_pos[1] - midpoint_y
    return _normalize_features(
        distances,
        relative_x,
        relative_y,
        lander_speed[0],
        lander_speed[1],
        angle,
        power,
        midpoint_x,
        midpoint_y,
    )


def _nn_action(inputs: List[float], gene: List[float]):
    if len(inputs) != INPUT_SIZE:
        raise ValueError(f"expected {INPUT_SIZE} features, got {len(inputs)}")
    if len(gene) != EXPECTED_GENE_LEN:
        raise ValueError(f"gene length mismatch: {len(gene)} != {EXPECTED_GENE_LEN}")

    cursor = 0
    w1 = gene[cursor:cursor + INPUT_SIZE * HIDDEN_SIZE]
    cursor += INPUT_SIZE * HIDDEN_SIZE
    b1 = gene[cursor:cursor + HIDDEN_SIZE]
    cursor += HIDDEN_SIZE
    w2 = gene[cursor:cursor + HIDDEN_SIZE * OUTPUT_SIZE]
    cursor += HIDDEN_SIZE * OUTPUT_SIZE
    b2 = gene[cursor:cursor + OUTPUT_SIZE]

    hidden = []
    for j in range(HIDDEN_SIZE):
        acc = b1[j]
        for i in range(INPUT_SIZE):
            acc += inputs[i] * w1[i * HIDDEN_SIZE + j]
        hidden.append(math.tanh(acc))

    outputs = []
    for k in range(OUTPUT_SIZE):
        acc = b2[k]
        for j in range(HIDDEN_SIZE):
            acc += hidden[j] * w2[j * OUTPUT_SIZE + k]
        outputs.append(acc)

    angle_norm = math.tanh(outputs[0])
    power_norm = 2 * math.tanh(outputs[1]) - 1
    return angle_norm, power_norm


def _gene_controller(state, heights, landing_segment):
    x, y, hs, vs, fuel, rotate, power = state
    try:
        features = _sensor_features(heights, (x, y), (hs, vs), rotate, power, landing_segment)
        angle_norm, power_norm = _nn_action(features, BEST_GENE)
    except Exception:
        return None

    target_angle = round(_clamp(angle_norm, -1.0, 1.0) * 90.0 / 15.0) * 15.0
    target_angle = _clamp(target_angle, -90.0, 90.0)
    target_power = int(round(_clamp(power_norm, 0.0, 1.0) * 4.0))
    return int(target_angle), target_power


def _heuristic_controller(state, landing_segment):
    x, y, hs, vs, fuel, rotate, power = state
    left, right = sorted([landing_segment[0][0], landing_segment[1][0]])
    mid_x = (left + right) / 2
    target_angle = 0
    target_power = 3

    if x < left - 200:
        target_angle = -20
    elif x > right + 200:
        target_angle = 20
    else:
        target_angle = 0

    if y - landing_segment[0][1] < 200:
        target_power = 4
    elif vs < -35:
        target_power = 4
    elif vs < -25:
        target_power = 3
    elif vs > 20:
        target_power = 1
    else:
        target_power = 2

    if abs(hs) > 40:
        target_power = max(target_power, 4)

    if abs(x - mid_x) < 50 and abs(hs) < 10:
        target_angle = 0

    return target_angle, target_power


def main():
    surface_points = [tuple(map(int, input().split())) for _ in range(int(input()))]
    heights, landing_segment = _build_ground(surface_points)

    use_gene = len(BEST_GENE) == EXPECTED_GENE_LEN
    if not use_gene:
        print("Using heuristic controller (gene missing or invalid)", file=sys.stderr, flush=True)

    while True:
        x, y, hs, vs, fuel, rotate, power = map(int, input().split())
        state = (float(x), float(y), float(hs), float(vs), fuel, float(rotate), float(power))

        command = None
        if use_gene:
            command = _gene_controller(state, heights, landing_segment)
        if command is None:
            command = _heuristic_controller(state, landing_segment)

        target_angle, target_power = command
        print(f"{int(target_angle)} {int(target_power)}")


if __name__ == "__main__":
    main()
