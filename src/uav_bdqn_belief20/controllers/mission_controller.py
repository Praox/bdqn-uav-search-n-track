from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from uav_bdqn_belief20.envs.drone_memory import DroneMemory

STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

SEARCH = 0
TRACK = 1


@dataclass
class ControllerConfig:
    search_belief_weight: float = 2.0
    search_uncertainty_weight: float = 1.0
    search_distance_weight: float = 0.05
    track_value_weight: float = 2.0
    track_progress_weight: float = 0.5
    track_distance_weight: float = 0.10


class MissionController:
    """Low-level controller that only uses drone memory.

    SEARCH: choose a cell according to belief + uncertainty - distance.
    TRACK: choose a known non-completed target according to value + progress - distance.
    """

    def __init__(self, cfg: ControllerConfig | None = None):
        self.cfg = cfg or ControllerConfig()

    def act(self, mission: int, drone_pos: np.ndarray, memory: DroneMemory) -> int:
        if mission == TRACK:
            action = self._track_action(drone_pos, memory)
            if action is not None:
                return action
        return self._search_action(drone_pos, memory)

    def _search_action(self, drone_pos: np.ndarray, memory: DroneMemory) -> int:
        best_cell = None
        best_score = -1e18
        g = memory.grid_size
        for r in range(g):
            for c in range(g):
                cell = np.array([r, c], dtype=np.int64)
                belief = float(memory.belief[r, c])
                uncertainty = 1.0 - float(memory.visited[r, c])
                dist = self._dist(drone_pos, cell)
                score = (
                    self.cfg.search_belief_weight * belief
                    + self.cfg.search_uncertainty_weight * uncertainty
                    - self.cfg.search_distance_weight * dist
                )
                if score > best_score:
                    best_score = score
                    best_cell = cell
        if best_cell is None:
            return STAY
        return self._move_toward(drone_pos, best_cell)

    def _track_action(self, drone_pos: np.ndarray, memory: DroneMemory) -> int | None:
        candidates = [t for t in memory.known_targets.values() if not t.completed]
        if not candidates:
            return None
        target = max(
            candidates,
            key=lambda t: (
                self.cfg.track_value_weight * float(t.value)
                + self.cfg.track_progress_weight * float(t.progress)
                - self.cfg.track_distance_weight * self._dist(drone_pos, t.pos)
            ),
        )
        return self._move_toward(drone_pos, target.pos)

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> int:
        return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))

    @staticmethod
    def _move_toward(pos: np.ndarray, goal: np.ndarray) -> int:
        dr = int(goal[0]) - int(pos[0])
        dc = int(goal[1]) - int(pos[1])
        if abs(dr) >= abs(dc) and dr != 0:
            return DOWN if dr > 0 else UP
        if dc != 0:
            return RIGHT if dc > 0 else LEFT
        return STAY
