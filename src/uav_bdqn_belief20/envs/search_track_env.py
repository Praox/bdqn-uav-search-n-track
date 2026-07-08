from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from uav_bdqn_belief20.controllers.mission_controller import (
    MissionController,
    SEARCH,
    TRACK,
    STAY,
    UP,
    DOWN,
    LEFT,
    RIGHT,
)
from uav_bdqn_belief20.envs.drone_memory import DroneMemory


@dataclass
class EnvConfig:
    grid_size: int = 20
    n_value1_targets: int = 3
    n_value2_targets: int = 1
    sensor_radius: int = 2
    detection_probability: float = 1.0
    track_radius: int = 1
    track_required: int = 3
    max_steps: int = 150
    macro_steps: int = 5
    seed: int | None = None

    step_penalty: float = -0.01
    new_cell_bonus: float = 0.01
    revisit_penalty: float = -0.005
    #detect_bonus: float = 0.50
    detect_value1_bonus: float = 0.30
    detect_value2_bonus: float = 1.00
    #track_progress_bonus: float = 0.05
    track_progress_value1_bonus: float = 0.03
    track_progress_value2_bonus: float = 0.12
    #complete_bonus: float = 2.00
    complete_value1_bonus: float = 2.0
    complete_value2_bonus: float = 8.0
    #for cost tracking
    track_step_penalty: float = -0.02
    all_targets_bonus: float = 3.00
    boundary_penalty: float = -0.05
    unknown_track_penalty: float = -0.02
    
    


class SearchTrackBelief20Env:
    """Single-UAV search/track with local sensing and 20x20 drone memory.

    The environment stores hidden truth. The BDQN and controller see only the drone memory.
    """

    SEARCH = SEARCH
    TRACK = TRACK
    action_dim = 2
    primitive_action_dim = 5

    MOVES = {
        STAY: (0, 0),
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
    }

    def __init__(self, config: EnvConfig | None = None, controller: MissionController | None = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.controller = controller or MissionController()
        #self.observation_shape = (5, self.cfg.grid_size, self.cfg.grid_size)
        self.observation_shape = (6, self.cfg.grid_size, self.cfg.grid_size)
        self.reset()

    def _detect_reward(self, value: int) -> float:
        if int(value) == 1:
            return self.cfg.detect_value1_bonus
        return self.cfg.detect_value2_bonus


    def _track_progress_reward(self, value: int) -> float:
        if int(value) == 1:
            return self.cfg.track_progress_value1_bonus
        return self.cfg.track_progress_value2_bonus


    def _complete_reward(self, value: int) -> float:
        if int(value) == 1:
            return self.cfg.complete_value1_bonus
        return self.cfg.complete_value2_bonus
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        g = self.cfg.grid_size
        self.t = 0
        self.drone_pos = np.array([self.rng.integers(g), self.rng.integers(g)], dtype=np.int64)
        self.target_values = np.array(
            [1] * self.cfg.n_value1_targets + [2] * self.cfg.n_value2_targets,
            dtype=np.int64,
        )
        n_targets = len(self.target_values)
        forbidden = {tuple(self.drone_pos)}
        positions = []
        while len(positions) < n_targets:
            p = (int(self.rng.integers(g)), int(self.rng.integers(g)))
            if p not in forbidden:
                positions.append(p)
                forbidden.add(p)
        self.target_pos = np.array(positions, dtype=np.int64)
        self.detected = np.zeros(n_targets, dtype=bool)
        self.completed = np.zeros(n_targets, dtype=bool)
        self.track_progress = np.zeros(n_targets, dtype=np.int64)
        self.memory = DroneMemory(grid_size=g, n_targets=n_targets)
        # At reset the drone knows its starting cell is visited.
        self.memory.mark_visited([tuple(self.drone_pos)])
        self.last_mission = SEARCH
        self.last_primitive_actions: list[int] = []
        return self._obs(), self._info()
    
    
    def action_mask(self) -> np.ndarray:
        """Return valid high-level actions for the current drone memory.

        action 0 = SEARCH
        action 1 = TRACK

        TRACK is only valid if the drone already knows at least one
        non-completed target in its own memory.
        """
        has_trackable_target = any(
            not target.completed
            for target in self.memory.known_targets.values()
        )

        return np.array(
            [True, has_trackable_target],
            dtype=bool,
        )
        
    def step(self, mission: int):
        assert 0 <= int(mission) < self.action_dim
        mission = int(mission)
        self.last_mission = mission
        self.last_primitive_actions = []
        total_reward = 0.0
        terminated = False
        truncated = False
        executed = 0

        for k in range(self.cfg.macro_steps):
            primitive = self.controller.act(mission, self.drone_pos, self.memory)
            self.last_primitive_actions.append(int(primitive))
            reward, terminated, truncated = self._low_level_step(primitive, mission)
            total_reward += (0.99 ** k) * reward
            executed += 1
            if terminated or truncated:
                break
        info = self._info()
        info["macro_executed_steps"] = executed
        return self._obs(), float(total_reward), terminated, truncated, info

    def _low_level_step(self, primitive_action: int, mission: int):
        self.t += 1
        reward = self.cfg.step_penalty
        if mission == TRACK:
            reward += self.cfg.track_step_penalty

        if mission == TRACK and not self._has_known_uncompleted_target():
            reward += self.cfg.unknown_track_penalty

        dr, dc = self.MOVES[int(primitive_action)]
        new_pos = self.drone_pos + np.array([dr, dc], dtype=np.int64)
        clipped = np.clip(new_pos, 0, self.cfg.grid_size - 1)
        if np.any(clipped != new_pos):
            reward += self.cfg.boundary_penalty
        self.drone_pos = clipped

        pos = tuple(self.drone_pos)
        if self.memory.visited[pos] < 0.5:
            reward += self.cfg.new_cell_bonus
        else:
            reward += self.cfg.revisit_penalty

        reward += self._observe_and_update_memory()
        
        if mission == TRACK:
            reward += self._track_update_if_possible()

        terminated = bool(np.all(self.completed))
        if terminated:
            reward += self.cfg.all_targets_bonus
        truncated = self.t >= self.cfg.max_steps
        return float(reward), terminated, truncated

    def _observe_and_update_memory(self) -> float:
        """Local sensor update: this is the only place hidden target positions affect memory."""
        reward = 0.0
        visible = self._cells_in_radius(self.drone_pos, self.cfg.sensor_radius)
        self.memory.mark_visited(visible)
        empty_cells: list[tuple[int, int]] = []

        for cell in visible:
            target_ids_here = [i for i, p in enumerate(self.target_pos) if tuple(p) == cell and not self.completed[i]]
            if not target_ids_here:
                empty_cells.append(cell)
                continue
            for i in target_ids_here:
                detected_now = self.rng.random() < self.cfg.detection_probability
                if detected_now:
                    if not self.detected[i]:
                        self.detected[i] = True
                        #reward += self.cfg.detect_bonus * float(self.target_values[i])
                        reward += self._detect_reward(int(self.target_values[i]))
                    self.memory.add_or_update_target(
                        target_id=i,
                        pos=cell,
                        value=int(self.target_values[i]),
                        step=self.t,
                    )
        self.memory.suppress_empty_cells(empty_cells, factor=0.2)
        return float(reward)

    def _track_update_if_possible(self) -> float:
        candidates = []
        for target_id, target in self.memory.known_targets.items():
            if target.completed:
                continue
            if self._dist(self.drone_pos, target.pos) <= self.cfg.track_radius:
                candidates.append(target_id)
        if not candidates:
            return 0.0
        candidates.sort(
            key=lambda i: (
                -int(self.memory.known_targets[i].value),
                self._dist(self.drone_pos, self.memory.known_targets[i].pos),
            )
        )
        i = int(candidates[0])
        self.track_progress[i] += 1
        #reward = self.cfg.track_progress_bonus * float(self.target_values[i])
        value = int(self.target_values[i])
        reward = self._track_progress_reward(value)
        if self.track_progress[i] >= self.cfg.track_required and not self.completed[i]:
            self.completed[i] = True
            #reward += self.cfg.complete_bonus * float(self.target_values[i])
            reward += self._complete_reward(value)
        self.memory.update_target_progress(i, int(self.track_progress[i]), bool(self.completed[i]))
        return float(reward)

    def _obs(self) -> np.ndarray:
        g = self.cfg.grid_size
        drone = np.zeros((g, g), dtype=np.float32)
        drone[tuple(self.drone_pos)] = 1.0
        known_target_value = self.memory.known_target_value_map()
        time_remaining = np.full(
            (g, g),
            1.0 - float(self.t) / float(self.cfg.max_steps),
            dtype=np.float32,
        )
        return np.stack(
            [
                drone,
                self.memory.belief,
                known_target_value,
                self.memory.completed_map,
                self.memory.visited,
                time_remaining,
            ],
            axis=0,
        ).astype(np.float32)

    def _info(self) -> Dict:
        completed_value = int((self.completed.astype(np.int64) * self.target_values).sum())
        detected_value = int((self.detected.astype(np.int64) * self.target_values).sum())
        
        detected_value1 = int(((self.detected) & (self.target_values == 1)).sum())
        detected_value2 = int(((self.detected) & (self.target_values == 2)).sum())

        completed_value1 = int(((self.completed) & (self.target_values == 1)).sum())
        completed_value2 = int(((self.completed) & (self.target_values == 2)).sum())
        return {
            "t": self.t,
            "drone_pos": self.drone_pos.copy(),
            "detected": int(self.detected.sum()),
            "completed": int(self.completed.sum()),
            "known_targets": len(self.memory.known_targets),
            "known_uncompleted": sum(1 for t in self.memory.known_targets.values() if not t.completed),
            "visited_ratio": float(self.memory.visited.mean()),
            
            "target_values": self.target_values.copy(),
            "target_pos": self.target_pos.copy(),  # debug only; not exposed to controller or BDQN.
            "track_progress": self.track_progress.copy(),
            "last_mission": int(self.last_mission),
            "last_primitive_actions": list(self.last_primitive_actions),
            
            "completed_value": completed_value,
            "detected_value": detected_value,
            "detected_value1": detected_value1,
            "detected_value2": detected_value2,
            "completed_value1": completed_value1,
            "completed_value2": completed_value2,
            
        }

    def _has_known_uncompleted_target(self) -> bool:
        return any(not t.completed for t in self.memory.known_targets.values())

    def _cells_in_radius(self, center: np.ndarray, radius: int) -> list[tuple[int, int]]:
        g = self.cfg.grid_size
        cr, cc = int(center[0]), int(center[1])
        out = []
        for r in range(max(0, cr - radius), min(g, cr + radius + 1)):
            for c in range(max(0, cc - radius), min(g, cc + radius + 1)):
                if abs(r - cr) + abs(c - cc) <= radius:
                    out.append((r, c))
        return out

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> int:
        return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))
