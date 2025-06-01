# envs/gridworld.py
from __future__ import annotations
import numpy as np


class GridWorld:
    """20×20 hall with 1 or 2 exits, identical to the paper’s environment.

    Observation = 4-dim float vector: [agent_x, agent_y, goal_x, goal_y] scaled to [0,1].
    """

    ACTIONS = {
        0: np.array([0, 1]),    # up
        1: np.array([1, 0]),    # right
        2: np.array([0, -1]),   # down
        3: np.array([-1, 0])    # left
    }

    def __init__(self, size: int = 20, exits: int = 2, max_steps: int = 400):
        assert exits in (1, 2)
        self.size = size
        self.max_steps = max_steps
        self.exit_cols = [5] if exits == 1 else [5, 15]
        self._build_hall_mask()

    # ------------------------------------------------------------------ helpers
    def _build_hall_mask(self):
        mid = self.size // 2
        self.hall_mask = np.zeros((self.size, self.size), dtype=bool)
        self.hall_mask[mid - 1: mid + 1, :] = True
        for c in self.exit_cols:
            self.hall_mask[:, c] = False   # carve exits vertically

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.state, self.goal]) / self.size  # scaled

    def _random_hall_coord(self):
        mid = self.size // 2
        col = np.random.randint(0, self.size)
        row = np.random.choice([mid - 1, mid])
        return np.array([col, row])

    def _random_outside_coord(self):
        while True:
            c = np.random.randint(0, self.size, 2)
            if not self.hall_mask[tuple(c)]:
                return c

    # ------------------------------------------------------------------ public API
    def reset(self):
        self.state = self._random_hall_coord()
        self.goal = self._random_outside_coord()
        self.steps = 0
        return self._obs()

    def step(self, action: int):
        self.steps += 1
        move = self.ACTIONS[action]
        nxt = np.clip(self.state + move, 0, self.size - 1)
        collided = self._is_blocked(self.state, nxt)
        if not collided:
            self.state = nxt
        done = np.all(self.state == self.goal) or self.steps >= self.max_steps or collided
        reward = 0.0 if np.all(self.state == self.goal) else -1.0
        info = {"collided": bool(collided)}
        return self._obs(), reward, done, info

    # expert (A*)
    def optimal_action(self):
        dx, dy = self.goal - self.stat
