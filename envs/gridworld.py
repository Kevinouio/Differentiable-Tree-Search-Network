from __future__ import annotations
import numpy as np


class GridWorld:
    """20×20 hall environment (1-exit or 2-exit) exactly matching the paper.

    Observation: 4-float vector (agent_x, agent_y, goal_x, goal_y) ∈ [0,1].
    Action space: 0-up, 1-right, 2-down, 3-left.
    A terminal episode ends on goal, collision with wall, or max_steps.
    """

    ACTIONS = {
        0: np.array([0, 1], dtype=int),    # up
        1: np.array([1, 0], dtype=int),    # right
        2: np.array([0, -1], dtype=int),   # down
        3: np.array([-1, 0], dtype=int),   # left
    }

    def __init__(self, size: int = 20, exits: int = 2, max_steps: int = 400):
        assert exits in (1, 2)
        self.size = size
        self.max_steps = max_steps
        self.exit_cols = [5] if exits == 1 else [5, 15]
        self._build_hall_mask()

    # ------------------------------------------------------------------ helpers
    def _build_hall_mask(self):
        """Binary mask — True where the 2-row hall wall blocks movement."""
        mid = self.size // 2
        self.hall_mask = np.zeros((self.size, self.size), dtype=bool)
        self.hall_mask[mid - 1: mid + 1, :] = True
        # carve vertical corridors at exit columns
        for c in self.exit_cols:
            self.hall_mask[:, c] = False

    def _obs(self) -> np.ndarray:
        return (np.concatenate([self.state, self.goal]).astype(np.float32) / self.size)

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

        reached_goal = np.all(self.state == self.goal)
        done = reached_goal or collided or self.steps >= self.max_steps
        reward = 0.0 if reached_goal else -1.0
        return self._obs(), reward, done, {"collided": bool(collided)}

    # ------------------------------------------------------------------ expert policy (for data generation)
    def optimal_action(self):
        dx, dy = self.goal - self.state
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3
        else:
            return 0 if dy > 0 else 2

    # ------------------------------------------------------------------ utils
    def _is_blocked(self, cur: np.ndarray, nxt: np.ndarray) -> bool:
        """Returns True if movement from *cur* → *nxt* crosses the wall.
        Crossing is only allowed through designated exit columns."""
        mid = self.size // 2
        # Did we attempt to move between the two hall rows?
        crossing_rows = (cur[1] != nxt[1]) and (mid - 1 <= max(cur[1], nxt[1]) <= mid)
        # If crossing, block unless x-coordinate is one of the exits.
        blocked = crossing_rows and (cur[0] not in self.exit_cols)
        return blocked
