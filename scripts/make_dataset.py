"""make_dataset.py
===================
Creates an **offline Grid-World dataset** compatible with `dtsn.train`.
Highlights versus the simple demo:
• Guarantees every episode terminates (A* expert).  
• Supports 1-exit *or* 2-exit halls.  
• Optional ε-greedy noise for sub-optimal behaviour.  
• Caps episode length to avoid infinite loops.  
Run:
    python make_dataset.py --episodes 1000 --exits 2 --epsilon 0.05
"""

from __future__ import annotations

import heapq
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Grid-World Environment
# -----------------------------------------------------------------------------

class GridWorld:
    """20×20 hall environment with configurable exits."""

    ACTIONS = {0: np.array([0, 1]),   # up
               1: np.array([1, 0]),   # right
               2: np.array([0, -1]),  # down
               3: np.array([-1, 0])}  # left

    def __init__(self, size: int = 20, exits: int = 2):
        assert exits in (1, 2)
        self.size = size
        # exit positions along the central vertical wall (row = size//2)
        mid = size // 2
        self.exit_cols = [5] if exits == 1 else [5, 15]
        # build hall mask: two central rows are walls except exits
        self.hall = np.zeros((size, size), dtype=bool)
        self.hall[mid - 1: mid + 1, :] = True
        for c in self.exit_cols:
            self.hall[:, c] = False  # carve vertical corridor at exit col

    # ---------------- core API ----------------
    def reset(self):
        self.state = self._random_hall_coord()
        self.goal = self._random_outside_coord()
        self.steps = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        nxt = self.state + self.ACTIONS[action]
        nxt = np.clip(nxt, 0, self.size - 1)
        # wall collision → stay
        if self._is_wall(self.state, nxt):
            nxt = self.state
        self.state = nxt
        done = np.all(self.state == self.goal) or self.steps >= 400
        reward = 0.0 if np.all(self.state == self.goal) else -1.0
        return self._obs(), reward, done, {}

    # ---------------- expert path (A*) ----------------
    def optimal_action(self) -> int:
        """A* on Manhattan distance that respects hall walls."""
        open_set: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, tuple(self.state)))
        g = {tuple(self.state): 0}
        parent = {}
        goal = tuple(self.goal)
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                break
            for a, move in self.ACTIONS.items():
                nxt = (current[0] + move[0], current[1] + move[1])
                if not (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size):
                    continue
                if self._is_wall(np.array(current), np.array(nxt)):
                    continue
                tentative = g[current] + 1
                if tentative < g.get(nxt, 1e9):
                    g[nxt] = tentative
                    f = tentative + self._manhattan(nxt, goal)
                    heapq.heappush(open_set, (f, nxt))
                    parent[nxt] = (current, a)
        # backtrack one step to choose current action
        cur = goal
        if cur not in parent:  # already at goal
            return 0
        while parent[cur][0] != tuple(self.state):
            cur = parent[cur][0]
        return parent[cur][1]

    # ---------------- helpers ----------------
    def _obs(self):
        return np.concatenate([
            self.state / self.size,
            self.goal / self.size,
        ]).astype(np.float32)

    def _random_hall_coord(self):
        mid = self.size // 2
        col = np.random.randint(0, self.size)
        row = np.random.choice([mid - 1, mid])
        return np.array([col, row])

    def _random_outside_coord(self):
        while True:
            c = np.random.randint(0, self.size, 2)
            if not self.hall[tuple(c)]:
                return c

    def _is_wall(self, cur: np.ndarray, nxt: np.ndarray):
        # moving between hall rows is blocked except at exit cols
        mid = self.size // 2
        crossing = (cur[1] != nxt[1]) and (mid - 1 <= max(cur[1], nxt[1]) <= mid)
        blocked = crossing and (cur[0] not in self.exit_cols)
        return blocked

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------

def generate_dataset(episodes: int, exits: int, epsilon: float, out_path: Path):
    env = GridWorld(exits=exits)
    obs_l, act_l, r_l, next_l, q_l = [], [], [], [], []

    for _ in range(episodes):
        s = env.reset()
        traj = []
        done = False
        while not done:
            a = env.optimal_action()
            if np.random.rand() < epsilon:
                a = np.random.randint(0, 4)  # exploration noise
            s2, r, done, _ = env.step(a)
            traj.append((s, a, r, s2))
            s = s2
        # Monte-Carlo return
        G = 0.0
        for s, a, r, s2 in reversed(traj):
            G += r
            obs_l.append(s)
            act_l.append(a)
            r_l.append(r)
            next_l.append(s2)
            q_l.append(G)

    data = {
        "obs": torch.tensor(obs_l, dtype=torch.float32),
        "action": torch.tensor(act_l, dtype=torch.long),
        "reward": torch.tensor(r_l, dtype=torch.float32),
        "next_obs": torch.tensor(next_l, dtype=torch.float32),
        "q": torch.tensor(q_l, dtype=torch.float32),
    }
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(data, fh)
    print(f"Saved {len(obs_l)} transitions → {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--exits", type=int, choices=[1, 2], default=2)
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="ε-greedy noise in behaviour policy")
    parser.add_argument("--out", type=str, default="data/dataset.pkl")
    args = parser.parse_args()

    generate_dataset(args.episodes, args.exits, args.epsilon, Path(args.out))