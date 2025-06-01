# make_dataset.py
# ------------------------------------------------------------
# Generates an offline dataset for D-TSN training
# ------------------------------------------------------------
import pickle
from pathlib import Path

import numpy as np
import torch


# -----------------------------------------------------------------
# Simple 20×20 grid-world with one exit (replace with Procgen if you like)
# -----------------------------------------------------------------
class GridWorld:
    ACTIONS = {0: np.array([0, 1]),   # up
               1: np.array([1, 0]),   # right
               2: np.array([0, -1]),  # down
               3: np.array([-1, 0])}  # left

    def __init__(self, size=20):
        self.size = size
        self.hall_mask = self._hall_mask()

    def _hall_mask(self):
        m = np.zeros((self.size, self.size), dtype=bool)
        m[self.size//2-1:self.size//2+1, :] = True
        return m

    def reset(self):
        # start inside the hall
        while True:
            s = np.random.randint(0, self.size, 2)
            if self.hall_mask[tuple(s)]:
                break
        # random goal outside the hall
        while True:
            g = np.random.randint(0, self.size, 2)
            if not self.hall_mask[tuple(g)]:
                break
        self.state, self.goal = s, g
        return self._obs()

    def _obs(self):
        # concatenate agent coords and goal coords → obs_dim = 4
        return np.concatenate([self.state / self.size,
                               self.goal  / self.size]).astype(np.float32)

    def step(self, action):
        nxt = self.state + self.ACTIONS[action]
        # keep inside grid
        nxt = np.clip(nxt, 0, self.size - 1)
        reward = -1.0
        done = False
        # collide with wall around hall
        if self.hall_mask[tuple(nxt)]:
            self.state = nxt
        else:
            # must exit through opening (col 10 here)
            if self.state[1] == self.size//2 - 1 and nxt[1] == self.size//2 and nxt[0] == 10:
                self.state = nxt
            elif not self.hall_mask[tuple(self.state)]:
                self.state = nxt
        if np.all(self.state == self.goal):
            reward = 0.0
            done = True
        return self._obs(), reward, done, {}

    # deterministic expert: A* on Manhattan grid
    def optimal_action(self):
        dx, dy = self.goal - self.state
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3
        else:
            return 0 if dy > 0 else 2


# -----------------------------------------------------------------
# Generate trajectories
# -----------------------------------------------------------------
NUM_EPISODES = 1000
env = GridWorld()
obs_buf, act_buf, rew_buf, next_buf, q_buf = [], [], [], [], []

for _ in range(NUM_EPISODES):
    s = env.reset()
    episode = []
    terminal = False
    while not terminal:
        a = env.optimal_action()
        s2, r, terminal, _ = env.step(a)
        episode.append((s, a, r, s2))
        s = s2

    # Monte-Carlo return
    G = 0.0
    for s, a, r, s2 in reversed(episode):
        G += r
        obs_buf.append(s)
        act_buf.append(a)
        rew_buf.append(r)
        next_buf.append(s2)
        q_buf.append(G)

# -----------------------------------------------------------------
# Dump tensors
# -----------------------------------------------------------------
dataset = {
    "obs":       torch.tensor(obs_buf, dtype=torch.float32),
    "action":    torch.tensor(act_buf, dtype=torch.long),
    "reward":    torch.tensor(rew_buf, dtype=torch.float32),
    "next_obs":  torch.tensor(next_buf, dtype=torch.float32),
    "q":         torch.tensor(q_buf, dtype=torch.float32),
}

Path("data").mkdir(exist_ok=True)
with open("data/dataset.pkl", "wb") as fh:
    pickle.dump(dataset, fh)

print("Dataset saved to data/dataset.pkl")
