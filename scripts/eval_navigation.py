# scripts/eval_navigation.py
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from envs.gridworld import GridWorld
from dtsn.model import Encoder, Transition, Reward, Value
from dtsn.search import DTSNSearch


def load_agent(ckpt_path: Path, latent_dim: int, action_dim: int, device):
    # create empty nets with correct shapes then load weights
    dummy_obs_dim = 4
    enc = Encoder(dummy_obs_dim, latent_dim).to(device)
    trn = Transition(latent_dim, action_dim).to(device)
    rew = Reward(latent_dim, action_dim).to(device)
    val = Value(latent_dim).to(device)

    sd = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(sd["encoder"])
    trn.load_state_dict(sd["transition"])
    rew.load_state_dict(sd["reward"])
    val.load_state_dict(sd["value"])

    return DTSNSearch(enc, trn, rew, val, action_dim=action_dim,
                      max_iters=10, temperature=1.0)


@torch.no_grad()
def evaluate(agent: DTSNSearch, env: GridWorld, episodes: int, device):
    """Return success-rate and collision-rate over N episodes."""
    success, collisions = 0, 0

    for _ in range(episodes):
        obs_np = env.reset()                                    # fresh episode
        obs    = torch.tensor(obs_np, dtype=torch.float32, device=device)

        done = False
        while not done:
            q_vec, _ = agent._search_single(obs)
            act = int(torch.argmax(q_vec).item())

            obs_np, _, done, info = env.step(act)
            obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

            if info.get("collided", False):
                collisions += 1
                break  # terminate early on collision

        if np.all(env.state == env.goal):
            success += 1

    return success / episodes, collisions / episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--exits", type=int, choices=[1, 2], default=1)
    parser.add_argument("--logdir", default="runs/eval_nav")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorld(exits=args.exits)

    agent = load_agent(
        Path(args.checkpoint),
        latent_dim=64,          # keep in sync with training
        action_dim=4,
        device=device,
    )

    succ, coll = evaluate(agent, env, args.episodes, device)
    print(f"Success rate:   {succ*100:.1f} %")
    print(f"Collision rate: {coll*100:.1f} %")

    # TensorBoard
    writer = SummaryWriter(log_dir=args.logdir)
    writer.add_scalar("success_rate", succ, 0)
    writer.add_scalar("collision_rate", coll, 0)
    writer.close()
