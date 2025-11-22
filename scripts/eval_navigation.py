# scripts/eval_navigation.py
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# Allow running as `python scripts/eval_navigation.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
@torch.no_grad()
def evaluate(agent: DTSNSearch, env: GridWorld, episodes: int, device,
             render: bool = False, max_steps: int | None = None):
    """Return success-rate and collision-rate over N episodes. If render=True, stream episodes."""
    success, collisions, timeouts = 0, 0, 0

    ep_iter = range(episodes) if render else tqdm(range(episodes), desc="Evaluating episodes")
    for ep in ep_iter:
        obs_np = env.reset()                                    # fresh episode
        obs    = torch.tensor(obs_np, dtype=torch.float32, device=device)
        steps = 0
        timed_out = False

        if render:
            print(f"\nEpisode {ep+1}/{episodes} start -> state={env.state}, goal={env.goal}")

        done = False
        while not done:
            q_vec, _ = agent._search_single(obs)
            act = int(torch.argmax(q_vec).item())

            obs_np, _, done, info = env.step(act)
            obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            steps += 1

            if render:
                status = "collided" if info.get("collided", False) else "step"
                print(f"  step {steps:03d}: act={act}, state={env.state}, status={status}")

            if info.get("collided", False):
                collisions += 1
                break  # terminate early on collision

            if max_steps is not None and steps >= max_steps:
                timed_out = True
                break

        if np.all(env.state == env.goal):
            success += 1
            if render:
                print(f"Episode {ep+1} SUCCESS in {steps} steps.")
        else:
            if timed_out:
                timeouts += 1
            if render:
                outcome = "TIMEOUT" if timed_out else "FAILURE"
                print(f"Episode {ep+1} {outcome} in {steps} steps.")

    return success / episodes, collisions / episodes, timeouts / episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--exits", type=int, choices=[1, 2], default=1)
    parser.add_argument("--logdir", default="runs/eval_nav")
    parser.add_argument("--render", action="store_true",
                        help="Stream per-step actions/states instead of only a progress bar.")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Per-episode step cap during evaluation (default: 50).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorld(exits=args.exits)

    agent = load_agent(
        Path(args.checkpoint),
        latent_dim=64,          # keep in sync with training
        action_dim=4,
        device=device,
    )
    agent.max_iters = 20

    succ, coll, tout = evaluate(agent, env, args.episodes, device,
                                render=args.render, max_steps=args.max_steps)
    print(f"Success rate:   {succ*100:.1f} %")
    print(f"Collision rate: {coll*100:.1f} %")
    print(f"Timeout rate:   {tout*100:.1f} %")

    # TensorBoard
    writer = SummaryWriter(log_dir=args.logdir)
    writer.add_scalar("success_rate", succ, 0)
    writer.add_scalar("collision_rate", coll, 0)
    writer.close()
