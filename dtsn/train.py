"""dtsn.train
================
Offline‑RL training loop for Differentiable Tree‑Search Network.
Now uses the *hook* API in `search.search()` to compute the true
REINFORCE term from Section 3.6 → 3.7 of the paper.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# Allow running as `python dtsn/train.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtsn.model import Encoder, Transition, Reward, Value
from dtsn.search import DTSNSearch
from dtsn.losses import (
    q_loss,
    cql_loss,
    transition_consistency_loss,
    reward_consistency_loss,
    reinforce_term,
)
from dtsn.logger import TBLogger

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _load_config(path: str | Path) -> SimpleNamespace:
    import yaml
    with open(path, "r", encoding="utf8") as fh:
        return SimpleNamespace(**yaml.safe_load(fh))


def _load_dataset(pkl_path: str | Path) -> TensorDataset:
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    return TensorDataset(d["obs"], d["action"], d["reward"], d["next_obs"], d["q"])


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def train(cfg_path: str | Path = "configs/config.yaml"):
    # ---------- config ----------
    cfg = _load_config(cfg_path)
    cfg.learning_rate = float(cfg.learning_rate)
    cfg.batch_size = int(cfg.batch_size)
    cfg.max_iters = int(cfg.max_iters)
    print("Using config:", cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---------- data ----------
    dataset = _load_dataset(cfg.dataset_path)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    obs_dim = dataset.tensors[0].shape[-1]
    action_dim = cfg.action_dim

    # ---------- networks ----------
    encoder = Encoder(obs_dim, cfg.latent_dim).to(device)
    transition = Transition(cfg.latent_dim, action_dim).to(device)
    reward_net = Reward(cfg.latent_dim, action_dim).to(device)
    value_net = Value(cfg.latent_dim).to(device)

    # EMA target encoder (for consistency loss)
    enc_t = Encoder(obs_dim, cfg.latent_dim).to(device)
    enc_t.load_state_dict(encoder.state_dict())
    ema_tau = cfg.ema_tau

    search = DTSNSearch(
        encoder, transition, reward_net, value_net,
        action_dim=action_dim, max_iters=cfg.max_iters, temperature=cfg.temperature,
    )

    params = list(encoder.parameters()) + list(transition.parameters()) + \
             list(reward_net.parameters()) + list(value_net.parameters())
    optim = torch.optim.Adam(params, lr=cfg.learning_rate)

    # ---------- logging ----------
    logger = TBLogger(run_name=f"grid_bs{cfg.batch_size}_lr{cfg.learning_rate}")
    global_step = 0

    # ------------------------------------------------------------------ epochs
    for epoch in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.epochs}")
        for obs, act, rew, next_obs, q_target in pbar:
            obs, act = obs.to(device), act.to(device)
            rew, next_obs, q_target = rew.to(device), next_obs.to(device), q_target.to(device)

            # ------------------------------------------------ search (with hook)
            step_losses: list[list[torch.Tensor]] = [[] for _ in range(cfg.max_iters)]

            def hook(q_vec: torch.Tensor, b: int, t: int):
                """Collect λ1 L_Q + λ2 L_D per expansion (for Eq. 11 telescope)."""
                chosen = q_vec[act[b]]
                l_q = F.mse_loss(chosen, q_target[b], reduction='none')
                l_d = torch.logsumexp(q_vec, dim=0) - chosen
                step_losses[t].append(cfg.lambda_q * l_q + cfg.lambda_d * l_d)

            q_vecs, log_probs = search.search(obs, step_hook=hook)
            # q_vecs (B,A)  log_probs (B,T)

            # ------------------------------------------------ losses
            chosen_q = q_vecs.gather(1, act.unsqueeze(1)).squeeze(1)
            loss_q = q_loss(chosen_q, q_target)
            loss_cql = cql_loss(q_vecs, act)

            with torch.no_grad():
                h_next_true = enc_t(next_obs)
            h_pred = transition(encoder(obs), act)
            loss_t = transition_consistency_loss(h_pred, h_next_true)
            r_pred = reward_net(encoder(obs), act)
            loss_r = reward_consistency_loss(r_pred, rew)

            # ---------- REINFORCE term ----------
            reinforce_loss = torch.tensor(0.0, device=device)
            if cfg.lambda_reinforce > 0:
                # stack to (batch, T) so Rt = L_T − L_{t−1} per trajectory
                step_loss_tensor = torch.stack([torch.stack(sl) for sl in step_losses], dim=1)
                step_rewards = torch.zeros_like(step_loss_tensor)
                step_rewards[:, 0] = step_loss_tensor[:, 0]
                step_rewards[:, 1:] = step_loss_tensor[:, 1:] - step_loss_tensor[:, :-1]
                reinforce_loss = reinforce_term(log_probs, step_rewards) * cfg.lambda_reinforce

            total = (cfg.lambda_q * loss_q + cfg.lambda_d * loss_cql +
                     cfg.lambda_t * loss_t + cfg.lambda_r * loss_r + reinforce_loss)

            # ------------------------------------------------ optimise
            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step()

            # EMA
            with torch.no_grad():
                for pt, p in zip(enc_t.parameters(), encoder.parameters()):
                    pt.data.mul_(1 - ema_tau).add_(p.data, alpha=ema_tau)

            # log
            logger.log(global_step, total_loss=total, q_loss=loss_q,
                       cql=loss_cql, trans_cons=loss_t, rew_cons=loss_r)
            global_step += 1
            pbar.set_postfix(loss=f"{total.item():.4f}")

        # checkpoint each epoch
        ckpt = {
            "encoder": encoder.state_dict(),
            "transition": transition.state_dict(),
            "reward": reward_net.state_dict(),
            "value": value_net.state_dict(),
        }
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(ckpt, Path("checkpoints") / f"dtsn_epoch{epoch+1}.pt")

    logger.close()


if __name__ == "__main__":
    train()
