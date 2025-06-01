"""dtsn.train
================
Batch-parallel offline RL training loop for Differentiable Tree Search Network.
Assumes dataset tensors are stored in `data/dataset.pkl`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

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


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _load_config(path: str | Path) -> SimpleNamespace:
    import yaml
    with open(path, "r", encoding="utf8") as fh:
        return SimpleNamespace(**yaml.safe_load(fh))


def _load_dataset(pkl_path: str | Path) -> TensorDataset:
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    return TensorDataset(d["obs"], d["action"], d["reward"], d["next_obs"], d["q"])


# ----------------------------------------------------------------------------
# main training entry
# ----------------------------------------------------------------------------

def train(cfg_path: str | Path = "configs/config.yaml"):
    cfg = _load_config(cfg_path)
    cfg.learning_rate = float(cfg.learning_rate)
    print(f"Using config: {cfg}")
    cfg.batch_size    = int(cfg.batch_size)
    cfg.max_iters     = int(cfg.max_iters)
    # (cast any other numeric field you might accidentally quote)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dataset = _load_dataset("data/dataset.pkl")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    obs_dim = dataset.tensors[0].shape[-1]
    action_dim = cfg.action_dim

    # build nets
    encoder = Encoder(obs_dim, cfg.latent_dim).to(device)
    transition = Transition(cfg.latent_dim, action_dim).to(device)
    reward_net = Reward(cfg.latent_dim, action_dim).to(device)
    value_net = Value(cfg.latent_dim).to(device)

    # target encoder for consistency
    enc_t = Encoder(obs_dim, cfg.latent_dim).to(device)
    enc_t.load_state_dict(encoder.state_dict())
    ema_tau = 0.005

    search = DTSNSearch(
        encoder, transition, reward_net, value_net,
        action_dim=action_dim, max_iters=cfg.max_iters, temperature=cfg.temperature,
    )

    params = list(encoder.parameters()) + list(transition.parameters()) + \
             list(reward_net.parameters()) + list(value_net.parameters())
    optim = torch.optim.Adam(params, lr=cfg.learning_rate)
    logger = TBLogger(run_name=f"gridworld_bs{cfg.batch_size}_lr{cfg.learning_rate}")
    global_step = 0


    # ------------------------------------------------------------------
    for epoch in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.epochs}")
        for obs, act, rew, next_obs, q_target in pbar:
            obs = obs.to(device)
            act = act.to(device)
            rew = rew.to(device)
            next_obs = next_obs.to(device)
            q_target = q_target.to(device)

            # ==== forward search (batched) ====
            # ==== forward search (batched, but looped) ====
            q_vecs_list, log_probs_list = [], []
            for obs_i in obs:                       # iterate over batch dim
                q_vec_i, log_probs_i = search._search_single(obs_i)
                q_vecs_list.append(q_vec_i)
                log_probs_list.append(log_probs_i)
            q_vecs  = torch.stack(q_vecs_list)       # (B, A)
            log_probs = torch.stack(log_probs_list)  # (B, T)

            # q_vecs: (B, A); log_probs: (B, T)

            # ==== losses ====
            # supervised Q for chosen actions
            chosen_q = q_vecs.gather(1, act.unsqueeze(1)).squeeze(1)
            loss_q = q_loss(chosen_q, q_target)
            loss_cql = cql_loss(q_vecs, act)

            # world-model consistency
            with torch.no_grad():
                h_next_true = enc_t(next_obs)
            h_pred = transition(encoder(obs), act)
            loss_t = transition_consistency_loss(h_pred, h_next_true)
            r_pred = reward_net(encoder(obs), act)
            loss_r = reward_consistency_loss(r_pred, rew)

            # REINFORCE term (optional) – use negative supervised loss diff
            reinforce_loss = torch.tensor(0.0, device=device)
            if cfg.lambda_reinforce > 0:
                # compute per-step BC loss (no grad) inside tree
                step_losses = []
                for t in range(cfg.max_iters):
                    step_losses.append(loss_q.detach())  # placeholder: proper L_t requires hook inside search
                rewards = [step_losses[0]] + [step_losses[i] - step_losses[i-1] for i in range(1, len(step_losses))]
                reinforce_loss = reinforce_term(log_probs.flatten(), torch.tensor(rewards, device=device)) * cfg.lambda_reinforce

            total = (cfg.lambda_q * loss_q + cfg.lambda_d * loss_cql +
                     cfg.lambda_t * loss_t + cfg.lambda_r * loss_r + reinforce_loss)

            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()

            # ==== logging ====

            logger.log(global_step, total_loss=total, q_loss=loss_q, cql=loss_cql, trans_cons=loss_t, rew_cons=loss_r,)
            global_step += 1
            

            # EMA update
            with torch.no_grad():
                for pt, p in zip(enc_t.parameters(), encoder.parameters()):
                    pt.data.mul_(1 - ema_tau).add_(p.data, alpha=ema_tau)

            pbar.set_postfix({"loss": f"{total.item():.4f}"})

        # checkpoint
        ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
        torch.save({
            "encoder": encoder.state_dict(),
            "transition": transition.state_dict(),
            "reward": reward_net.state_dict(),
            "value": value_net.state_dict(),
        }, ckpt_dir / f"dtsn_epoch{epoch+1}.pt")
    logger.close()



if __name__ == "__main__":
    train()