"""dtsn.losses
================
Utility loss functions used to train Differentiable‑Tree‑Search‑Network.
All losses operate on *tensors* so gradients propagate end‑to‑end.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Supervised Q‑value regression (behaviour‑cloning)
# -----------------------------------------------------------------------------

def q_loss(pred_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    """L_Q – mean‑squared error between predicted & target Q(s,a)."""
    return F.mse_loss(pred_q, target_q)


# -----------------------------------------------------------------------------
# Conservative Q‑Learning penalty to down‑weight out‑of‑distribution actions
# -----------------------------------------------------------------------------

def cql_loss(q_sa: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """L_D – Conservative Q‑Learning soft‑max gap.

    Args
    -----
    q_sa : (batch, action_dim) tensor of Q(s, a) for *all* discrete actions.
    actions : (batch,) tensor of integer actions sampled from the dataset.
    """
    logsumexp = torch.logsumexp(q_sa, dim=-1)  # soft‑max over actions
    idx = torch.arange(q_sa.size(0), device=q_sa.device)
    chosen_q = q_sa[idx, actions]
    return (logsumexp - chosen_q).mean()


# -----------------------------------------------------------------------------
# World‑model consistency losses
# -----------------------------------------------------------------------------

def transition_consistency_loss(h_pred: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    """L_T – latent‑transition consistency (Eq. 19)."""
    return F.mse_loss(h_pred, h_true.detach())  # detach to stop gradients through target encoder


def reward_consistency_loss(r_pred: torch.Tensor, r_true: torch.Tensor) -> torch.Tensor:
    """L_R – one‑step reward prediction loss (Eq. 20)."""
    return F.mse_loss(r_pred, r_true)


# -----------------------------------------------------------------------------
# REINFORCE term with telescoping‑sum baseline for node‑selection policy
# -----------------------------------------------------------------------------

def reinforce_term(log_probs, step_rewards) -> torch.Tensor:
    """Variance-reduced REINFORCE objective (Eq. 11).

    Args
    -----
    log_probs : tensor/list of log π_θ(n_t | τ_t) for each expansion step.
                Shape (T,) or (B, T).
    step_rewards : per-step rewards r_t = L_t − L_{t‑1}. Accepts list, (T,) or (B, T).
    """
    # coerce to tensors
    rewards = torch.stack(step_rewards, dim=0) if isinstance(step_rewards, list) else step_rewards
    lp = torch.stack(log_probs, dim=0) if isinstance(log_probs, list) else log_probs

    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(0)  # (1, T)
    if lp.dim() == 1:
        lp = lp.unsqueeze(0)            # (1, T)

    assert lp.shape == rewards.shape

    # returns R_t = sum_{i=t..T} r_i, computed along the last axis
    returns = torch.flip(torch.cumsum(torch.flip(rewards, dims=[-1]), dim=-1), dims=[-1])
    return -(lp * returns.detach()).mean()
