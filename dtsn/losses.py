import torch
import torch.nn.functional as F


def q_loss(pred_q, target_q):
    """L_Q: Supervised Q-value regression loss."""
    return F.mse_loss(pred_q, target_q)


def cql_loss(q_values, actions, action_dim):
    """L_D: Conservative Q-Learning loss.

    Args:
        q_values: Q(s, a) for actions in batch
        actions: true actions from dataset
        action_dim: total number of actions
    """
    logsumexp = torch.logsumexp(q_values, dim=-1)
    batch_indices = torch.arange(q_values.size(0), device=q_values.device)
    chosen_q = q_values[batch_indices, actions]
    return (logsumexp - chosen_q).mean()


def transition_consistency_loss(h_t, a_t, h_tp1_pred, h_tp1_target):
    """L_T: Transition model latent consistency loss."""
    return F.mse_loss(h_tp1_pred, h_tp1_target)


def reward_consistency_loss(r_pred, r_true):
    """L_R: Reward prediction consistency loss."""
    return F.mse_loss(r_pred, r_true)


def reinforce_term(log_probs, rewards):
    """REINFORCE gradient estimator with telescoping baseline."""
    # rewards: [T], log_probs: [T]
    assert len(log_probs) == len(rewards)
    returns = []
    R = 0
    for r in reversed(rewards):
        R = R + r
        returns.insert(0, R)

    loss = 0
    for log_prob, R in zip(log_probs, returns):
        loss += -log_prob * R  # negate for gradient descent
    return loss / len(log_probs)
