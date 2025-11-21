import torch
import torch.nn.functional as F
from typing import Tuple, List, Callable, Optional

from dtsn.tree_node import TreeNode

# -----------------------------------------------------------------------------
# Differentiable Best‑First Search – stochastic & fully differentiable
# -----------------------------------------------------------------------------

class DTSNSearch:
    """Differentiable Tree‑Search module used by **D‑TSN**.

    Mirrors Appendix A of Mittal & Lee (2024): best‑first expansion, stochastic
    node policy, and optional REINFORCE hooks.
    """

    def __init__(self, encoder, transition, reward, value, *,
                 action_dim: int, max_iters: int = 10, temperature: float = 1.0):
        self.encoder = encoder
        self.transition = transition
        self.reward = reward
        self.value = value
        self.action_dim = action_dim
        self.max_iters = max_iters
        self.temperature = temperature

    # ------------------------------------------------------------------ public
    def search(self, obs_batch: torch.Tensor,
               step_hook: Optional[Callable[[torch.Tensor, int, int], None]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch search.

        Parameters
        ----------
        obs_batch : (B, obs_dim) tensor
        step_hook : optional ``hook(q_vec, b, t)`` to collect intermediate losses.

        Returns
        -------
        q_vecs   : (B, action_dim)
        log_probs: (B, max_iters)
        """
        if step_hook is None:
            q_vecs, logps = torch.vmap(
                lambda o: self._search_single(o, None), randomness="different"
            )(obs_batch)
            return q_vecs, torch.stack(logps, dim=0)

        # slower path with hooks
        qs, lps = [], []
        for b, obs in enumerate(obs_batch):
            q, lp = self._search_single(obs, lambda qv, t: step_hook(qv, b, t))
            qs.append(q); lps.append(lp)
        return torch.stack(qs), torch.stack(lps)

    # ------------------------------------------------------------------ single
    def _search_single(self, obs: torch.Tensor,
                       step_hook: Optional[Callable[[torch.Tensor, int], None]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = obs.device
        root = TreeNode(self.encoder(obs))
        open_set: List[TreeNode] = [root]
        log_probs: List[torch.Tensor] = []

        for t in range(self.max_iters):
            # ------------ select node -------------
            path_vals = torch.stack([
                (node.reward + self.value(node.latent)).view(())  # make 0-D
                for node in open_set
            ]).view(-1)                                     # (K,)
            probs = torch.softmax(path_vals / self.temperature, dim=0)
            m = torch.distributions.Categorical(probs)
            idx_t = m.sample()
            idx   = int(idx_t)   # keep; will now be safe

            log_probs.append(m.log_prob(idx_t))
            node = open_set.pop(int(idx_t))

            # ------------ expand all actions ------
            for a in range(self.action_dim):
                a_t = torch.tensor(a, device=device)
                h_next = self.transition(node.latent, a_t)
                r_next = self.reward(node.latent, a_t)
                child = TreeNode(latent=h_next,
                                reward=node.reward + r_next,
                                depth=node.depth + 1)
                node.children[a] = child
                open_set.append(child)

            if step_hook is not None:
                # Run a full backup on the partially expanded tree so L_t matches Eq. (6)
                self._backup(root)
                step_hook(self._root_q_vector(root, device), t)

        # ------------ backup ---------------------
        self._backup(root)
        q_vec = self._root_q_vector(root, device)
        return q_vec, torch.stack(log_probs)

    # ------------------------------------------------------------------ helpers
    def _backup(self, node: TreeNode) -> torch.Tensor:
        if node.is_leaf():
            return self.value(node.latent)
        q_children = []
        for child in node.children.values():
            r = child.reward - node.reward
            q_children.append(r + self._backup(child))
        node._q_vec = torch.stack(q_children)
        return node._q_vec.max()

    def _root_q_vector(self, root: TreeNode, device) -> torch.Tensor:
        q_vec = torch.full((self.action_dim,), float('-inf'), device=device)
        # for a, child in root.children.items():
        #     r = child.reward
        #     q_vec[a] = r + (self.value(child.latent) if child.is_leaf() else child._q_vec.max())
        for a, child in root.children.items():
            r = child.reward
            # if no q-vector has been backed up yet, fall back to the value function
            if child._q_vec is None:
                q = self.value(child.latent)
            else:
                q = child._q_vec.max()
            q_vec[a] = r + q
        return q_vec
