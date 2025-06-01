import torch
import torch.nn.functional as F
from typing import Tuple, List

from dtsn.tree_node import TreeNode

# -----------------------------------------------------------------------------
# Batched Differentiable Best-First Search (vmap friendly)
# -----------------------------------------------------------------------------

class DTSNSearch:
    """Batched, fully-differentiable tree search used by D-TSN.

    * `search(obs_batch)` accepts **(B, obs_dim)** and returns:
        • `q_vecs`  – tensor (B, action_dim)
        • `log_probs` – tensor (B, max_iters)  (log π for REINFORCE)

    The implementation wraps the *single-trajectory* logic with
    `torch.vmap`, so you get GPU-vectorised expansion without rewriting
    the whole algorithm for batched open-sets.
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

    # ------------------------------------------------------------------
    # public API – batch search
    # ------------------------------------------------------------------
    def search(self, obs_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorised search over a batch of observations.

        Parameters
        ----------
        obs_batch : torch.Tensor (B, obs_dim)
            Batch of raw observations.

        Returns
        -------
        q_vecs : torch.Tensor (B, action_dim)
        log_probs : torch.Tensor (B, max_iters)
        """
        # vmap over the 0-dim using a closure around *self*
        q_vecs, logps = torch.vmap(self._search_single,in_dims=0,out_dims=0,randomness="different")(obs_batch)
        # stack log_probs list → (B, T)
        log_probs = torch.stack(logps, dim=0)
        return q_vecs, log_probs

    # ------------------------------------------------------------------
    # single-trajectory search (was the old `search` method)
    # ------------------------------------------------------------------
    def _search_single(self, obs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Search for a single observation (no batch dim)."""
        device = obs.device
        h_root = self.encoder(obs)
        root = TreeNode(h_root)
        open_set = [root]
        log_probs: List[torch.Tensor] = []
        assert self.max_iters > 0, "max_iters is zero"


        # ---------------- Expansion ----------------
        for _ in range(self.max_iters):
            # --- before sampling ---------------------------------
            path_vals = torch.stack([
                node.reward.detach() + self.value(node.latent)     # (1,) tensor
                for node in open_set
            ]).squeeze(-1)      # <- NEW: remove the extra dim so shape == (num_open,)

            probs = torch.softmax(path_vals / self.temperature, dim=0)
            m     = torch.distributions.Categorical(probs)
            idx_t = m.sample()
            log_probs.append(m.log_prob(idx_t))      # <- NEW
            idx   = int(idx_t)                       # python int

            node  = open_set.pop(idx)


            # expand all actions
            for a in range(self.action_dim):
                a_t = torch.tensor(a, device=device)
                h_next = self.transition(node.latent, a_t)
                r_next = self.reward(node.latent, a_t)
                child = TreeNode(
                    latent=h_next,
                    reward=node.reward + r_next,
                    depth=node.depth + 1,
                )
                node.children[a] = child
                open_set.append(child)

        # ---------------- Backup ----------------
        self._backup(root)
        q_vec = self._root_q_vector(root, device)
        return q_vec, torch.stack(log_probs)

    # ------------------------------------------------------------------
    # helpers (unchanged logic)
    # ------------------------------------------------------------------
    def _backup(self, node: TreeNode) -> torch.Tensor:
        if node.is_leaf():
            return self.value(node.latent)
        q_children = []
        for a, child in node.children.items():
            r = child.reward - node.reward  # immediate reward tensor
            q_children.append(r + self._backup(child))
        node._q_vec = torch.stack(q_children)  # save for parent use
        return node._q_vec.max()

    def _root_q_vector(self, root: TreeNode, device) -> torch.Tensor:
        q_vec = torch.full((self.action_dim,), float('-inf'), device=device)
        for a, child in root.children.items():
            r = child.reward  # root reward is 0
            if child.is_leaf():
                q_vec[a] = r + self.value(child.latent)
            else:
                q_vec[a] = r + child._q_vec.max()
        return q_vec