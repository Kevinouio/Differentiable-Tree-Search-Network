import torch


class TreeNode:
    """Lightweight container for a single node in the latent search tree."""

    def __init__(self, latent: torch.Tensor, reward: torch.Tensor | None = None, depth: int = 0):
        # Latent state for this node (shape: [latent_dim])
        self.latent: torch.Tensor = latent

        # Accumulated *tensor* reward along the path to this node so gradients can flow.
        if reward is None:
            reward = torch.zeros(1, device=latent.device, dtype=latent.dtype)
        self.reward: torch.Tensor = reward  # shape: [1]

        self.depth: int = depth
        self.children: dict[int, "TreeNode"] = {}

    # Convenience helpers
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self):
        return (
            f"TreeNode(depth={self.depth}, reward={self.reward.item():.3f}, "
            f"num_children={len(self.children)})"
        )
