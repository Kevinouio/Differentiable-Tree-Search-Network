class TreeNode:
    def __init__(self, latent, reward=0.0, depth=0):
        """
        A single node in the latent search tree.

        Args:
            latent (torch.Tensor): The latent state vector for this node.
            reward (float): Accumulated reward along the path to this node.
            depth (int): Depth of the node in the search tree.
        """
        self.latent = latent
        self.reward = reward
        self.depth = depth
        self.children = {}  # action -> TreeNode

    def is_leaf(self):
        return len(self.children) == 0

    def __repr__(self):
        return f"TreeNode(depth={self.depth}, reward={self.reward:.2f}, num_children={len(self.children)})"
