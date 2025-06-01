import torch
import torch.nn.functional as F
from dtsn.tree_node import TreeNode

class DTSNSearch:
    def __init__(self, encoder, transition, reward, value, action_dim,
                 max_iters=10, temperature=1.0):
        self.encoder = encoder
        self.transition = transition
        self.reward = reward
        self.value = value
        self.action_dim = action_dim
        self.max_iters = max_iters
        self.temperature = temperature

    def search(self, obs):
        h_root = self.encoder(obs)
        root = TreeNode(h_root)
        open_set = [root]

        for _ in range(self.max_iters):
            path_vals = torch.stack([
                node.reward + self.value(node.latent) for node in open_set
            ])

            probs = F.softmax(path_vals / self.temperature, dim=0)
            idx = torch.multinomial(probs, 1).item()
            node_to_expand = open_set.pop(idx)

            for a in range(self.action_dim):
                a_tensor = torch.tensor([a], device=obs.device)
                h_next = self.transition(node_to_expand.latent, a_tensor)
                r_next = self.reward(node_to_expand.latent, a_tensor)

                new_node = TreeNode(
                    latent=h_next,
                    reward=node_to_expand.reward + r_next.item(),
                    depth=node_to_expand.depth + 1
                )
                node_to_expand.children[a] = new_node
                open_set.append(new_node)

        return self._backup(root)

    def _backup(self, node):
        if not node.children:
            return self.value(node.latent)

        q_vals = []
        for a, child in node.children.items():
            r = self.reward(node.latent, torch.tensor([a], device=node.latent.device))
            q_val = r + self._backup(child)
            q_vals.append(q_val)

        return torch.stack(q_vals).max()
