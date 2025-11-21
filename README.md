# Differentiable Tree-Search Network (D-TSN)

This repository contains a PyTorch implementation of the Differentiable Tree-Search Network (D-TSN), an offline reinforcement learning (RL) agent that learns to reason over latent states using a differentiable tree structure. The model integrates value estimation, transition dynamics, and reward prediction in a unified framework, enabling powerful planning capabilities in a differentiable and end-to-end trainable manner.

## 🧠 Algorithm Overview

The D-TSN architecture consists of the following core components:

- **Encoder**: Maps observations to latent representations.
- **Transition Model**: Predicts the next latent state for each action.
- **Reward Model**: Predicts immediate rewards from latent transitions.
- **Value Model**: Estimates expected return from latent states.
- **Differentiable Tree Search (DTSNSearch)**: Builds a recursive latent search tree using learned models and aggregates values using softmax-weighted backups.

During training, D-TSN minimizes a combination of:

- Temporal-difference Q loss
- Conservative Q-learning (CQL) regularization
- Consistency losses for transitions and rewards
- A REINFORCE-style term derived from soft value backups

---

## 🖼 Architecture Diagram

![DTSN Architecture Placeholder](docs/dtsn_architecture.png)

> Replace this with a diagram showing the flow from `obs → latent → tree → value`, with branches showing transitions and value aggregation at each depth.

---

## 🧪 Running the Code

### 🛠 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Differentiable-Tree-Search-Network.git
   cd Differentiable-Tree-Search-Network
   ```

2. Create a conda environment:

   ```bash
   conda create -n DSTN python=3.10
   conda activate DSTN
   pip install -r requirements.txt
   ```

3. (Optional) Add your own offline dataset under `data/`.

---

### 🚆 Training

Train the D-TSN model on a dataset:

```bash
python dtsn/train.py --config configs/nav1.yaml
```

This will save checkpoints under `checkpoints/` and logs to TensorBoard under `runs/`.

---

### 🔍 Evaluation

Evaluate a trained model on a navigation environment:

```bash
python scripts/eval_navigation.py \
    --checkpoint checkpoints/dtsn_epoch50.pt \
    --episodes 200 --exits 1
```

You can adjust `--exits`, `--episodes`, and other evaluation parameters as needed.

---

## 📁 Repo Structure

```
dtsn/
├── model.py             # Encoder, Transition, Reward, Value
├── search.py            # Differentiable tree search logic
├── losses.py            # Loss functions including REINFORCE term
├── logger.py            # TensorBoard logger
├── tree_node.py         # Tree node data structure
├── train.py             # Training script
scripts/
├── eval_navigation.py   # Evaluation script
configs/
├── nav1.yaml            # Sample config file
data/
├── ...                  # Offline datasets
```

---

## 📄 Citation

If you use this codebase or ideas from this implementation, please cite the original D-TSN paper:


---

## 📜 License

MIT License. See [`LICENSE`](LICENSE) for details.

---
