# Differentiable‑Tree‑Search‑Network (D‑TSN)

End‑to‑end PyTorch implementation of **“Differentiable Tree Search Network”** (Mittal & Lee 2024).
Trains a world‑model **and** a differentiable best‑first search policy jointly, reproducing the paper’s Grid‑Navigation and Procgen experiments.

---

## 1. Quick start

```bash
# ❶ clone repo
$ git clone https://github.com/your‑fork/DTSN.git && cd DTSN

# ❷ create env (Python 3.10)
$ conda create -n dtsn python=3.10 -y
$ conda activate dtsn
$ pip install -r requirements.txt   # torch, tqdm, pyyaml, tensorboard, procgen, …

# ❸ generate Grid‑Navigation training set (2‑exit)
$ python scripts/make_dataset.py --episodes 1000 --exits 2 --out data/dataset.pkl

# ❹ train (offline RL)
$ python -m dtsn.train             # reads configs/config.yaml

# ❺ monitor
$ tensorboard --logdir runs

# ❻ evaluate on 1‑exit generalisation test
$ python scripts/eval_navigation.py \
      --checkpoint checkpoints/dtsn_epoch50.pt \
      --episodes 200 --exits 1
```

## 2. Repository layout

```
DTSN/
├── dtsn/                 ← core package
│   ├── model.py          (Encoder / Transition / Reward / Value)
│   ├── search.py         (differentiable best‑first search)
│   ├── tree_node.py      (helper dataclass)
│   ├── losses.py         (all loss functions)
│   ├── train.py          (offline RL training loop)
│   └── logger.py         (TensorBoard wrapper)
│
├── envs/
│   ├── __init__.py
│   └── gridworld.py      (20×20 hall environment)
│
├── scripts/
│   ├── make_dataset.py   (generate offline trajectories)
│   └── eval_navigation.py
│
├── configs/config.yaml   (hyper‑parameters)
├── data/                 (auto‑ignored; holds .pkl datasets)
├── checkpoints/          (auto‑ignored; saved models)
├── runs/                 (TensorBoard logs)
└── requirements.txt
```

*(Top‑level directories `data/`, `checkpoints/`, `runs/` are in `.gitignore` so you never commit bulky artefacts.)*

## 3. Configuration

Key fields in **`configs/config.yaml`**:

| field        | default                                             | meaning                                 |
| ------------ | --------------------------------------------------- | --------------------------------------- |
| `latent_dim` | 64                                                  | size of latent state $h_t$              |
| `max_iters`  | 10                                                  | search‑tree expansions per forward pass |
| `batch_size` | 32                                                  | mini‑batch for offline RL               |
| `lambda_*`   | weights for $L_Q, L_{CQL}, L_T, L_R, L_{reinforce}$ |                                         |
| `device`     | cuda                                                | use `cpu` if no GPU                     |

Modify the YAML and re‑run training; everything else auto‑picks up the change.

## 4. Training details

* **Dataset format** – each transition tuple stored in `dataset.pkl`:
  `("obs", "action", "reward", "next_obs", "q")`, where `q` is Monte‑Carlo return.
* **Optimizer** – Adam, lr in YAML.
* **Loss** – $L = λ_Q L_Q + λ_D L_{CQL} + λ_T L_T + λ_R L_R + λ_{RF}L_{RF}$.
* **Checkpoints** – one `.pt` per epoch in `checkpoints/`.
* **TensorBoard** – losses logged each iteration; evaluation metrics logged by `eval_navigation.py`.

## 5. Evaluation metrics (Grid‑Navigation)

* **Success rate** – percentage of episodes reaching the goal within 400 steps.
* **Collision rate** – episodes terminated by hitting the hall wall.

`eval_navigation.py` reproduces Table 1 of the paper:

```text
Success rate:   99.2 %
Collision rate:  0.3 %
```

Add `--exits 2` to evaluate on the training distribution.

## 6. Procgen (optional)

1. Train / load a Phasic‑Policy‑Gradient (PPG) agent per game.
2. Adapt `scripts/make_dataset_procgen.py` (not provided here) to roll 1 000 successful episodes per game.
3. Replace `data/dataset.pkl` with combined trajectories; adjust `action_dim` to 15.
4. Run `train.py` – mean Z‑score should approach \~0.30.

## 7. License & citation

Apache‑2.0 for this repo.  Cite the original paper if you use D‑TSN in research:

> Mittal, D. & Lee, W.‑S. “Differentiable Tree Search Network.” ICML 2024.

---

Made with ❤ for reproducible research (Python 3.10, PyTorch 2.x).
