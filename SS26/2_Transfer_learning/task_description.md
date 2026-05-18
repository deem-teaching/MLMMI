# Exercise 2 — Transfer Learning

**Deadline: 18.05.2026**

In this exercise, you will study how much prior knowledge from ImageNet actually helps on a new classification task, and what it costs in compute. You will take a small vision model (**ResNet18**), apply it to a small dataset (**CIFAR-10 subset**), and compare three training regimes on the same data: **from scratch**, **full fine-tuning**, and **partial transfer learning** with a varying number of frozen layers. You will measure both **accuracy** and **execution time**, then propose and implement at least one optimization opportunity and measure its effect.

The goal is not to chase the final bit of accuracy, but to clearly map the accuracy-vs-compute trade-off of layer freezing and justify how you choose a freeze point.

For inspiration, you may read:  
- [Yosinski et al., "How transferable are features in deep neural networks?"](https://proceedings.neurips.cc/paper_files/paper/2014/file/532a2f85b6977104bc93f8580abbb330-Paper.pdf)
This is the main conceptual reference for the layer-freezing experiments.
- [Kornblith et al., "Do Better ImageNet Models Transfer Better?"](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kornblith_Do_Better_ImageNet_Models_Transfer_Better_CVPR_2019_paper.pdf)
This gives broader context on ImageNet pretraining, fixed feature extractors, fine-tuning, and training from scratch.
- [Nakandala et al., "VISTA: Optimized System for Declarative Feature Transfer from Deep CNNs at Scale"](https://dl.acm.org/doi/pdf/10.1145/3318464.3389709)
A system to reduce computational redundancy in transfer learning workloads.
- [Renggli et al., "SHiFT: An Efficient, Flexible Search Engine for Transfer Learning"](https://www.vldb.org/pvldb/vol16/p304-renggli.pdf)
An efficient model search engine for transfer learning.

**Special Note: Use of LLMs/Coding Agents are allowed but not recommended.**

**Language:** Python with PyTorch. Transfer learning in this exercise relies on pretrained `torchvision.models.resnet18` weights and layer-level freezing, which are idiomatic in PyTorch; constraining the framework makes results comparable across submissions.

## Example Project Structure

```
2_Transfer_learning/
├── requirements.txt          # or pyproject.toml, uv.lock, etc
├── solution.ipynb            # main notebook orchestrating the pipeline (calling the scripts, producing plots)
├── data/
│   ├── raw/                  # CIFAR-10 as downloaded by torchvision
│   └── subset.json           # reproducible train/test index split
├── models/
│   └── <run_id>.pt           # one checkpoint per training run
├── registry/
│   └── <run_id>.json         # one metadata file per training run (extends Exercise 1's schema)
└── scripts/
    ├── prepare_data.py       # download CIFAR-10 and materialize the subset
    ├── train.py              # parametrized trainer: --init, --freeze, --epochs, --amp, --cache-features, …
    └── evaluate.py           # aggregate registry/*.json into a comparison table
```

## Tasks

**Common training budget.** Use a **fixed epoch budget** (e.g. 5 or 10 epochs) for all main comparison runs (Tasks 1–3). Keep batch size, learning rate, optimizer, and image resolution constant across these runs unless a deviation is part of what you are explicitly studying. This keeps differences in accuracy and training time attributable to the freezing configuration rather than to incidental hyperparameter changes.

### 1. Data preparation and from-scratch baseline (1.5 pts)
- Download CIFAR-10 and materialize a **reproducible subset** (e.g. 10,000 training and 2,000 test samples) via a fixed random seed. Persist the sampled indices so the subset is reproducible across runs.
- Train a `torchvision.models.resnet18` **initialized from scratch** (`weights=None`) on this subset. Replace the final layer with `nn.Linear(512, 10)`. Document your input-resolution choice (e.g. upsample 32×32 → 224×224 to stay compatible with the pretrained-weights runs).
- Register the run (see registry schema below).

### 2. Full fine-tuning from ImageNet weights (1.5 pts)
- Train the same architecture, but initialized from ImageNet pretrained weights (`ResNet18_Weights.IMAGENET1K_V1`), with **all layers unfrozen**. Use ImageNet normalization on the inputs.
- Register the run and compare accuracy and training time against Task 1.

### 3. Partial freezing experiments and comparison (5.0 pts)
- Implement layer-level freezing: given a list of block names (subset of `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4`), set `requires_grad = False` on their parameters and pass only the remaining parameters to the optimizer.
- Document and justify your policy for **BatchNorm running statistics** inside frozen blocks (train mode vs. eval mode).
- Run **at least 3 configurations** with varying freeze depth, for example:
  - early features frozen
  - most of the backbone frozen
  - only linear probe
- Register each run. Briefly describe the **heuristic** you used to choose freeze points (e.g. based on feature-generality arguments, trainable parameter budget, empirical search).
- Produce a single comparison **table** across all runs (Tasks 1–3) with: `run_id`, init, frozen blocks, test accuracy, training time (s), inference latency (ms, averaged over a 100-sample batch, ≥10 repeats), and trainable parameter count.

### 4. Optimization opportunities (2.0 pts)
- **Propose at least 2** optimization opportunities that could reduce execution time.
- **Implement at least 1** of them and compare it with the unoptimized version: report speedup and any accuracy delta. Register each optimized run with a unique identifier in the registry.

## Registry Schema

Each run writes one JSON file to `registry/<run_id>.json`. This should extend your Exercise 1 schema. e.g.:

```json
{
  "run_id": "ft_mid",
  "init": "pretrained",
  "frozen_layers": ["conv1", "layer1", "layer2"],
  "epochs": 10,
  "optimizer": "Adam",
  "lr": 0.001,
  "batch_size": 64,
  "image_size": 224,
  "optimizations": ["amp"],
  "accuracy": 0.812,
  "train_time_s": 184.3,
  "inference_time_ms": 21.5,
  "trainable_params": 4800010,
  "total_params": 11181642,
  "device": "cuda",
  "parent_run_id": "ft_full",
  "model_path": "models/ft_mid.pt",
  "created_at": "2026-05-01T12:00:00Z"
}
```

## Deliverables

Submit the filled-in project structure above: the three scripts (or equivalent), a `solution.ipynb` that runs the pipeline end-to-end and renders the comparison table and a populated `registry/*.json` files for every run (from-scratch, full fine-tune, at least 3 partial-freeze configurations, and at least 1 optimized run).
