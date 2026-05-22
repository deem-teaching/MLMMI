# Exercise 3 — Optimizing a Model Serving System

**Deadline: 1st of June, 11.59pm**

In this exercise, you build a model serving system from scratch and quantify the effect of two classes of optimizations: black-box (request batching, prediction caching) and workload-aware (preprocessing sharing across models). The use case is fixed; the server, the benchmark, and the analysis are yours to design.

For inspiration:
- [Crankshaw et al., "Clipper: A Low-Latency Online Prediction Serving System" (NSDI 2017)](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf)
  Black-box prediction serving: request batching, prediction caching, model containerization. Closest match for Task 3.
- [Lee et al., "PRETZEL: Opening the Black Box of Machine Learning Prediction Serving Systems" (OSDI 2018)](https://www.usenix.org/system/files/osdi18-lee.pdf)
  White-box, pipeline-level optimization: shared operators across pipelines, end-to-end planning. Closest match for Task 5.

**Special Note: Use of LLMs/Coding Agents are allowed but not recommended.**

**Language:** Python. We expect FastAPI / Flask / aiohttp for the server, `sentence-transformers` + scikit-learn + PyTorch for the models, and any HTTP client for the load generator.

## Fixed use case

**Text topic classification** on the 20 Newsgroups dataset.

- **Preprocessing (the expensive shared step):** a forward pass through `sentence-transformers/all-MiniLM-L6-v2`, producing a 384-dim sentence embedding.
- **Heads:** four models trained on top of those embeddings, all interchangeable behind the same encoder:
  - `logreg` — logistic regression
  - `rf` — random forest
  - `hgb` — HistGradientBoosting (one-vs-rest)
  - `mlp` — small PyTorch MLP head (the DNN)

## Tasks

### Step 0 (0 pts) — train and register the four models

Train (or load) each of the four model classes on the same sentence embeddings and write a registry entry per model. How you arrive at the weights is up to you:

- random initialisation, no fitting,
- a single training run with sane defaults, or
- a small grid-search HPO.

The registry should be a directory of JSON files (one per model) containing at least: `model_id`, `algorithm`, `accuracy`, `model_path`. Your server in Task 1 loads from this registry at startup.

### Task 1 (2.0 pts) — Inference server baseline

Build the bare server. `POST /predict` taking `{ "text": "...", "model_id": "..." }` returns the prediction. It must:

- Load every model in the registry at startup.
- Maintain an explicit **request queue** between the HTTP handler and the model invocation path (even if the queue is effectively size 1 at this point — the abstraction will pay off in Task 3).
- Invoke the encoder and the requested head once per request.

No batching, no caching, no preprocessing sharing yet.

### Task 2 (2.0 pts) — RPS-sweep benchmark

Write a load generator that drives the server with **open-loop Poisson arrivals at a target rate**, for a fixed duration, and reports **mean** and **p95** latency. Use a fixed RNG seed so the experiment is reproducible.

Run it as an RPS sweep against the baseline server: increase the target arrival rate step by step until **latency takes off** — the throughput bottleneck. Plot **latency vs. RPS** (mean and p95).

### Task 3 (2.0 pts) — Generic ("black-box") optimizations

These optimizations don't depend on what the model does or what the workload looks like, only on the shape of the request API.

- **Request queue + micro-batching.** Gather pending requests for up to `max_wait_ms` or until `max_batch_size`, then run the encoder and the head on the whole batch in one go. Both knobs must be configurable.
- **Prediction cache.** LRU keyed by `(model_id, input)`. Expose hit / miss counts on `/metrics`.

Document your scheduling policy under overload (block vs. drop, FIFO vs. priority, what happens when a batch contains mixed `model_id`s).

### Task 4 (2.0 pts) — Rerun the benchmark across repeat rates

Rerun the RPS sweep from Task 2 against the optimized server, at **three repeat rates**: **0%, 10%, 20%** of the requests sampled from a small hot pool (the rest unique). Produce **one plot per repeat rate** (3 plots total). Each plot:

- X-axis: target requests per second.
- Y-axis: latency (mean and p95).
- One curve per server configuration: `baseline` and `+batching+cache`.

### Task 5 (2.0 pts) — Workload-aware optimization: preprocessing sharing

The encoder dominates wall time, and the same input text can legitimately be served by different heads (think: an A/B comparison, or a router that asks two cheap models and falls back to a third). Exploit this.

Implement **both** of:

- **Cross-model embedding reuse.** A separate LRU keyed by `text` alone. When `(model_id, text)` misses the prediction cache but the embedding of `text` is already known (because some earlier request used a different `model_id` on the same text), skip the encoder.
- **In-batch deduplication.** When the same `text` appears more than once in a single micro-batch — across one or several `model_id`s — pass it through the encoder exactly once and fan the embedding back out to every waiter.

Add a `+preproc share` curve to each of your three Task 4 plots and write a short analysis: where time is spent in the baseline, which mechanism each curve's improvement is attributable to, and at least one configuration where an optimization does **not** help (and why). Document the cache keys and the eviction policy.

## Deliverables

- Your server code, the load generator and the registry of trained models
- A `solution.ipynb` containing the **3 plots** (one per repeat rate, each showing mean + p95 latency vs. RPS, with one curve per server configuration: `baseline`, `+batching+cache`, `+preproc share`) and the **short analysis** for Task 5.
