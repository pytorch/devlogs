---
title: "Graph-based CPU Offloading for TorchTitan Frontier Model Training"
date: 2026-06-23
author: "Michael Lazos (@mlazos)"
tags: [torchtitan, distributed, memory, performance, torch.compile]
---

> **TL;DR** – We added a graph-based CPU activation-offloading pass to torchtitan's graph_trainer with agent-tunable knobs and a user-customizable offload policy function. On dense models you can reclaim 10% of peak memory for under 1% throughput loss, scaling to 33% (Llama3) / 38% (Qwen3) with 20% throughput loss. Our implementation achieves SOL PCIe transfer bandwidth of ~300 GB/s.

## Background: Frontier Training, MoE, and Activation Checkpointing

In model training, the forward pass is run followed by the backward pass to perform updates on the model parameters according to the error gradients. The backward pass computes gradients via the chain rule, and to do so it needs the activations the forward pass produced. Every activation must therefore still be available when backward reaches it, which leaves two options: **save** it in GPU memory (costs HBM) or **recompute** it during backward (costs FLOPs).

Activation checkpointing &mdash; saving some activations and recomputing the rest, often selectively (SAC) &mdash; is the standard policy for trading one against the other.

Activation memory scales with batch size, sequence length, hidden dimension, and depth. For frontier models with long context and large micro-batches it routinely exceeds parameters and optimizer states combined, and is the largest single consumer of HBM per step.

Mixture-of-Experts models differ in that they carry many more parameters but activate only a few experts per token, so they run with expert parallelism and the all-to-all communication that comes with it. As a result, activations for these type of models tend to be smaller at a given scale.

### How Offloading Fits into Existing Memory Policies

Activation offloading is a third option alongside save and recompute. Instead of keeping an activation in HBM or discarding it, you move it to CPU DRAM while it sits idle between forward and backward, then stream it back over PCIe before backward needs it. The three policies spend different resources: saving spends HBM, recompute spends FLOPs, offload spends PCIe bandwidth. They compose &mdash; offload sits on top of SAC and they can operate on exclusive groups of tensors.

Offloading's appeal is that, unlike recompute, it can be nearly free: if the copy overlaps with compute, you pay almost nothing in wall-clock time. The limit is bandwidth. Transfers utilize the PCIe link, whose peak bandwidth is 300 GB/s. As long as there is more idle compute than traffic, transfers hide completely; once traffic approaches that ceiling, they serialize and throughput drops.

Before this work, CPU offloading was limited to eager which utilizes saved tensor hooks &mdash; these would offload and reload tensors at the site of save and reuse in the backward. This works, but is often suboptimal because it does not enable the global reordering and selection optimizations that a graph-based pass provides.

## Our Contributions

### A Graph-based Offloading Pass

The pass operates on the traced graph rather than reacting to tensors at runtime. With the whole forward and backward visible ahead of time, it can decide what to offload, when to start each copy-back so it lands just in time, and how to handle activations that are views into shared storage.

### View Replay

Activations are often views &mdash; reshapes, slices, transposes &mdash; that share storage with a base tensor rather than standalone allocations. A naive offloader can't move a view without orphaning everything else pointing at that storage, so only materialized base tensors are eligible, capping savings at about 4% of peak memory for Llama 8B.

View replay offloads the base storage and records the chain of view ops, replaying them after reload to reconstruct the view chain before use in the backward. Making views eligible for reload raises the ceiling to ~33% for Llama 8B.

### Pinned Memory Pool in PyTorch Core

H2D/D2H transfers want pinned (page-locked) host memory for full PCIe bandwidth, but allocating it per tensor via `cudaHostAlloc` costs ~3 ms each &mdash; roughly 465 ms/step across 155 tensors on Llama3 8B, and most of the early ~50% overhead.

A reusable pinned pool (`offload_ops.py`, the `pinned_memory_pool()` context manager) allocates once in the first training step and recycles buffers on subsequent steps, lowering overhead by 22%. A backward keepalive list returns each buffer to the pool once its H2D copy completes, so the pool doesn't drain until the context is exited.

### Automatic NUMA Detection in GraphTrainer

During initialization, the graph trainer applies NUMA binding using PyTorch's NUMA support. It first determines which NUMA node the GPU is attached to by reading the GPU's PCI address from the operating system's device information. It then restricts every thread in the process to run only on the CPU cores belonging to that node.

The binding is applied at the node level and is best-effort: if it cannot be established, training proceeds without it. The main purpose is to achieve maximum bandwidth &mdash; when pinned host memory resides on the same NUMA node as the GPU, device-to-host transfers reach roughly 350 GB/s, compared with about 120 GB/s when a transfer has to cross NUMA nodes.

### Three Configuration Knobs: Budget, Defer, Prefetch

**Budget** is the target amount of memory to offload; we offload the largest tensors until that budget is reached.

**Defer** is how many layers to wait before the offload is launched.

**Prefetch** is how many layers ahead from first use the reload is launched.

Larger defer gives the copy more time to hide behind compute (less throughput loss) but holds the activation's GPU buffer longer (less effective savings). Dense models tolerate the large window (defer=8); MoE does not (defer=1), where defer>=3 causes 24%+ overhead or NCCL timeouts.

Three suggested presets:

| Preset | Dense | MoE |
|--------|-------|-----|
| **Conservative** | budget=5 GB, defer=8 | budget=100 GB, defer=1 |
| **Balanced** | budget=10 GB, defer=8 | &mdash; |
| **Max savings** | budget=100 GB, defer=1 | budget=100 GB, defer=1 |

## How It Works

Enable CPU offloading with `--compile.sac_and_offload`.

The `sac_and_offload` policy runs in two phases:

**Phase 1 &mdash; SAC tagging** (`memory_policy.py`): Every forward node is classified:
- Compute-intensive ops (mm, linear, SDPA, flex_attn, topk, collectives, DeepEP ops) &rarr; `MUST_SAVE`
- Everything else &rarr; `PREFER_RECOMPUTE`
- Cross-layer-boundary recomputable nodes get promoted to `MUST_SAVE`

**Phase 2 &mdash; CPU offload tagging** (`cpu_offload.py:tag_all_offloadable_activations`): The offloading pass takes the `MUST_SAVE` nodes and decides which to offload to CPU instead of keeping on GPU with the following heuristics:

1. Skips the last layer (activations consumed immediately in backward, no benefit)
2. Filters candidates: must be a `call_function`, not a collective, and a contiguous tensor >= 4096 bytes
3. Budget selection: sorts candidates largest-first, then greedily picks them until the CPU budget is exhausted

## Results and Observations

All runs on 8xH100 96 GB, with variations across budget, defer, batch size, and sequence length.

### Tradeoff Across Models

<img src="/devlogs/images/distributed/offload_tradeoff_across_models.png" alt="Memory vs throughput tradeoff across Llama3, Qwen3, and DSv3">

As we begin to offload more activations, the compute is no longer able to hide the transfer time and memory transfers begin to slow down the end-to-end performance. Qwen3 has the better frontier and no cliff in the measured range; Llama3 has a sharp one. DSv3, due to smaller activations with the MoE architecture, has the lowest number of activations to offload (more nodes are recompute).

### How Efficiency Scales with Budget and Defer Settings

**Llama3 8B** (defer=8, bs=1, seqlen=8192)

<img src="/devlogs/images/distributed/offload_llama3_efficiency.png" alt="Llama3 8B efficiency scaling with budget">

**Qwen3 14B** (bs=4, seqlen=4096)

<img src="/devlogs/images/distributed/offload_qwen3_efficiency.png" alt="Qwen3 14B efficiency scaling with budget">

DSv3 is omitted here because its efficiency was constant across budgets.

### Pareto Frontier for Qwen3 14B and Llama3 8B

<img src="/devlogs/images/distributed/offload_pareto_frontier.png" alt="Pareto frontier for Qwen3 14B and Llama3 8B">

### Maximum Offload per Model

<img src="/devlogs/images/distributed/offload_max_per_model.png" alt="Maximum offload per model">

The tensor fractions make the dense/MoE gap concrete: DSv3 reaches only 129 of 335 tensors and 6.8 GB moved, versus Qwen3's 273 of 351 and 28.8 GB.

### Key Observations

- **Numerics verified:** the transformation is bitwise identical.
- **TPS cliff at ~12 GB budget (Llama3).** 10-12 GB drops 60-70 TPS for 0.46 GiB of extra savings. No cliff on Qwen3, which has enough compute per step to completely hide transfers.
- **View replay enables ~82% of savings** (4% ceiling without it, 33% with).
- **The pinned pool cut overhead from ~50% to ~22%.**
- **Longer sequences overlap better:** seq=8192 at 20.3% overhead vs seq=4096 at 23.9%.
- **MoE needs defer=1; dense uses defer=8.** DSv3 at defer>=3 hits 24%+ overhead or NCCL timeouts. DSv3 has substantially smaller activations than the dense models due to only activating a subset of experts and running with 8-way expert parallelism.

## Future Work

The torchtitan work is complete. This can be built on further with the following:

- Productionizing an offloading pass in core PyTorch when a full joint graph isn't available (there has been some interest in this from Ads and ByteDance); this is in progress in collaboration with Ads.
- Research into automatic offload policies &mdash; selecting budget, defer, and which tensors to offload without manually specified policy. Our research intern Nurlan Nazaraliyev this summer will be investigating this.