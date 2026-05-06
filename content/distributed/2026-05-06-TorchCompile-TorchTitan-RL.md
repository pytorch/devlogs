---
title: "torch.compile for TorchTitan RL: 6x Faster Unified RL Training"
date: 2026-05-06
author: "Lucas Kabela (@lucaskabela)"
tags: [torchtitan, rl, torch.compile, distributed, performance]
---

> **TL;DR** – We enabled `torch.compile` across the full RL training loop in TorchTitan, achieving a **6x end-to-end speedup** (from 446s to 70s) on Qwen3 0.6B for GSM8K. Thanks to TorchTitan RL using a single unified model definition for both training and inference, we can share compiled artifacts across the trainer and generator, reducing startup time while leveraging performance improvements to make this possible.

## What makes TorchTitan RL different?

Most RL frameworks (Verl, OpenRLHF, etc.) maintain separate model definitions for training vs. inference. This means:

- Duplicated code to keep in sync
- Separate optimization paths for each
- No opportunity to share compilation work

TorchTitan RL uses **one model definition** across both the Trainer (TorchTitan) and Generator (vLLM). `torch.compile` traces the model once and reuses it in both contexts, enabling fullgraph optimizations that span the entire RL loop and reducing compilation time vs. compiling each independently.

**Challenges:** Due to the unified definition, we needed to handle interoperability with vLLM and particular DTensor operations. This included defining how to capture `weak_ref` for cudagraph management of DTensors, as well as fixing codegen paths that would be otherwise undiscovered.

## Results

Qwen3 0.6B on GSM8K, TP=4 on 8 H100, 10 training steps:

|                    | No Compile (baseline) | + Separate Compile & Piecewise CUDAGraphs | + Batching | + Fullgraph CUDAGraphs & Shared Compile |
| ------------------ | --------------------- | ----------------------------------------- | ---------- | --------------------------------------- |
| **Total Time**     | 446.0s                | 205.0s                                    | 120.0s     | 70.4s                                   |
| **Startup Time**   | 24.3s                 | 79.1s                                     | 84.3s      | 47.9s                                   |
| **Generator Time** | 262.4s                | 22.0s                                     | 17.9s      | 5.4s                                    |
| **Trainer Time**   | 157.0s                | 103.3s                                    | 17.8s      | 17.1s                                   |

> Total Time = Startup + Generator + Trainer + weight sync. Startup is compilation/and cudagraph capture overhead. Weight sync time is negligible.

### Key takeaways

- **Generator: 48x faster** (from 262s to 5.4s) — CUDAGraphs + fullgraph capture eliminate Python overhead
- **Trainer: 9x faster** (from 157s to 17.1s) — batching + compile fuse operations
- **Shared compile cuts startup by 36s** vs. compiling trainer and generator independently

## What we shipped

- **Compile for Trainer** — `torch.compile` on the policy model ([#2568](https://github.com/pytorch/torchtitan/pull/2568))
- **Compile for Generator** — vLLM `support_torch_compile` + DTensor CUDAGraph support ([#2486](https://github.com/pytorch/torchtitan/pull/2486), [#2710](https://github.com/pytorch/torchtitan/pull/2710))
- **Batched RL training** — varlen sequence packing ([#2906](https://github.com/pytorch/torchtitan/pull/2906), Joe Cummings)
- **Fullgraph CUDAGraphs + shared compile** — Reduce from 50s to 15s compilation time for vLLM ([#3145](https://github.com/pytorch/torchtitan/pull/3145))
- **Bitwise determinism** — numeric parity between trainer and generator paths ([#2358](https://github.com/pytorch/torchtitan/pull/2358))
- **`.compile()` migration** — aligned with PT2 programming model ([#2688](https://github.com/pytorch/torchtitan/pull/2688))

## Try it

```bash
python torchtitan/experiments/rl/grpo.py --module rl --config rl_grpo_qwen3_0_6b
```

Full setup: [`torchtitan/experiments/rl`](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/rl)

## What's next

There are still a number of unexplored integrations and optimizations to be made, including:

- CUDAGraph for trainer (estimate saving ~1.5-2x performance)
- Regional/bitwise Inductor (selective optimization preserving numeric parity)
- Compile on one rank for scaled training (> 1 Node, run on Mast)
- Scaling to MOE models and investigating dynamism
