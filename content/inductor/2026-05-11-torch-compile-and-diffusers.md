---
title: "torch.compile and Diffusers: A Hands-On Guide to Peak Performance"
date: 2026-05-11
author: "Sayak Paul (@sayakpaul), Animesh Jain (@anijain2305), Benjamin Bossan (@BenjaminBossan)"
tags: [torch.compile, diffusers, regional-compilation, dynamic-shapes, quantization, lora]
---

> **TL;DR** – `torch.compile` delivers a ~1.5x speedup on Flux-1-Dev with no quality loss. Use `compile_repeated_blocks` to cut compile latency 7x (67s → 9.6s) while keeping the speedup, enable `dynamic=True` to avoid recompiles on shape changes, and combine with CPU offloading, NF4 quantization, and LoRA hot-swap without giving up the compiled kernels.

## Background / Motivation

Diffusion pipelines are heavy: Flux-1-Dev in bf16 weighs ~33 GB and a single image takes 6.7s on an H100. `torch.compile` can fuse kernels and strip Python overhead, but applying it naively to a real pipeline runs into four practical issues:

1. **Compile latency.** First-call JIT cost — 67.4s for the full DiT.
2. **Graph breaks.** Any unsupported op silently slices the graph and leaves speedup on the table.
3. **Recompilations.** Shape specialization forces a fresh compile when the user changes resolution.
4. **DtoH syncs.** Mostly absent in Diffusers pipelines, but worth checking.

The compute is dominated by the **denoiser (a DiT)**. Text encoders (CLIP, T5) and the VAE decoder are tiny slices of total runtime, so compiling only the DiT avoids unnecessary overhead.

## Design / Approach

### Vanilla compilation — the 1.5x speedup

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

pipe.transformer.compile(fullgraph=True)

out = pipe(
    prompt="A cat holding a sign that says hello world",
    guidance_scale=3.5,
    num_inference_steps=28,
    max_sequence_length=512,
).images[0]
```

H100 latency: **6.7s → 4.5s** (~1.5x), no image-quality regression. `fullgraph=True` is the model-author's friend — it raises on any graph break instead of silently falling back.

### Regional compilation — pay 7x less to compile

A DiT is a stack of identical Transformer layers. Compile one, reuse the kernels for the rest:

```python
pipe.transformer.compile_repeated_blocks(fullgraph=True)
```

| Metric                | Full compile | Regional compile |
|-----------------------|-------------:|-----------------:|
| Compile latency       | 67.4s        | **9.6s**         |
| Warm start (cached)   | —            | 2.4s             |
| Runtime speedup       | 1.5x         | **1.5x**         |

Same runtime speedup, dramatically faster startup. Combine with [compile caching](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) for sub-3s warm starts.

### Dynamic shapes — avoid recompiles on resolution changes

```python
pipe.transformer.compile_repeated_blocks(fullgraph=True, dynamic=True)
```

PyTorch generally recommends `mark_dynamic` for selective dynamism, but for diffusion DiTs `dynamic=True` works well across the common shape distribution.

### Composing with memory-saving features

**CPU offloading** — keep peak VRAM at 22.7 GB:

```python
pipe.enable_model_cpu_offload()
pipe.transformer.compile_repeated_blocks(fullgraph=True)
```

**NF4 quantization (bitsandbytes)** — DiT + T5 quantized to 4-bit drops peak memory to 15 GB. The compiler fuses around the 4-bit ops and recovers the quantization overhead: 7.3s → **5.0s** (1.5x).

**Quantization + offloading** — 12.2 GB peak, 12.2s → **9.8s** with compile.

### LoRA hot-swap — switch adapters without recompiling

Swapping LoRA tensors normally changes the weight identities the compiler keyed on, triggering a recompile. The PEFT hot-swap path pre-declares the max rank and reuses the compiled graph:

```python
pipe.enable_lora_hotswap(target_rank=max_rank)
pipe.load_lora_weights(<lora-1>)
pipe.transformer.compile(fullgraph=True)

# Subsequent swaps reuse the compiled kernels:
pipe.load_lora_weights(<lora-2>, hotswap=True)
```

Caveats: declare `max_rank` up front (use the largest rank across adapters), all adapters must target the same layer set (or a subset of the first), and text-encoder hot-swap isn't supported yet.

## Results / Benchmarks

H100, Flux-1-Dev, 28 steps:

| Scenario                  | Baseline | With `torch.compile` | Peak Mem | Speedup |
|---------------------------|---------:|---------------------:|---------:|--------:|
| Vanilla                   | 6.7s     | 4.5s                 | 33 GB    | 1.5x    |
| Regional compile          | 6.7s     | 4.5s                 | 33 GB    | 1.5x    |
| CPU offload               | 21.5s    | 18.7s                | 22.7 GB  | 1.15x   |
| NF4 quantization          | 7.3s     | 5.0s                 | 15 GB    | 1.5x    |
| NF4 + CPU offload         | 12.2s    | 9.8s                 | 12.2 GB  | 1.24x   |

Compile latency (regional): **67.4s → 9.6s cold, 2.4s warm.**

## Operational hardening

Diffusers runs nightly CI dedicated to `torch.compile` health, covering graph breaks, unintended recompilations across common shapes, and compatibility with every quantization backend and offloading mode. Benchmarks (latency + peak memory) are captured alongside CI and exported to a [consolidated CSV](https://huggingface.co/datasets/diffusers/benchmarks/blob/main/collated_results.csv); design lives in [diffusers#11565](https://github.com/huggingface/diffusers/pull/11565). Tracker: [torch.compile-labeled issues](https://github.com/huggingface/diffusers/issues?q=label%3Atorch.compile).

## Open questions / Future work

- **Text-encoder LoRA hot-swap** — currently DiT-only.
- **More aggressive dynamic-shape coverage** — `dynamic=True` works for the common Flux shape set; harder cases (e.g. video models with variable temporal dims) still need `mark_dynamic` tuning.
- **Compile-time budget for end users** — 9.6s cold start is acceptable for servers but still noticeable in interactive UIs; further caching wins are on the table.

## References

- Original post: [torch.compile and Diffusers: A Hands-On Guide to Peak Performance](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
- [Regional compilation tutorial](https://pytorch.org/tutorials/recipes/regional_compilation.html)
- [torch.compile caching guide](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
- [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
- [Diffusers x compilation docs](https://huggingface.co/docs/diffusers/main/en/optimization/fp16#torchcompile)
- [Diffusers quantization backends](https://huggingface.co/blog/diffusers-quantization)
- [LoRA hot-swap docs](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference#hotswapping)
- [Benchmark script](https://gist.github.com/sayakpaul/91fa328e949c71dc4420ebb50eb35ca3)
