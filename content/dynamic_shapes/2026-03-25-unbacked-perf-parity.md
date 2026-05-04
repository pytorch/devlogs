---
title: "Unbacked Dynamic Shapes Shouldn't Be Slower — Now They Aren't"
author: Laith Sakka (@laithsakka)
date: 2026-03-25
tags: [dynamic_shapes, unbacked, performance, vllm, torchbench, inductor]
---

![Backed vs unbacked perf parity](/images/dynamic_shapes/2026-03-25-perf-parity-header.jpg)

> **TL;DR** – Unbacked dynamic shapes had 2x–20% slowdowns on TorchBench
> and ~30% regressions on vLLM.  We fixed the root causes — now unbacked
> matches backed across all tested models and configurations.

## Motivation

[These regressions were blocking adoption](./2026-01-20-backed-to-unbacked.md)
in Frontier workloads like vLLM.  Demand for unbacked shapes is growing —
just in the past week, multiple users needed them to control
recompilations — so the gap was not acceptable.

We've now solved this: unbacked matches backed across all HuggingFace
TorchBench models (up to 2x faster) and 30+ vLLM models across multiple
configurations.

## The invariant and the task

The key idea behind this work is simple:

> **For a given graph G and guard set E, unbacked shapes must match the
> performance of backed shapes, given identical optimization hints.**

Any optimization or decision available in backed mode should also be
available in unbacked mode under the same guards.  There is no fundamental
reason for divergence — if it happens, it's a bug in how decisions are
made.

In the vLLM setting, *E* is effectively fixed (often empty, or
range-bounded), so there is no reason for the performance divergence.

> **The task was clear: given the same guards and appropriate hints for
> unbacked shapes, optimization decisions should be identical between
> modes.**

## Approach

We approached this in three steps: make the gap measurable, fix the
foundation, then eliminate the remaining divergences.

### 1. Make the gap measurable (TorchBench unbacked mode)

We first added an [unbacked mode](https://github.com/pytorch/pytorch/pull/172719)
to TorchBench to enable fast iteration and direct measurement.  This
required fixing a number of data-dependent errors along the way.

The gap was large and consistent:

| Model | Regression |
|-------|-----------|
| MegatronBertForCausalLM | 0.86x |
| BartForCausalLM | 0.80x |
| BertForMaskedLM | 0.75x |
| T5Small | 0.72x |
| MobileBertForMaskedLM | 0.53x |

### 2. Fix the foundation — size-hinting refactor

The first real fix was at the foundation.  The size-hinting APIs were
fragmented, inconsistent, and easy to misuse.  Instead of chasing
individual regressions, we simplified the system with a clear goal: ensure
that guardless optimizations work reliably for unbacked shapes and do not
diverge when hints are available.

We replaced the existing size-hinting APIs with two primitives:

- **`optimization_hint`** (~200 call sites migrated) — Always produces an
  optimization hint, even for unbacked symbols.  This ensures
  optimizations are never silently skipped and enforces consistency across
  symbolic expressions and shape env constraints.
- **`guarding_hint_or_throw`** — Used only for guards.  Throws on
  unbacked, used when callers want to guard on hints.

This refactor fixes many existing optimization gaps and ensures that
future optimizations work correctly for unbacked shapes by default.

Meaningful optimization hints are required for unbacked symbols to enable
correct optimization decisions.  These are provided via `override_hint` in
`mark_unbacked`, or via `torch._dynamo.optimization_hint()` for
data-dependent cases.

### 3. Close the remaining gaps — individual optimization fixes

After fixing the foundation, I re-ran TorchBench and still saw
regressions.  The approach was straightforward: compare Inductor-generated
code between backed and unbacked, identify where decisions diverge, fix,
and repeat.

**Optimizations skipped or using dummy decisions for unbacked shapes:**

Several optimizations were either skipped entirely for unbacked symbols or
not using the proper size-hinting APIs:

- [Triton autotuning](https://github.com/pytorch/pytorch/pull/175220)
- [Memory estimation fusion heuristics](https://github.com/pytorch/pytorch/pull/175221)
- [Reduction splitting](https://github.com/pytorch/pytorch/pull/175835)
- [`pad_mm`](https://github.com/pytorch/pytorch/pull/175824)

Switching to `optimization_hint` and `all_unbacked_hinted` across all of
these made optimization decisions consistent between backed and unbacked
modes.

**Symbolic reasoning gaps in reshape indexing:**

In `_dynamic_reshape_indexer`, comparisons like `u0 * 12 > u0` could not
be proven symbolically (e.g., because `u0` could be 0), causing the
compiler to take a slow fallback path — even when the edge cases were
irrelevant in practice.
[Improving symbolic reasoning](https://github.com/pytorch/pytorch/pull/175232)
for these cases recovered ~20% performance.

Two additional changes were needed to achieve identical Inductor output
code but had minimal performance impact:
[min/max bounds](https://github.com/pytorch/pytorch/pull/176313) for
`mark_unbacked` to enable correct 32-bit indexing decisions, and
[runtime assertion skipping](https://github.com/pytorch/pytorch/pull/175871).

## Final state

With these changes, backed and unbacked shapes achieve **identical
performance** on ALL HuggingFace TorchBench models.  Notably, these
models did not rely on significant guard-based optimizations.

Furthermore, unbacked performance was validated on vLLM across a wide
range of models and configurations — including 29 text models, multimodal,
FP8, NVFP4, CUDA graphs on/off, and mixed static + dynamic paths — with
**no regressions observed**.

## Protection from future regression

To ensure this remains true, a
[periodic Inductor test](https://github.com/pytorch/pytorch/pull/177034)
was added that enforces performance parity between backed and unbacked
shapes on a set of HuggingFace TorchBench models.

As vLLM moves to unbacked by default, its performance dashboards will
also serve as a continuous signal and line of defense against regressions.

## What's next

- Now that unbacked performance matches backed, the next step is a safe,
  progressive migration plan for vLLM from backed to unbacked with minimal
  disruption.
- The main remaining piece for unbacked is better APIs for specifying
  dynamic shapes specs and dispatch — this will be a focus in Q2.

## Acknowledgments

Special thanks to Jason Ansel ([@jansel](https://github.com/jansel)) for
his prompt diff reviews throughout the process. Thanks to Bob Ren
([@bobrenjc93](https://github.com/bobrenjc93)) for adding the early work
of hint override that `optimization_hint` depends on, and to Colin Peppler
([@colinpeppler](https://github.com/colinpeppler)) for adding the logic of
consistent optimization hint generation that was consolidated in
`optimization_hint`.

## References

- [TorchBench unbacked mode PR #172719](https://github.com/pytorch/pytorch/pull/172719)
- [Performance parity test PR #177034](https://github.com/pytorch/pytorch/pull/177034)
- [Backed to Unbacked — background post](./2026-01-20-backed-to-unbacked.md)
