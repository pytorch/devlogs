---
title: "Slaying Framework Data-Dependent Errors Dragon 🐉"
author: Laith Sakka (@laithsakka)
date: 2025-10-29
tags: [dynamic_shapes, unbacked, dde, export, guard_free, torch.compile]
---

# Slaying Framework Data-Dependent Errors Dragon 🐉

*By [@laithsakka](https://github.com/laithsakka)*

> **TL;DR** – Framework DDE dragon has been slain! 🐉 We've eliminated the
> vast majority of framework data-dependent errors — reducing user issues
> by over **85%** — and unlocked **specialization-free full graph capture**
> that *just works*. This lays the groundwork for emerging unbacked use
> cases in vLLM, MoE graphs, and PT2-Frontier.

## Tackling Data-Dependent Errors

Data-dependent errors (DDEs) have long been a major pain point for framework export users, as detailed in [the previous post](./2025-07-08-guard-free-dynamic-shapes.md). Six months ago, we launched an initiative to eliminate these issues by implementing explicit unbacked semantics — explicitly defining how code should behave when inputs are unbacked.

That work is now complete. We've moved into the maintenance phase, and many previously error-prone operations—such as reshaping, slicing and narrowing, selection, contiguity checks, and broadcasting checks—are now fully DDE-free. In total we addressed 270+ code branches. And the old, complex `guard_size_oblivious`/`size-like` mechanism has been completely deprecated.

This marks a major milestone: we can now capture specialization-free graphs much more reliably, providing a smoother and more predictable user experience. The growing number of use cases leveraging unbacked dynamic shapes—like deterministic compilation, vLLM, pre-compile APIs, and PT2-Frontier—highlights the importance of specialization-free graphs.

## What This Means for Users and Developers?

### 1. Improved User & Developer Experience

- Reports of DDEs from PyTorch users have dropped by **85%** (from 35+ to just 5).
- We closed 30+ GitHub issues related to DDEs — many were no longer reproducible DDEs, while others involved outdated ideas to deal with DDE underlying problems that this work already resolved.
- A study of 50 open-source models identified DDEs as the dominant exportability issue. We cut framework-related DDEs by 50–55% initially, and then eliminated the remaining sources of DDEs in those models—making exporting these models much simpler and faster.
- Exporting complex models now saves two or more weeks of development time, making the workflow faster and more efficient.

### 2. Reduced Technical Complexity

The previous `guard_size_oblivious` / `size-like` system was the first step toward eliminating DDEs, and our work was significantly influenced by it. However, it often made the code's behavior difficult to reason about and introduced multiple layers of technical overhead to maintain:

**Size-like annotation and propagation:**
Users had to manually call `_check_size()` to mark size-like dimensions, and the framework then had to correctly propagate those annotations across operations. Any missed annotation or propagation failure broke the system's guarantees around DDE elimination.

**Dependence on symbolic reasoning:**
The system relied on a hint-free symbolic evaluator to infer relationships among dynamic shapes. DDE elimination depended on the evaluator's ability to reason correctly about these relationships under certain input constraints; if inference failed or remained incomplete, DDEs would persist.

With explicit unbacked semantics, all of this complexity has been removed: No manual `_check_size()` calls. No propagation of "size-likeness." No reliance on symbolic evaluation. The result is a simpler, more deterministic, and more predictable system — that achieves better DDE elimination.

### 3. Enabling Sound, Non-Constrained Graphs

In the past, users often had to insert `torch._check` calls to constrain the graph and avoid DDEs, then manually remove or ignore those checks later to generalize exported graphs. It was a fragile and frustrating workaround.

With unbacked semantics, that's no longer necessary. Users can now produce fully general, unconstrained graphs directly—without resorting to these manual hacks.

## What's Next for Unbacked Dynamic Shapes

Support for unbacked dynamic shapes remains a key theme in our dynamic shapes roadmap—especially as their importance grows with upcoming features such as deterministic compilation, compile-on-one-rank, PT2-Frontier, and ensuring vLLM soundness.

There's still significant work ahead. Our focus remains on advancing support for unbacked shapes while continuing to address urgent user needs in distributed settings. Key remaining areas include:

- **Improve the performance of unbacked shapes** (initially using vLLM as a proxy) to match that of backed dynamic shapes.
- **Improve size hinting consistency** by making size hinting for unbacked symbols consistent with the shape environment. And/or allow users to control their hints. This issue often leads to compile failures for users of AOTInductor during autotune.
- **Ensure unbacked shapes are guard-free** unless explicitly requested by users. And introduce unbacked striding policies that let users control behavior around layout properties of input tensors.
- **Build infrastructure to support pre-compile API.** Namely, hooks that allow users to mark dynamism and provide invariants without changing model code. These hooks will be leveraged by the upcoming precompile dispatch APIs.
- **Address remaining less-frequent DDE sources.** A few GitHub issues remain open. And we will continue monitoring reports and using the fuzzer to uncover unhandled call sites—a method that has already proven effective at finding many of these branches.
- **Enable unbacked shapes for model inputs in export API.**
- **Harden runtime assertion lowering** by making sure they are added when expected.

## Thanks!

A big thank you to Brian Hirsh ([@bdhirsh](https://github.com/bdhirsh)) for the long discussions and early-stage guidance that helped shape this project. Similar thanks goes to Bob Ren ([@bobrenjc93](https://github.com/bobrenjc93)) and Aaron Orenstein ([@aorenste](https://github.com/aorenste)) for their support through long discussion and diff reviews.

Pian Pawakapan ([@pianpwk](https://github.com/pianpwk)) deserves special recognition for addressing DDEs across several operations — notably slicing, stride ordering, expand — and for leading the exportability benchmark, identifying crucial DDE sources along the way.

Finally, Colin Peppler ([@colinpeppler](https://github.com/colinpeppler)) has been instrumental in continuously reporting and tracking user DDE issues in addition to addressing many DDEs in many ops.

## References

- [Guard-Free Dynamic Shapes — original initiative post](./2025-07-08-guard-free-dynamic-shapes.md)
- [`torch/fx/experimental/symbolic_shapes.py`](../../../torch/fx/experimental/symbolic_shapes.py) — symbolic shape infrastructure
- [Backed to Unbacked — broader context](./2026-01-20-backed-to-unbacked.md)
