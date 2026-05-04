---
title: "Guard-Free Dynamic Shapes"
author: Laith Sakka (@laithsakka), Brian Hirsh (@bdhirsh), Angela Yi (@angelayi), Colin Peppler (@colinpeppler), Bob Ren (@bobrenjc93), Avik Chaudhuri (@avikchaudhuri), Aaron Orenstein (@aorenste), Pian Pawakapan (@pianpwk)
date: 2025-07-08
tags: [dynamic_shapes, unbacked, dde, guard_free, export, torch.compile]
---


> **TL;DR** – Data-dependent errors (DDEs) are the dominant barrier to
> exporting models with dynamic shapes. There is widespread consensus that
> DDEs are a significant issue for export — among the various errors
> observed, data-dependent errors are the most dominant. We launched an
> initiative to eliminate them via **explicit unbacked semantics** —
> explicitly defining how code should behave when inputs are unbacked. In
> the first half, we generalized more than 60% of the relevant code,
> achieving a **55% reduction** in framework DDEs and saving weeks of
> engineering time on complex model exports.

![DDE reduction progress](/devlogs/images/2025-07-08-dde-progress.jpg)

## Why are guard-free dynamic shape graphs necessary?

### 1. Exporting data-dependent operations

When exporting models, graph breaks aren't allowed. Therefore, when a
non-guardable dynamic size is introduced through data-dependent operations
like `.item()`, we have to keep it in the graph and trace through the
program without adding guards to be able to export the program.

### 2. Exporting with specialized inputs

Sometimes, the example inputs used during model export can cause
unnecessary specialization in the exported graph. For instance, using a
contiguous example input might result in a graph that's only valid for
contiguous inputs. In those situations we want tracing modes that allow
avoiding guards on dynamic shapes whenever possible by taking guard-free
general paths.

### 3. Generic graphs (e.g., vLLM fallback graph)

The need for a single graph that handles all inputs becomes more prominent
in pre-compile schemes. One example is vLLM, which pre-compiles multiple
optimized graphs alongside a single dynamic graph expected to handle
**all** input shapes. However, vLLM currently unsoundly drops all guards,
which can lead to soundness issues in production. To make this approach
sound, we need to ensure that dynamic shapes in the graph aren't
constrained by guards — using guard-free dynamic shapes.

### 4. Reducing recompilation

0/1 specializations can lead to a large number of recompilations,
especially for models with many dynamic inputs (e.g., list/dict values).
In this case the main motivation for guard-free shapes is avoiding the
recompilations.

## The issue of data-dependent errors

Ok, so let's just avoid the guards! Well, the devil is in DDEs, which
show up when `torch.compile` attempts to branch on sizes that we are not
allowed to guard.

Consider the following simple scenario, where it's not allowed to guard
`x`:

```python
do_fast_path() if (x == 0) else do_slow_path()
```

To avoid this branch, users might add a check `torch._check(x != 0)`.
However, this introduces a runtime constraint that `x != 0`, which is
**not necessary** and **not always trivial**. Instead, we want to capture
a single generic graph without adding constraints and without requiring
user intervention, by always taking the slow path.

`guard_size_oblivious` was an early attempt to address this non-invasively.
However, it fell short of fully solving the problem due to its limitations.
We took a more aggressive yet more predictable approach, generalizing many
of the ideas introduced by `guard_size_oblivious`. We often utilized
`guard_or_false` and `guard_or_true` to achieve this.

The code above can be written as:

```python
do_fast_path() if guard_or_false(x == 0) else do_slow_path()
```

Here, `guard_or_false` says: "If you can't determine whether `x == 0`
because you're unable or not allowed to guard on it, just return `False`."
By doing so, we avoid throwing data-dependent errors on this branch.

## Explicit unbacked (guard-free) semantics

The goal of the new approach is to explicitly define the branch behavior
when guards on dynamic shapes are not allowed. Although this sometimes
leads to deviation from eager execution, the differences are usually not
harmful.

A simple example: calling `t.contiguous()`. When a tensor has dynamic
sizes and strides that are not guardable, it's ambiguous whether the
tensor is contiguous or not. In fact it can be seen as a dynamic tensor
that represents both contiguous and not-contiguous tensors.

When presented with such a tensor, we always clone the tensor if we can't
confirm (without adding guards) that it's contiguous. This means that,
unlike eager execution, `t.contiguous()` might clone the input tensor
during runtime even if it's already contiguous.

The use of `guard_or_false` and `guard_or_true` in framework code covers a
wide range of cases. For example:

```python
# Contiguity checks — if we can't guard, assume not contiguous (safe path)
is_contiguous = guard_or_false(stride == expected_stride)

# Broadcasting — if we can't guard, assume sizes differ (general path)
needs_broadcast = guard_or_true(size_a != size_b)

# Zero-element fast paths — if we can't guard, skip the fast path
if guard_or_false(numel == 0):
    return empty_result
```

This pattern generalizes across reshaping, slicing, narrowing, selection,
expand, and many more operations — each time picking the safe, general
path when guard-free shapes are involved.

## Moving the needle

To prioritize our efforts, we focused on three key indicators:

1. **Existing size-oblivious handling** — we targeted generalizing the
   handling at locations using `guard_size_oblivious`, where we had
   previously encountered data-dependent errors.
2. **User issues** — Colin Peppler
   ([@colinpeppler](https://github.com/colinpeppler)) triaged user cases,
   while Pian Pawakapan ([@pianpwk](https://github.com/pianpwk)) developed
   queries to identify DDE sources.
3. **OSS export benchmark analysis** — Pian Pawakapan's work on the top 50
   model DDEs helped us pinpoint the most common problematic operators.

Our efforts yielded strong positive signals:

- **60% reduction in `guard_size_oblivious` usage** — a significant step
  toward generalizing and removing this complex mechanism from the
  codebase.
- **Complex model export case study** — we can now export a model that
  previously required 31 `torch._check` calls with zero checks. This
  saved an estimated 2+ weeks of engineering effort.
- **55% reduction in framework DDEs** on the OSS export benchmark of 50
  open-source models.
- **Several guaranteed DDE-free operations** — `contiguous()`, `reshape()`,
  `infer_size()`, `expand()`, broadcasting, and more no longer throw
  data-dependent errors.

## What's next

- Achieve 90–100% generalization for `guard_size_oblivious`.
- Avoid all framework DDEs in the OSS export benchmark.
- Address vLLM dynamic shapes soundness issues by tracing vLLM with
  guard-free dynamic shapes. Early experiments show that models like Llama
  can already be traced with non-guardable sizes — we plan to expand to
  more models.
- Harden `backed_size_oblivious` (a best-effort mode that allows
  specializations only when not avoidable).

## Acknowledgments

Many of the ideas and directions explored in this project are
generalizations of Edward Yang's
([@ezyang](https://github.com/ezyang)) thoughts on unbacked semantics and
the implications of `guard_size_oblivious` behavior. Special thanks to
Edward — without his initial work we would not have achieved this.

## References

- [`torch/fx/experimental/symbolic_shapes.py`](../../../torch/fx/experimental/symbolic_shapes.py) — symbolic shape infrastructure
- [Followup: Slaying Framework DDEs](./2025-10-29-slaying-framework-ddes.md) — completion of this initiative
