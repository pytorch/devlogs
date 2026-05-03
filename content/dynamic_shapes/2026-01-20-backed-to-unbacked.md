---
title: "Backed to Unbacked: From Guardable to Guardless Shapes in PyTorch"
author: Laith Sakka (@laithsakka), Aditya Venkataraman (@aditvenk), Bob Ren (@bobrenjc93)
date: 2026-01-20
tags: [dynamic_shapes, unbacked, backed, torch.compile, frontier]
---

# Backed to Unbacked: From Guardable to Guardless Shapes in PyTorch

*By [@laithsakka](https://github.com/laithsakka), [@aditvenk](https://github.com/aditvenk), [@bobrenjc93](https://github.com/bobrenjc93)*

![Hint stamp](./images/2026-01-20-backed-unbacked-header.jpg)

> **TL;DR** – We expect unbacked dynamic shapes to become the dominant
> shape mechanism for Frontier-style workloads due to their better
> predictability and controllability. However, some blockers remain for
> their ideal usage, most notably the performance gap, which is a primary
> focus for the first half of 2026.

## Origins

Recently, unbacked dynamic shapes have become a hot topic. But many people
still don't fully understand (1) what backed vs unbacked dynamic shapes
actually are, and (2) why that choice matters for performance, UX, and
Frontier.

In this post, I'll walk through a simplified story of how we got here, why
unbacked shapes are becoming more important, and what's still blocking
them. This post is divided into three parts. Feel free to skip to
Section 3 if you are already familiar with backed and unbacked shapes.

1. The Emergence of Backed Shapes
2. The Emergence of Unbacked Shapes
3. Unbacked is a better fit for the Frontier.

## 1. The Emergence of Backed Shapes

### 1.1 Endless recompilations

We created PT2 as a JIT drop-in, plug-and-play method to accelerate ML
programs — simply by adding a decorator. People started using
`torch.compile` and life was good — until we began hitting recompilations
that were killing performance for some important use cases.

Consider the following simple function:

```python
def func(x):
    return torch.ones(x)
```

When compiled with `x=10`, Dynamo inserted a guard checking that "the
input `x` is exactly 10" and generated a graph hard-coded to return a
tensor of size 10. Next, when called with `x=20`, the guard failed,
triggering another compilation for size 20, and so on.

For workloads with many varying inputs, this led to endless recompilations
and became a serious performance problem. So, what did we do? We stopped
hard-coding (specializing) sizes in the graph and made them dynamic by
representing them symbolically in the compiled graph without strict value
guards. Conceptually:

```python
def func_symbolic(s0):
    return torch.ones(s0)
```

### 1.2 Back it with a hint

Of course, it wasn't that simple! Once we represented sizes symbolically,
the compiler encountered issues whenever it needed to branch based on
those sizes. Branching could occur in framework/compiler code (e.g.,
contiguity checks) or in user code, as shown in this example:

```python
if x < 1024:
    # path A
else:
    # path B
```

With symbolic shapes, the compiler no longer had a concrete value for `x`
to decide which path to take during compilation. To enable the compiler to
make this decision, we **backed** the dynamic shape (e.g., `s0`) with a
**hint** — a concrete value from the example input used in compilation,
and allowed the compiler to use it to decide what branch to take.

For instance, if the example value at compile time was 10, then we use 10
as the hint for `x`. During compilation, `path A` is picked based on the
hint, and Dynamo then adds a guard ensuring that the condition of the
taken branch (`x < 1024`) is satisfied.

This approach effectively solved the "endless recompilations" problem;
instead of recompiling for each new concrete value we see for `x`, we only
compile once for each branch taken.

We named these **backed dynamic shapes** because they are backed by a
hint that guides branch selection. Another term for these is **guardable
shapes**, as we are allowed to introduce guards that constrain them within
the Dynamo graph.

## 2. The Emergence of Unbacked Shapes

### 2.1 Data-dependent ops

In a different use case, we encountered functions that use data-dependent
operations, for example:

```python
def func(x):
    u = x.item()
```

Here, `u` is a scalar value that depends on the data of `x`. The question
arose: how do we represent the output `u` inside the compiled graph? At
compile time, we do not know the concrete value of `u`.

Initially, the trivial option was to give up and trigger a **graph
break** — namely, force the compilation to stop, and split the compiled
graph into two compiled graphs, executing the data-dependent operation
eagerly (`.item()` here), getting a concrete value for `u`. This would
then resume compiling the second graph with a known integer input
representing the value of `u` for the next code section.

This was problematic for export and other use cases requiring a single,
full graph that captures the entire model. Furthermore, data-dependent
heavy code would result in many graph breaks, hurting performance and
significantly increasing compile time.

### 2.2 Unbacked dynamic shapes

To keep the data-dependent operation within the graph without graph
breaking, we represented its output symbolically — with a different type
of shape. Unlike backed shapes, we do not have a hint to use for
resolving branching on `u`. That's why we call these **unbacked dynamic
shapes**.

A significant challenge with unbacked dynamic shapes was handling
branching with absence of the hint; without a hint, the compiler couldn't
determine which branch to take, and the default behavior was to **throw a
data-dependent error (DDE)**. For example:

```python
def func(x):
    u = x.item()
    y = .. if u == 0 else ..
```

This was one of the most painful UX aspects of dynamic shapes. We
addressed this by teaching framework code how to handle these branches —
automatically picking the general path — and by providing APIs for users
to write DDE-friendly branching. This work resolved the single most
common reason for export failures in the framework.

### 2.3 Unbacked inputs

While unbacked shapes were originally introduced to support data-dependent
operations, over time users began deliberately choosing unbacked shapes
for primary graph inputs as well (effectively dropping the hints and
treating those inputs as if they came from data-dependent ops).

The main motivations were twofold: (1) to avoid branch-induced
recompilations, and (2) to compile graphs that work across all relevant
input shapes. With unbacked shapes, general branches that are valid for
all inputs are selected without imposing shape constraints (unbacked
semantics). This ensures no recompilations occur ever, and a single
compiled graph can handle a broad range of input shapes.

Unbacked shapes can also be referred to as **guardless dynamic shapes**.
Not only do they lack a "hint for the purpose of guarding", but they are
also not allowed to have guards.

> Note: unbacked dynamic shapes *can* have hints in the form of
> "optimization hints", but those hints can only be used for guardless
> optimizations such as auto-tuning.

## 3. Unbacked is a Better Fit for the Frontier

### 3.1 Predictability, determinism, and control

Backed dynamic shapes — while a strong fit for drop-in JIT optimization —
conflict with Frontier's focus on determinism, predictability, and
control. This is precisely where unbacked shapes excel: they are naturally
aligned with deterministic behavior, explicit control of shape
constraints, predictable compilation outcomes, and enabling
pre-compilation workflows.

- **Predictability.** Backed shapes are difficult to reason about ahead
  of time. You cannot know what constraints will be imposed on dynamic
  input ranges without actually compiling, nor is it clear which example
  inputs are required to generate enough graphs to cover the full input
  space. By contrast, unbacked shapes are highly predictable and
  controllable. Users can explicitly request, for example: "Compile one
  graph with `x` unconstrained and another with `x` in `[0, 100]`." If
  compilation succeeds, it is known that the compiler has not introduced
  any additional, hidden constraints.

- **Determinism.** This predictability is tightly coupled with
  determinism: the output graph for backed dynamic shapes depends heavily
  on the specific example inputs used during compilation, and is very
  sensitive to source changes (including added or removed optimizations),
  since different branches may or may not be taken.

- **Pre-compile.** This also creates challenges for pre-compilation since
  we want to ensure that a reasonable, finite set of graphs collectively
  covers the entire relevant input space. With backed shapes, the
  compiler's implicit constraints and dependence on example inputs make it
  difficult to know whether the precompiled graphs truly cover all
  intended inputs.

A concrete example is the vLLM use case, where multiple graphs are
precompiled for different input ranges and are expected to work reliably
within those specified ranges. Backed shapes are **fundamentally unsound
here** — leading to continuous issues because the compiler can silently
introduce restrictive shape constraints. Unbacked shapes, by design, are
the correct tool for this scenario.

### 3.2 Are backed about to die?

Not at all. While Frontier clearly points toward unbacked shapes, backed
shapes are far from obsolete. For the typical PyTorch user who just wants
a drop-in JIT optimization, backed dynamic shapes remain an excellent
choice: they provide strong performance benefits without requiring deep
model understanding or the overhead of manually compiling and managing
multiple unbacked graphs.

### 3.3 Remaining blockers for Frontier's unbacked shapes

There are three major issues with unbacked shapes that must be addressed
before they are ready for Frontier.

The first is the **framework DDE problem**, which we have resolved in the
last year.

**Performance.** The second challenge is performance regressions when
using unbacked shapes. These regressions are mostly due to shortcomings in
the inductor's handling of unbacked shapes. On vLLM models, using unbacked
shapes led to around a 30% performance drop. And in a separate experiment
comparing backed and unbacked shapes on TorchBench Hugging Face models, we
observed regressions up to 85%. Improving unbacked performance and closing
this gap is a key goal for the first half of 2026.

**Expressive dispatch.** Finally, we need APIs that allow users to compile
multiple unbacked graphs with explicit shape constraints and automatically
dispatch among those graphs, especially in pre-compilation workflows.
While this is possible today, it relies on manual workarounds and custom
dispatch logic that should ideally be automated.

## References

- [`torch/fx/experimental/symbolic_shapes.py`](../../../torch/fx/experimental/symbolic_shapes.py) — symbolic shape infrastructure
- [`torch/_dynamo/variables/builder.py`](../../../torch/_dynamo/variables/builder.py) — unbacked symbol creation
