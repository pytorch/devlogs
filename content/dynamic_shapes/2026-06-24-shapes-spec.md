---
title: "ShapesSpec: A Unified, Descriptive Dynamic-Shapes API"
author: Laith Sakka (@laithsakka), Xiao Fu (@fxdawnn)
date: 2026-06-24
tags: [dynamic_shapes, unbacked, export]
---

> **TL;DR** – A new dynamic shapes API is available and ready to use! It provides a unified, consistent way for specifying dynamic specs across compile, export and make_fx, brings native unbacked support to torch.export and make_fx, and completes the unbacked story [described earlier](./2026-01-20-backed-to-unbacked.md) by providing unified, predictable, declarative control over the shapes — and dispatch behavior — of compiled artifacts.

## Motivating example

Consider the example below: the user has a function project(x, w) with a fast path for small batches and a general matmul path. The user wants to compile a dynamic-shape artifact that takes the fast path.

The ShapesSpec says x has dynamic shape [B, D] and w has shape [D, D], and the assumption B < 32 commits this artifact to the fast branch for small sizes.

```python
def project(x, w):
    if guard_or_false(x.shape[0] < 32):
        return (x.unsqueeze(-1) * w).sum(dim=-2)       # fast path
    return x @ w                                       # general path


B = ShapeVar("B", optimization_hint=8)   # optimization_hint guides guardless optimizations
D = ShapeVar("D", optimization_hint=64)

spec = ShapesSpec(
    ParamsSpec({
        "x": TensorSpec([B, D]),
        "w": TensorSpec([D, D]),
    }),
    assumptions=[B < 32],          # commits this artifact to the fast path
)

example = (torch.randn(8, 64), torch.randn(64, 64))

compiled = torch.compile(project, dynamic_shapes=spec)
ep       = torch.export.export(project, args=example, dynamic_shapes=spec)
gm       = make_fx(project, dynamic_shapes=spec, tracing_mode="fake")(*example)
```

The new API isn't syntactic sugar over mark_unbacked — it's a real spec language, with the following features:

## Properties

**Unified across compiler entry points.** The same API works for torch.compile, torch.export, make_fx, and python_export (not yet landed) — potentially more in the future. One spec object, every entry point.

**Unbacked only.** Unbacked shapes align naturally with the idea of a spec: the user states properties of the shapes up front, and the compiler is bound by them with a guarantee that no silent specializations happen.

By contrast, dynamic markings for backed shapes act more like hints — the compiler can still specialize silently during export time. Existing validations like min/max constraints help in narrow cases but are not complete solutions.

**Assumptions up front.** The new API allows specifying constraints the user already knows about their shapes and the relations between dynamic inputs — ranges, equalities, divisibility, ordering — as assumptions. Assumptions are used to (1) avoid DDEs and remove the need to sprinkle torch._check calls inside the model, and (2) generate specialized kernels that pick up optimizations based on the assumed properties.

**Derived expressions.** A LeafSpec can also be a derived expression, e.g.

```python
ParamsSpec({
    "x": TensorSpec([B,         STATIC]),     # B is bound here (bare)
    "y": TensorSpec([2 * B + 1, STATIC]),     # implies y.shape[0] == 2*B + 1
})
```

The compiler reads B from x.shape[0] and implicitly assumes y.shape[0] == 2 * B + 1 — equivalent to writing that equality into assumptions=[...].

**Dynamic shapes as a property of the compiled function.** The new API allows users to specify dynamic shapes as a property of the compiled function by attaching the spec to the function definition.

```python
@dynamic_spec(spec)  # spec defined in the example above
def project(x, w):
    if guard_or_false(x.shape[0] < 32):
        return (x.unsqueeze(-1) * w).sum(dim=-2)
    return x @ w


compiled = torch.compile(project)
ep       = torch.export.export(project, args=example)
gm       = make_fx(project, tracing_mode="fake")(*example)
```

**Complete spec language.** Supports tensors (TensorSpec), scalar int arguments (IntVar / ShapeVar), dicts (DictSpec), lists / tuples (SeqSpec), and user-defined objects of any Python class (ObjectSpec).

## Export now supports unbacked

This is a long-overdue addition. Unbacked shapes are a natural fit for export because they guarantee no silent specialization during export time — which is an important export property. The current workaround is backed_size_oblivious, which is incomplete.

While NamedDim + backed_size_oblivious is export-time sound, it's clunky. The new API replaces it with a cleaner story:

- Assumptions are a first-class field on ShapesSpec, instead of torch._check calls scattered through the model or embedded constraints.
- Invariants the model already establishes with torch._check are honored automatically — with NamedDim you had to restate them as explicit assumptions.
- No need to explicitly turn on backed_size_oblivious.

## Dispatching to multiple specialized artifacts

A common use case is to compile several specialized artifacts of the same function and dispatch to them depending on input at runtime. For now dispatch is the user's responsibility — the spec gives you the building block (specialized artifacts, cleanly separated), but selecting among them at runtime is yours to wire up. A higher-level dispatch abstraction is plausible follow-up work.

**How to dispatch.** Say you have a function f and want specialized artifacts for B == 1, B == 3, a B > 100 version, plus a generic fallback. With the new API this is a few lines — compile one torch.compile wrapper per assumption set and dispatch on the same predicates at runtime:

```python
def f(x, weight):                            # stand-in for your function
    ...

B = ShapeVar("B")
params = {
    "x":      TensorSpec([B, STATIC, STATIC]),
    "weight": TensorSpec([STATIC]),
}

cases = [
    (1,   [lambda B: B == 1]),
    (3,   [lambda B: B == 3]),
    (256, [lambda B: B > 100]),
    (7,   []),                               # generic fallback
]

kernels = []
for example_B, predicates in cases:
    spec = ShapesSpec(
        params=params,
        assumptions=[p(B) for p in predicates],
    )
    kernels.append(
        torch.compile(
            f,
            dynamic_shapes=spec,
            isolate_recompiles=True,         # each spec gets its own cache bucket
        )
    )

def dispatch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    for kernel, (_, predicates) in zip(kernels, cases):
        if all(p(x.shape[0]) for p in predicates):
            return kernel(x, weight)
    raise AssertionError(f"no kernel for B={x.shape[0]}")
```

Note that isolate_recompiles=True matters here. See [Dynamo: isolate recompiles for torch.compile](../dynamo/2026-05-04-Dynamo-Isolate-Recompiles.md) to understand why it is needed.
