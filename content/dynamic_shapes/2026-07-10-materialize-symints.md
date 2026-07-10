---
title: "Stop Passing Raw SymInts to FX Graph Nodes — Use materialize_symints"
author: Laith Sakka (@laithsakka)
date: 2026-07-10
tags: [dynamic_shapes, fx, symint, correctness]
---

> **TL;DR** – If you're creating FX graph nodes that take symbolic shape
> values (`SymInt`, `SymFloat`, `SymBool`) as arguments, use the new
> `Graph.materialize_symints` / `create_size_node` / `create_stride_node` /
> `create_storage_offset_node` helpers instead of passing raw symbolic
> values directly. This fixes a common, subtle class of correctness bugs.

## The bug pattern

This is a pattern that's easy to write and hard to catch. Today it emits a
warning; it will become a hard error soon, once we've fixed some dependencies
in torch edge (executorch):

```python
# ❌ WRONG — raw SymInts passed directly as arguments
graph.call_function(
    torch.ops.aten.empty_strided.default,
    args=(val.size(), val.stride()),   # size()/stride() may be SymInts!
    kwargs={"dtype": val.dtype, "device": val.device},
)
```

Under dynamic shapes, `val.size()` and `val.stride()` return `SymInt`
values, not proper FX nodes. When those get stored on a node's `args`, they
are baked into the graph as opaque symbolic objects rather than as real
graph nodes. This breaks:

- **Symbolic reasoning** — downstream passes traverse the graph node-by-node.
  A `SymInt` buried in an arg tuple is invisible to them; it isn't a
  producer they can see, reason about, or rewrite.
- **Graph serialization** — a raw `SymInt` isn't a serializable FX
  `Argument`, so graphs carrying them can't round-trip.
- **Correctness** — passes that transform the graph silently skip these
  values, and the symbol may end up undefined when the graph is codegen'd
  or replayed (the classic `NameError: name 's48' is not defined` in
  `fx_graph_runnable` repros).

The bug is insidious because it often works at trace time (the Python value
is correct), then produces wrong behavior — or a hard failure — during
later graph transformations or replay.

## The fix

Materialize symbolic values as proper graph nodes before you use them as
args. There are two entry points depending on what you have in hand.

**If you have a tensor node and want one of its size/stride/offset symbols**,
use the targeted helpers:

```python
# ✅ CORRECT — emit an aten.sym_size.int node instead of a raw SymInt
dim_node = graph.create_size_node(node, dim)     # node.size(dim)
stride_node = graph.create_stride_node(node, dim)  # node.stride(dim)
offset_node = graph.create_storage_offset_node(node)  # node.storage_offset()
```

**If you have arbitrary `SymInt` values** (e.g. a whole `size()`/`stride()`
tuple, possibly with symbolic expressions like `s0 * 4`), use
`materialize_symints`. It takes a list and returns a list where each
`SymInt` becomes a `Node` and each plain `int` passes through unchanged:

```python
# ✅ CORRECT — materialize the whole size/stride tuples as graph nodes
with graph.inserting_before(output_node):
    size = graph.materialize_symints(val.size())
    stride = graph.materialize_symints(val.stride())
    n.replace_all_uses_with(
        graph.call_function(
            torch.ops.aten.empty_strided.default,
            args=(size, stride),
            kwargs={"dtype": val.dtype, "device": val.device},
        )
    )
```

> **Watch the insertion point.** Like every other node-creation API, these
> emit at the graph's current insertion point, which by default is the end
> of the graph — *after* the `output` node, leaving the new nodes orphaned.
> Scope the call with `with graph.inserting_before(graph.output_node()):`
> so the nodes land in the body and get wired into the graph.

## Available helpers

| Helper | Returns | Use case |
| --- | --- | --- |
| `Graph.materialize_symints(values)` | `list[Node \| int]` | Lower a list of `SymInt`/`int` values into FX subgraphs rooted at the existing producers of their symbols; plain ints pass through |
| `Graph.materialize_symint(value)` | `Node \| int` | Single-value convenience wrapper |
| `Graph.create_size_node(tensor, dim)` | `Node` | A node for `tensor.size(dim)` |
| `Graph.create_stride_node(tensor, dim)` | `Node` | A node for `tensor.stride(dim)` |
| `Graph.create_storage_offset_node(tensor)` | `Node` | A node for `tensor.storage_offset()` |

## create_size_node vs. materialize_symints: live query vs. freeze

These look similar but have different semantics, and picking the wrong one
is its own subtle bug.

- **`create_size_node(%x, 0)`** emits `aten.sym_size.int(%x, 0)` — a *live
  query* on `%x`. If a later pass mutates `%x`'s layout and `FakeTensorProp`
  re-runs, the node's `meta["val"]` is overwritten with the **new** stride.
  Use this when you want the value to track the producer.

- **`materialize_symints([...])`** walks the sympy expression and rebuilds
  it from the existing producer of each *symbol* (typically an input
  placeholder). The result is "what this symbol is at runtime," which is
  determined by the graph inputs and is **independent** of later layout
  changes to `%x`. Use this when you want to *freeze* the trace-time value.

A concrete example: in Inductor's `joint_graph` pass we want the frozen
size/stride of a to-be-eliminated meta tensor, so we use
`materialize_symints` — using `create_size_node(n, d)` there would query `n`
and pin it alive, blocking the `eliminate_dead_code()` that's supposed to
remove it.

## A real before/after

From the export runtime-assertions pass — replacing a hand-rolled
`sym_size.int` call with the helper:

```python
# Before
dim_node = module.graph.call_function(
    torch.ops.aten.sym_size.int, (node, dim), {},
)

# After
dim_node = module.graph.create_size_node(node, dim)
```

The helper also carries the tensor metadata forward onto the new node
(`meta["val"]` / `meta["example_value"]`), which the raw `call_function`
form did not.

## Migration status

- **Now:** Passing raw `SymInt`/`SymFloat`/`SymBool` values to
  `Graph.create_node(op=call_function/call_method/call_module)` emits a
  warning:

  ```
  Raw SymInt value (s0) passed as argument to Graph.create_node(...).
  Use create_*_node() helpers for tensor metadata queries or
  materialize_symints() for general symbolic expressions.
  ```

- **Soon:** This becomes a `RuntimeError` once all downstream consumers
  (including executorch) have migrated. That flip is staged as a follow-up
  so executorch's PyTorch pin can roll forward first.

## What you should do

1. If you maintain FX passes that create nodes with shape-dependent
   arguments, look for raw `SymInt` usage — anywhere you pass
   `x.size()` / `x.stride()` / `x.storage_offset()` (or expressions derived
   from them) into `args`/`kwargs`.
2. Replace with `create_size_node` / `create_stride_node` /
   `create_storage_offset_node` for direct tensor-metadata queries, or
   `materialize_symints(...)` for general symbolic expressions — remembering
   to scope the insertion point.
3. Run your tests — the warning will surface any remaining instances.

---

Diff: D107938876 &nbsp;|&nbsp; PR: [pytorch/pytorch#186665](https://github.com/pytorch/pytorch/pull/186665)

Questions? Ping me or drop a comment.
