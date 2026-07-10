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
> values directly. This fixes a common, subtle class of correctness bugs —
> we've already found **3 in PyTorch Inductor** and **6 across executorch's
> ARM backend passes**. **Passing raw symbolic values will become a hard
> error soon**, so migrate now.

## The bug pattern

This is a pattern that's easy to write and hard to catch. Today it emits a
warning; it will become a hard error soon, once we land the executorch fixes:

```python
# `val` is the example/fake tensor stored on a node's meta — under dynamic
# shapes its size()/stride() are SymInts, not plain ints.
val = n.meta["val"]   # e.g. FakeTensor with shape (s0, s1)

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
graph nodes.

**The core invariant this breaks:** every `SymInt` in a graph must have a
*producer* — the graph has to know where the value comes from at runtime.
A symbol like `s0` is only meaningful because something produces it: an
input integer placeholder, or an input tensor's size (`x.size(0)`), stride,
or storage offset. If you store the raw `SymInt` directly, you throw that
provenance away — you keep the symbol but lose the "where do I get it from"
link. The correct representation is a *reference to the node that produces
it*: the placeholder node for an integer input, or a `x.size(index)` /
`x.stride(index)` node for a tensor dimension. That's exactly what the
helpers below emit.

Concretely, storing raw `SymInt`s breaks:

- **Symbolic reasoning** — downstream passes traverse the graph node-by-node.
  A `SymInt` buried in an arg tuple is invisible to them; it isn't a
  producer they can see, reason about, or rewrite.
- **Graph serialization** — export's serializer records each symbol by its
  *producer* (the input or node it comes from). A raw `SymInt` in an arg
  tuple has no producer node, so there's nothing to reference on the way out
  and nothing to re-bind on deserialize — the graph can't round-trip.
- **Correctness** — passes that transform the graph silently skip these
  values, and because the symbol has no producer node it can end up
  undefined when the graph is codegen'd or replayed (the classic
  `NameError: name 's48' is not defined` in `fx_graph_runnable` repros).

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

These look similar, and in the common case they emit the *same* node — but
they are not equivalent, and picking the wrong one is its own subtle bug.

The distinction is **who the producer is**:

- **`create_size_node(%x, 0)`** unconditionally emits
  `aten.sym_size.int(%x, 0)` — a *live query* on `%x`, whatever `%x` is.

- **`materialize_symints([s])`** looks up the actual *producer* of the
  symbol `s` (by scanning placeholders / unbacked bindings) and roots the
  subgraph there — typically the input placeholder the symbol originates
  from.

> **If `%x` is itself the producer, there is no difference.** When `%x` is
> the input placeholder the symbol originates from (`s0 == x.size(0)`), both
> calls emit the exact same `aten.sym_size.int(%x, 0)` node. The two only
> come apart in the case below.

They diverge when **`%x` is not the producer** — when it's some intermediate
node that merely happens to carry that shape. Then `create_size_node` pins a
live query on that intermediate `%x`: if a later pass mutates `%x`'s layout
and `FakeTensorProp` re-runs, the node's `meta["val"]` is overwritten with
the **new** stride. `materialize_symints`, by contrast, roots at the
symbol's true origin (the graph input), so the value is "what this symbol is
at runtime" — determined by the inputs and **independent** of later layout
changes to `%x`. Use `create_size_node` when you want a live query on a
specific node; use `materialize_symints` when you want to *freeze* the
trace-time value.

A concrete example: in Inductor's `joint_graph` pass we want the frozen
size/stride of a to-be-eliminated meta tensor, so we use
`materialize_symints` — using `create_size_node(n, d)` there would query `n`
and pin it alive, blocking the `eliminate_dead_code()` that's supposed to
remove it.

> **TODO:** We could potentially auto-materialize raw SymInt args instead of
> hard-erroring on them. For now we ask callers to do it explicitly, because
> of exactly the distinction above — only the caller knows whether they want
> a live query (`create_size_node`) or a frozen value (`materialize_symints`),
> and picking wrong is a silent correctness bug.

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

PR: [pytorch/pytorch#186665](https://github.com/pytorch/pytorch/pull/186665)
