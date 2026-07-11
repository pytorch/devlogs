---
title: "Making PT2 Symbolic Tracing Reliable for Distributed Workloads"
author: "Sanket Purandare (@sanketpurandare)"
date: 2026-07-10
tags: [dynamic_shapes, unbacked, distributed, dtensor, flex_attention, inductor, tracing]
---

> **TL;DR** – To capture a whole distributed training step as one FX graph, PT2
> has to trace models whose **batch and sequence dimensions are unbacked
> SymInts**. When we tried this on the TorchTitan DeepSeek-V3 MoE trainer, every
> layer of the stack either silently specialized those dims to concrete ints or
> blew up with a data-dependent error (DDE). This post walks through the 11
> fixes — spanning ATen meta kernels, ProxyTensor, the ShapeEnv, Inductor,
> collective bucketing, DTensor, and FlexAttention — that make each layer keep
> tensor **semantics symbolic** while allowing **hints for policy decisions
> only**. That contract is what unblocked end-to-end symbolic tracing for the
> **graph trainer** and **expert-parallel (EP) compute/communication overlap**.

## Background / Motivation

The **graph trainer** captures an entire training step — forward, backward,
optimizer, *and* collectives — as a single FX graph, then rewrites that graph to
go faster: bucketing collectives, chunking work across the token grid, and
overlapping expert-parallel (EP) communication with expert compute.

The workload driving this effort is EP-overlap for a large MoE model
(DeepSeek-V3). To overlap the all-to-all / all-gather / reduce-scatter traffic of
expert parallelism with compute, the graph pass **chunks the token grid** into
slices. Those slices are not nice round numbers — they are *derived symbolic
expressions* over the unbacked batch/sequence symbols, things like `u0 // 2`,
`2 * (u0 // 2)`, and `(u1 + 1) // 2`.

For the captured graph to be valid across variable-length batches, the batch and
sequence dimensions must stay **symbolic and unbacked through the entire trace**.
The moment any layer specializes them to a concrete integer, two things break:

1. The graph is only correct for the one shape it happened to be traced with.
2. Rewrites that depend on *derived* symbolic extents (EP-overlap chunking) can no
   longer be expressed at all.

But PT2's tracing path was full of spots that quietly assumed a concrete integer.
They failed in two recurring ways:

- **Forced specialization** — a path calls `.numel()`, `sizes()`, `int64_t` shape
  plumbing, or `guarding_hints_or_throw()`, and turns an unbacked symbol into a
  concrete value behind your back.
- **Spurious DDE guards** — a path evaluates a Python `bool` on an unbacked
  expression (`8 < 2*u0`, a stride comparison, a contiguity check) and raises
  `GuardOnDataDependentSymNode`, aborting the trace even though the op is perfectly
  representable symbolically.

Every subsystem the trainer touches — matmul/SDPA meta kernels, FlexAttention,
DTensor layout propagation, collective bucketing, Inductor codegen — had at least
one of these. This is the story of removing them, layer by layer, and of the
single contract that ties the fixes together.

## Design / Approach

### The one contract

One principle runs through all 11 fixes and is the main reusable takeaway:

> **Semantics stay symbolic; hints are for policy only.**
> A value that describes *runtime tensor behavior* — a size, a stride, a split, a
> reshape — must remain symbolic. An optimization hint may be consumed *only* where
> a Python integer is genuinely required to **choose an optimization** (bucket byte
> sizing, foreach grouping, layout scoring). On those policy paths, an *unhinted*
> unbacked symbol **fails fast with a clear error** rather than guessing.

A hint is never a promise that a dimension really is that value. Everything else
falls out of this:

- **Guard-or-false over hard bools.** Unprovable relations prune candidates or take
  the safe branch (`TORCH_GUARD_OR_FALSE`, `guard_or_false`, `false_if_dde=True`)
  instead of raising DDE.
- **Bind symbols at their real binding site.** Fallback kernels and nested traces
  register fresh unbacked symbols through the normal `compute_unbacked_bindings`
  path instead of create-then-rediscover.
- **Don't lie about layout.** When you take a symbolic shortcut, emit `torch._check`
  assertions for the relations you're assuming.
- **Isolate nested traces.** A nested `make_fx` / bucketing trace owns only the
  symbols it produces; snapshot and restore ambient ShapeEnv state around it.

The rest of this section walks the stack bottom-up, because if a lower layer
specializes, nothing above it can stay symbolic.

### 1. ATen / FakeTensor meta kernels — keep the math symbolic

The meta and decomposition kernels are the foundation. Two of them were still
forcing concrete sizes on the matmul/attention path.

The **folded matmul** path (ND @ 1D/2D) forced concrete sizes twice — an
empty-tensor `numel()` check and a `DimVector` fold/view built from `sizes()`.
Reworked to the symbolic equivalents (`sym_numel()`, `sym_sizes()`,
`sym_strides()`, `reshape_symint()`, `_unsafe_view_symint()`, symbolic output
resize), with contiguity guarded through `TORCH_GUARD_OR_FALSE`. It also lets fake
view metadata recover from *hinted* unbacked contiguity relations — e.g. a shape
like `2 * (u0 // 2)` whose stride is expressed through `u0` — by emitting symbolic
equality checks that hold at the traced hint instead of specializing `u0`.
([#183397](https://github.com/pytorch/pytorch/pull/183397), fixes
[torchtitan#3322](https://github.com/pytorch/torchtitan/issues/3322))

**SDPA** lowers through batched matmul, and the broadcasted bmm path still
converted batch dims to `int64_t` before expand/reshape/view — breaking rank-4
inputs with an unbacked batch symbol, including math SDPA. Carried symbolic sizes
through with `sym_sizes()`, `SymDimVector`, `infer_size_symdimvector()`,
`expand_symint()`, `reshape_symint()`, `_unsafe_view_symint()`. Separately,
`aten.size.default` has a non-symbolic `int[]` schema, so running it on a fake
tensor with unbacked sizes *forces* the symbol; during torch-function metadata
tracing we redirect it to `aten.sym_size.default` so downstream consumers get
symbolic size proxies (this covers cuDNN SDPA fake outputs whose batch dim stays
symbolic). ([#183398](https://github.com/pytorch/pytorch/pull/183398), fixes
[torchtitan#3324](https://github.com/pytorch/torchtitan/issues/3324))

Finally, trace tooling serializes FakeTensor storage metadata to JSON. We keep the
`size` field symbolic and add a *separate* `size_hint` field only when every free
symbol in the storage-size expression has an explicit hint override — giving
diagnostic/policy tooling a concrete expected extent without ever specializing the
shape. ([#183839](https://github.com/pytorch/pytorch/pull/183839), fixes
[#183835](https://github.com/pytorch/pytorch/issues/183835))

### 2. Inductor codegen — bind and order unbacked extents without guarding

Several Inductor codegen paths needed to reason about unbacked extents without
turning policy into semantic guards
([#183840](https://github.com/pytorch/pytorch/pull/183840)):

- **Stride ordering:** a stride-specific symbolic `>=` proof (ordinary guarded
  comparisons plus divisibility reasoning) proves layouts like `u0 * 256 >= 256`
  without hints; when `require_strides` shows the current layout already satisfies
  the requested order, freeze it directly instead of calling
  `guarding_hints_or_throw()` or bailing on every unbacked stride.
- **Fallback outputs:** temporarily re-enable fresh unbacked symbol tracking while
  rerunning the fallback fake kernel — that call *is* the binding site for output
  size/stride symbols that wrapper code later references, so they should go through
  the normal pending-symbol / `compute_unbacked_bindings()` path.
- **C++ wrapper:** treat input unbacked symbols as already declared before emitting
  output bindings (otherwise a fallback output can redeclare `u0`), and emit C++
  integer division instead of Python-only `__floordiv__` for `DivideByKey`.
- Stop making the post-copy stride-order sanity check prove data-dependent
  inequalities on a layout that's already been materialized.

(fixes [#183834](https://github.com/pytorch/pytorch/issues/183834),
[#185341](https://github.com/pytorch/pytorch/issues/185341))

### 3. Collective bucketing — isolate and tolerate symbols

Bucketing rewrites collectives with nested `make_fx` traces, and it turned out to
be a hotspot for *both* failure modes.

First, isolation. Bucketing reuses the surrounding `FakeTensorMode` / `ShapeEnv`.
When that env already has pending fresh unbacked symbols from an outer dynamic
trace, `compute_unbacked_bindings` tries to account for symbols that have nothing
to do with the bucket's inputs and outputs, and errors. The fix snapshots and
clears pending/ignorable fresh unbacked symbols around the nested bucketing trace,
then restores the ambient state afterward — so bucketing is responsible only for
the symbols it produces.
([#183495](https://github.com/pytorch/pytorch/pull/183495), fixes
[#183679](https://github.com/pytorch/pytorch/issues/183679))

Then, tolerance — the diff that crystallized the whole project's contract.
Bucketing uses tensor metadata for two distinct purposes:

- **Semantic** shape paths — `x.numel()`, `x.shape[0] // group_size`, split sizes,
  `torch.empty` extents, narrow offsets/lengths, reshape shapes — stay symbolic, so
  hinted chunk expressions like `u0 // 2` survive through all-gather and
  reduce-scatter merge graphs.
- **Optimization-policy** paths — bucket byte-size accounting and foreach grouping —
  may use concrete hints, but *unhinted* unbacked symbols fail fast with a clear
  bucketing error instead of falling back to heuristics or forcing a guarding
  specialization.

([#183544](https://github.com/pytorch/pytorch/pull/183544), fixes
[#183676](https://github.com/pytorch/pytorch/issues/183676))

### 4. Dynamic Shapes / ShapeEnv core — the substrate

`rebind_unbacked` already recorded equivalences when a retraced binding mapped an
unbacked symbol to another symbol or to a constant, but it asserted that any
non-symbol replacement with free symbols was invalid. That is too strong for
legitimate *derived* unbacked shapes like `(u1 + 1) // 2`: the old symbol still has
a concrete binding relationship and should be eliminated in favor of that
expression. We now record the replacement via `_eliminate_unbacked` (the existing
path for replacing an unbacked symbol by a non-symbol expression). Notably, this
fixed the FlexAttention/HOP reproducer *without* restoring the previously-reverted
broad HOP fake-trace suppression that had caused cond / AOTInductor / Executorch /
FlexAttention regressions.
([#183837](https://github.com/pytorch/pytorch/pull/183837), fixes
[#183677](https://github.com/pytorch/pytorch/issues/183677))

Non-strict tracing can also receive *raw* SymInt inputs from an outer fake-tensor
trace — this is the FlexAttention `BlockMask` path used by graph chunking, where
tensor sizes and raw SymInt captures refer to the same outer unbacked sequence
length. We taught `ShapeEnv` to **transfer a foreign unbacked SymInt expression
into the local ShapeEnv**: it first rebuilds the expression from any
already-transferred foreign symbols, and for any unresolved remainder mints one
opaque local unbacked symbol for the whole foreign expression, recording its
source, range, and hint. A shared cache means tensor dimensions and scalar
captures that originate from the same foreign symbols preserve their sharing. Guard
printing now emits `torch.sym_max` / `torch.sym_min` for SymPy `Max`/`Min` so
guards evaluate symbolically instead of via Python truthiness. Raw foreign wrapping
stays gated to non-strict tracing, and data-dependent *branching* on the
transferred symbol still fails through the normal DDE path rather than specializing
on a hint. ([#187273](https://github.com/pytorch/pytorch/pull/187273), fixes
[#187272](https://github.com/pytorch/pytorch/issues/187272))

### 5. ProxyTensor / make_fx / Regional Inductor — preserve provenance at the boundary

Metadata should be preserved at the boundary where a value *becomes* an FX proxy,
not rediscovered later in FlexAttention or some other downstream subsystem
([#187231](https://github.com/pytorch/pytorch/pull/187231)):

- Route each `node.meta["unbacked_bindings"]` entry along its full output path to
  the proxy that owns the tensor leaf, so provenance survives when multiple inputs
  share token-grid dimensions.
- Record `meta["val"]` for primitive **scalar** placeholders that already have FX
  proxies — the scalar analogue of the existing tensor/SymInt placeholder path — so
  scalar captures stay visible to subgraph consumers without Flex-specific patching.
- Pass `donate_graph_module=True` in Regional Inductor to avoid deepcopying graph
  modules it already owns; deepcopying symbolic fake tensor metadata can call
  `numel()` on symbolic sizes and strides.

(fixes [#187230](https://github.com/pytorch/pytorch/issues/187230))

### 6. DTensor — recognize equivalent symbolic layouts

Compiled DTensor paths can see symbolic local layout metadata that is semantically
valid but not syntactically identical to what was saved during forward
propagation. For `to_local()` backward, AOTAutograd can produce a local gradient
stride like `(Max(1, u3), 1)` while the saved DTensor spec uses `(u1, 1)`;
recomputing the global gradient stride forced `compute_global_tensor_info()` to
evaluate symbolic stride relations and raised a DDE for the default same-placement
backward.

We now reuse the original spec for that default backward when the strides are
compatible: provable equality is accepted directly, contiguous symbolic forms like
`Max(1, u*)` go through `check_contiguous_sizes_strides(..., false_if_dde=True)`,
and otherwise we emit `torch._check` assertions for the required stride equalities
before taking the shortcut. Placement changes and uneven channels-last shards keep
the recomputation path so autograd can repair the physical layout. The same class
of issue in `aten.t` sharding propagation is fixed by registering transpose with
`allow_unbacked_sharding=False`, so unproven candidates (`8 < 2*u0`) are pruned
instead of being evaluated as a Python bool.
([#187026](https://github.com/pytorch/pytorch/pull/187026), fixes
[#187025](https://github.com/pytorch/pytorch/issues/187025))

### 7. FlexAttention HOP — the top of the stack

Finally, the FlexAttention `BlockMask` path itself — handling unbacked symbolic
predicates and scalar shape captures. This is the direct consumer of everything
below it, and the workload that motivated the whole stack.
([#183838](https://github.com/pytorch/pytorch/pull/183838), fixes
[#183833](https://github.com/pytorch/pytorch/issues/183833))

## Results / Benchmarks

With the full stack landed:

- The **DeepSeek-V3 MoE trainer traces end-to-end** with unbacked batch/sequence
  dimensions, through both the SDPA and FlexAttention paths.
- **EP-overlap chunking works**: derived symbolic extents (`u0 // 2`,
  `2 * (u0 // 2)`, `(u1 + 1) // 2`) flow through matmul, attention, DTensor
  layouts, collective bucketing, and Inductor codegen without specialization or DDE
  errors.
- The **graph trainer** stack is unblocked — whole-step graph capture and rewrite
  is now possible on this workload.

Each fix landed with targeted regression coverage: fake-tensor unit tests
(rank-3/4 unbacked matmul, math + cuDNN SDPA, symbolic `aten.size.default`,
hinted-view contiguity), dynamic-shapes tests (foreign-ShapeEnv transfer,
rebind-to-expression, symbolic `sym_max`/`sym_min` guard printing), Inductor
unbacked-symint tests (stride ordering, fallback binding reuse), collective
bucketing trace tests, and DTensor compile tests. On top of the unit coverage, the
TorchTitan graph_trainer H100 integration run
(`aot_fx_trace_deepseek_v3_sdpa_full_inductor_ep_overlap_moe_seq`) exercises the
full path end-to-end.

## Open questions / Future work

- **Generalize raw unbacked SymInt inputs beyond non-strict tracing.** Raw foreign
  wrapping is currently gated to non-strict tracing; general Dynamo still raises the
  existing unsupported case until the broader guard-propagation behavior is
  verified.
- **Harden remaining ATen ops** that still assume concrete shapes, as new
  distributed workloads exercise more of the operator surface.
- **Document the contract.** Fold the "semantics symbolic / hints for policy only"
  rule into contributor-facing docs so new codegen and sharding paths follow it by
  default instead of reintroducing specialization.

## References

- [#183397](https://github.com/pytorch/pytorch/pull/183397) — [ATen][FakeTensor] Handle unbacked dims in folded matmul; fixes [torchtitan#3322](https://github.com/pytorch/torchtitan/issues/3322)
- [#183398](https://github.com/pytorch/pytorch/pull/183398) — [ATen][ProxyTensor] Preserve unbacked batch dims in SDPA tracing; fixes [torchtitan#3324](https://github.com/pytorch/torchtitan/issues/3324)
- [#183495](https://github.com/pytorch/pytorch/pull/183495) — [Inductor][Bucketing] Isolate bucketing traces from ambient unbacked symbols; fixes [#183679](https://github.com/pytorch/pytorch/issues/183679)
- [#183544](https://github.com/pytorch/pytorch/pull/183544) — [Inductor][Bucketing] Make collective bucketing tolerate hinted unbacked SymInts; fixes [#183676](https://github.com/pytorch/pytorch/issues/183676)
- [#183837](https://github.com/pytorch/pytorch/pull/183837) — [Dynamic Shapes] Rebind unbacked symbols to derived expressions; fixes [#183677](https://github.com/pytorch/pytorch/issues/183677)
- [#183838](https://github.com/pytorch/pytorch/pull/183838) — [Inductor][HOP] Handle unbacked FlexAttention predicates; fixes [#183833](https://github.com/pytorch/pytorch/issues/183833)
- [#183839](https://github.com/pytorch/pytorch/pull/183839) — [FakeTensor] Add hinted symbolic storage size metadata; fixes [#183835](https://github.com/pytorch/pytorch/issues/183835)
- [#183840](https://github.com/pytorch/pytorch/pull/183840) — [Inductor] Handle hinted and fallback unbacked symbols; fixes [#183834](https://github.com/pytorch/pytorch/issues/183834), [#185341](https://github.com/pytorch/pytorch/issues/185341)
- [#187026](https://github.com/pytorch/pytorch/pull/187026) — [DTensor] Preserve symbolic local layouts without DDE guards; fixes [#187025](https://github.com/pytorch/pytorch/issues/187025)
- [#187231](https://github.com/pytorch/pytorch/pull/187231) — [PT2] Preserve symbolic metadata across tracing; fixes [#187230](https://github.com/pytorch/pytorch/issues/187230)
- [#187273](https://github.com/pytorch/pytorch/pull/187273) — [Dynamo] Trace raw unbacked SymInt inputs; fixes [#187272](https://github.com/pytorch/pytorch/issues/187272)
- [Slaying Framework Data-Dependent Errors Dragon](./2025-10-29-slaying-framework-ddes.md) — related unbacked-shapes work
- [Stop Passing Raw SymInts to FX Graph Nodes](./2026-07-10-materialize-symints.md) — related SymInt-provenance work
