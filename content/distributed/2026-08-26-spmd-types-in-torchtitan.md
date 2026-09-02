---
title: "SPMD types in TorchTitan"
date: 2026-08-26
author: "Pian Pawakapan (@pianpwk)"
tags: [torchtitan, distributed, dtensor, spmd, sharding]
---

> **TL;DR**
>
> TorchTitan now uses [`spmd_types`](https://github.com/meta-pytorch/spmd_types) as its default backend for distributed model computation. Authors specify sharding contracts and collectives explicitly, with optional typechecking to catch distributed-correctness errors during development.
> At runtime, the typechecking machinery can be erased so forward and backward execute on plain tensors. Coupled FWD-BWD typing and support for both global and local SPMD also make distributed behavior easier to express and reason about.
> Across our repeated debug-model benchmarks, `spmd_types` improved eager throughput by up to 46% and Inductor throughput by 5-9% over `partial_dtensor`, without a meaningful peak-memory increase. FSDP2 continues to use DTensor as the persistent storage representation outside model computation.

TorchTitan aims for a unified model definition across single-GPU execution and any
1D-to-ND composition of parallelisms, for both training and inference. A model's `forward()`
should describe its computation as if running on a single device, while composing well
with any combination of parallelisms: FSDP, CP, TP, PP, or EP.

Historically, this meant keeping the forward definition free of distributed code and collectives,
while DTensor provided the global SPMD (GSPMD) programming model. Model code read like
single-device code. At compute time, DTensors carried a `DeviceMesh` and `Replicate/Shard/Partial`
placements describing their distribution across GPUs, while sharding propagation (shard prop)
computed and implicitly inserted any required collectives to make the program correct.
Previous designs also relied on DTensor-based context managers to locally handle
[context parallelism](https://github.com/pytorch/torchtitan/pull/4218),
[loss parallelism](https://github.com/pytorch/torchtitan/pull/3694), and
[vocab-parallel embedding](https://github.com/pytorch/torchtitan/blob/cdf0c9908698631600ca15243e9a5f3dc1b4fe1f/tests/unit_tests/cpu/test_embedding.py#L133-L148).

GSPMD abstractions kept model code clean, but as Titan model support and use cases expanded, we encountered more problems with DTensor
in compute regions, especially concerning runtime overhead, BWD expressivity, and local region bugs.

## DTensor pains in FWD/BWD

**Runtime overhead.** As a tensor subclass, DTensor intercepts every compute operation at the `__torch_dispatch__` level,
where it runs sharding propagation to check input placements and implicitly reshard or insert collectives as needed,
executes the operation on inner tensors, and then rewraps the result with output placements and other metadata.
Shard prop caching makes repeated operations cheaper, improving steady-state performance,
while compilation or CUDA graphs move this work out of runtime, although overhead is still paid during warmup.
In eager mode, shard prop can be prohibitively expensive,
especially in inference scenarios where varying input token counts frequently cause cache misses,
or for placements such as `_StridedShard`, whose redistribution planning on large meshes can take 10-20 seconds per uncached operation.

**Implicit redistributions.** When an operator's input placements don't match existing sharding rules, DTensor calculates
the cheapest way to redistribute inputs to get a globally correct output, implicitly adding collectives if needed.
This can be desirable during early authoring, when distributed correctness takes priority over explicit control,
but it is also often a source of surprising sharding decisions and performance regressions, especially when rules are missing on
the framework side. In upstream PyTorch, we have since added `ExplicitRedistributionContext` to error out when detected,
and have more than doubled operator coverage this past year, but historically this has been a source of pain.

**Forward and backward placements are independent.** By design, sharding propagation is executed separately for the FWD and BWD
of the same operation. Gradient placements therefore aren't constrained by activation or parameter placements:
the autograd graph enforces global shape consistency, but sharding and local shapes are opaque to it.
Occasionally this means gradient reductions can occur in surprising spots, and in extreme cases FWD/BWD sharding can be on
different tensor dimensions.

As `__torch_dispatch__` intercepts after the autograd graph has been constructed, coupled FWD-BWD behavior has been difficult
to retroactively enforce on the framework side, and difficult to author on the user side without swapping DTensor out in favor of
custom autograd functions.

**Local-region boundaries.** DTensor APIs like `local_map` or `to_local` let authors drop into local regions when compute shapes are inexpressible with standard sharding (e.g. MoE EP, uneven multimodal shapes), or when computation is not GSPMD
(e.g. TP vocab-parallel CE or embedding). Placements are completely untracked inside local regions, making re-annotation
on exit bug-prone, and incoming gradient placements difficult to reason about.

## `spmd_types`: erasure mode, explicit-only, coupled FWD-BWD, local+global SPMD

Motivated by these pain points, we developed `spmd_types`, a standalone companion library that preserves
DTensor-style GSPMD placement-based authoring with different distributed characteristics:

**Explicit collectives.** As of August 2026, `spmd_types` does not insert communication implicitly. Authors write
the required collective directly—for example, `spmd.redistribute(x, "tp", src=spmd.S(i), dst=spmd.R)` for an
all-gather—and the typechecker verifies that its source and destination types are valid.

**Erasure mode.** Shard-propagation rules are used for validation rather than runtime planning. During development,
authors annotate model inputs and parameters on mesh axes like DP, CP, TP, and EP, then run the program with
typechecking enabled to validate operator inputs, outputs, and collectives. Fake process groups allow multi-GPU setups
to be typechecked on a single device. In production training, the typechecker is disabled,
leaving the validated plain-tensor program and its explicit collectives with no added overhead.

**Coupled forward and backward behavior.** Differentiable operations such as `redistribute`, `convert`, and
`all_gather` are implemented with custom autograd functions. The transition selected in forward therefore fixes the
corresponding backward collective, rather than asking shard propagation to make a new placement decision during
backward. Typechecking observes the forward call at the `__torch_function__` level and validates this contract before
the autograd function executes. Forward and backward types are paired as follows:

| Forward | Backward |
| --- | --- |
| Replicate (`R`) | Partial (`P`) |
| Partial (`P`) | Replicate (`R`) |
| Invariant (`I`) | Invariant (`I`) |
| Varying (`V`) | Varying (`V`) |

On each mesh axis, `R` means replicated data whose independently computed gradients are `P`; `P` means pending contributions whose gradient is `R`.
`I` means identical data *and* identical computation on every rank, so its gradient remains `I`. `V` only promises that values may vary by rank and
therefore remains `V` in backward.

`S(i)` (the global SPMD refinement of `V`) is mirrored in BWD. Collective types are similarly tied:

| Fwd Type | Forward | Bwd Type | Backward |
| --- | --- | --- | --- |
| `R -> V` | `convert(R,V)` | `V -> P` | `convert(V,P)` |
| `R -> P` | `convert(R,P)` | `R -> P` | `convert(R,P)` |
| `I -> R` | `convert(I,R)` | `P -> I` | `all_reduce(I)` |
| `V -> R` | `all_gather(R)` | `P -> V` | `reduce_scatter()` |
| `V -> V` | `all_to_all()` | `V -> V` | `all_to_all()` |
| `V -> R` | `all_reduce(src=V,R)` | `P -> R` | `all_reduce(R)` |
| `V -> V` | `reduce_scatter(src=V)` | `V -> R` | `all_gather(R)` |
| `P -> R` | `all_reduce(R)` | `P -> R` | `all_reduce(R)` |
| `P -> V` | `reduce_scatter()` | `V -> R` | `all_gather(R)` |
| `P -> I` | `all_reduce(I)` | `I -> R` | `convert(I,R)` |
| `V -> I` | `all_gather(I)` | `I -> V` | `convert(I,V)` |
| `I -> V` | `convert(I,V)` | `V -> I` | `all_gather(I)` |

A Megatron-TP MLP example with just the `forward()`:

```python
# x: sequence-sharded if SP is on; invariant otherwise
# w1: colwise TP; w2: rowwise TP

# annotate types (optional at runtime)
x_tp = spmd.S(1) if enable_sp else spmd.I
x_BLD = spmd.assert_type(x_BLD, {tp_axis: x_tp})
w1_DF = spmd.assert_type(w1_DF, {tp_axis: spmd.S(1)})
w2_FD = spmd.assert_type(w2_FD, {tp_axis: spmd.S(0)})

# compute:
# enter TP region -> colwise -> silu -> rowwise -> all-reduce or reduce-scatter back
x_BLD = spmd.redistribute(x_BLD, tp_axis, src=x_tp, dst=spmd.R)
x1_BLF = x_BLD @ w1_DF
hidden_BLF = F.silu(x1_BLF)
out_BLD = hidden_BLF @ w2_FD
out_BLD = spmd.redistribute(out_BLD, tp_axis, src=spmd.P, dst=x_tp)
```

With sequence parallelism on, inter-block activations are `S(1)`. Forward executes an all-gather, colwise matrix multiply,
SiLU, rowwise matrix multiply, and reduce-scatter. Backward is the mirror image: the forward `S(1) -> R` all-gather fixes a
`P -> S(1)` reduce-scatter in backward, while the forward `P -> S(1)` reduce-scatter fixes an `S(1) -> R` all-gather in backward.
Weights and intermediate activations retain the same shard dimension in their gradients.

With sequence parallelism off, inter-block activations are `I`. The forward `I -> R` conversion is a local no-op, but the corresponding `P -> I` is a gradient all-reduce in BWD. The MLP compute itself is unchanged.

<img src="/devlogs/images/distributed/spmd-tp-mlp-forward-backward.svg" alt="Megatron TP MLP forward and backward placements, with sequence parallelism on and off" style="width: 100%; max-width: 1000px;">

**Global and local SPMD.** DTensor-style GSPMD works best when placements can be expressed using
standard sharding, with APIs such as `to_local` and `local_map` providing escape hatches when they cannot. Common
examples include MoE expert parallelism and multimodal encoder computation with uneven sharding, and TP vocab-parallel
cross-entropy or embedding, where per-rank computation doesn't match the global view.

Decoupled forward and backward behavior caused problems for these local-region APIs: the incoming `grad_output`
placement and the reductions required when leaving the region were not always clear, although explicit gradient-placement
APIs ([PyTorch #155181](https://github.com/pytorch/pytorch/pull/155181),
[#173454](https://github.com/pytorch/pytorch/pull/173454), and
[#175867](https://github.com/pytorch/pytorch/pull/175867)) and
[SPMD typechecking inside `local_map`](https://github.com/pytorch/pytorch/pull/181398) have recently improved this behavior for DTensor.

SPMD types supports transitioning into local SPMD typechecking under a `spmd.local()` context or `spmd.local_map`
wrapper. Within the region, `spmd.S(i)` becomes `spmd.V` (data that varies across ranks), while `spmd.R/I/V/P`
continue to track whether values are identical or varying and whether forward or backward reductions are required.
Only the tensor's sharding relationship to the logical global tensor is discarded.

It is also useful to adopt only local SPMD: the typechecker can validate existing distributed computation and
collectives without requiring the framework to be rewritten in GSPMD.

For more detail, see [ezyang's detailed writeup in the `spmd_types` local SPMD types documentation](https://github.com/meta-pytorch/spmd_types/blob/main/docs/local_spmd_types.md).

```python
# Dense transformer compute on the DP/CP/TP mesh.
x = norm(attention(x))
x, ... = router_and_permute(x)

# Enter the EFSDP/EP mesh and use local SPMD semantics.
with spmd.set_current_mesh(sparse_mesh), spmd.local():
    x = spmd.all_to_all(...)  # dispatch
    x = grouped_experts(x, ...)
    x = spmd.all_to_all(...)  # combine

# Return to dense computation and continue the transformer block.
x = token_combine(x, ...)
x = next_layer(...)
```

### Mesh reinterpretation

Unlike a DTensor, a plain tensor in SPMD types is not permanently associated with one `DeviceMesh`.
`spmd.set_current_mesh()` pushes an ambient mesh onto a thread-local stack. Typechecking and collectives resolve mesh
axis names against it, allowing operations like `spmd.redistribute(tensor, "tp", src=spmd.S(0), dst=...)` to be written
without threading process groups through every module.

This is useful because a training step naturally uses several meshes:

1. FSDP/HSDP stores parameters on a storage mesh, treating FSDP as a unified axis separate from HSDP's replicate axis.
2. At compute time, the same tensors are interpreted on a DP/CP/TP mesh for DP batch sharding, CP all-gathers, and TP collectives.
3. MoE expert computation reinterprets them on a sparse EFSDP/EP view of the same global ranks, repurposing the dense mesh for EP.
4. After expert computation, tensors return to the dense mesh for the rest of the transformer block.
5. After forward and backward, they can be interpreted on a loss or global mesh to reduce losses, gradient statistics, and logged scalars across ranks.

## TorchTitan integration

TorchTitan has enabled `spmd_types` as the default backend for eager pretraining and RL, using `spmd.*` collectives
and an optional typechecking mode across all current models, parallelisms, and runtime features. While the previous
default backend used DTensor compute for TP, `spmd_types` covers every mesh axis except PP (DP/CP/TP/EFSDP/EP),
building on the [Config-Based Sharding work](https://docs.pytorch.org/devlogs/distributed/2026-08-17-config-based-sharding-full-dtensor/).
The model computation and distributed behavior remain unchanged: both backends emit the same underlying collective
patterns—which is important for GraphTrainer's compilation passes—and produce bitwise-identical numerics.

With config-based sharding, most `spmd.redistribute` collectives are currently injected at module boundaries. This design is
still evolving, and the collectives may eventually move explicitly into `forward()`:

```python
feed_forward_cfg.sharding_config = ShardingConfig(
    in_src_shardings={"x": attn_x_layout},
    in_dst_shardings={
        "x": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
    },
)
feed_forward_cfg.w1.sharding_config = colwise_config()
feed_forward_cfg.w3.sharding_config = colwise_config()
feed_forward_cfg.w2.sharding_config = rowwise_config(output_sp=enable_sp)
```

The feed-forward config owns the input redistribution, the colwise configs own the `w1`/`w3` parameter shardings,
and the rowwise config declares a `P` output followed by either a sequence-parallel reduce-scatter or a non-SP
all-reduce. During model initialization, TorchTitan reads these contracts, shards the model state, and installs
module-boundary wrappers while leaving the model's `forward()` parallelism-invariant.

TorchTitan has also moved toward explicit collective implementations for TP vocab-parallel embedding and
cross-entropy, replacing DTensor's `MaskPartial` and the implicit `loss_parallel()` context.

## Performance comparison

Erasure mode is intended to remove DTensor's runtime dispatch and shard-propagation overhead without requiring compilation or cudagraphs. The measurements below use TorchTitan's debug models on 8 H100 GPUs, LBS 8, and context length 2048. Llama 3 uses selective AC; the MoE configurations use full AC.

| Model and parallel configuration | Execution mode | `partial_dtensor` TPS [range] | `spmd_types` TPS [range] | Paired change [95% CI] | Peak memory, partial / SPMD |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama 3, FSDP=4, TP=2 | Eager | 48,517 [47,632, 48,732] | 69,758 [68,244, 70,094] | **+43.69%** [+42.00%, +47.50%] | 0.49 / 0.49 GiB |
| Llama 3, FSDP=4, TP=2 | Eager + CUDA graphs | 224,294 [218,588, 232,758] | 230,286 [221,063, 233,392] | **+1.35%** [+0.18%, +6.75%] | 0.52 / 0.52 GiB |
| Llama 3, FSDP=4, TP=2 | Compile (`inductor`) | 102,578 [101,309, 105,472] | 112,368 [109,610, 114,913] | **+9.47%** [+6.10%, +10.99%] | 0.42 / 0.42 GiB |
| Qwen3 MoE, FSDP=4, TP=2, EP=8 | Eager | 39,074 [38,110, 39,888] | 54,272 [48,598, 56,854] | **+41.89%** [+22.26%, +47.27%] | 0.69 / 0.69 GiB |
| Qwen3 MoE, FSDP=4, TP=2, EP=8 | Eager + CUDA graphs | 55,699 [47,076, 59,291] | 48,972 [44,836, 57,007] | **-0.83%** [-23.39%, +2.75%] | 0.71 / 0.71 GiB |
| Qwen3 MoE, FSDP=4, TP=2, EP=8 | Compile (`inductor`) | 81,222 [79,185, 86,625] | 86,374 [80,493, 91,370] | **+5.45%** [+3.01%, +9.17%] | 0.73 / 0.73 GiB |
| DeepSeek V3, FSDP=4, TP=2, EP=8 | Eager | 33,128 [32,742, 33,290] | 48,376 [45,580, 48,686] | **+46.32%** [+36.15%, +47.85%] | 0.85 / 0.86 GiB |
| DeepSeek V3, FSDP=4, TP=2, EP=8 | Eager + CUDA graphs (HybridEP, custom all-gather off) | 67,437 [57,856, 72,466] | 71,214 [67,340, 73,661] | **+7.45%** [-3.16%, +12.89%] | 0.80 / 0.80 GiB |
| DeepSeek V3, FSDP=4, TP=2, EP=8 | Compile (`inductor`) | 61,226 [60,728, 62,052] | 64,515 [62,808, 67,108] | **+4.68%** [+1.63%, +8.29%] | 0.79 / 0.79 GiB |

The repeated runs show large eager gains of 42-46%. Inductor gains are smaller at 5-9%, and CUDA graphs largely erase the runtime-dispatch difference. Llama 3 retains a small gain, while the Qwen3 and DeepSeek V3 confidence intervals include zero. Peak memory is effectively unchanged.

Inductor compile perf has a clearer gap because TorchTitan's compile usage is not full-graph but regional, and DTensor subclass-related overhead can still be obvious, when outside of compiled regions. TorchTitan's chunked loss compiles the inner loss function, but leaves the chunking wrapper, applied on DTensor activations in eager. This leaves obvious tensor subclass wrap/unwrap overhead, particularly for the chunking [`local_map()` call](https://github.com/pytorch/torchtitan/blob/eae4563ade4bc6e877f5e181388e64fcf8e0ec48/torchtitan/components/loss.py#L625-L635):

```python
def _chunk_local(t):
    chunk_len = t.shape[0] // num_chunks
    return tuple(
        chunk.contiguous()
        for chunk in torch.split(t, [chunk_len] * num_chunks, dim=0)
    )

local_map(
    _chunk_local,
    out_placements=(t.placements,) * num_chunks,
    in_placements=(t.placements,),
    device_mesh=t.device_mesh,
)(t)
```

We see an overall >20% slowdown for loss computation, and this demonstrates how local-global SPMD transitions aren't so seamless with DTensor, both in terms of authoring (refactor and wrap a simple chunk call because it's not GSPMD; chunk then shard != shard then chunk) as well as overhead (8 added `_FromTorchTensor` calls from the profile, 1 per chunk):

<img src="/devlogs/images/distributed/spmd-types-chunked-loss-local-map-profile.png" alt="Profiler trace for chunked loss; local_map unwraps input DTensor, chunks, and rewraps eight DTensor outputs" style="width: 100%; max-width: 1200px;">

Shape-varying inference decoding is another likely beneficiary because DTensor's shard-propagation cache keys on shape. Preliminary
eager TitanRL measurements showed substantial gains, but a controlled public benchmark has not landed yet; compiled or padded
CUDA graph execution can also amortize much of this overhead.

## DTensor / FSDP2 status

TorchTitan is deprecating the previously default `partial_dtensor` backend in favor of `spmd_types`, removing DTensor
from forward and backward computation. In the near term, however, FSDP2 will continue to use DTensor for persistent
state. A useful mental model is: **DTensor at rest, `spmd_types` in compute.**

FSDP2 continues to use DTensor as the rest-time/storage representation for sharded parameters and gradients. This remains useful
for preserving global shape and sharding metadata, integrating with Distributed Checkpointing (DCP),
and existing implementations like grad-norm compute.
The [`spmd_types` + FSDP2 upstream integration](https://github.com/pytorch/pytorch/pull/181519) bridges the two representations:
FSDP2 reads parameter annotations and translates them into DTensor placements for storage, then restores the
`spmd_types` annotations when parameters are materialized for computation if typechecking is enabled. The corresponding
backward types also tell FSDP2 how incoming gradients should be interpreted and which reductions it owns.

For SimpleFSDP in GraphTrainer, we have explored replacing its internal DTensor representation with plain tensors and
`spmd_types`. The compiler passes only need to recognize the resulting collective patterns, so they do not fundamentally
depend on DTensor. Even there, however, a DTensor representation at rest remains useful today for DCP.

Longer term, FlexShard may replace FSDP2 for these use cases. FlexShard explicitly separates persistent storage layouts
from temporary compute layouts and does not require DTensor as the model-compute representation. Until that path matures,
TorchTitan keeps a deliberate storage/compute separation.

## Acknowledgments

Many thanks to Chien-Chin Huang ([@fegin](https://github.com/fegin)), Edward Z. Yang
([@ezyang](https://github.com/ezyang)), and Tianyu Liu ([@tianyu-l](https://github.com/tianyu-l)) for their design work,
implementation, reviews, and discussions throughout this project, and Pei Zhang
([@zpcore](https://github.com/zpcore)) and Vishal Nandavanam
([@vishal9-team](https://github.com/vishal9-team)) for being core contributors to the design and implementation of
`spmd_types`.
