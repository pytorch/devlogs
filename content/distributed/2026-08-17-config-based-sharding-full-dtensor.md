---
title: "Config-Based Sharding and Full DTensor Adoption in TorchTitan"
date: 2026-08-17
author: "Chien-Chin Huang (@fegin)"
tags: [torchtitan, distributed, dtensor, sharding, spmd, moe]
---

> **TL;DR**
>
> 1. TorchTitan now adopts a declarative approach: ALL sharding (SPMD parallelization) is expressed in configuration.
> 2. We created a full DTensor mode where every tensor is a DTensor that shards on all activated mesh axes (DP, CP, TP; EP for MoE experts).
> 3. Full DTensor removes the ambiguity where a tensor could be a plain local tensor or a DTensor sharded on only some axes - the source of correctness bugs.
> 4. Available via `--parallelism.spmd_backend=full_dtensor`, verified bit-identical to the legacy path across FSDP/HSDP/CP/TP/EP, at performance parity.
> 5. TorchTitan is transitioning to [spmd_types](https://github.com/meta-pytorch/spmd_types); the work here (config-based sharding and full DTensor) is the foundation that makes it possible.

## Motivation

### Why full DTensor?

DTensor was meant to make distributed training easier, and it did - but only where we used it. TorchTitan adopted DTensor primarily for two use cases: (1) to express TP (and later EP), and (2) to work with DCP via `fully_shard`. This fits the legacy parallelization API (for example, `fully_shard` assumes a DTensor parameter is sharded on at most one mesh axis), but using DTensor only partially exposes TorchTitan to several model-authoring difficulties. Two stand out:

- Plain tensors get mixed with DTensors.
- The DTensor sharding spec does not describe every mesh axis.

Plain tensors mixed with DTensors: DTensor does not work well with plain `torch.Tensor` - it errors out when some arguments are plain tensors. When we annotate only some axes, plain tensors and DTensors can end up in the same computation. The most common case is computation outside the model, such as loss and metric computation. The model output is a DTensor if TP is enabled, but inputs and labels stay plain `torch.Tensor` because we ONLY specified TP sharding. So one has to be careful writing this code. The error is at least explicit (DTensor asserts when it meets a plain tensor), but it is annoying and error-prone.

This first issue is mostly outside the model code. The more general problem is that a DTensor's sharding spec covers only some mesh axes: it says how the tensor is sharded on TP (and EP), but leaves DP and CP out. Take CP as an example: because we don't specify the CP axis in the DTensor sharding spec, we gather K/V a different way - module hooks replace the input tensors with a DTensor sharded only on the sequence axis, run attention, and restore the output back to a plain tensor. But look closely at CP and TP and they share the same structure:

- CP's inputs (K and V) are sharded and must be gathered to compute the correct result.
- TP's inputs (activations and parameters) are sharded too, and its output needs an all-reduce or reduce-scatter to be correct.
- Both have a micro-pipelining optimization to hide the communication (ring attention vs. async TP).

So why do we implement them two different ways? With full DTensor, the ambiguity shrinks and the two parallelisms share one interface. CP and TP are just the clearest example: once every axis lives in the tensor's type, all SPMD parallelisms compose through the same redistribution mechanism instead of a separate one per parallelism.

Note that `fully_shard` (FSDP) stays a wrapper-style exception. We argue FSDP is not a true computation parallelization - it does not change the math of any op - so it is better treated as a storage optimization (distributed storage): at rest the parameters are stored sharded across the data-parallel ranks (`Shard(0)`), outside the computation, and are all-gathered before they are used. The data-parallel axis itself is still represented within the DTensor sharding spec (activations are DP-sharded, parameters are DP-replicated at compute); only the parameter all-gather/reduce-scatter is left to the `fully_shard` wrapper.

### Why config-based sharding?

Beyond the partial-annotation problem, the original TorchTitan design has a second issue: the parallelization logic is imperative and scattered. Each model has an `apply_tp` function that applies TP to its modules during the parallelization phase. There are manual `to_local/from_local` calls (or `local_map`) in the module forward to handle regions that DTensor can't express (for example, FlexAttention). The sharding intent is tangled into both the model code and the infra. We want to minimize this fragmentation.

To solve the issues, we added a declarative, config-based sharding API to TorchTitan. All the sharding, including the `local_map` logic, is embedded in a custom `Module.parallelize()` and driven by the sharding config. The user's main job is to decide the sharding config for each module. If a region (module) is not compatible with DTensor, the user just declares the sharding at the region boundary, and `Module` converts the DTensor to a local plain tensor at the declared placement.

Config-based sharding is backend-agnostic: the same sharding config drives the partial-annotated DTensor path (legacy, TorchTitan's current default), full DTensor mode, and the [spmd_types](https://github.com/meta-pytorch/spmd_types) backend. The sharding declarations do not change; only the backend that consumes them does. We treat config-based sharding as the prerequisite and infrastructure for all three - it gives one unified UX for every kind of parallelization, and it is what makes TorchTitan's move to `spmd_types` possible (see Next Step section).

## Design and Implementation

### Module Config

We made every component in TorchTitan a `Configurable` subclass - a class that lets users configure the object before materializing it. `Module` is no exception: `Module` inherits from both `nn.Module` and `Configurable`. This lets users customize a `Module`'s behavior, e.g., the number of embeddings in an `Embedding` module, or the type of `TokenDispatcher` in a MoE module. Deferring materialization also matters for parallelization: the model is first built from its config on the meta device, then `parallelize()` applies the sharding, and only then are the weights materialized on the device. The following example is the `FeedForward` module implementation. To construct a `FeedForward` module, one only needs to first create a `FeedForward.Config` object which contains three `Linear` configurations: `w1`, `w2`, and `w3`. Then a `feed_forward_cfg.build()` call returns the `FeedForward` module based on the configuration.

```python
class FeedForward(Module):
    """SwiGLU feed-forward module shared across models.

    Config takes the **final** hidden_dim (no internal 2/3 scaling).
    Use compute_ffn_hidden_dim() for Llama3/4-style dim computation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w1: Linear.Config
        w2: Linear.Config
        w3: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.w1 = config.w1.build()
        self.w2 = config.w2.build()
        self.w3 = config.w3.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

Because every module is built from a `Config`, we can attach the sharding declarations to that same `Config` - which is how config-based sharding works, and the subject of the next section.

### Config-based Sharding

On top of the Module Config infrastructure, we added the sharding configuration to `Module.Config`.

```python
class ShardingConfig:
    state_shardings: dict[str, SpmdLayout] = field(default_factory=dict)
    in_src_shardings: dict[str, SpmdLayout] | None = None
    in_dst_shardings: dict[str, SpmdLayout] | None = None
    out_src_shardings: SpmdLayout | tuple[SpmdLayout, ...] | None = None
    out_dst_shardings: SpmdLayout | None = None
    local_map: LocalMapConfig | None = None
```

Each field declares one kind of boundary. `state_shardings` gives the placement of the module's parameters and buffers. The `in_*` and `out_*` fields are (`src`, `dst`) redistribution pairs: src is the placement the tensor arrives with, dst is the placement we want, and if the two differ, that difference will incur a collective (all-gather, reduce-scatter, or all-reduce). `in_src_shardings / in_dst_shardings` cover the module's inputs; `out_src_shardings / out_dst_shardings` cover its outputs. `local_map` is for regions computed on local tensors (e.g., FlexAttention), as described earlier.

This lets users specify the module logic and its parallelization sharding (and `local_map`) all in the config. When `Module.parallelize()` is called, each `Module` recursively parallelizes its submodules and then parallelizes itself from its own sharding config. Concretely, `parallelize()` wraps the module's forward as: 1) redistribute the inputs (`in_src` -> `in_dst`), 2) run the forward, 3) redistribute the outputs (`out_src` -> `out_dst`). No information beyond the sharding config is required except the mesh, provided by `ParallelismDims` (discussed next). The following is an example sharding config for `FeedForward.w2`.

Note that the placements in the config use `spmd_types` placements (`spmd.R, spmd.S(n), spmd.P`). This is deliberate - the same config drives both backends. Under full DTensor these types are translated into DTensor placements (`Replicate`, `Shard`, `Partial`); under the `spmd_types` backend they are used directly. So the sharding declarations do not change when we switch backends, only how they are consumed.

```python
# spmd.R and spmd.I are Replicate for DTensor, meaning that the tensor is replicated on all ranks.

# spmd.P is Partial for DTensor, meaning that the tensor is partial and a reduction (all-reduce) is required to get the full tensor.

# spmd.S(n) is Shard(n) for DTensor, meaning that the tensor is sharded on N-th dimension across all the ranks.

feed_forward_cfg.w2.sharding_config = ShardingConfig(
    state_shardings={
        "weight": {
            DP: spmd.R,
            CP: spmd.R,
            TP: spmd.S(1),
         },
        "bias": {
            DP: spmd.R,
            CP: spmd.R,
            TP: spmd.R,
         },
    },
    out_src_shardings={
       DP: spmd.S(0),
       CP: spmd.S(1),
       TP: spmd.P,
    },
    out_dst_shardings={
       DP: spmd.S(0),
       CP: spmd.S(1),
       TP: spmd.I,
    },
)
```

W2 is a linear layer, so `state_shardings` annotates its two states, `weight` and `bias`. `weight` is sharded on dim 1 under TP, so `spmd.S(1)`; DP and CP do not shard the states, so `spmd.R`. For the output, DP shards the batch dimension (dim 0) and CP shards the sequence dimension (dim 1); under TP the output is Partial and needs an all-reduce, so `out_src_shardings` uses `spmd.P` and `out_dst_shardings` uses `spmd.I`.

In the legacy imperative sharding approach, this was `RowwiseParallel(output_layouts=Shard(1))`. The config above says the same thing and, unlike the legacy sharding approach, it also covers the DP and CP axes.

Note that we declare both src and dst even though DTensor could infer src from the tensor itself. Declaring both sides keeps the contract uniform for `spmd_types`, where the type is erased and the source cannot be inferred.

We always annotate every parallelism's sharding in the config, so one config covers DP, TP, CP, EP, or any combination of them: at runtime `Module.parallelize()` applies only the axes that are actually enabled and ignores the rest. Write the sharding once, run it under any parallelism layout and any parallelism backend (legacy, full DTensor, and `spmd_types`). But how does `parallelize()` know which axes are enabled?

### ParallelismDims

ParallelismDims is an object in TorchTitan that records which parallelisms are enabled (DP, CP, TP, EP, PP). It also contains the corresponding DeviceMesh for each parallelism. As a result, the only extra information, not in the sharding config, we need to provide to `Module.parallelize()` is ParallelismDims. When parallelizing a tensor (state or activation), `Module` first looks up the corresponding sharding config. The keys of the sharding config tell `Module` which parallelisms this tensor may be sharded on. In the above example, the parallelisms are DP, CP and TP. `Module` then consults ParallelismDims to know 1) what parallelisms are actually enabled and 2) what is the DeviceMesh for these parallelisms. With the sharding (from sharding config) and DeviceMesh, `Module` can parallelize the tensor correctly.

We designed config-based sharding in a way that it covers both the legacy TP-annotated only mode and full DTensor mode. ParallelismDims controls what parallelism is enabled. So for the legacy TP-annotated only mode, ParallelismDims will pretend only TP is enabled. As a result, `Module` will only annotate TP but not DP and CP even if they are actually enabled.

With this config-based sharding, we are able to annotate both states and activations - the core part of full DTensor design.

### Other Changes

While the previous sections cover the core design of full DTensor, there are numerous other changes needed to enable full DTensor, too many to cover here. We list only a few important ones.

**TorchTitan Trainer Change:**

TorchTitan trainer must parallelize the inputs and labels on all the SPMD dimensions. Before this work, the inputs and labels were always plain `torch.Tensor`. Now they are DTensors with all SPMD dimensions annotated.

**FSDP2 Change:**

FSDP2 (`fully_shard`) was designed to be aware of TP-sharded DTensors, but if a DTensor is passed with more than one dimension parallelized, FSDP2 doesn't know how to handle it. More importantly, the DeviceMesh used by FSDP2 (we call it the storage DeviceMesh) and the DeviceMesh used by `model.forward()` (we call it the compute DeviceMesh) are different, so FSDP2 needs to know how to convert from the storage DeviceMesh to the compute DeviceMesh. The old design just assumes the compute DeviceMesh is TP if the tensor is a DTensor. This is not true anymore. So we also changed `fully_shard`'s API to accept a `DataParallelMeshDims` argument that tells it which mesh axes are data-parallel, so it can fold those axes out of the storage mesh to recover the compute mesh.

**PP Change:**

PP splits the model into stages and sends the activations from one stage to the next stage. Before this work, these activations were plain `torch.Tensor`, so the cross-stage send/recv just moved plain tensors around. Under full DTensor the activations become DTensors, so PP now must understand DTensor to send and receive them between stages. For the full detail, please reference the [RFC](https://github.com/pytorch/pytorch/issues/172419) from Sanket Purandare (@sanketpurandare).

**MoE Change:**

MoE does not fit the dense sharding pattern directly. The routed experts live on a different mesh than the dense path: the dense modules are sharded on DP/CP/TP, but the experts are sharded on the expert-parallel mesh (EP, plus their own data-parallel axis). On top of that, the token dispatch between experts is a data-dependent all-to-all, which DTensor cannot express well. So we rewrote the MoE modules to work with config-based sharding like every other module: the expert modules declare their sharding on the expert mesh through the same sharding config, and the parts DTensor cannot express, such as the token dispatch, are handled with `local_map`. With these changes, MoE is parallelized through the config just like the dense modules. See the [PR](https://github.com/pytorch/torchtitan/pull/3386) from Jessica Zhong (@acisseJZhong), and the PRs before it.

## Result and Issues

### Performance and Bit-wise Parity

We verified the full DTensor against the legacy path (`spmd_backend=default`) with the same seed and deterministic settings. The `loss` and `grad_norm` are bit-wise identical for llama3 (dense) and qwen3 (MoE) across every parallelism combination we tested - FSDP, HSDP, TP, CP (no ring attention), EP, and their combinations. Full DTensor does not change the math.

For performance, we compared full DTensor and legacy with `torch.compile` on, at the smallest non-debug model size. Full DTensor matches legacy throughput within noise on llama3 (FSDP and FSDP+TP) and on eager Qwen3 MoE, and it uses the same memory. The one exception is Qwen3 MoE FSDP+EP with compile, which is about 2% slower. So full DTensor is at performance parity, with that single case as the exception.

### Issues

Getting the main logic and infra to work is only part of the story. We hit many issues along the way. We do not list them all here; instead we list a few representative ones and their status.

| Issue/Issue Category | Status |
| --- | --- |
| Missing DTensor op support (e.g., multi-dim `loss_parallel`, `clone` keeping Partial) | Fixed (PyTorch) |
| Wrong gradient / metric reduction (`dist_reduce`, `clip_grad_norm`) | Fixed |
| MoE global-vs-local token count under full DTensor | Fixed (TorchTitan) |
| `_StridedShard` from view / flatten / redistribute (DP+CP+TP) | Correctness fixed; eager redistribute slow (unsolved) |
| DTensor + `torch.compile` (CP+compile, TP+EP+compile) | Unsolved |

Most issues were either missing DTensor op support or wrong reductions. We fixed them, many of them upstream in PyTorch, and the loss stays bit-wise identical.

Under full DTensor, one tensor can be sharded on more than one mesh axis at the same time. For example, DP shards an activation on the batch dimension, and CP shards the same activation on the sequence dimension. In the legacy design this did not happen inside a single tensor during the computation: only the TP axis was specified in the DTensor spec.

Full DTensor puts every axis into the spec, so a tensor sharded on several axes at once becomes the normal case. This is more powerful, but the layouts get complicated, and the two unsolved issues below both come from this multi-axis sharding.

The first is `_StridedShard`. When the tensor dimensions sharded by those two axes get flattened together - like the DP + CP case - we need `_StridedShard` to describe it. The result is correct, but the redistribute planner is slow to find the path for `_StridedShard` tensors, so the first step can take 5-10 minutes.

The second is `torch.compile`, which does not handle full DTensor's multi-axis sharding well. Full DTensor + CP + compile hangs, and full DTensor + TP + EP + compile fails to compile because it cannot resolve the dynamic token counts from the MoE dispatch.

We did not deep-dive the root cause of the 2% slowdown, nor pursue fixes for the compile failures. Both are specific to how DTensor represents and compiles multi-axis sharding, and we are moving to `spmd_types`, which does not use that representation and its type checking is not part of the compiled graph - so chasing them would not help the direction we are heading. `spmd_types` computes on local tensors with explicit collectives, so there is no DTensor sharding propagation during runtime and no DTensor for the compiler to lower (see Next Step).

## Next Step (already underway)

The goal is to transition TorchTitan into an `spmd_types`-based parallelization training framework. While full DTensor offers a significant improvement over TorchTitan's legacy parallelization, it still inherits several issues, such as eager per-op dispatch overhead, and exacerbates others like strided sharding (`_StridedShard`). Conversely, `spmd_types` addresses these by design through a deliberate trade-off: it foregoes implicit redistribution in favor of explicit, configuration-driven control.

TorchTitan, [when this post was published](https://github.com/pytorch/torchtitan/tree/0d2438f954c2581cbf22b45f45835bbfa8e0c8db), currently supports `spmd_types`, `full_dtensor`, and the legacy backend; however, we are going to deprecate both `full_dtensor` and the legacy backend soon.

## Acknowledgments

This work was done together with Tianyu Liu ([@tianyu-l](https://github.com/tianyu-l)), Pian Pawakapan ([@pianpwk](https://github.com/pianpwk)), Edward Z. Yang ([@ezyang](https://github.com/ezyang)), Sanket Purandare ([@sanketpurandare](https://github.com/sanketpurandare)), and Jessica Zhong ([@acisseJZhong](https://github.com/acisseJZhong)).
