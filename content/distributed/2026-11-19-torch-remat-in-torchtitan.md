---
title: "Making torch_remat work in TorchTitan"
date: 2026-11-19
author: "Animesh Jain (@anijain2305)"
tags: [torchtitan, distributed, activation-checkpointing, remat, memory]
---

> **TL;DR** – TorchTitan now uses [`torch_remat`](https://github.com/meta-pytorch/remat) for activation checkpointing. Named regions in model code replace aten-level SAC policies, but distributed training made it trickier than "replay forward": collectives needed their own regions, side effects and nondeterministic routing had to be pinned down during replay, and the missing early-stop optimization led to a case where recomputing used *more* memory than stashing.

TorchTitan now uses [`torch_remat`](https://github.com/meta-pytorch/remat) for activation checkpointing. The idea behind activation checkpointing is simple. Drop some tensors during forward, then recompute them when backward needs them. You save memory by doing more work. But in distributed training, running forward again can also mean running collectives again, updating counters again (side effects), and making routing decisions again. This post walks through the choices we made while integrating `torch_remat` in Titan, and the distributed surprises I ran into along the way.

Quick disclaimer: I did not build `torch_remat`. The credit goes to Edward Yang, with the earlier `torch.utils.checkpoint` foundations built by Jeffrey Wan.

The post is organized as follows:

* Background - What Gets Stashed for Backward?
* Background - Why `torch_remat`?
* Deciding What a Region Is
* Collectives - Save or Recompute?
* Side Effects and Determinism
* Early Stop
* Results and the Memory-Performance Tradeoff

## Background - What Gets Stashed for Backward?

First, a quick recap of the problem activation checkpointing solves.

An operation's backward formula often needs tensors from its forward pass. For a linear layer, as shown in the figure, backward needs input `X` and `W`. Autograd keeps references to those tensors through `save_for_backward`, keeping them alive until backward uses them (the figure is pseudo code).

![A linear layer stashes X and W for backward via save_for_backward](/devlogs/images/distributed/remat_save_for_backward.png)

Different operations need different tensors for backward. For example:

![Tensors stashed for backward by RMSNorm, Linear, SiLU, Linear and a residual add](/devlogs/images/distributed/remat_ffn_stash_examples.png)

A note on terminology. I'll deliberately use the word **"stash"** for keeping a tensor alive for backward. The word "save" is overloaded in activation checkpointing. **Saving a tensor** means holding onto it and using memory. **Saving memory** means reducing how much memory we use. Those two meanings pull in opposite directions, which makes "saved activations" confusing in a discussion about memory savings. Throughout this post, **stashed tensors** are tensors we keep alive for later use, and **stashed memory** is the memory they occupy.

**How does activation checkpointing help?** Here's a contrived example: 4 layers, each stashing two tensors for backward (dark blue and light blue). In the top chart, with no activation checkpointing, every layer stashes both tensors. Each one is allocated during the forward pass and freed after its use in the backward pass. The stash grows through the whole forward pass, so the peak is right at the end of it.

![Memory timeline with and without activation checkpointing](/devlogs/images/distributed/remat_memory_timeline.png)

With activation checkpointing, we stash only the dark blue tensor and recompute the light blue one. That's the bottom chart, where grey boxes are short-lived allocations: the tensor exists briefly in forward, then again when it's recomputed in backward. The peak is now just the dark blue stash, plus one grey box right at the start of the backward pass for the first layer's recompute.

## Background - Why torch_remat?

PyTorch's `torch.utils.checkpoint` supports two common approaches to activation checkpointing.

* **Full activation checkpointing.** Recompute the entire checkpointed block during backward. This saves memory, but repeating expensive operations like matmuls adds substantial work.
* **Selective activation checkpointing (SAC).** Replay the block, but retain the outputs of selected operations so those operations can be skipped during recomputation. A `policy_fn` decides what to stash and what to recompute.

SAC is widely used, but we got two frequent complaints.

* **Overhead.** It uses `TorchDispatchMode` to intercept `aten` operations, which adds CPU overhead.
* **The policy lives separately.** The policy lives apart from the model code and works at the `aten` level. A policy expressing "stash the attention output projection" therefore requires counting matmuls and knowing which particular aten matmul corresponds to the attention output. For example, this is what a policy looks like in Titan SAC:

```python
# A policy separate from the model code.
def policy_fn(ctx, op, *args, **kwargs):
    if op == torch.ops.aten.mm.default:
        mm_count[ctx] += 1
        return SAVE if mm_count[ctx] % 2 == 0 else RECOMPUTE
    ...
```

`torch_remat` lets us express that choice at the same level as the source code (instead of the aten ops in SAC). Users express this directly in the model code through **regions**. We wrap a function in `remat.region` and specify whether to recompute it.

```python
out = remat.region(
    self.inner_attention,
    self.remat_region_name("inner_attention"),
    recompute=self.remat_should_recompute("inner_attention"),
)(q, k, v)
```

Here, the region has a meaningful name, and TorchTitan's configuration controls its recompute policy. There is no `TorchDispatchMode`, so it is fast. Being able to specify which regions to save and recompute directly in the model code is a big UX win.

**Saved region output.** There's one more detail that took me a while to appreciate. If we skip a region during replay (because it was marked save), its output must still be available for the operations that follow it. This is shown in the figure below, where region `f` is marked save and the following op is recomputed. So tensor `y` has to be kept alive.

![A saved region's output y must be stashed so the following op can be replayed](/devlogs/images/distributed/remat_replay_tax.png)

In full AC, this intermediate tensor `y` might never have been stashed, because no backward formula ever needed it. But here, we have to stash it for replay. `torch_remat` provides `recompute_needs_tensor` to keep these tensors alive for replay. Essentially, this is the memory tax we must pay to skip a region in recompute. Most importantly, these tensors do count toward peak memory (this will be relevant later in the Early Stop section).

## Deciding What a Region Is

The first question was where to put the region boundaries. Around every operation? Around a few operations together? Should some operations always be recomputed?

We used two questions to guide the decision: how much time would recomputing this operation cost, and how much stashed memory would it let us release? Roughly, that gives us a recompute cost per byte. A cheap operation that holds onto a large tensor is a good candidate for recomputation. An expensive operation that saves little memory is a good candidate for stashing. With this, we came up with a list of ops that were cheap to always recompute (like `residual_add`) and never made regions for them.

`local_compile` (Felipe Mello and Jessica Zhong) also influenced these choices. Some operations that looked borderline, like norms, became cheap enough after compilation that we simply let them recompute.

That gave us a starting point for compute operations. But a linear layer in distributed training also includes collectives, and those needed a closer look.

## Collectives - Save or Recompute?

A collective changes the layout of a tensor, so the memory cost depends on which side of it we stash. As shown in the figure, tensor parallelism gives us two useful examples.

![ColumnParallelLinear and RowParallelLinear: where to stash relative to the collective](/devlogs/images/distributed/remat_tp_collectives.png)

* **ColumnParallelLinear** runs an all-gather followed by a matmul. If we stash the matmul output but recompute the all-gather, we can stash just the sharded input and gather it again during backward, avoiding stashing the full gathered input.
* **RowParallelLinear** runs a matmul followed by a reduce-scatter. If we stash the matmul output but recompute the reduce-scatter, we keep the large partial output alive. Stashing the reduce-scatter output as well lets replay use the smaller, reduced shard.

So I had to change my mental model. Users needed control over collectives as well as compute operations. We needed remat regions for collectives.

But there was a problem. TorchTitan's collectives were inserted implicitly by its resharding logic, so there was nowhere to wrap a region. We first had to make them explicit. `ColumnParallelLinear` took ownership of its input gather, `RowParallelLinear` of its output reduce-scatter, and the MoE TP-to-EP transitions became explicit. That work grew into its own side project, where we removed the implicit resharding logic from Titan completely (thanks to Pian Pawakapan for the great [SPMD types](/devlogs/distributed/2026-08-26-spmd-types-in-torchtitan/) foundation).

Once the collectives were explicit, we could wrap them in regions and let users choose what to stash and what to recompute.

## Side Effects and Determinism

Replaying forward needs to reproduce the computation (i.e., be bitwise equal to the original forward). I encountered two violations during this integration.

**Counters ran twice.** The auxiliary-loss code counted tokens during forward, then counted them again during recomputation. The existing full-AC path compensated with a divide-by-two hidden in the code. With `torch_remat`, we could check `is_recomputing` and skip the counter update during replay, removing that workaround.

**Routing needed to stay consistent.** The `top_k` operator is not deterministic. If recomputing the router's top-k selects different experts, backward follows a different computation from the original forward. This can be very hard to debug. So we made routing a fixed `recompute=False` region so replay reuses the original decisions. Users can tune the surrounding regions, but this one stays stashed for correctness.

## Early Stop

`torch.utils.checkpoint` has a neat optimization called **early stop**. During replay, it stops as soon as every tensor required by backward has been generated. The remaining trailing operations never run.

Look at how a MoE block ends. After the `combine`, one add joins the shared- and routed-expert outputs, and another adds the result back into the residual stream. The combine stashes tensors for backward, but the adds stash nothing. For `a + b`, backward simply passes `dY` to both inputs. Replay can therefore stop at the combine and skip both adds. When this was tried in TorchTitan, it improved performance by 24%.

`torch_remat`, unfortunately, **does not** support early stop, so we were missing that optimization. But the bigger surprise was memory. Remember that we left `add` operations outside regions because they were cheap to recompute? Replay now had to rerun them, and to replay them, we needed `recompute_needs_tensor` on the first residual add. Therefore the inputs to the first residual add (which were never needed for any backward computation, only for replay) are stashed and contribute to peak memory.

![Trailing adds as bare ops force their inputs to be stashed for replay; wrapping them in save regions removes that](/devlogs/images/distributed/remat_early_stop_adds.png)

The fix is simple. We wrapped the two add operations in always-saved regions. This prevented `torch_remat` from stashing the inputs of the first residual add, and also skipped the residual add computation during replay. This is one of the rare cases where recompute (without our fix) requires more memory than save. With every region set to save, activation memory dropped from **8.18 to 7.93 GiB** on Qwen3 MoE and from **9.11 to 8.67 GiB** on DeepSeek-V3 16B, both with FSDP4 + EP4.

## Results and the Memory-Performance Tradeoff

With these regions in place, we can explore different save/recompute choices and plot their memory usage against step time. The Pareto frontier gives us useful configurations.

![DeepSeek-V3 16B: peak memory vs step time across 128 remat policies, with the Pareto frontier](/devlogs/images/distributed/remat_pareto_dsv3.png)

This was possible with SAC too, but named regions make the choices easier to express and inspect. The example above comes from an agent exploring the remat configurations for DeepSeek-V3 16B. An exhaustive search gets expensive as the number of regions grows, so TorchTitan's [`ACTIVATION_REMAT.md`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/ACTIVATION_REMAT.md) also describes a greedy approach for finding good configurations with fewer runs.

In distributed training, parallelism settings and recompute choices change memory usage in discrete steps. The curve helps us see which configurations fit our memory budget and how much performance we will trade for extra headroom.

## Closing Thoughts

That's it for this post. I hope you learned something new from these examples. I started with the familiar idea of replaying forward to save memory, then found myself thinking about collectives, side effects, and tensors kept alive for computations that backward never needed (`recompute_needs_tensor`). If you also tend to think about one GPU first and found this post interesting, I shared a similar journey while integrating MXFP8 into TorchTitan. We are also exploring activation offloading, another optimization to reduce memory footprint. Stay tuned!

## References

- [`torch_remat`](https://github.com/meta-pytorch/remat)
- [TorchTitan activation remat docs](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/ACTIVATION_REMAT.md)
- [SPMD types in TorchTitan](/devlogs/distributed/2026-08-26-spmd-types-in-torchtitan/)
- [`torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html)
