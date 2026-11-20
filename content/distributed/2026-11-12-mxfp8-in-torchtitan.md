---
title: "Making MXFP8 Work in TorchTitan"
date: 2026-11-12
author: "Animesh Jain (@anijain2305)"
tags: [torchtitan, distributed, mxfp8, quantization, fsdp, pipeline-parallelism, tensor-parallelism]
---

> **TL;DR** – The kernels were the easy part of bringing MXFP8 to TorchTitan. The hard part was making it compose with distributed training. We moved quantized-weight lifetime into FSDP's lifecycle so each weight is quantized once per step under PP. We picked a stacked `[2, F, D]` layout for fused MoE gate/up weights so it works with both 32x32 quantization tiles and TP sharding. And we changed FSDP's gradient-dtype contract so weight-gradient accumulation can be fused into the `wgrad` matmul in fp32.

This post covers MXFP8 training support in TorchTitan, developed as part of a broader effort to improve TorchTitan's DeepSeek-V3 performance.

I had spent the past few years working on Dynamo, the `torch.compile` frontend, mostly thinking about what happens on one GPU. So when I joined the MXFP8 effort, I figured I would spend most of my time on kernels. But I soon realized that the TorchAO team had already done much of that work (thanks to Vasiliy Kuznetsov). My hardest problems showed up when I tried to make MXFP8 compose with FSDP, tensor parallelism, and pipeline parallelism.

That's also where I had the most fun. Thanks to Tianyu Liu for patiently pointing out the composability issues in my PRs that my coding agent had happily glossed over. In this post, I will go over these composability issues. If you also tend to reason about what happens on one GPU first, you might find this post interesting. The post is organized as follows:

* **MXFP8 basics**
* **MXFP8 recipe**
* **Lifecycle of quantized parameters**: composing with FSDP and pipeline parallelism
* **Fused gate and up projections in MoE experts**: composing with tensor parallelism
* **In-place weight-gradient accumulation**: composing `autograd.Function` with FSDP and pipeline parallelism

## MXFP8 Basics

**The format.** MXFP8 is the hardware-native FP8 format we used on Blackwell. Every block of 32 consecutive values gets its own scale, stored in the `e8m0` format. If one value is much larger than the rest, it can affect the precision of its own block without forcing an entire row or column to use the same scale.

Hopper already had fast FP8 matmuls, but blockwise scaling inside those matmuls had to be emulated in software. To keep the overhead low, Hopper recipes preferred per-row or per-column scales (figure below). On Blackwell, the Tensor Cores apply the scales directly as part of the matmul. That makes fine-grained scaling much more practical for training, and MXFP8 is becoming the norm.

![Per-row scaling on Hopper vs one scale per 32 values with MXFP8 on Blackwell](/devlogs/images/distributed/mxfp8_per_row_vs_block_scale.png)

**The linear layer.** A linear layer has three matmuls: one in the forward pass (`fprop`) and two in the backward pass (`dgrad` and `wgrad`). Using PyTorch's weight convention, where `W` has shape `[out_features, in_features]`, here is how they look with quantization:

![fprop, dgrad and wgrad matmuls with MXFP8 quantization of X, W and dY](/devlogs/images/distributed/mxfp8_linear_quantization.png)

Notice how the same tensors show up in different roles. `fprop` uses `W` transposed, while `dgrad` uses `W`. `X` moves from the left operand in `fprop` to the right operand in `wgrad`. And `dY` is transposed in `wgrad`.

With MXFP8, the math stays the same. What changes is that each matmul needs its scales grouped along its **reduction dimension**. For `fprop`, we reduce over input features, so we quantize `X` in `1x32` blocks along its rows. For `wgrad`, we quantize that same `X` in `32x1` blocks down its columns. The same logic applies to `dY`. We'll come to the 32x32 blocks for weights later.

**What gets quantized.** Our MXFP8 recipe targets the large matmuls with learned weights. In DeepSeek-V3, that includes the attention projections and the gate, up, and down projections in both the shared and routed experts. The snapshot below comes from [Ed's interactive DeepSeek-V3 walkthrough](https://deepseek-v3.ezyang.com/studies/02-hopper-memory.html). It shows a Hopper FP8 recipe, but it's a good map of where quantization comes into play (look for `e4m3`, or better, visit the interactive walkthrough).

![DeepSeek-V3 layer showing which matmuls run in e4m3](/devlogs/images/distributed/mxfp8_dsv3_quantized_layers.png)

## MXFP8 Recipe

Different users run MXFP8 with different recipes. For TorchTitan, we wanted to support what seemed to be the most common pattern among the recipes we looked at: 32x32 blocks for weights and 1x32 blocks for forward activations.

**Why 32x32 for weights?** Let's look at the linear layer formulas again. If the forward pass computes `Y = X @ W_q.T`, where `W_q` is the quantized weight, then the corresponding `dgrad` formula is `dX = dY @ W_q`. That's the same quantized weight in both passes, just in a different orientation. If we quantize the weight separately for each orientation, the same entry can round to two different values. Getting this right matters because `dX` is the gradient we pass back to the preceding layer. With one scale per `32x32` tile, transposing the weight keeps the same values together under the same scale, so `dgrad` can use exactly the quantized weights that `fprop` used. This [paper](https://arxiv.org/abs/2607.24953) is a good read to dive deeper.

**So why not use 32x32 blocks for activations too?** An activation matrix has one row per token. A 32x32 tile would make 32 tokens share a scale. If token 32 has an outlier that changes the scale, it can change how token 1 gets rounded, and with it, token 1's output. **That's a causal leak** (see this excellent [blog post](https://matx.com/research/leaky_quantization)). With 1x32 blocks across features, each token's forward scales depend only on its own activations.

Note that practitioners use several MXFP8 recipes. TorchTitan currently supports the one that seemed most common among those we looked at. If your workload calls for a different recipe, you'll need to add it. TorchTitan already has stubs for defining how tensors are quantized, so you have a place to start.

One open question is whether deterministic rounding introduces enough bias to affect training. Daniel Haziza led a lively discussion on this in the final session of the MoE Summit. Depending on how much that bias matters in practice, we may want to add stochastic rounding to the recipe.

## Lifecycle of Quantized Parameters

OK, let's get into the interesting composability stuff. First, a quick primer.

**PP + FSDP primer.** With pipeline parallelism (PP), each rank works on only its slice of the model. That usually leaves enough room to keep the stage's parameters unsharded across all microbatches in a step. FSDP all-gathers each weight before its first forward use and, with `reshard_after_forward=False`, keeps it unsharded for all the microbatches. Similarly, gradients remain unsharded and get reduce-scattered after the last microbatch. Effectively, this gives us a ZeRO-1-like pattern during the step.

Before our sprint, TorchTitan's MXFP8 support relied on a TorchAO `autograd.Function`. It did not compose well with PP and FSDP because it had no visibility into the FSDP runtime (or flags like `reshard_after_forward`). As a result, it quantized the weights in forward, and **again in backward**, for every microbatch. Not only did this add performance overhead, we also paid in memory, because this design kept the unsharded bf16 parameter alive when ideally only the MXFP8 parameter would be. Overall, it looked something like this:

![Before: the bf16 weight stays alive and is quantized in every microbatch's forward and backward](/devlogs/images/distributed/mxfp8_pp_fsdp_before.png)

To fix this, we redrew the boundary between TorchAO and TorchTitan. TorchAO would supply the kernels, which lined up with where the AO team wanted to go. TorchTitan would own the `autograd.Function`, which still handles the forward and backward computations. More importantly, we moved weight preparation and lifetime management into FSDP's lifecycle. We used a tensor subclass that implements `fsdp_post_all_gather`, which gives it full visibility into the FSDP runtime. This fixed both the memory and the performance problems above, resulting in something like this:

![After: the weight is quantized once after all-gather, the bf16 copy is released, and the MXFP8 weight is reused across microbatches](/devlogs/images/distributed/mxfp8_pp_fsdp_after.png)

**FSDP (no PP) composability.** So far, we've focused on PP, where `reshard_after_forward=False` lets us quantize each weight once and reuse it across all microbatches. That's the minimum work possible. But does this design still make sense in the typical FSDP setup, where `reshard_after_forward=True`?

Here, FSDP unshards and quantizes the weight separately for forward and backward. Our kernel produces the shared MXFP8 data and both scale layouts each time. That's more than the minimum work. So what is the minimum? Without activation checkpointing, forward only needs the `fprop` scales and backward only needs the `dgrad` scales. With checkpointing, backward needs both (for forward recomputation and for `dgrad`), so only the original forward produces unused scales.

So our approach does some extra work in a typical FSDP setup. Fortunately, the overhead is minimal. Remember that the quantization kernel produces the MXFP8 data as well as both sets of scales, so the extra cost of producing both scales instead of one is very small. Avoiding it would require passing more context into the post-all-gather hook, including whether backward will recompute the forward. We chose to keep the hook simple and prepare both scale layouts.

**Extensibility.** We got to test how well this refactor extended to two other components.

* **DistMoE.** Our new open-source MoE kernels support MXFP8 and need the same FSDP lifecycle guarantees discussed above. Sanket Purandare had already explored the integration early through draft PRs and design docs, which gave us confidence that it would work. By the time DistMoE was ready to land, TorchTitan's abstractions were in place. We could adapt the quantization recipe to DistMoE's requirements while keeping FSDP in charge of the MoE weights' lifetime.
* **GraphTrainer.** The tensor subclasses also helped us compose with GraphTrainer. Their `__tensor_flatten__` and `__tensor_unflatten__` methods let GraphTrainer unpack the quantized tensor into its constituent tensors and reconstruct it, preserving the representation we built for FSDP.

*Side note: this is still suboptimal. You might be wondering why we still all-gather in `bf16`. If so, look into FlexShard, Wei Feng's work in this area.*

*Side note: Andrew Or has also extended this design to work for the Generator in TitanRL.*

## Fused Gate and Up Projections in MoE Experts

A common optimization in MoE experts is to fuse the gate and up projections before SwiGLU. You'll often see the fused weight called `w13`, after the two weights it combines, `w1` and `w3`. The idea is simple. Instead of running two matmuls, `X @ w1.T` and `X @ w3.T`, we run one matmul, `X @ w13.T`, then split the output back into gate and up activations.

The fusion seems innocuous at first sight. However, how we lay out these parameters decides whether they compose with MXFP8 quantization and tensor parallelism (TP). Suppose `w1` and `w3` each have shape `[F, D]`. Here are three ways to lay them out.

![Three layouts for the fused w13 weight: concatenate, interleave, and stack](/devlogs/images/distributed/mxfp8_w13_layouts.png)

**Layout A: concatenate the weights.** Put all of `w1` before all of `w3` in a `[2F, D]` tensor. This works for MXFP8: with the boundary aligned to 32 rows, every 32x32 tile stays within one projection. But a contiguous two-way TP split along the fused dimension gives rank 0 all of `w1` and rank 1 all of `w3`. For SwiGLU, each rank needs matching slices of both projections so it can multiply the corresponding gate and up activations locally.

**Layout B: interleave the rows.** Alternate rows from `w1` and `w3`. Now a contiguous TP split gives each rank matching slices of both projections. But a 32x32 tile now mixes rows from `w1` and `w3`. If the two weights have different ranges, an outlier in one projection can set the scale for the other and make its values round more coarsely. So it does not compose with quantization.

**Layout C: stack the weights.** Give the fused weight shape `[2, F, D]` with TP sharding along dimension 1. Each rank gets matching slices of `w1` and `w3`, and every 32x32 quantization tile stays entirely within `w1` or `w3`.

This example shows how easy it is to miss these details when you're working on one GPU. Even a simple optimization takes some care to get right once you combine it with distributed training.

## In-place Weight-Gradient Accumulation

This is another optimization that is simple to understand but tricky to implement because of how it interacts with distributed training.

**The idea.** Each microbatch produces a weight gradient, and PyTorch autograd normally adds it to the running gradient in a separate `AccumulateGrad` operation. But `wgrad` is already a matmul, as we saw in the linear backward earlier. We can fuse the add into its epilogue and write directly into the running gradient. For MXFP8, `scaled_mm` becomes `scaled_addmm_`. For BF16, `mm` becomes `addmm_`.

Simple, right? Nope. When we looked closely, the existing setup was doing something suboptimal (some would call it wrong, from an accuracy point of view):

![Before: wgrad is rounded to bf16, cast back to fp32, then added in a separate kernel](/devlogs/images/distributed/mxfp8_wgrad_before.png)

The `wgrad` matmul accumulated in fp32 internally, but the result was then rounded to bf16. That bf16 result was then cast back up to fp32 and added to the running fp32 gradient. So we not only paid for two extra kernels (the cast and the add), we also threw away precision with a pointless round trip through bf16. This is what we wanted instead:

![After: the first microbatch writes an fp32 gradient directly; later microbatches accumulate in place with addmm_](/devlogs/images/distributed/mxfp8_wgrad_after.png)

So why the unnecessary round trip? Because of the FSDP contract: a gradient had to have the same dtype as its parameter. The parameter is bf16 during compute (`mixed_precision_policy.param_dtype`), so the matmul produced a bf16 gradient. FSDP then upcast it to fp32 because it reduces gradients in `mixed_precision_policy.reduce_dtype`, which is fp32 here. The key question to work through is: what dtype is each unsharded parameter and gradient at each point in a training step?

*Side note: Felipe Mello independently stumbled on the same issue during his TitanRL sprint, which is quite amazing in itself. In the Megatron perf-max sprint, we had profile traces from the Megatron team to help us identify these bottlenecks, but Felipe found the same issue in TitanRL without any such traces.*

After some lively discussion between the FSDP and TorchTitan teams, we changed the FSDP contract. FSDP now supports `grad_dtype`, so a parameter's gradient can have a different dtype from the parameter itself. With the new contract, the unsharded gradient is in fp32, the unnecessary round trip is gone, and the fusion is just a matter of writing an `autograd.Function`.

The main change here was the FSDP contract. Big credit to Wei Feng for landing this BC-breaking contract change while ensuring no regressions for internal customers.

Like the last section, this shows how a simple fusion can lead you down a rabbit hole. To land a design you're happy with, you end up reasoning about the entire training step and its distributed interactions.

## Closing Thoughts

That's it for my MXFP8 journey and its composability struggles. I hope you learned something from this post. I tried to keep things high level and palatable. If you are interested in learning more, feel free to reach out. Many thanks to the TorchTitan team for bringing me along on this rewarding journey.

## References

- [Interactive DeepSeek-V3 walkthrough](https://deepseek-v3.ezyang.com/studies/02-hopper-memory.html)
- [Paper on consistent weight quantization across fprop/dgrad](https://arxiv.org/abs/2607.24953)
- [Leaky quantization (MatX)](https://matx.com/research/leaky_quantization)
- [TorchTitan](https://github.com/pytorch/torchtitan)
