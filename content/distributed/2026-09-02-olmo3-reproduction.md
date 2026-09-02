---
title: "Olmo3 reproduction in TorchTitan"
date: 2026-09-02
author: "Ruisi Zhang (@ruisizhang123)"
tags: [torchtitan, distributed, model factory]
---

> **TL;DR** – I have reproduced 23% of Olmo3’s pre-training loss curve with TorchTitan. The training takes 4 days running on 512H100 GPUs. The on-the-fly downstream task evaluations further confirm TorchTitan’s ability to train capable LLMs. This marks the first step in our model factory efforts, which aims to enable agents to use TorchTitan to hillclimb scaling ladders, explore new research ideas, and validate them across increasing model scales. Stay tuned for more updates.

## Introduction

Olmo3 is Ai2's fully open family of language models ([link](https://allenai.org/blog/olmo3)), with training data, code, checkpoints, and recipes spanning pretraining, mid-training, long-context training, and post-training. This transparency provides a well-specified reference system against which we can validate our Olmo3 implementation in TorchTitan, including the data pipeline, checkpointing, training infrastructure, and evaluation stack.

With this goal in mind, I re-implemented Olmo3 in TorchTitan using the same pretraining data and hyperparameters, and evaluated intermediate checkpoints on MMLU, WikiText-2, ARC Challenge, ARC Easy, and HellaSwag. During training, the Olmo3 7B model shows consistent learning progress: perplexity decreases smoothly while accuracy on downstream benchmarks steadily improves; losses decreased to match the referenced loss value from Olmo3’s provided [wandb](https://wandb.ai/ai2-llm/Olmo-Hybrid-7B/reports/Olmo-3-7B-vs-Olmo-Hybrid--VmlldzoxNjA5NDIyNg) at the same training steps. These trends provide an initial validation that our TorchTitan reproduction follows the expected pretraining trajectory.


In Olmo3’s released code, they mostly used HSDP(FSDP2+DDP) and torch.compile. My modifications sit at the infrastructure level, including (1) using Varlen FlashAttention3 instead of FlashAttention2 in Olmo; (2) using HSDP (dp_shard=16 and dp_replicate=32) instead of (dp_shard=8 and dp_replicate=64) in Olmo paper in mast; (3) reusing a significant portion of TorchTitan common components to test TorchTitan's correctness. It includes RoPE, RMSNorm, and QKV Linears, etc.

## Model performance and throughput results

On the model performance side, the Olmo3-7B pre-training is learning in a meaningful way with increased accuracy and decreased perplexity. I also cross-validated the loss and downstream task performance with Olmo3’s official wandb, which shows TorchTitan’s implementation is able to match Olmo3’s loss and downstream performance after 323k steps training. I will cover bit-wise loss equivalence later. Here, I use the 6T [dolma3](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B) dataset to train the model.

{{< figure src="/devlogs/images/distributed/olmo3_original_loss.png" caption="*Original Olmo3 training loss, 1.9763 at the 323k step*" >}}

{{< figure src="/devlogs/images/distributed/olmo3_torchtitan_loss.png" caption="*TorchTitan Olmo3 training loss, 2.0029 at the 323k step*" >}}

| Task | ARC Challenge ↑ | ARC Easy ↑ | MMLU ↑ |
| --- | ---: | ---: | ---: |
| Our task performance | 0.6962 | 0.8607 | 0.5448 |
| Olmo3's original task performance | 0.6902 | 0.8421 | 0.5129 |

{{< figure src="/devlogs/images/distributed/olmo3_torchtitan_task_perform.png" caption="*TorchTitan Olmo3 task performance*" >}}


On the training infrastructure side, we use HSDP + torch.compile to train Olmo3 model from scratch on 512H100 GPUs. Here, we have dp_shard degree = 16 (fsdp2), dp_replicate degree =  32, block-level compile. The MFU is pretty stable over time to be at ~47%. There are some fluctuations after the job runs for a few days. I’ll cover the details and debugging later. The profiled trace also shows the FSDP(All gather, Reduce scatter)/DDP(All reduce) related communication in FWD/BWD pass are able to be overlapped by computation.

{{< figure src="/devlogs/images/distributed/olmo3_torchtitan_mfu.png" caption="*TorchTitan Olmo3 MFU*" >}}

{{< figure src="/devlogs/images/distributed/olmo3_torchtitan_trace.png" caption="*TorchTitan Olmo3 profile trace*" >}}


## Bitwise identical with Olmo3 pre-training

It’s possible to achieve bitwise identicality between Olmo3-core and TorchTitan after synchronizing their op implementations. Here, I pinned to Torch 2.13 since TorchTian requires higher version to use ParallelDim, spmd-types, etc.

Specifically, I’ll need to revise these things to ensure bitwise identicality. It is mostly to help us understand what are the implementation differences between TorchTitan and Olmo3 that would incur numerics unidentical. The difference is benign from the loss and downstream task performance we observed in previous sections.

- RoPE, RMSNorm, and Fused QKV op implementations are different. I sync'ed TorchTitan's implementation with Olmo-core.

- Gradient norm reduction order between TorchTitan and Olmo-core are different. I sync'ed TorchTitan's implementation with Olmo-core.

- No torch.compile, the lowered fx graphs are different for Olmo-core and TorchTitan since their implementation structures are different.

{{< figure src="/devlogs/images/distributed/olmo3_bitwise_loss.png" caption="*Olmo3 bitwise loss comparison*" >}}


## Fluctuations of MFU in later stage of training

The MFU plot after running for 4days is shown above. The training job restarted twice for different reasons: (1) At 170k step, the job restarted because of a NCCL timeout error; (2) At 261k step, the job restarted because it got preempted. The per-step low MFU can come from various sources, e.g., profiling & checkpointing at every 1k step, communication & computation stragglers, etc. A healthy MFU is ~47% for our case. Let’s say MFU below 35% is considered as a low MFU step, the distribution of these low MFU steps over time is shown below:

{{< figure src="/devlogs/images/distributed/olmo3_low_mfu.png" caption="*Low MFU training steps*" >}}


At ~170k step, a NCCL failure occurs: The failure was an RDMA transport error (IBV_WC_RETRY_EXC_ERR(12), vendor_err=129 on mlx5_3, rank 153 → rank 169), followed by NCCL timeout. The flight recorder trace that rank 153 dumped covers the last 202s before the failure, and the first collective to stall is the mesh_dp_replicate all-reduce (PG 2058, 32 ranks on 32 different hosts). Rank 169 is Rank 153's ring successor in exactly that group.

The low MFU also happened in other periods of the training. Specifically, in 150k-180k, there are over 150 low MFU steps in 10k step bins. Not all of them stem from the previous mesh_dp_replicate all-reduce NCCL timeout error. Not a surprise tho – These communication stragglers come from different process groups and are of different communication types. It is only when they happen at a close time window that we can find the straggler root-cause hosts have overlaps.


At step 162k and 163k, the low MFU (28.62%; 22.5%) is both because of FSDP comm straggler in Rank 17-31 resides in Host #2 3.

{{< figure src="/devlogs/images/distributed/olmo3_low_mfu_step162.png" caption="" >}}
{{< figure src="/devlogs/images/distributed/olmo3_low_mfu_step163.png" caption="" >}}

At step 176k and 178k, the low MFU (22.96%; 28.94%) is because of FSDP comm straggler in  Rank 96-127 (Host#12-#15), Rank 176-191 (Host#22 #23), Rank 400-431(Host#50-53) for step 176k; and Rank 176-191 (Host#22 #23) and Rank 336-351 (Host #42 #43) for step 178k.

{{< figure src="/devlogs/images/distributed/olmo3_low_mfu_step176.png" caption="" >}}
{{< figure src="/devlogs/images/distributed/olmo3_low_mfu_step178.png" caption="" >}}


## Future work

We leverage this step to validate TorchTitan’s capability to train capable LLMs. As of the next step:

- Build scaling ladders for MoE models that provide the scaffolding needed for agentic hill climbing.

- Establish model factories that enable agents to explore and validate research ideas directly in TorchTitan.
