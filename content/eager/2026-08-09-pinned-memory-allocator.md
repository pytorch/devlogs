---
title: "Pinned memory: what it is for, and why nobody gives it back"
author: Edward Yang (@ezyang)
date: 2026-08-09
tags: [eager, cuda, memory, pinned-memory, cuda-graphs]
---

> **Disclosure.** This post was drafted by Claude (Anthropic's coding
> assistant) with editing from ezyang.

For quite some time, I used to think of pinned memory as something you
sprinkled around your code to let you do async transfers to GPU.  In fact, the
old Caffe2 used to
[pin *every single* CPU tensor](https://github.com/pytorch/pytorch/blob/df0c69f32d269f8cdc136c9c65d791b6b86ef5e3/caffe2/core/context_gpu.cu#L304),
even if it never actually participated in GPU compute.  Under this regime, you might imagine that you
need an allocator for pinned memory that lets you allocate and free pinned
memory as necessary.

However, if you think carefully about the implications of async transfers on
pinned memory lifetime, as well as the implications for CUDA graphs, it turns
out that you don't... really want to ever free pinned memory.  The tl;dr:

1. CUDA graphs forbid it: a captured graph bakes the host
   buffer's address into its nodes, so any pinned buffer that participates in a
   graphed copy must stay alive at that address for as long as the graph is
   replayed.

2. Re-pinning is expensive: `cudaHostAlloc` costs milliseconds,
   so freeing in steady state just means paying that again on the next iteration.

3. The workloads that use pinned memory have a fixed working set: same
   shapes every step, so releasing between steps buys nothing.

The rest of this post works through the mechanics.

By the way, as a shortcut to believing this claim, it's worth looking at how
Megatron-LM and vLLM actually use pinned memory, because neither of them ever
releases it. Both have their own pinned pools. Both allocate up front and hold
the memory until the process exits. One of them has a config flag whose entire
documented purpose is to stop pinned buffers from being freed.

## What pinned memory is actually for

Pinned (page-locked) host memory is memory the OS has promised not to swap or
move, which lets the GPU's DMA engines read and write it directly by physical
address. This buys you four things:

* **Bandwidth.** A copy from pageable memory has to be staged through a
  driver-internal bounce buffer. Pinned memory goes straight over the interconnect
  and typically gets you 2-3x on PCIe, more on NVLink C2C.

* **Asynchrony.** `copy_(non_blocking=True)` from *pageable* memory does not
  actually overlap with anything: the driver has to stage the copy, so it blocks.
  Only from pinned memory does `non_blocking=True` return before the transfer
  completes and let the copy overlap with compute. If you were trying to hide
  transfer latency behind compute and forgot to pin, you got nothing.

* **Direct-to-hardware paths.** RDMA NICs and GPUDirect Storage require
  registered, page-locked buffers. There is no pageable option.

* **D2H inside a CUDA graph.** More on this below, but briefly: an async copy
  touching pageable memory behaves synchronously, and synchronizing operations
  are not permitted during graph capture.

## The complication: you don't own the buffer when you think you do

When you issue

```python
gpu_tensor.copy_(cpu_pinned, non_blocking=True)
```

the call returns immediately and the DMA is still in flight. The buffer is not
safe to reuse when the Python statement finishes, nor when the last reference
drops; it is safe to reuse when the copy actually completes on the stream, and
only the GPU knows when that is.  If you recycle the buffer early, you get
silent data corruption.

PyTorch's pinned memory allocator helps you avoid this use-after-free problem.
Every `copy_(non_blocking=True)` involving pinned memory records a CUDA event
against the host block
([aten/src/ATen/native/cuda/Copy.cu:481-486](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/cuda/Copy.cu#L481-L486)):

```cpp
const auto& host_tensor = (dst_device == kCPU ? dst_tensor : src_tensor);
auto* ptr = (dst_device == kCPU ? dst : src);
auto* ctx = host_tensor.storage().data_ptr().get_context();
at::getHostAllocator(at::kCUDA)->record_event(ptr, ctx, stream.unwrap());
```

Note that it keys on both the data pointer and the storage context, so a slice
of a pinned tensor, or one built with `from_blob` over allocator memory, still
attributes to the right block. On free, the block is parked in an event queue
and is not returned to the free list until every recorded event has retired
([aten/src/ATen/core/CachingHostAllocator.h:428-500](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/core/CachingHostAllocator.h#L428-L500)). Setting
`PYTORCH_ALLOC_CONF=pinned_use_background_threads:true` moves that polling to a
helper thread so it stays off the allocation fast path.

Because frees in Python are always implicit (via refcount), my personal
opinion is that making sure freed buffers don't get reused too early is a very
important anti-footgun that PyTorch's pinned memory allocator provides.

## CUDA graphs turn buffers into permanent fixtures

A CUDA graph bakes device *and host* addresses into the captured nodes. If a
D2H copy is part of the graph, it writes to one fixed host address on every
replay, forever. Two consequences:

- The buffer must be pinned, because a pageable async copy cannot be captured.
- It must be the *same* buffer on every replay. You cannot allocate a fresh
  staging tensor each step, because the graph does not know about it.

This is why graph-friendly code works the way it does. If you want to read
something back from the device (loss, grad norm, a NaN check, sampled token
ids, KV cache lengths) and you want the whole step to stay inside one graph,
you allocate a pinned staging buffer once at startup, capture the D2H copy into
it, and read it on the host after the replay. The alternative is a
device-to-host sync in the middle of your step, which un-graphs everything.

Megatron has a config flag for exactly this
([megatron/core/model_parallel_config.py:480](https://github.com/NVIDIA/Megatron-LM/blob/6518b75ecb93ad27f8c3e4d8512860faae7e7bb2/megatron/core/model_parallel_config.py#L480)):

```python
cpu_offloading_retain_pinned_cpu_buffers: bool = False
"""If True, the pinned CPU buffers are retained after offloading and reused for the
   next iteration. It is useful for cuda graphs capture.
"""
```

A flag whose entire purpose is: do not give the pinned memory back, because
CUDA graphs need stable addresses.

PyTorch's host allocator encodes the same invariant internally. Blocks
allocated during stream capture are never recycled
([aten/src/ATen/core/CachingHostAllocator.h:481-493](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/core/CachingHostAllocator.h#L481-L493)):

```
// If the block was ever used, block->event_count_ will be above
// 0 and thus can never be recycled by
// process_events_for_specific_size. Thus, this block will never
// be returned again. "Leaking" memory like this is intentional
// to avoid subtle cuda graph problems described here: ...
```

## Some examples of pinned memory usage

You don't have to use PyTorch's pinned memory allocator; it's relatively
simple to call `cudaHostRegister` yourself.

**vLLM** allocates its host-side input buffers once at worker init, sized to
`max_num_reqs`, and reuses them every step for the duration of the process
([vllm/v1/worker/gpu_model_runner.py](https://github.com/vllm-project/vllm/blob/b22afe45ac797ae58e67a7a3ad79ee5714024420/vllm/v1/worker/gpu_model_runner.py),
some twenty `pin_memory=PIN_MEMORY` allocations, mostly at init). This is the
graph-friendly input-prep pattern: fill the pinned buffer on the host, one
async H2D, replay the graph.

For KV offload, vLLM goes further and bypasses PyTorch's allocator entirely
([vllm/v1/simple_kv_offload/cuda_mem_ops.py:23-32](https://github.com/vllm-project/vllm/blob/b22afe45ac797ae58e67a7a3ad79ee5714024420/vllm/v1/simple_kv_offload/cuda_mem_ops.py#L23-L32)):

```python
def pin_tensor(tensor: torch.Tensor) -> None:
    """Pin a CPU tensor via cudaHostRegister.

    This bypasses PyTorch's CUDACachingHostAllocator which rounds
    every ``pin_memory=True`` allocation up to the next power of 2
    (e.g. 100 GB becomes 128 GB).
    """
    err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
```

**Megatron** has its own pinned pool,
[`OffloadTensorPool`](https://github.com/NVIDIA/Megatron-LM/blob/6518b75ecb93ad27f8c3e4d8512860faae7e7bb2/megatron/core/pipeline_parallel/fine_grained_activation_offload.py#L115),
which it builds as an abstraction over PyTorch's pinned memory pool:

- It keys pools on the exact `(shape, dtype)` rather than on size buckets, so
  there is no rounding waste at all.
- It keeps every tensor it has ever created in an `'all'` list, and `free()`
  just moves the tensor to a `'free'` deque. Nothing is ever released during
  training (so we never exercise PyTorch's pool event handling). The pool only grows.
- It does no event tracking whatsoever. Safety comes from choreography *around*
  the pool: dedicated `d2h_stream` / `h2d_stream`, explicit `torch.cuda.Event`
  objects, `wait_event` and `wait_stream` at the right points, and an
  `Event(external=True)` for graph interop.

All three implementations (PyTorch's, vLLM's, Megatron's) have one thing in
common: **none of them ever return pinned memory to the OS during steady-state
operation.** They allocate a working set, reuse it, and hold it until the
process exits.

## So, does deleting the tensor unpin it? No

When the last reference to a pinned tensor drops, PyTorch records events on any
streams that touched it, and once those retire, the block goes on a free list
([`maybe_cache_block`, CachingHostAllocator.h:808](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/core/CachingHostAllocator.h#L808)). Nothing calls
`cudaFreeHost`. The pages stay page-locked and registered with the driver.

```python
import gc, torch

MiB = 1024 * 1024

def report(label):
    s = torch.cuda.host_memory_stats()
    print(f"{label:20} owned={s['allocated_bytes.current']//MiB:5} MiB"
          f"  checked_out={s['active_bytes.current']//MiB:5} MiB"
          f"  cudaFreeHost calls={s['num_host_free']}")

t = torch.empty(256 * MiB, dtype=torch.uint8, pin_memory=True)
report("after alloc")

del t
gc.collect()
torch.cuda.synchronize()
report("after del")          # checked_out -> 0, owned stays at 256 MiB

torch.accelerator.empty_host_cache()
report("after empty_cache")  # owned drops, num_host_free increments
```

`active_bytes` is memory checked out to callers; `allocated_bytes` is memory
the allocator owns, active plus cached. Deleting a tensor moves bytes from the
first to the second, and only `empty_cache` removes them from the second.

When reasoning about memory allocations in PyTorch, it's important to
distinguish three different things. Releasing the tensor is refcounted and
deterministic. Making the block reusable is deferred, but only until the
stream events retire, and that deferral is correctness, not laziness.
Returning pages to the OS never happens on its own.

## When you do want it back, and how

"Allocate once and hold" is right for a training job that owns the box. There
are two cases where it is not:

- **Phase changes.** Colocated RL where a trainer and a generator alternate, or
  train-then-eval, genuinely have different pinned working sets. The
  steady-state argument assumes a steady state.
- **Pinned memory is a machine-global resource.** Unlike GPU memory,
  over-pinning degrades the whole host (page cache, other tenants) and there is
  no per-process accounting to catch it. Holding 100 GB pinned is fine for a
  job that owns the node and antisocial for a long-lived service that does not.

For those cases, you can force returning pinned memory to the OS:

```python
torch.accelerator.empty_host_cache()   # public, device-generic, 2.12+
torch._C._host_emptyCache()            # CUDA-only, private, since 2.5
```

**`torch.cuda.empty_cache()` does not do this.** It only calls
`_cuda_emptyCache` for the device allocator, and there is deliberately no
`torch.cuda.empty_host_cache()`.

`empty_cache` only reclaims blocks sitting on a free list, so it is safe to call
at any time; it just may free less than you hoped if there are still async
transfers going on.

If a full flush is too blunt (every subsequent allocation of that size pays a
fresh multi-millisecond `cudaHostAlloc`), bound the cache instead:

```
PYTORCH_ALLOC_CONF=pinned_max_cached_size_mb:512
```

Blocks above this size are freed as soon as their copy events retire rather than
cached. There is also `pinned_max_round_threshold_mb`, which disables the
power-of-two rounding above a given size; it postdates vLLM's `cudaHostRegister`
workaround above and would address the same 100 GB-becomes-128 GB problem.

One thing that is never released under any circumstances: if you enable
`pinned_reserve_segment_size_mb`, that slab is explicitly skipped by `free_block`
([aten/src/ATen/cuda/CachingHostAllocator.cpp:95](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/cuda/CachingHostAllocator.cpp#L95)) and lives until the process
exits, by design.

## Unpinning memory

If you want to think of pinned memory as something that happens to normal CPU memory,
you are welcome to directly use `cudaHostRegister` / `cudaHostUnregister` on an ordinary CPU tensor,
reachable from Python via the public `torch.cuda.cudart()`. PyTorch ships a
four-line wrapper at
[`torch/cuda/_pin_memory_utils.py`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/torch/cuda/_pin_memory_utils.py)
(private, but the underlying call is not), and distributed checkpointing uses
it for staging buffers. You get fully explicit lifetime, no caching, no rounding; but you
have to make sure you get your events correct.

## Can I use pinned memory without CUDA?

`_pin_memory` is an accelerator-specific API and will error with no
accelerator present (e.g., a CPU-only build).  This can be somewhat
inconvenient if you sometimes like to test with CPU device.  I don't really
have any advice besides irritating `t.pin_memory() if
torch.accelerator.is_available() else t` tests.

Note that technically you can get page-locked host memory with no accelerator
involved: e.g., `mlock`-ed buffers, or the page-aligned allocations that
`O_DIRECT` / libaio NVMe offload wants. This has nothing to do with CUDA and
PyTorch doesn't provide APIs for it.

## Conclusion

Hopefully, this answers some questions you might have about pinned memory and
how to go about working with it.  It is indeed quite easy to roll your own
pinned memory allocator directly from Python, and you should feel free to do
so if appropriate (though I doubt you'd need this post if you do.) But the
built-in pinned memory allocator I think is also quite serviceable, and you
should mostly consider not using it if you want exact-size fit instead of
power-of-two buckets, or if you promise to take care of deallocations
yourself.

## References

- [aten/src/ATen/core/CachingHostAllocator.h](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/core/CachingHostAllocator.h) -- generic implementation, plus
  [`Note [HostAllocator design]` at line 168](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/core/CachingHostAllocator.h#L168)
- [aten/src/ATen/cuda/CachingHostAllocator.cpp](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/cuda/CachingHostAllocator.cpp) -- CUDA specialization
- [aten/src/ATen/native/cuda/Copy.cu:481](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/cuda/Copy.cu#L481) -- where copies record host events
- [torch/cuda/_pin_memory_utils.py](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/torch/cuda/_pin_memory_utils.py) -- explicit register/unregister
- [Pinned memory options in the CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
- [Graph-based CPU Offloading for TorchTitan](/devlogs/distributed/2026-06-23-cpu-offloading/) -- which needed its own pinned pool too
- [expose host_emptyCache to python (#134919)](https://github.com/pytorch/pytorch/pull/134919)
- [Introduce a unified API to empty the host cache memory (#171270)](https://github.com/pytorch/pytorch/pull/171270)
