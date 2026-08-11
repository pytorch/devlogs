---
title: "Host-to-device syncs are bad too"
author: Edward Yang (@ezyang)
date: 2026-08-11
tags: [eager, cuda, performance, profiling]
---

> **Disclosure.** This post was drafted by Claude (Anthropic's coding
> assistant) with editing from ezyang.

Once you get a bit experienced with writing performant PyTorch code you know
to avoid device-to-host syncs: calling `.item()` is an easy way to become CPU
bound in kernel launches afterwards, since we have to wait for the GPU work to
finish before we can get the result to the CPU.  And with more time, you might
find out about a number of API footguns in PyTorch's API that invisibly cause
DtoH syncs, like `x[bool_tensor]`, which implicitly triggers a `torch.nonzero`
sync.

But what you might not be expecting is that host-to-device syncs are
bad too!  And these syncs can also show up in sneaky ways, like
`torch.tensor(0, device="cuda")` or `x[:, (0, 2, 3)]`.  The goal of this post is
to educate you on this class of problem.

## Why is a host-to-device sync bad?

A host-to-device (or a device-to-host) *transfer* is not inherently bad.  With
the use of pinned memory, you can set up these transfers to occur
asynchronously, so that work on CPU/GPU doesn't block on the transfer
occurring.

A *sync*, however, is different from a transfer.  In eager mode the host and
the device are a producer/consumer pair: Python enqueues kernels into a
stream, and the GPU drains them.  In a healthy steady state the host runs
some distance *ahead*--it has already launched the kernels the GPU will
execute over the next several milliseconds.  That lead is what lets you get
away with slow Python in your training loop at all: as long as the queue
never empties, host-side dispatch overhead is completely hidden.

A `cudaStreamSynchronize` zeroes out that lead, costing you two things:

- **The host wait itself.**  This is the part that shows up in a profile--a
  red bar on the CPU timeline--and it is mostly not the problem.  The GPU was
  busy that whole time; nothing was wasted.
- **The bubble afterwards.**  From the instant the sync returns, the host has
  no lead, so GPU idle time is directly exposed to Python overhead until the
  lead is rebuilt.  This is the real cost, and it is the one you have to
  compute yourself, because no profiler attributes it back to the sync that
  caused it: it shows up as diffuse GPU idle scattered across the rest of the
  step.

## What causes host-to-device syncs?

It is easily understandable why a device-to-host transfer requires a sync: you
have to wait for the GPU to finish computing a result before you can get it to
the host.  But why do we sometimes need a sync for host-to-device transfers?
To answer this question, we have to take a little detour into describing how
PyTorch gets information from host to GPU when executing kernels.

Ordinarily, pending kernel launch information are stored on a pushbuffer.
This channel is fully asynchronous, and can store arbitrary data!
For example, in `torch.full((n,), 3.0, device='cuda')`, the
`3.0` becomes a field of a `FillFunctor` struct that is passed by value
to the kernel ([`aten/src/ATen/native/cuda/FillKernel.cu:24`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/cuda/FillKernel.cu#L24)).  Most other
kernels that accept Python numbers operate the same way.
TensorIterator even extends it to 0-dim CPU *tensors*: for a pointwise op,
`x + torch.tensor(1.0)` reads the value on the host, drops the operand, and
folds it into the functor ([`aten/src/ATen/native/cuda/Loops.cuh:193`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/cuda/Loops.cuh#L193)).

However, this channel is small: kernel parameters are capped at 32 KB (or 4 KB
if your CUDA is old and decrepit), so it is only good for scalars--bulk data
has to take the other channel.  (PyTorch actually bumps into this cap in the
`foreach` ops and fused optimizers--there's logic to split a single giant
kernel launch into multiple kernel launches to get around this restriction;
[`MultiTensorApply.cuh:21`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/cuda/MultiTensorApply.cuh#L21).)

> Fun fact: CUDA also secretly uses the pushbuffer in other non-apparent cases
> as well.  For example, if you do a `cpu_tensor.to("cuda")` and the
> `cpu_tensor` is not pinned but small enough (e.g., less than 64KB), the
> usermode driver may just put the contents of your tensor on the pushbuffer.
> Unfortunately, this doesn't help you at all because we always do a stream
> synchronize after this operation.  (You could unsafely drop the synchronize,
> but NVIDIA reserves the right to delay actually reading your non-pinned CPU
> memory).  When people on the Internet write "non-pinned memory is copied
> into a pinned staging buffer", I believe they are referring to this copy to
> the pushbuffer.  But you only do this for small tensors, since copying big
> CPU tensors is expensive.

The other channel is to have the GPU DMA the memory from the CPU, which is what
something like `.to("cuda")` will do (modulo the note above).  It is not
documented how exactly the closed source usermode driver implements
`cudaMemcpyAsync` in this case, but Opus and Sol jointly ran a number of
experiments on a GB200 and we think that concretely what happens in this case
is that when the source is `>64KB`:

1. The target stream is drained (sync!)  We have to drain
   the target stream before the DMA because another kernel might be using the
   memory you're about to DMA into.

2. The copy engine DMAs the virtual address from the host (with the GPU's MMU
   doing per-page translation, thus accounting for the fact that the unpinned
   could get paged out.)  If you're lucky, you can do it without faulting
   (otherwise you'll spend a bunch of time handling `handle_mm_fault` faults,
   on order of 3ms per GB).

In the case of `.to("cuda")` (without `non_blocking=True`), we will then
immediately issue another stream synchronize.  In practice, this doesn't
seem to be strictly necessary, since we block on the DMA completing anyway,
but async CUDA APIs reserve the right to be asynchronous with respect the host
memory too (even if in practice they're not.)

The upshot of all of this is that a HtoD will cause a sync.  Either because we
need to make sure the destination memory is safe to DMA into, or because no
one can guarantee the source CPU tensor will not get modified before the HtoD
transfer finishes.

So, pop quiz!  Which of these sync?

```python
torch.full((3,), 1.0, device="cuda")           # no copy: 1.0 is a kernel argument
torch.arange(3, device="cuda")                 # no copy: computed on device
x + 1.0                                        # no copy
x + torch.tensor(1.0)                          # no copy: 0-dim CPU operand gets lifted
torch.tensor([0.0, 1.0, 2.0], device="cuda")   # memcpy, and a full stream sync
```

Three numbers through `torch.full` cost nothing.  Three numbers through
`torch.tensor` drain your entire launch queue.  Which channel you get is
decided by which operator you called, not by how much data you're moving--and
there is nothing at the call site to tell you which one you picked.

## Advanced indexing trap

PyTorch indexing has two regimes.  *Basic* indexing--integers, slices,
ellipsis--is resolved entirely on the host: the result is a view, computed by
fiddling with sizes, strides, and storage offset, and no kernel runs at all.
*Advanced* indexing--indexing with a tensor, or with anything that must
become one--runs a gather kernel, and the indices are an input *tensor* to
that kernel, not part of its argument list.

Which brings us to the trap: fancy indexing looks like the first channel but
is the second.  `t[:, (0, 2, 3)]` reads like an argument list--three small
integers, known at the call site, morally no different from the `3.0` you
passed to `torch.full`.  But actually, we convert the Python tuple
`(0, 2, 3)` into a CUDA tensor to perform the advanced indexing kernel, and
that conversion is `torch.tensor([0, 2, 3], device=t.device)` in disguise--a
pageable CPU tensor, a blocking HtoD copy, a full stream sync.

Honestly, this is terrible API design: so little syntax separates the
regimes.  `t[:, 1:3]` is a view; `t[:, (1, 3)]` drains your launch queue.
Nothing at the call site tells you that you crossed the line--the expression
that syncs is the one that *looks* most like passing a couple of constants.

## A case study

Here is a story that happens in every large training codebase eventually.

Somebody profiles the step, notices that the CPU spends tens of milliseconds
blocked inside `loss.item()`, and does the textbook fix: allocate a pinned
host buffer once, `copy_(loss, non_blocking=True)` into it, record a CUDA
event, and don't actually read the Python float until the next step
boundary.  The profile confirms the block on `.item()` is gone.  But the
median step time doesn't change.

The rule of syncs is that your code doesn't get any better until you've
fixed *all* the syncs.  Imagine, if you will, a metrics helper that
accumulates per-parameter-group statistics into a single
`[num_groups, 8]` CUDA tensor.  Seven of the eight columns want a sum
reduction; column 1 (`max`) wants a max.  Here's some code that does it:

```python
SUM_COLS = (0, 2, 3, 4, 5, 6, 7)

def reduce_grad_stats(local: torch.Tensor):        # local: [G, 8] on cuda
    sums = local[:, SUM_COLS].sum(dim=0)           # <-- this line
    gmax = local[:, 1].max()
    ...
```

That first line is the advanced indexing trap: a full `cudaStreamSynchronize`.
But it's a lot harder to see!  And if you don't find it, none of your async
changes will materialize into concrete gains.

As Claude, I want to make one last observation: this bug lived in the
observability code, and that is not a coincidence.  Metrics, logging, and
stats aggregation are the parts of a training step that nobody compiles,
nobody CUDA-graphs, and nobody benchmarks in isolation, because "it's just a
few tiny reductions, it can't matter."  The tiny reductions didn't matter.
The host stall next to them did.

## Which indexing forms sync, and which don't

If you know what advanced indexing is, you can predict which syntax forms are
bad.  But here's a table in case you don't:

| Expression | What happens |
|---|---|
| `t[3]`, `t[:, 1]`, `t[2:5]`, `t[..., :1]` | View metadata only.  No kernel, no sync. |
| `t.select`, `t.narrow`, `t.unbind`, `t.split`, `t.chunk` | Same--views. |
| `t[:, (0, 2, 3)]`, `t[:, [0, 2, 3]]` | Builds a pageable CPU int64 tensor, then a blocking H2D copy: **`cudaStreamSynchronize`**. |
| `t[:, np.array([0, 2, 3])]` | Same. |
| `t[:, idx]` where `idx` is already a CUDA tensor | Gather kernel.  No sync. |
| `t[:, idx_cpu]` where `idx_cpu` is a CPU int tensor | Blocking copy of the index to the device ([`IndexingUtils.h:61`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/IndexingUtils.h#L61)): **sync**. |
| `t[cuda_bool_mask]` | `nonzero` must report its output size to the host to size the result: **sync** (a DtoH one, inherent to data-dependent shapes). |
| `t[cpu_bool_mask]` | Actually optimized: the `nonzero` output is written to *pinned* memory and copied async ([`IndexingUtils.h:52`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/IndexingUtils.h#L52)). |

That last row deserves a double take: didn't we just say `nonzero` syncs?
Only when the mask is a *CUDA* tensor, where the host must wait to learn the
output size.  A CPU mask's `nonzero` is an ordinary host computation--no
stream involved--so the only hazard left is the HtoD copy of the resulting
indices, and that is exactly what got optimized
([#156384](https://github.com/pytorch/pytorch/pull/156384)): somebody
noticed this class of bug and fixed it for boolean masks:

```cpp
if (ensure_same_device && index.device() != self.device()) {
  bool non_blocking = index.is_cpu() && self.device().is_cuda();
  auto out = at::empty({0}, index.options().dtype(kLong).pinned_memory(non_blocking));
  nonzero = at::nonzero_out(out, index).to(self.device(), non_blocking);
}
```

...and then ten lines later, the integer-index branch does the naive thing.
Constructing the index tensor pinned and copying it `non_blocking` would be
a small, well-precedented upstream change--the event mechanism from earlier
already makes it safe to free the pinned staging buffer immediately.  I
think this is worth doing; it turns a 50 ms cliff into a 56-byte copy.

## Finding these

The good news is that this class of bug is mechanically detectable, because
PyTorch instruments its synchronizing operations
([`c10/cuda/CUDAFunctions.h:86`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/c10/cuda/CUDAFunctions.h#L86)):

```python
torch.cuda.set_sync_debug_mode("error")
...
sums = local[:, SUM_COLS].sum(dim=0)
# RuntimeError: called a synchronizing CUDA operation
```

Run your metrics and logging code under this once.  It is not
exhaustive--the docs warn that `torch.distributed` and `torch.sparse` aren't
covered--but it catches every unpinned H2D and every D2H
`.item()`/`.cpu()`, which is the bulk of what you're hunting.

CUDA graph capture is an even stricter detector: an unpinned host copy
inside a capture region hard-errors
([`Copy.cu:454`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/cuda/Copy.cu#L454)):

```cpp
if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
  TORCH_CHECK(host_tensor->is_pinned(),
      "Cannot copy between CPU and CUDA tensors during CUDA graph capture ...");
}
```

The parts of your step that are captured or compiled get strong static
guarantees about host/device interaction for free.  The parts that
aren't--precisely the parts written by people who weren't thinking about the
stream at all--are where you should point `set_sync_debug_mode`.

## Conclusion

PyTorch eager earned the love of its users by making it extremely easy to
weave in and out of GPU and CPU computation for debugging.  But in modern,
performant training stacks, it's important to ban syncs to prevent bubbles in
your kernel execution and to make your code CUDA graphable.  These syncs can
be quite subtle, so use machine assistance when possible to hunt them down!

## References

- [`torch/csrc/autograd/python_variable_indexing.cpp:139`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/torch/csrc/autograd/python_variable_indexing.cpp#L139), [`:284`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/torch/csrc/autograd/python_variable_indexing.cpp#L284)--sequence index
  becomes a tensor, on the indexed tensor's device
- [`torch/csrc/utils/tensor_new.cpp:436`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/torch/csrc/utils/tensor_new.cpp#L436), [`:465`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/torch/csrc/utils/tensor_new.cpp#L465)--unpinned host allocation, then
  `.to(device, non_blocking=false)`
- [`aten/src/ATen/native/cuda/Copy.cu:454`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/cuda/Copy.cu#L454), [`:486`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/cuda/Copy.cu#L486), [`:488`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/cuda/Copy.cu#L488)--capture check,
  host-allocator event recording, and the blocking path
- [`c10/cuda/CUDAFunctions.h:78`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/c10/cuda/CUDAFunctions.h#L78)--`memcpy_and_sync`
- [`aten/src/ATen/native/IndexingUtils.h:52`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/IndexingUtils.h#L52), [`:61`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/aten/src/ATen/native/IndexingUtils.h#L61)--the pinned bool-mask path and
  the unpinned integer path next to it
- [`torch.cuda.set_sync_debug_mode`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html)
- [Pinned memory: what it is for, and why nobody gives it back](/devlogs/eager/2026-08-09-pinned-memory-allocator/)
