---
title: "Cross-stream memory management: wait_event instead of record_stream"
date: 2026-09-29
author: "Wei Feng (@weifengpy)"
tags: [eager, cuda, streams, memory, fsdp]
---

> **TL;DR** – The CUDA caching allocator only knows which stream a tensor was
> *allocated* on. When a producer stream allocates a buffer and a consumer
> stream reads it, keep a Python reference to the buffer and record a CUDA
> event on the consumer, then make the producer stream `wait_event` on it
> *before* you drop the reference. Don't reach for `Tensor.record_stream`: it's
> correct, but it hands memory reuse to the allocator's event polling, which
> makes peak memory depend on how far the CPU runs ahead.

Overlapping communication with compute comes down to a producer and a consumer
on two CUDA streams. In FSDP2's forward pass, the all-gather stream (the
producer) fetches each layer's full parameters, and the compute stream (the
consumer) runs forward on them. While compute runs layer 1, the all-gather for
layer 2 is already in flight, so communication hides behind compute.

<img src="/devlogs/images/eager/cross-stream-pipeline.svg" style="max-width: 100%" alt="Producer-consumer pipeline: all-gathers on the AG stream run ahead of forward compute on the compute stream">

Getting this overlap is easy. Getting it without corrupting memory, and
without making memory usage unpredictable, takes a little care. This post walks
through the options on a single GPU, with no `torch.distributed` involved,
then shows how FSDP2 uses the recommended pattern. The same reasoning applies
to anything else that moves data on a side stream, such as activation
offloading with separate H2D/D2H streams.

The full runnable script is at the [end of this post](#appendix-the-demo-script).
In every figure, the top lanes are CUDA streams and the bottom lanes are memory
blocks. **Color identifies the memory block** a kernel writes or reads.

## Streams, events, and the caching allocator

A CUDA stream is a FIFO queue of GPU work. Work on one stream runs in order;
work on different streams may overlap. Two primitives order work across
streams, and **both are GPU-side: the CPU never blocks on them**:

- `s.wait_stream(other)`: work enqueued on `s` from now on waits for
  everything *currently* enqueued on `other`.
- `s.wait_event(e)`: finer-grained. Work enqueued on `s` from now on waits
  for the single point on another stream where `e = other.record_event()` was
  recorded.

<img src="/devlogs/images/eager/cross-stream-basics.svg" style="max-width: 100%" alt="wait_stream orders stream q after everything already enqueued on p; wait_event orders q after only the point where event e was recorded">

The [CUDA caching allocator](/devlogs/eager/2026-06-01-cuda-caching-allocator/)
tags every block with the stream it was **allocated** on, and only hands a
freed block to later allocations **on that same stream**. When you `del` a
tensor, the allocator reasons: "every kernel on the allocation stream that
touches this block was enqueued before the `del`, so any kernel enqueued later
on that stream, which is the only stream that can get this block, runs after
them." That argument holds within a single stream and says nothing about
other streams. The allocator **has no idea that another stream is still
reading the block**.

## The bug: the allocator only sees one stream

Here is the naive pipeline. The producer "all-gathers" each layer's
parameters on `ag_stream` (simulated with `torch.full`), and the consumer runs
a matmul on the default stream. `wait_stream` makes sure forward starts after
the all-gather that produced its parameters.

```python
prev_params = all_gather(ag_stream, layer_id=1)

for layer_id in range(2, NUM_LAYERS + 1):
    default.wait_stream(ag_stream)              # fwd L waits for AG L
    result = forward(prev_params, x, layer_id - 1)

    del prev_params                             # matmul is still running!

    prev_params = all_gather(ag_stream, layer_id)
```

`del prev_params` runs on the CPU right after the matmul is *enqueued*, long
before it finishes. The block was allocated on `ag_stream`, so the allocator
checks only `ag_stream`, decides the block is reusable, and gives it straight
back to the next all-gather. `AG L2` lands in block A and overwrites layer 1's
parameters while `fwd L1` is still reading them. So do `AG L3` and `AG L4`,
because nothing orders `ag_stream` after the compute stream:

<img src="/devlogs/images/eager/cross-stream-bug.svg" style="max-width: 100%" alt="Bug: after del L1 the allocator hands block A to AG L2, then AG L3 and AG L4, while fwd L1 is still reading it">

This isn't a theoretical race. On an H100 it corrupts every run:

```text
=== BUGGY: no lifetime management ===
  trial  0: CORRUPT  max_diff=10876.0  (expected 0)
  trial  1: CORRUPT  max_diff=10854.0  (expected 0)
  trial  2: CORRUPT  max_diff=10882.0  (expected 0)
  [FAIL] 50/50 trials corrupted, at most 1 param blocks per run (4 layers)
```

The correct value per element is `HIDDEN * 1 = 4096`, and each weight
overwritten by layer `k` adds `k - 1` to it. A `max_diff` above `2 * 4096` is
only possible if `fwd L1` read many weights that `AG L4` had already overwritten
(4.0 instead of 1.0).

Whether and how often it corrupts depends on kernel timing, so a bug like this
can pass on one GPU or model size and fail on another. Don't rely on "it
passed once".

## The textbook fix: `record_stream`, and why we avoid it

The API designed for this is
[`Tensor.record_stream`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html):
tell the allocator that another stream also used the tensor.

```python
    result = forward(prev_params, x, layer_id - 1)
    prev_params.record_stream(default)          # "default also used this block"
    del prev_params
    prev_params = all_gather(ag_stream, layer_id)
```

It is correct (`0/50 trials corrupted`). Here's what happens inside the allocator:

1. [`recordStream`](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/c10/cuda/CUDACachingAllocator.cpp#L2667-L2678)
   adds `default` to the block's `stream_uses` set.
2. On `del`, because `stream_uses` isn't empty, the block is
   [not freed](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/c10/cuda/CUDACachingAllocator.cpp#L2586-L2604).
   Instead,
   [`insert_events`](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/c10/cuda/CUDACachingAllocator.cpp#L4492-L4508)
   records a CUDA event on each recorded stream and parks the block.
3. Normally, the block returns to the pool only when a *later* allocation calls
   [`process_events`](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/c10/cuda/CUDACachingAllocator.cpp#L4530)
   (from
   [`prepare_for_malloc`](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/c10/cuda/CUDACachingAllocator.cpp#L1794-L1811)),
   and `cudaEventQuery` reports that the event has completed.

So reuse is no longer decided by the stream order you wrote. It's decided by
**whether the GPU happens to have finished by the time the CPU asks for its
next allocation**. In a healthy pipeline the CPU runs ahead of the GPU, so
the answer is usually "not yet":

<img src="/devlogs/images/eager/cross-stream-record-stream.svg" style="max-width: 100%" alt="record_stream: every allocation on the AG stream happens before the previous block's event is polled, so each layer gets a new block">

```text
=== RECORD_STREAM: correct, allocator polls events ===
  [PASS] 0/50 trials corrupted, at most 4 param blocks per run (4 layers)
```

Four layers, four blocks. The CPU enqueued all four all-gathers before `fwd
L1` finished, so every allocation found the previous block's event still
pending and carved out a fresh one. The number of live buffers is bounded by
**how far the CPU runs ahead**, not by the two-deep pipeline you designed.
This leads to three practical problems:

- **Peak memory is nondeterministic.** It changes with CPU speed, kernel
  timing, and anything else that shifts the race. FSDP1 frees its unsharded
  parameters
  [with `record_stream`](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/torch/distributed/fsdp/_flat_param.py#L1828-L1832),
  and ships a CPU-side
  ["rate limiter"](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/torch/distributed/fsdp/fully_sharded_data_parallel.py#L351-L360)
  (`limit_all_gathers=True`, on by default) that bounds the growth by blocking
  the CPU thread, capping all-gathered memory at two FSDP instances.
- **Hidden syncs under memory pressure.** Blocks parked on events can't be
  reused, so you run out of cached memory sooner. If no cached block fits and
  `cudaMalloc` fails, the allocator's OOM-retry path calls
  [`release_cached_blocks`](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/c10/cuda/CUDACachingAllocator.cpp#L4210-L4220),
  which blocks the CPU on *every* outstanding `record_stream` event
  (`cudaEventSynchronize`) before it can reclaim those blocks.
- **It's easy to miss a stream.** Correctness depends on every consumer
  stream being recorded, and nothing in the code tells you when one is
  missing.

The
[`record_stream` docstring](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/torch/_tensor_docs.py#L3994-L4005)
says the same thing and points to the alternative:

> You can safely use tensors allocated on side streams without
> `record_stream`; you must manually ensure that any non-creation stream uses
> of a tensor are synced back to the creation stream before you deallocate the
> tensor.

`record_stream` still has its place: a one-off handoff in a cold path, or a
library that returns a side-stream tensor to callers who shouldn't have to
think about streams. For a steady-state pipeline where you own the whole
lifetime, sync back to the creation stream yourself. The next two sections
show how.

## Fix 1: stall the producer with `wait_event`

The direct way to "sync back to the creation stream" is to record an event
after forward and make `ag_stream` wait on it before the block can be handed
out again:

```python
    result = forward(prev_params, x, layer_id - 1)
    fwd_done = default.record_event()

    ag_stream.wait_event(fwd_done)              # 1. order ag_stream after fwd
    del prev_params                             # 2. only then release the block
    prev_params = all_gather(ag_stream, layer_id)
```

This is correct and uses exactly one block. On the CPU, the next all-gather
gets the same address immediately. That's fine: its kernel is queued behind
the `wait_event`, so it can't write until `fwd L1` is done. But it serializes
the pipeline. `AG L2` can't start until `fwd L1` finishes, so the compute
stream sits idle waiting for every all-gather:

<img src="/devlogs/images/eager/cross-stream-stall.svg" style="max-width: 100%" alt="Stall: the AG stream waits for each forward before reusing the single block, so all-gather and forward are serialized">

**The invariant.** Every allocation that could be handed this block runs on
the block's allocation stream, so it's enough that **the `wait_event` is
enqueued on the allocation stream before the block is released**. Put
`wait_event` *before* `del`, not after. If you `del` first, any allocation
that sneaks in between (a hook, another module, another library) can get the
block, and its kernels won't be ordered after the consumer.

## Fix 2 (recommended): keep a reference, defer the `wait_event`

To keep the overlap, don't release layer 1's block when you're done
*enqueueing* its forward. Keep a Python reference to it. While that reference
is alive, the allocator can't recycle the block, so `AG L2` gets a
**different block** and starts right away. Release L1 one step later, again
with `wait_event` first:

```python
keepalive, keepalive_event = None, None
prev_params = all_gather(ag_stream, layer_id=1)

for layer_id in range(2, NUM_LAYERS + 1):
    if keepalive is not None:
        ag_stream.wait_event(keepalive_event)   # 1. order ag_stream after fwd
        del keepalive                           # 2. only then release the block

    default.wait_stream(ag_stream)
    result = forward(prev_params, x, layer_id - 1)
    fwd_done = default.record_event()

    keepalive, keepalive_event = prev_params, fwd_done   # don't free yet
    prev_params = all_gather(ag_stream, layer_id)        # gets a different block
```

<img src="/devlogs/images/eager/cross-stream-keepalive.svg" style="max-width: 100%" alt="Keepalive: holding the L1 reference makes AG L2 use a second block; the AG stream waits on fwd_done only right before block A is reused">

The producer only waits right before it reuses block A for `AG L3`, and by
then `fwd L1` is usually finished or close to it. The compute stream never
waits for buffer reuse. It only waits for the all-gather it actually needs,
and that wait is hidden as long as each all-gather is shorter than a forward.
The cost is explicit and
fixed: **one extra buffer buys the overlap**, and the two blocks alternate:

```text
=== PIPELINED: keep ref + wait_event ===
  [PASS] 0/50 trials corrupted, at most 2 param blocks per run (4 layers)
```

This generalizes to deeper pipelines: to allow N buffers in flight, keep N
references in a queue and wait-then-drop the oldest. The memory bound is a
number you chose, not an outcome of CPU run-ahead.

| Approach | Correct | Producer overlaps consumer | Param blocks (demo, 4 layers, max per run) | Block reuse is decided by |
|---|---|---|---|---|
| No lifetime management | ✗ | ✓ | 1 | nothing (it's a race) |
| `record_stream` | ✓ | ✓ | 4, grows with CPU run-ahead | allocator polling events on later mallocs |
| Stall: `wait_event` | ✓ | ✗ | 1 | stream order |
| Keepalive + `wait_event` | ✓ | ✓ | 2, fixed by design | stream order + your reference |

## How FSDP2 uses this

FSDP2 creates four side streams in
[`FSDPCommContext.lazy_init`](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L82-L109),
`all_gather_copy_in_stream`, `all_gather_stream`, `reduce_scatter_stream`
and `all_reduce_stream`, which together with the compute stream make five.
On CUDA, FSDP2 never calls `record_stream`. Cross-stream buffers are kept alive
by references and synced back to their allocation stream with `wait_event` or
`wait_stream`. The two handoffs that need overlap keep their keepalive in
these tuples, which are exactly "a reference plus an event":

```python
class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.Event | None  # all-gather copy-out

class ReduceScatterState(NamedTuple):
    reduce_scatter_input: torch.Tensor
    event: torch.Event | None  # reduce-scatter event
    allocation_stream: torch.Stream  # Owns the input allocation
    param_group: FSDPParamGroup  # Identifies the owning FSDP root
```

### Forward: copy-in → all-gather → copy-out

Each layer's unshard runs in three stages on three streams:

1. Copy the local shards into one flat buffer on the **copy-in stream**.
2. All-gather in place on the **all-gather stream**. The all-gather input is
   a view of its output, so this is one allocation, made on the copy-in stream.
3. Copy out into the unsharded parameters on the **compute stream**, right
   before forward.

<img src="/devlogs/images/eager/cross-stream-fsdp2-forward.svg" style="max-width: 100%" alt="FSDP2 forward: copy-in, all-gather and copy-out on three streams; the copy-out event of L1 gates reuse of L1's all-gather buffer">

The all-gather buffer is produced on the copy-in stream and last read by the
copy-out on the compute stream, so it's a producer-consumer handoff. In
[`wait_for_unshard`](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L540-L556),
FSDP2 records `all_gather_copy_out_event` after the copy-out and saves
`AllGatherState(all_gather_result, event)` as the keepalive instead of
freeing the buffer. When the next layer reaches `wait_for_unshard`, which is
after its own copy-in has already allocated a different block,
[`release_all_gather_state_for_comm_reuse`](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L144-L154)
makes **both** all-gather streams `wait_event` on the saved event and only then
drops the reference. Since input and output share one allocation, one event
covers both. (The figure only draws the copy-in stream's wait. The all-gather
stream's wait is enqueued after `AG L2`, so by the time it runs it's already
satisfied.) This is the keepalive pattern with the copy-in stream as the
allocation stream. (See also
[Note: Overlapping all-gather copy-in and all-gather](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L65-L74).)

### Backward: reduce-scatter in the other direction

In backward the roles flip. The reduce-scatter input is
[allocated on the compute stream](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L670-L705)
(the producer, which copies gradients into it) and read by the reduce-scatter
on `reduce_scatter_stream` (the consumer), which records `reduce_scatter_event`.
FSDP2
[appends a `ReduceScatterState`](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L804-L811)
holding the input, the event, and the allocation stream. When a later module's
post-backward needs room, it
[pops the oldest state](https://github.com/pytorch/pytorch/blob/1a0b56b882d23423ac97d2799f6c4c6fd2957836/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L716-L744),
calls `allocation_stream.wait_event(event)`, and only then does `del oldest`.

This is also the "N buffers in flight" generalization in practice. By default
FSDP2 keeps one reduce-scatter input alive. As in Fix 1 there's a single
buffer, but the wait is deferred to the next module's post-backward, just
before L3's gradients are copied into block A. So `bwd L3` still overlaps
`RS L4`, and the compute stream stalls only when `RS L4` takes longer than
`bwd L3`:

<img src="/devlogs/images/eager/cross-stream-fsdp2-backward-1buf.svg" style="max-width: 100%" alt="FSDP2 backward with one reduce-scatter input buffer: the compute stream waits for RS L4 before copying L3's gradients into the same block">

`FSDPModule.set_reduce_scatter_max_input_buffers(2)` (experimental) turns it
into Fix 2. L3's copy-in gets a second block, and by the time block A is
reused for L2, the `wait_event` on `RS L4` is typically already satisfied.
The cost is one extra reduce-scatter input buffer, a known amount of memory:

<img src="/devlogs/images/eager/cross-stream-fsdp2-backward-2buf.svg" style="max-width: 100%" alt="FSDP2 backward with two reduce-scatter input buffers: L3's copy-in uses a second block, and the wait on RS L4 before reusing block A is typically already satisfied">

## Checklist

When a buffer crosses streams:

1. **Name the allocation stream.** That is the only stream whose later
   allocations can reuse the block, and so the only stream that needs to be
   ordered.
2. **Record an event on the consumer** after its last use of the buffer.
3. **Hold a Python reference** (keepalive) for as long as you want the
   producer to keep running ahead.
4. **`wait_event` on the allocation stream, then `del`**, in that order.
5. **Choose how many buffers may be in flight.** One means stall, two means
   ping-pong, N means a queue. Make it a number, not a side effect of CPU
   run-ahead.

## Appendix: the demo script

<details>
<summary><code>cross_stream_demo.py</code> (single GPU, no <code>torch.distributed</code>)</summary>

```python
"""
Producer-consumer pipelining across CUDA streams — single-GPU demo.

Simulates FSDP2's all-gather → forward pipeline: a producer stream
"fetches" each layer's parameters while a consumer stream runs forward
compute (matmul) on the previous layer's parameters.

Run:  python cross_stream_demo.py
Requires: 1 CUDA GPU.
"""
import torch

HIDDEN = 4096
BATCH = 4096
PARAM_SIZE = HIDDEN * HIDDEN  # 16M floats = 64 MB
NUM_LAYERS = 4


# Addresses of every all-gather output, to count how many distinct blocks
# a run uses
param_ptrs: list[int] = []


def all_gather(stream: torch.cuda.Stream, layer_id: int) -> torch.Tensor:
    """Simulate AG: allocate and fill a param buffer on `stream`."""
    with torch.profiler.record_function(f"AG L{layer_id}"):
        with torch.cuda.stream(stream):
            params = torch.full((PARAM_SIZE,), float(layer_id), device="cuda")
    param_ptrs.append(params.data_ptr())
    return params


def forward(params: torch.Tensor, x: torch.Tensor, layer_id: int) -> torch.Tensor:
    """Simulate forward: matmul with the layer's weight matrix."""
    with torch.profiler.record_function(f"fwd L{layer_id}"):
        weight = params.view(HIDDEN, HIDDEN)
        return x @ weight


def check_results(results: list[tuple[torch.Tensor, int]]) -> float:
    max_diff = 0.0
    for result, layer_id in results:
        expected = float(HIDDEN) * layer_id
        max_diff = max(max_diff, (result - expected).abs().max().item())
    return max_diff


# ---------------------------------------------------------------------------
# Buggy: no lifetime management
# ---------------------------------------------------------------------------

def buggy(x: torch.Tensor) -> float:
    ag_stream = torch.cuda.Stream()
    default = torch.cuda.default_stream()
    results = []

    # Prefetch L1
    prev_params = all_gather(ag_stream, layer_id=1)

    for layer_id in range(2, NUM_LAYERS + 1):
        # Forward previous layer (enqueue only, no CPU sync)
        default.wait_stream(ag_stream)
        result = forward(prev_params, x, layer_id - 1)
        results.append((result, layer_id - 1))

        # Free previous — matmul still running on GPU
        del prev_params

        # Prefetch next — may reuse freed block while matmul reads it
        prev_params = all_gather(ag_stream, layer_id)

    # Last layer
    default.wait_stream(ag_stream)
    result = forward(prev_params, x, NUM_LAYERS)
    results.append((result, NUM_LAYERS))
    del prev_params

    torch.cuda.synchronize()
    return check_results(results)


# ---------------------------------------------------------------------------
# Stall: the producer waits for each forward before reusing the block
# ---------------------------------------------------------------------------

def stall_producer(x: torch.Tensor) -> float:
    ag_stream = torch.cuda.Stream()
    default = torch.cuda.default_stream()
    results = []

    prev_params = all_gather(ag_stream, layer_id=1)

    for layer_id in range(2, NUM_LAYERS + 1):
        default.wait_stream(ag_stream)
        result = forward(prev_params, x, layer_id - 1)
        fwd_done = default.record_event()
        results.append((result, layer_id - 1))

        # Stall: AG waits for forward before the block can be reused
        ag_stream.wait_event(fwd_done)
        del prev_params
        prev_params = all_gather(ag_stream, layer_id)

    default.wait_stream(ag_stream)
    result = forward(prev_params, x, NUM_LAYERS)
    fwd_done = default.record_event()
    results.append((result, NUM_LAYERS))
    ag_stream.wait_event(fwd_done)
    del prev_params

    torch.cuda.synchronize()
    return check_results(results)


# ---------------------------------------------------------------------------
# Keepalive: hold a Python ref + wait_event before del
# ---------------------------------------------------------------------------

def keep_ref(x: torch.Tensor) -> float:
    ag_stream = torch.cuda.Stream()
    default = torch.cuda.default_stream()
    results = []
    keepalive = None     # previous iteration's params (ref kept alive)
    keepalive_event = None

    prev_params = all_gather(ag_stream, layer_id=1)

    for layer_id in range(2, NUM_LAYERS + 1):
        # Free previous iteration's keepalive (wait_event first, then del)
        if keepalive is not None:
            ag_stream.wait_event(keepalive_event)
            del keepalive

        # Forward current layer
        default.wait_stream(ag_stream)
        result = forward(prev_params, x, layer_id - 1)
        fwd_done = default.record_event()
        results.append((result, layer_id - 1))

        # Don't del prev_params — save as keepalive
        keepalive = prev_params
        keepalive_event = fwd_done

        # Prefetch next — gets DIFFERENT block because keepalive holds ref
        prev_params = all_gather(ag_stream, layer_id)

    # Free last keepalive
    if keepalive is not None:
        ag_stream.wait_event(keepalive_event)
        del keepalive

    # Last layer
    default.wait_stream(ag_stream)
    result = forward(prev_params, x, NUM_LAYERS)
    fwd_done = default.record_event()
    results.append((result, NUM_LAYERS))
    ag_stream.wait_event(fwd_done)
    del prev_params

    torch.cuda.synchronize()
    return check_results(results)


# ---------------------------------------------------------------------------
# record_stream: correct, but reuse is decided by the allocator polling events
# ---------------------------------------------------------------------------

def record_stream(x: torch.Tensor) -> float:
    ag_stream = torch.cuda.Stream()
    default = torch.cuda.default_stream()
    results = []

    prev_params = all_gather(ag_stream, layer_id=1)

    for layer_id in range(2, NUM_LAYERS + 1):
        default.wait_stream(ag_stream)
        result = forward(prev_params, x, layer_id - 1)
        results.append((result, layer_id - 1))

        # Tell the allocator the default stream also used this block
        prev_params.record_stream(default)
        del prev_params

        prev_params = all_gather(ag_stream, layer_id)

    default.wait_stream(ag_stream)
    result = forward(prev_params, x, NUM_LAYERS)
    results.append((result, NUM_LAYERS))
    prev_params.record_stream(default)
    del prev_params

    torch.cuda.synchronize()
    return check_results(results)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_trials(label: str, fn, x: torch.Tensor, trials: int) -> None:
    corrupted = 0
    max_blocks = 0
    for t in range(trials):
        param_ptrs.clear()
        diff = fn(x)
        max_blocks = max(max_blocks, len(set(param_ptrs)))
        if diff > 1e-6:
            corrupted += 1
            if corrupted <= 3:
                print(f"  trial {t:2d}: CORRUPT  max_diff={diff:.1f}  (expected 0)")
    status = "PASS" if corrupted == 0 else "FAIL"
    print(f"  [{status}] {corrupted}/{trials} trials corrupted, "
          f"at most {max_blocks} param blocks per run ({NUM_LAYERS} layers)\n")


if __name__ == "__main__":
    TRIALS = 50

    torch.cuda.synchronize()
    x = torch.ones(BATCH, HIDDEN, device="cuda")
    _w = torch.empty(PARAM_SIZE, device="cuda"); del _w
    torch.cuda.synchronize()

    print(f"Config: HIDDEN={HIDDEN}, BATCH={BATCH}, NUM_LAYERS={NUM_LAYERS}")
    print(f"Expected per element: HIDDEN * layer_id\n")

    print("=== BUGGY: no lifetime management ===")
    run_trials("buggy", buggy, x, TRIALS)

    print("=== RECORD_STREAM: correct, allocator polls events ===")
    run_trials("record_stream", record_stream, x, TRIALS)

    print("=== STALL: wait_event (correct but serialized) ===")
    run_trials("stall", stall_producer, x, TRIALS)

    print("=== PIPELINED: keep ref + wait_event ===")
    run_trials("pipelined", keep_ref, x, TRIALS)
```

</details>

Output on one H100:

```text
Config: HIDDEN=4096, BATCH=4096, NUM_LAYERS=4
Expected per element: HIDDEN * layer_id

=== BUGGY: no lifetime management ===
  trial  0: CORRUPT  max_diff=10876.0  (expected 0)
  trial  1: CORRUPT  max_diff=10854.0  (expected 0)
  trial  2: CORRUPT  max_diff=10882.0  (expected 0)
  [FAIL] 50/50 trials corrupted, at most 1 param blocks per run (4 layers)

=== RECORD_STREAM: correct, allocator polls events ===
  [PASS] 0/50 trials corrupted, at most 4 param blocks per run (4 layers)

=== STALL: wait_event (correct but serialized) ===
  [PASS] 0/50 trials corrupted, at most 1 param blocks per run (4 layers)

=== PIPELINED: keep ref + wait_event ===
  [PASS] 0/50 trials corrupted, at most 2 param blocks per run (4 layers)
```

## References

- [`torch.Tensor.record_stream`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)
  and [CUDA semantics: CUDA streams](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams)
- [When does fragmentation occur in the CUDA caching allocator?](/devlogs/eager/2026-06-01-cuda-caching-allocator/)
- [FSDP & CUDACachingAllocator: an outsider newb perspective](https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486)
  (Jane Xu)
- [[RFC] Per-Parameter-Sharding FSDP (FSDP2)](https://github.com/pytorch/pytorch/issues/114299)
