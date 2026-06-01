---
title: "When does fragmentation occur in the CUDA caching allocator?"
author: Edward Yang (@ezyang)
date: 2026-06-01
tags: [eager, cuda, memory]
---

> **Disclosure.** This post was drafted by Claude (Anthropic's coding
> assistant) with editing from ezyang.

In an ideal world, users of CUDA memory in PyTorch programs should be able to
abstract the allocator behavior as: there is a fixed amount of GPU memory,
whenever you allocate this available memory goes down, and when you free the
available memory goes back up.

Unfortunately, the internal implementation of the CUDA caching allocator means
that certain allocation patterns can give rise to fragmentation, where
even though there is "technically" enough free space to store a requested
allocation, the CUDA caching allocator is unable to actually serve the request.

There are many modern use cases where users wish to use as much memory that
their GPUs provide as possible, while needing to ensure we do not OOM.  Users
are often penny-inching allocations in this situation, and find it very
surprising when PyTorch reserves more memory than they expect under the
abstract model of the allocator.

This is especially common in LLM serving, where every megabyte of GPU memory
that isn't nailed down by model weights or CUDA graph buffers can be used for
KV cache.  Modern disaggregated serving involves CUDA graphing distinct graphs
for each batch size.  It's important for these graphs to share the same memory
pool.  But sharing a pool means the allocator's internal bookkeeping needs to
be correct before each recording. And the way the allocator manages
memory--splitting and merging blocks--can go wrong in ways that depend on
allocation order.

In this post, we'll walk through some small laboratory examples where this
fragmentation happens, and then demonstrate *why* expandable segments fixes
these examples.  It's important to have a mental model for what exactly we
mean by "fragmentation", because some fragmentation can be solved with
expandable segments (especially those related to recording CUDA graphs), while
others cannot.

## Segments, blocks, and splitting

The caching allocator organizes GPU memory in two levels. **Segments**
are contiguous regions obtained from CUDA (`cudaMalloc` or virtual memory
mapping). **Blocks** are sub-regions within a segment that track
individual allocations.

When a request comes in, the allocator finds a free block that's large
enough. If the block is bigger than needed, it **splits** the block: the
front portion serves the allocation, the back portion becomes a new free
block. When a block is freed, the allocator tries to **merge** it with
its immediate neighbors--but only if the neighbor is also free. Two free
blocks separated by an allocated block cannot merge.

```python
import gc, torch

MiB = 1024 * 1024

def alloc(n, mib, pool, dev):
    with torch.cuda.use_mem_pool(pool, dev):
        return [
            torch.empty(int(mib * MiB), dtype=torch.uint8, device=dev)
            for _ in range(n)
        ]

def free(ts):
    ts.clear()

def layout(pool):
    for s in torch.cuda.memory_snapshot(pool.id):
        blocks = " | ".join(f"{b['size']//MiB}M {b['state']}" for b in s["blocks"])
        print(f"  seg {s['total_size']//MiB}M: [{blocks}]")

pool = torch.cuda.MemPool()
dev = torch.device("cuda:0")

t = alloc(1, 32, pool, dev)
layout(pool)  # one 32M block

free(t)

ts = alloc(2, 16, pool, dev)
layout(pool)  # 32M segment split into two 16M blocks

del ts[0]
layout(pool)  # first block inactive, second still active; can't merge

free(ts)
layout(pool)  # both free and adjacent; merged back to 32M
```

How segments are obtained depends on whether expandable segments are
enabled. The behavior is quite different in each case.

## Without expandable segments

Run scripts in this section with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.

Without expandable segments, each `cudaMalloc` call creates a separate
segment. For allocations between 1 MiB and 10 MiB, the allocator
requests a 20 MiB segment regardless of the actual size. For allocations
>= 10 MiB, the segment is rounded up to the nearest 2 MiB.

The key constraint: **blocks in different segments can never merge**.
Each `cudaMalloc` is an independent allocation from CUDA's perspective.
A free 16 MiB block in one segment cannot combine with a free 16 MiB
block in another segment to serve a 32 MiB request.

This is where allocation order matters. Let's walk through two
scenarios step by step.

**Small then large (bad order):**

```python
import gc, torch

MiB = 1024 * 1024

def alloc(n, mib, pool, dev):
    with torch.cuda.use_mem_pool(pool, dev):
        return [
            torch.empty(int(mib * MiB), dtype=torch.uint8, device=dev)
            for _ in range(n)
        ]

def free(ts):
    ts.clear()

def reserved(pool):
    return sum(s["total_size"] for s in torch.cuda.memory_snapshot(pool.id))

def layout(pool):
    for s in torch.cuda.memory_snapshot(pool.id):
        blocks = " | ".join(f"{b['size']//MiB}M {b['state']}" for b in s["blocks"])
        print(f"  seg {s['total_size']//MiB}M: [{blocks}]")

dev = torch.device("cuda:0")
pool = torch.cuda.MemPool()

small = alloc(8, 16, pool, dev)
print("after 8x16M:", reserved(pool) // MiB, "MiB")
layout(pool)
free(small)
print("after free:", reserved(pool) // MiB, "MiB")
layout(pool)
large = alloc(4, 32, pool, dev)
print("after 4x32M:", reserved(pool) // MiB, "MiB")
layout(pool)
free(large)
```

Step by step:

1. **Allocate 8x16 MiB.** Each 16 MiB request triggers a separate
   `cudaMalloc`. Since 16 MiB >= 10 MiB, each segment is rounded up to
   16 MiB (nearest 2 MiB multiple). Result: eight separate 16 MiB
   segments, each containing one allocated block. 128 MiB reserved.

2. **Free all.** Each segment now has one 16 MiB free block. But the
   segments are separate `cudaMalloc` allocations--they can't merge with
   each other. The pool still holds 128 MiB of reserved memory across
   eight independent segments.

3. **Allocate 4x32 MiB.** The allocator looks for a free block >= 32 MiB.
   Every existing free block is only 16 MiB, and blocks can't span
   segments. None of the existing segments can serve the request. The
   allocator calls `cudaMalloc` four more times for 32 MiB each. Result:
   256 MiB reserved--eight stale 16 MiB segments plus four new 32 MiB
   segments.

**Large then small (good order):**

```python
import gc, torch

MiB = 1024 * 1024

def alloc(n, mib, pool, dev):
    with torch.cuda.use_mem_pool(pool, dev):
        return [
            torch.empty(int(mib * MiB), dtype=torch.uint8, device=dev)
            for _ in range(n)
        ]

def free(ts):
    ts.clear()

def reserved(pool):
    return sum(s["total_size"] for s in torch.cuda.memory_snapshot(pool.id))

def layout(pool):
    for s in torch.cuda.memory_snapshot(pool.id):
        blocks = " | ".join(f"{b['size']//MiB}M {b['state']}" for b in s["blocks"])
        print(f"  seg {s['total_size']//MiB}M: [{blocks}]")

dev = torch.device("cuda:0")
pool = torch.cuda.MemPool()

large = alloc(4, 32, pool, dev)
print("after 4x32M:", reserved(pool) // MiB, "MiB")
layout(pool)
free(large)
print("after free:", reserved(pool) // MiB, "MiB")
layout(pool)
small = alloc(8, 16, pool, dev)
print("after 8x16M:", reserved(pool) // MiB, "MiB")
layout(pool)
free(small)
```

Step by step:

1. **Allocate 4x32 MiB.** Four `cudaMalloc` calls, four 32 MiB
   segments. 128 MiB reserved.

2. **Free all.** Each segment has one 32 MiB free block. 128 MiB still
   reserved.

3. **Allocate 8x16 MiB.** The first 16 MiB request finds a 32 MiB free
   block. The allocator splits it: 16 MiB allocated, 16 MiB free
   remainder. The second 16 MiB request takes that remainder. Two
   allocations served from one segment, no new `cudaMalloc`. This
   repeats for each of the four segments. Result: still 128 MiB
   reserved, each segment now split into two 16 MiB blocks.

Same total work, half the memory. The classic workaround is to always
record in decreasing batch-size order: large allocations establish the
segments, smaller ones split within them. It works, but it's a leaky
abstraction.

## With expandable segments

Run scripts in this section with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### What cuMemMap gives us

Without expandable segments, the allocator calls `cudaMalloc` for each
segment. Each `cudaMalloc` returns an independent allocation that can
never be merged with another. This is the root cause of the
fragmentation above.

CUDA also has a separate virtual memory management APIs, which separates three
concerns:

- **`cuMemAddressReserve`**: reserves a contiguous range of *virtual*
  address space. This is cheap--no physical memory is committed. The
  allocator reserves enough to map essentially all GPU physical memory
  (1 1/8x of `totalGlobalMem`).
- **`cuMemCreate`**: allocates a chunk of *physical* memory (a
  "handle"). This is the expensive operation that actually consumes GPU
  memory.
- **`cuMemMap` + `cuMemSetAccess`**: maps a physical handle into the
  reserved virtual range, making it accessible.

The allocator creates one `ExpandableSegment` per (pool, stream) pair.
Each expandable segment owns one huge virtual reservation but starts
with zero physical memory mapped. As allocations arrive, physical pages
are mapped into the segment on demand and the segment grows. Because
everything is in one contiguous virtual address range, blocks within
the segment can always merge with their neighbors--the cross-segment
barrier from `cudaMalloc` doesn't exist.

### Physical page granularity

Physical memory is mapped in fixed-size pages: **2 MiB** for
`small_blocks`, **20 MiB** for `large_blocks` (configurable via
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments_page_size:<bytes>`). These
are the `segment_size` passed to `ExpandableSegment`. When the allocator
needs more memory, it calls `cuMemCreate` for one page and maps it at
the end of the segment.

This means there's rounding overhead. If you request 16 MiB from the
large pool (20 MiB pages), the allocator maps one 20 MiB page, serves
16 MiB, and the remaining 4 MiB becomes a free block. The next
allocation can use that 4 MiB remainder, or if it's larger, the
allocator maps another 20 MiB page and merges the free space.

Let's trace through 8x16 MiB allocations and watch the page mapping
at each step:

```python
import torch

MiB = 1024 * 1024

def layout(pool):
    for s in torch.cuda.memory_snapshot(pool.id):
        blocks = " | ".join(f"{b['size']//MiB}M {b['state']}" for b in s["blocks"])
        print(f"  seg {s['total_size']//MiB}M: [{blocks}]")

def reserved(pool):
    return sum(s["total_size"] for s in torch.cuda.memory_snapshot(pool.id))

pool = torch.cuda.MemPool()
dev = torch.device("cuda:0")
ts = []
for i in range(8):
    with torch.cuda.use_mem_pool(pool, dev):
        ts.append(torch.empty(16 * MiB, dtype=torch.uint8, device=dev))
    print(f"after alloc {i+1}:", reserved(pool) // MiB, "MiB mapped")
    layout(pool)
```

Step by step:

1. **First 16 MiB.** No physical memory yet. Map one 20 MiB page.
   Split: 16 MiB allocated, 4 MiB free. (20 MiB mapped)

2. **Second 16 MiB.** The 4 MiB free block is too small. Map another
   20 MiB page; it's adjacent in virtual space, so the allocator merges
   the 4 MiB free + 20 MiB newly mapped = 24 MiB free. Split off
   16 MiB, leaving 8 MiB free. (40 MiB mapped)

3. **Third 16 MiB.** 8 MiB free isn't enough. Map another 20 MiB page,
   merge to 28 MiB free, split off 16 MiB, leaving 12 MiB free.
   (60 MiB mapped)

4. **Fourth 16 MiB.** 12 MiB free isn't enough. Map another 20 MiB
   page, merge to 32 MiB free, split off 16 MiB, leaving 16 MiB free.
   (80 MiB mapped)

5. **Fifth 16 MiB.** The remainder from the previous step is exactly
   16 MiB--it fits without mapping a new page. (Still 80 MiB mapped)

6. **Sixth through eighth.** The cycle repeats: the remainder is now
   0 MiB, so a new page is needed, producing a 4 MiB remainder that
   grows by 20 MiB each step until it's consumed.

After all eight allocations, the segment has mapped 7 large pages:
140 MiB of physical memory for 128 MiB of allocations. The 12 MiB of
overhead is free space that can serve future allocations--not wasted.

### Why allocation order doesn't matter

Now free all eight tensors. Because every block lives in the same
segment, adjacent free blocks merge. The result is one contiguous free
block covering all the mapped physical memory. This is the key
difference from `cudaMalloc` segments: there are no segment boundaries
preventing merging.

From this single merged free block, 4x32 MiB allocations can be carved
out without mapping any new physical memory:

```python
import gc, torch

MiB = 1024 * 1024

def alloc(n, mib, pool, dev):
    with torch.cuda.use_mem_pool(pool, dev):
        return [
            torch.empty(int(mib * MiB), dtype=torch.uint8, device=dev)
            for _ in range(n)
        ]

def free(ts):
    ts.clear()

def reserved(pool):
    return sum(s["total_size"] for s in torch.cuda.memory_snapshot(pool.id))

def layout(pool):
    for s in torch.cuda.memory_snapshot(pool.id):
        blocks = " | ".join(f"{b['size']//MiB}M {b['state']}" for b in s["blocks"])
        print(f"  seg {s['total_size']//MiB}M: [{blocks}]")

dev = torch.device("cuda:0")
pool = torch.cuda.MemPool()

# Small-then-large: the same order that fragmented without expandable segments.
small = alloc(8, 16, pool, dev)
print("after 8x16M:", reserved(pool) // MiB, "MiB reserved")
layout(pool)  # one segment, blocks interleaved with free remainders

free(small)
print("after free:"); layout(pool)  # one merged free block

large = alloc(4, 32, pool, dev)
print("after 4x32M:", reserved(pool) // MiB, "MiB reserved")
layout(pool)  # no new pages mapped
free(large)
```

Allocation order doesn't matter because the intermediate state--how
blocks are split--is irrelevant once everything is freed. All that
matters is that enough physical memory is mapped in the segment, and
it's all contiguous in virtual space.

### Expandable segments don't eliminate all fragmentation

The above only holds when **everything is freed** before the next round
of allocations. If some blocks are still alive, they pin the splits in
place: free blocks on either side of a live block can't merge across it.
The segment has enough total free memory, but it's chopped into
non-contiguous pieces.

```python
import gc, torch

MiB = 1024 * 1024

def alloc(n, mib, pool, dev):
    with torch.cuda.use_mem_pool(pool, dev):
        return [
            torch.empty(int(mib * MiB), dtype=torch.uint8, device=dev)
            for _ in range(n)
        ]

def free(ts):
    ts.clear()

def reserved(pool):
    return sum(s["total_size"] for s in torch.cuda.memory_snapshot(pool.id))

def layout(pool):
    for s in torch.cuda.memory_snapshot(pool.id):
        blocks = " | ".join(f"{b['size']//MiB}M {b['state']}" for b in s["blocks"])
        print(f"  seg {s['total_size']//MiB}M: [{blocks}]")

dev = torch.device("cuda:0")
pool = torch.cuda.MemPool()

ts = alloc(4, 16, pool, dev)
print("four 16M:"); layout(pool)

# Free first and third, keep second and fourth alive.
t1, t3 = ts[1], ts[3]
ts[0] = None
ts[2] = None
print("free #0,#2:"); layout(pool)

# There is plenty of total free memory, but the largest existing free
# block is only 16M. A 32M allocation has to grow the segment.
big = alloc(1, 32, pool, dev)
print("32M alloc: reserved grew to", reserved(pool) // MiB, "MiB")
free(big)

# Now free everything; all blocks merge into one.
ts.clear()
del t1, t3
print("free all:"); layout(pool)

# 32M fits without growing.
big = alloc(1, 32, pool, dev)
print("32M after full free:", reserved(pool) // MiB, "MiB")
free(big)
```

For CUDA graph pools this is straightforward: graphs that share a pool
don't run concurrently, so everything should be freed between
recordings. As long as that holds, the "free everything -> full merge"
path applies and allocation order is irrelevant. But if your use case
has long-lived allocations interleaved with short-lived ones in the same
pool, expandable segments won't save you from fragmentation within the
segment.

For the typical CUDA graph pool use case--graphs that don't run
concurrently--everything should be freed between recordings. As long as
that holds, expandable segments eliminate fragmentation entirely.

### The 1 MiB loophole

The allocator routes allocations <= 1 MiB to a pool called
`small_blocks` and allocations > 1 MiB to `large_blocks`. These are
entirely separate: separate segments, separate free lists. A block in
`small_blocks` can never serve a request from `large_blocks`, and vice
versa. Even with expandable segments, each pool gets its own segment.

This means making a tensor *smaller* can *increase* total pool memory
if it crosses the 1 MiB boundary:

```python
import gc, torch

MiB = 1024 * 1024

def alloc(n, mib, pool, dev):
    with torch.cuda.use_mem_pool(pool, dev):
        return [
            torch.empty(int(mib * MiB), dtype=torch.uint8, device=dev)
            for _ in range(n)
        ]

def free(ts):
    ts.clear()

def reserved(pool):
    return sum(s["total_size"] for s in torch.cuda.memory_snapshot(pool.id))

dev = torch.device("cuda:0")

# --- Crossing the boundary breaks sharing ---
pool = torch.cuda.MemPool()
a = alloc(1, 2, pool, dev)       # 2M -> large_blocks
free(a)
b = alloc(1, 0.5, pool, dev)     # 0.5M -> small_blocks; can't reuse large segment
print("2M then 0.5M:", reserved(pool) // MiB, "MiB")  # 22 MiB with 20M/2M pages
free(b); del pool; torch.cuda.empty_cache()

# --- Staying above 1 MiB: sharing works ---
pool = torch.cuda.MemPool()
a = alloc(1, 2, pool, dev)       # 2M -> large_blocks
free(a)
b = alloc(1, 2, pool, dev)       # 2M -> large_blocks; reuses segment
print("2M then 2M:", reserved(pool) // MiB, "MiB")    # 20 MiB
free(b)
```

If you're recording multiple CUDA graphs into a shared pool, be aware
that crossing the 1 MiB boundary causes allocations to land in different
pools and breaks sharing.
