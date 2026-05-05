---

# Dynamo Isolate Recompiles for torch.compile
authors: Xiao Fu(@fxdawnn), William Wen(williamwen42), Animesh Jain(anijain2305), Laith Sakka(laithsakka)
tags: [dynamo, torch.compile, caching, recompilation]

> **TL;DR** – We introduce `isolate_recompiles=True` for `torch.compile`, which gives each invocation its own isolated cache bucket — solving recompile limit collisions in factory patterns and dynamic shapes dispatch by refactoring Dynamo's cache from per code-object to per `torch.compile()` invocation.
> 

## Background / Motivation

Multiple `torch.compile(fn, …)` wrappers can share the same underlying **code object (`fn`)**. In Python, a code object is created once per `def` statement as opposed to once per function invocation. This means that in factory patterns, or when compiling the same function with different compile options, every `torch.compile(fn, …)` invocation targeting functions from the same `fn` produces cache entries that land in **a single linked list** attached to that code object. This conflation of logically separate compilation contexts is the root cause of cache lookup interference and unclear recompilation boundaries.

### How Cache Lookup Interference Manifests

Dynamo's recompilation mechanism works as follows: each code object has a cache of previously compiled entries, each guarded by a set of assumptions (e.g., on tensor shapes or object attributes). When Dynamo receives a frame, it walks **all** cache entries attached to the code object (including those from unrelated `torch.compile()` calls) looking for one whose guards pass. If two `torch.compile` calls target functions that share the same code object but use different compile options (e.g., `dynamic=True` vs. `dynamic=False`), their cache entries are **not separated**. This leads to two concrete problems:

**Dynamic shape semantic collision:** Different `torch.compile()` invocations may have different recompilation expectations. For example, static shapes specialize on exact dimensions while dynamic shapes generalize (with fewer recompilations). Dynamo does not differentiate cache entries by these semantics, so a static-shape entry may interfere with a dynamic-shape lookup on the same code object.

**Recompile limit collision:** The recompile limit (`recompile_limit`/`cache_size_limit`, default 8) is counted across **all** entries on the code object. In factory patterns, `torch.compile` instance `A`'s compilations consume another instance `B`'s budget. Once the shared limit is exhausted, Dynamo suppresses compilation for the **entire** code object — even for `torch.compile()` calls that have never compiled a single entry.

### Existing Workarounds

There is an existing workaround: using `f.__code__.replace()` via `types.FunctionType` to create a clone with a distinct code object, giving each `torch.compile()` call its own cache. However, this is unintuitive, fragile across Python versions, and requires users to understand the internal distinction between Python function objects and code objects. As one user noted in the PyTorch Compile Q&A group: *"From a purely user perspective, I don't think it's right to require this trickery on top of the factory — it's unintuitive, and I'd never come up with it myself."*

We want to provide a cleaner, first-class interface so that users can engage with `torch.compile` without worrying about code object sharing. This also lays the groundwork for future use cases such as cleaner user-defined dispatching behavior and per-compile recompile limits.

## Design / Approach

### API

```python
opt_fn = torch.compile(fn, isolate_recompiles=True)
```

Each `torch.compile(..., isolate_recompiles=True)` call gets its own cache bucket. Minimal overhead when not used to ensure backward compatibility. Non-isolated calls behave exactly as before.

### Factory Pattern

With `isolate_recompiles`, each factory instance gets its own bucket:

```python
@cache
def factory(key):
    @torch.compile(fullgraph=True, isolate_recompiles=True)
    def frontend(x, n):
        return core(x) + n
    return frontend

factory("foo")(torch.ones(3), 3)  # compiles in bucket 0
factory("bar")(torch.ones(4), 3)  # compiles in bucket 1
factory("baz")(torch.ones(5), 3)  # compiles in bucket 2
```

### Static vs. Dynamic on the Same Function

When compiling the same function with different dynamic shape semantics, cache entries no longer interfere:

```python
opt_static = torch.compile(f, dynamic=False, isolate_recompiles=True)
opt_dynamic = torch.compile(f, dynamic=True, isolate_recompiles=True)

# Static entries live in bucket 0, dynamic entries in bucket 1
# Each exhausts its recompile_limit independently
opt_static(torch.randn(3, 3))
opt_static(torch.randn(4, 4))     # recompile in bucket 0 only
opt_dynamic(torch.randn(5, 5))    # compiles with dynamic shapes in bucket 1
opt_dynamic(torch.randn(6, 6))    # no recompile needed — dynamic shapes generalize
```

### Per-Compile Recompile Limit

`isolate_recompiles` can be combined with a per-compile `recompile_limit` to fine-tune each wrapper's budget independently of the global config:

```python
opt_a = torch.compile(f, isolate_recompiles=True, recompile_limit=2)
opt_b = torch.compile(f, isolate_recompiles=True, recompile_limit=16)
# opt_a allows 2 recompilations before falling back to eager
# opt_b allows 16 — useful for dispatch-heavy patterns with many shapes
```

### Implementation: Per-Compile Cache Map

The core structural change replaces a single flat linked list with a **keyed map of lists**.

**Before — single flat list per code object:**

All `torch.compile()` calls targeting the same code object share one flat linked list of cache entries.

```
ExtraState (per code object)
    │
    └── cache_entry_list: std::list<CacheEntry>   ← one shared list
          │
          ├── CacheEntry { guards, code, backend_A }
          │     _owner_list ──► &cache_entry_list
          │     next() ──► CacheEntry below
          │
          ├── CacheEntry { guards, code, backend_B }
          │     _owner_list ──► &cache_entry_list
          │     next() ──► CacheEntry below
          │
          └── CacheEntry { guards, code, backend_A }
                _owner_list ──► &cache_entry_list
                next() ──► None
```

**After — per-compile cache map:**

The `cache_entry_map` keys are `isolate_recompiles_id` values:

- **Key `-1`**: the default (non-isolated) bucket. All `torch.compile()` calls without `isolate_recompiles=True` share this bucket, preserving existing behavior.
- **Key `>= 0`**: each `torch.compile(..., isolate_recompiles=True)` call is assigned a unique monotonic id. Its cache entries live exclusively in its own bucket.

```
torch.compile(f, isolate_recompiles=True)
         │
         ▼
┌─────────────────────────────────────┐
│  ConvertFrameAssert                 │
│    _isolate_recompiles_id = N       │  ◄── allocated once, shared with clones
│    _clone_with_backend ─────────────┼──── preserves same id
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  _TorchDynamoContext (eval_frame.py)│
│    sets global id before fn call    │  ◄── save/restore around fn(*args)
│    restores previous id after       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  C++ eval_frame                     │
│    id = get_current_isolate_        │
│         recompiles_id()             │
│    lookup(extra, ..., id, ...)      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  ExtraState (per code object)                               │
│                                                             │
│  cache_entry_map: unordered_map<int64_t, list<CacheEntry>>  │
│  ┌─────┬───────────────────────────────────────────┐        │
│  │ -1  │ CacheEntry ↔ CacheEntry ↔ CacheEntry     │  non-  │
│  │     │ (_isolate_recompiles_id = -1)             │  iso.  │
│  ├─────┼───────────────────────────────────────────┤        │
│  │  0  │ CacheEntry ↔ CacheEntry                   │  opt_a │
│  │     │ (_isolate_recompiles_id = 0)              │        │
│  ├─────┼───────────────────────────────────────────┤        │
│  │  1  │ CacheEntry                                │  opt_b │
│  │     │ (_isolate_recompiles_id = 1)              │        │
│  └─────┴───────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Each bucket is still a `std::list<CacheEntry>` (LRU-ordered), so within a bucket the cache behaves exactly as before (guards are checked, entries are moved to front on hit, etc.).

### Key Implementation Details

- **Lookup with fallback:** An isolated compile searches its own bucket first, then falls back **read-only** to the default bucket (`-1`). This preserves backward compatibility as prior non-isolated compilations remain visible. New entries are always written to the isolated bucket only.
- **Recompile limit fallback:** We ensure that the `torch.compile()`-enabled region falls back to eager if no cached compiled code is found or if the recompile limit is reached.
- **Id propagation:** The `isolate_recompiles_id` is set/restored around each `torch.compile` invocation. The C++ `eval_frame` reads it via a single global read, which means zero overhead when not used.

## Open Questions / Future Work

- Cleaner user-defined dispatching behavior built on top of per-compile isolation.
- Potential integration with more granular per-compile configuration options beyond `recompile_limit`.
- Exploration of whether the fallback read from the default bucket (`-1`) should be configurable or opt-out in advanced use cases.

## References

- [Initial PR: isolate_recompiles implementation](https://github.com/pytorch/pytorch/pull/178351)
## Acknowledgements

Thanks Edward Yang for reviewing and the Twitter post for naming finalization! Thanks Milad Mohammadi for reviewing!
