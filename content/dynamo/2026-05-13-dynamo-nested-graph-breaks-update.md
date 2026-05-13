# Nested Graph Breaks: May 2026 Update

## Summary

`torch._dynamo.config.nested_graph_breaks = True` has been **enabled on all Dynamo and Inductor unit tests** (~250 test files). A sweep of the OSS benchmark models with graph breaks shows **81/82 passing** with NGB (the single regression is a pre-existing unstable model), with **graph break reductions of up to 67%** and graph merging in models with complex nested call structures (GNNs, detection models). **Dynamo tracing time is neutral or improved** for most models, and models with significant graph merging see **up to 15% runtime speedup** (8% geomean). The remaining goal is to set `nested_graph_breaks` to `True` by default.

## Recap

The nested graph break problem in `torch.compile` refers to the Dynamo limitation of only being able to resume from a graph break in the top-level frame. A graph break O(N) levels deep results in O(N) duplicate graph breaks, O(N^2) traced frames, and O(N) compiled graphs. By enabling Dynamo to resume at an arbitrary frame depth, we prevent duplicate graph breaks, reduce tracing to O(N) frames, and capture O(1) compiled graphs — increasing optimization opportunities by keeping more ops in the same graph.

## Test enablement

NGB has been enabled as the default for **all Dynamo and Inductor tests** — 105 Dynamo test files and 141 Inductor test files. Only 8 individual tests opt out.

## Benchmark results

### Setup

We ran 82 models across torchbench, huggingface, and timm suites in both training and inference modes. Each model was run with and without NGB using fresh Inductor caches, with interleaved runs to control for machine variance.

### Accuracy: 81/82 models pass

Of the 82 models that ran, 76 pass with both base and NGB. The remaining 6 failures are not related to NGB correctness: 5 fail on both base and NGB (pre-existing issues), and 1 (`vision_maskrcnn` training) changes from `pass` to `fail_accuracy` due to an Inductor codegen issue — `aot_eager` produces correct results with NGB.

### Graph break reduction

NGB reduces graph breaks in models where breaks occur in nested function calls. The reduction is most pronounced in models with complex nested call structures (GNN models, detection models) and in training mode where optimizer/grad scaler scaffolding adds breaks. Below we report total graph break events, the number of distinct compiled graphs, and per-graph op counts.

When NGB merges graphs, per-graph op counts increase because ops from multiple small graphs are combined into fewer larger ones. Without NGB, a break inside a nested call causes the inner function to be compiled as a separate top-level function — its graphs are cached and reused across call sites. With NGB, the inner function is inlined into the caller's compilation, so its ops appear in the same graph as the surrounding caller ops. The total ops *executed* as compiled code is the same either way, but NGB produces fewer, larger graphs, giving Inductor more fusion and scheduling opportunities.

**Models where NGB merges small graphs into larger ones** (H100, inductor, amp):
| Model | graph breaks | compiled graphs | per-graph ops | what changed |
|-------|---:|---:|---|---|
| AllenaiLongformerBase | 6 → 2 (−67%) | 7 → 3 | `[2,16,2,889,5,2,1028]` → `[20,896,1028]` | Small surrounding graphs absorbed into the two large model graphs |
| demucs | 5 → 2 (−60%) | 6 → 3 | `[2,13,24,34,2,2]` → `[39,36,2]` | 4 small graphs collapsed into 2 |
| detectron2_fcos (inference) | 19 → 12 (−37%) | 22 → 15 | `569 total ops` → `634 total ops` | 7 small graphs eliminated; leading graph grew from 2 to 13 ops |
| DistillGPT2 | 7 → 5 (−29%) | 5 → 3 | `[2,176,3,2,398]` → `[178,5,398]` | Small graphs absorbed into neighbors |
| basic_gnn_edgecnn | 7 → 6 (−14%) | 8 → 7 | `[2,11,0,11,11,2,2,78]` → `[13,0,15,15,2,2,78]` | GNN layer fragments merged with outer call ops |
| basic_gnn_gcn | 9 → 9 | 10 → 10 | `[2,4,1,7,6,6,6,2,2,48]` → `[2,4,14,4,14,4,14,2,2,48]` | Same break count but per-layer ops grow (6→14) by absorbing caller-side ops |

**Models where NGB reduces break count without changing graph structure** (the eliminated breaks come from training loop scaffolding — optimizer, grad scaler — not the model itself):
| Model | graph breaks | compiled graphs | per-graph ops |
|-------|---:|---:|---|
| resnet50 | 7 → 5 (−29%) | 3 → 3 | `[177, 2, 823]` — unchanged |
| densenet121 | 6 → 5 (−17%) | 2 → 2 | `[433, 2]` — unchanged |
| BERT_pytorch | 6 → 5 (−17%) | 2 → 2 | `[350, 2]` — unchanged |

### Dynamo tracing time

To isolate NGB's effect on Dynamo tracing (without Inductor overhead), we ran the same models with `backend="eager"` on H100 (3 interleaved runs each):

| Model | base | NGB | Δ |
|-------|---:|---:|---:|
| basic_gnn_edgecnn | 2.34s | 1.95s | **−17%** |
| OPTForCausalLM | 3.28s | 3.00s | −9% |
| basic_gnn_gcn | 1.47s | 1.43s | −3% (neutral) |
| demucs | 4.46s | 4.35s | −3% (neutral) |
| BartForCausalLM | 4.58s | 4.73s | +3% (neutral) |
| AllenaiLongformerBase | 9.55s | 12.26s | +28% |

Tracing time is neutral for most models. basic_gnn_edgecnn sees a −17% improvement from eliminating re-tracing overhead. AllenaiLongformerBase regresses +28% because NGB traces deeper into complex nested code that still eventually breaks — Dynamo spends more time analyzing the nested function before hitting the break, plus the cost of creating nested resume functions.

These benchmark models have shallow nesting (depth 1), so the quadratic re-tracing benefit from the previous post's microbenchmark (15x speedup at depth 100) doesn't manifest here. End-to-end cold-start compilation time (with Inductor) is unchanged because Inductor dominates — e.g., densenet121 spends ~113s compiling a 433-op graph regardless of Dynamo tracing.

### Runtime speedup

We measured Inductor runtime speedup (over eager) for the 5 models with the most graph break reduction, using 3 interleaved cold-start runs per configuration on H100 with cudagraphs enabled:

| Model | base speedup | NGB speedup | Δ | graph breaks |
|-------|---:|---:|---:|---|
| basic_gnn_gcn | 1.05x | 1.20x | **+15%** | 9 → 9 |
| basic_gnn_edgecnn | 1.75x | 1.99x | **+14%** | 7 → 6 |
| detectron2_fcos (inference) | 1.71x | 1.87x | **+9%** | 19 → 12 |
| AllenaiLongformerBase | 2.64x | 2.73x | **+4%** | 6 → 2 |
| demucs | 1.07x | 1.06x | neutral | 5 → 2 |

Geomean NGB/base ratio across these models: **+8%**. The GNN models benefit most because NGB merges per-layer graph fragments with caller-side ops into larger graphs, allowing Inductor to fuse element-wise ops across what were previously separate compilation boundaries. Demucs is neutral because its graph boundaries fall at convolutions, which are fusion barriers — Inductor already optimally fuses element-wise ops around each convolution within the original graphs, so merging them creates no new fusion opportunities.

### CI benchmark comparison

From the HUD benchmark API comparing main vs the NGB branch on H100: passrate is identical across all suites and compiler configs. Geomean speedup is within ±1% (performance neutral).

## What's next

1. Enable `nested_graph_breaks` by default in OSS
2. Continue to harden NGB

## Technical challenges since the previous post

Getting NGB to pass the full test suite required a number of bug fixes. Here are the most interesting ones.

### 1. Upper frame context managers during nested breaks ([#171823](https://github.com/pytorch/pytorch/pull/171823))

When a nested graph break occurs inside a context manager that was entered in an upper frame, the resume function needs to ensure that context manager is still active. Previously, upper frame context managers were not being restored:

```python
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    # After the break, torch.no_grad() from the upper frame must still
    # be active here. Without the fix, grad was incorrectly re-enabled.
    assert not torch.is_grad_enabled()
    return x + 2

@torch.compile
def outer(x):
    with torch.no_grad():
        return inner(x)
```

Without NGB, the graph break in `inner` causes the entire `outer` to break, and `torch.no_grad()` is handled at the top level. With NGB, `inner`'s resume function runs inside `outer`'s compiled code and must explicitly restore the `no_grad` state.

### 2. Resume function globals for nested closures ([#176906](https://github.com/pytorch/pytorch/pull/176906))

Resume functions for closures defined in inlined frames were getting the wrong `f_globals`. When Dynamo creates a resume function via `MAKE_FUNCTION`, the function inherits the globals dict of the frame that executes the bytecode — the root (outermost) frame. But for a closure defined in a different module, the resume function needs that module's globals:

```python
# module_a.py
HELPER_CONSTANT = torch.tensor([100.0])

def closure_with_graph_break(x):
    captured = x + 1
    def inner():  # inner is a closure — it captures 'captured'
        torch._dynamo.graph_break()
        return captured + HELPER_CONSTANT  # needs module_a's globals
    return inner()

# module_b.py
@torch.compile
def outer(x):
    # Without the fix, the resume function for inner() would get
    # module_b's globals, causing NameError on HELPER_CONSTANT.
    return closure_with_graph_break(x) + 2
```

The fix ensures `MAKE_FUNCTION` bytecode for resume closures uses the correct globals dict by inserting an explicit `LOAD_CONST` for the target module's globals.

### 3. Decorator interaction with graph breaks ([#177090](https://github.com/pytorch/pytorch/pull/177090))

Challenge #1 handles context managers entered via `with` statements — Dynamo can see the `with` block boundaries in the bytecode and knows to restore the context manager in the resume function. This fix handles the case where a context manager is used as a decorator instead:

```python
def fn(x):
    x = x + 1
    torch._dynamo.graph_break()
    assert not torch.is_grad_enabled()  # no_grad must still be active
    return x + 2

@torch.compile
def gn(x):
    # no_grad() used as a decorator — there is no `with` block,
    # so Dynamo doesn't know this is an active context manager.
    x = torch.no_grad()(fn)(x)
    assert torch.is_grad_enabled()  # grad must be re-enabled after fn returns
    return x
```

When `torch.no_grad()` is used as a decorator, there is no `with` statement in the bytecode, so Dynamo does not realize it should track the decorator as an active context manager. On a nested graph break inside `fn`, Dynamo fails to apply the context manager to the resume function, causing `no_grad` to be silently dropped. The fix traces the decorator as a context manager instead, effectively treating:

```python
x = torch.no_grad()(fn)(x)
```

as equivalent to:

```python
with torch.no_grad():
    x = fn(x)
```

so the NGB resume chain correctly preserves and restores the context manager state.

### 4. Step graph break stack corruption ([#177408](https://github.com/pytorch/pytorch/pull/177408))

When Dynamo doesn't know how to resume after a graph break, it performs a "step graph break": compile the graph traced up to the last point Dynamo knows it can resume from, then run the rest of the inner function in eager. The parent function still needs to resume compiled execution normally. Step graph breaks have a different codegen path than regular graph breaks, and the NGB implementation was incorrectly reconstructing the parent frame's operand stack:

```python
def inner(x):
    x = x + 1
    torch._dynamo.step_unsupported()
    return x + 1

@torch.compile
def fn(x):
    x = x + 1
    # The tuple construction puts x on the operand stack before
    # inner() is called. The step_unsupported() inside inner()
    # would corrupt x's stack slot during resume.
    y = (x, inner(x))
    return x, y
```

This bug was difficult to reproduce because it required all three conditions simultaneously — a step graph break (not a regular one), inside a nested call, with a non-empty parent operand stack. The bug produced silently wrong values — in this case, `x` in the return value would be wrapped in a spurious list (`[tensor([2., 3.])]` instead of `tensor([2., 3.])`), with no indication that the root cause was a bytecode generation bug.

### 5. contextlib.contextmanager __init__ breaks ([#177195](https://github.com/pytorch/pytorch/pull/177195))

Functions decorated with both `@contextmanager` and `@torch._disable_dynamo` caused graph breaks during `_GeneratorContextManager.__init__`, which NGB tried to trace through. This pattern appears in DDP's distributed bucket synchronization code:

```python
@contextmanager
@torch._disable_dynamo(recursive=False)
def my_ctx():
    yield

@torch.compile
def fn(x):
    x = x + 1
    # Entering my_ctx() calls _GeneratorContextManager.__init__,
    # which calls the generator function (my_ctx). Since my_ctx is
    # @_disable_dynamo, this triggers a graph break during __init__.
    # NGB would try to trace through the break, hitting internal errors.
    with my_ctx():
        x = x + 2
    return x + 3
```

The fix graph breaks before the context manager is initialized, rather than attempting to perform a nested graph break inside the context manager init.
