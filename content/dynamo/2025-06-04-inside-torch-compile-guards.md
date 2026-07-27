---
title: "Inside torch.compile Guards: How They Work, What They Cost, and Ways to Optimize"
date: 2025-06-04
author: "Animesh Jain (@anijain2305)"
tags: [dynamo, guards, performance, torch.compile]
---

> **TL;DR** – `torch.compile` bakes its input assumptions into every compiled graph as *guards*. A failed guard forces a recompilation (compile-time cost), and every call re-checks the guard set before dispatching (run-time cost). This post builds a mental model of guards and shows how to reduce both: `mark_dynamic` and a few config flags to cut recompilations, and the `skip_guard_eval_unsafe` stance plus the `guard_filter_fn` helpers to trim per-call guard overhead.

> **Note:** Originally written in June 2025 and lightly updated to reflect the current API — the guard-filter helpers described below have since shipped and are generally available.

`torch.compile` is PyTorch's just-in-time (JIT) compiler. It springs into action the first time a function is called with real inputs, lowering your Python code to an optimized graph. In doing so, it **specializes** that graph on specific properties of the inputs—such as shape, dtype, or device. These assumptions, called **guards**, are baked into the compiled artifact; if any guard is later violated, PyTorch must fall back and re-compile.

This post aims to give `torch.compile` users a clear mental model of how guards operate, how they influence both compile-time and run-time performance, and the techniques you can use to mitigate guard-related overhead.

## Building the Guard Programming Model

Lets look at a simple example and build a mental model for `torch.compile` and guards.

```python
import torch

class CoefficientPair:
    def __init__(self):
        self.a = 2
        self.b = 5

pair = CoefficientPair()

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, x):
        return self.norm(x) + pair.a + pair.b

mod = Mod()
opt_mod = torch.compile(mod)

x = torch.rand(4, 8)
opt_mod(x)
```

If you're not already using [**tlparse**](https://github.com/pytorch/tlparse/tree/main), consider adding it to your toolkit—it's invaluable when working with `torch.compile`. I'll show the key snippets here, but to see the complete picture, run the examples yourself in [**tlparse**](https://github.com/pytorch/tlparse/tree/main).

In this example, `torch.compile` produces the following graph (exported from **tlparse**).

```python
class GraphModule(torch.nn.Module):
    def forward(self, L_self_modules_norm_parameters_weight_: "f32[8][1]cpu", L_self_modules_norm_parameters_bias_: "f32[8][1]cpu", L_x_: "f32[4, 8][8, 1]cpu"):
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_x_ = L_x_

         # File: /home/anijain/local/pytorch/examples/guards/t0.py:17 in forward, code: return self.norm(x) + pair.a + pair.b
        layer_norm: "f32[4, 8][8, 1]cpu" = torch.nn.functional.layer_norm(l_x_, (8,), l_self_modules_norm_parameters_weight_, l_self_modules_norm_parameters_bias_, 1e-05);  l_x_ = l_self_modules_norm_parameters_weight_ = l_self_modules_norm_parameters_bias_ = None
        add: "f32[4, 8][8, 1]cpu" = layer_norm + 2;  layer_norm = None
        add_1: "f32[4, 8][8, 1]cpu" = add + 5;  add = None
        return (add_1,)
```

Here's one way to visualize the mapping from the original function to its compiled counterpart.

![Mapping from the original function to its compiled graph](/devlogs/images/dynamo/2025-06-04-guards-1.png)

However, this picture is still incomplete. As a JIT compiler, `torch.compile` specializes each extracted graph on a specific set of **input assumptions**—its guards. The graph remains valid only as long as those assumptions hold in later calls. For this example, the guards are as follows (again exported from the tlparse report).

```text
TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:629 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:617 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=1)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[4, 8], stride=[8, 1])
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False
| +- GuardManager: source=G, accessed_by=GlobalsGuardAccessor
| | +- GuardManager: source=G['pair'], accessed_by=DictGetItemGuardAccessor('pair')
| | | +- TYPE_MATCH: ___check_type_id(G['pair'], 10457280)
| | | +- GuardManager: source=G['pair'].a, accessed_by=GetAttrGuardAccessor(a)
| | | | +- EQUALS_MATCH: G['pair'].a == 2
| | | +- GuardManager: source=G['pair'].b, accessed_by=GetAttrGuardAccessor(b)
| | | | +- EQUALS_MATCH: G['pair'].b == 5
| | +- GuardManager: source=G['__import_torch_dot_nn_dot_modules_dot_module'], accessed_by=DictGetItemGuardAccessor('__import_torch_dot_nn_dot_modules_dot_module')
| | | +- ID_MATCH: ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_module'], 140190179552576)
| | | +- GuardManager: source=G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, accessed_by=GetAttrGuardAccessor(_global_forward_hooks)
| | | | +- TYPE_MATCH: ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 8830944)
| | | +- GuardManager: source=G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, accessed_by=GetAttrGuardAccessor(_global_backward_hooks)
| | | | +- TYPE_MATCH: ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 8830944)
| | | +- GuardManager: source=G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, accessed_by=GetAttrGuardAccessor(_global_forward_pre_hooks)
| | | | +- TYPE_MATCH: ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 8830944)
| | | +- GuardManager: source=G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, accessed_by=GetAttrGuardAccessor(_global_backward_pre_hooks)
| | | | +- TYPE_MATCH: ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 8830944)
| | +- GuardManager: source=G['__import_torch_dot_nn_dot_modules_dot_normalization'], accessed_by=DictGetItemGuardAccessor('__import_torch_dot_nn_dot_modules_dot_normalization')
| | | +- ID_MATCH: ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_normalization'], 140187515218784)
| | | +- GuardManager: source=G['__import_torch_dot_nn_dot_modules_dot_normalization'].F, accessed_by=GetAttrGuardAccessor(F)
| | | | +- ID_MATCH: ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_normalization'].F, 140187518692352)
| | | | +- GuardManager: source=G['__import_torch_dot_nn_dot_modules_dot_normalization'].F.layer_norm, accessed_by=GetAttrGuardAccessor(layer_norm)
| | | | | +- ID_MATCH: ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_normalization'].F.layer_norm, 140187516149472)
| +- GuardManager: source=L['self'], accessed_by=FrameLocalsGuardAccessor(key='self', framelocals_idx=0)
| | +- TYPE_MATCH: ___check_type_id(L['self'], 10415744)
| | +- GuardManager: source=L['self'].__dict__, accessed_by=GetGenericDictGuardAccessor
| | | +- GuardManager: source=L['self']._modules, accessed_by=DictGetItemGuardAccessor('_modules')
| | | | +- TYPE_MATCH: ___check_type_id(L['self']._modules, 8837568)
| | | | +- GuardManager: source=L['self']._modules['norm'], accessed_by=DictGetItemGuardAccessor('norm')
| | | | | +- TYPE_MATCH: ___check_type_id(L['self']._modules['norm'], 89776288)
| | | | | +- GuardManager: source=L['self']._modules['norm'].__dict__, accessed_by=GetGenericDictGuardAccessor
| | | | | | +- DICT_CONTAINS: not ___dict_contains('forward', L['self']._modules['norm'].__dict__)
| | | | | | +- GuardManager: source=L['self']._modules['norm'].eps, accessed_by=DictGetItemGuardAccessor('eps')
| | | | | | | +- EQUALS_MATCH: L['self']._modules['norm'].eps == 1e-05
| | | | | | +- GuardManager: source=L['self']._modules['norm']._parameters, accessed_by=DictGetItemGuardAccessor('_parameters')
| | | | | | | +- TYPE_MATCH: ___check_type_id(L['self']._modules['norm']._parameters, 8837568)
| | | | | | | +- GuardManager: source=L['self']._modules['norm']._parameters['bias'], accessed_by=DictGetItemGuardAccessor('bias')
| | | | | | | | +- TENSOR_MATCH: check_tensor(L['self']._modules['norm']._parameters['bias'], Parameter, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[8], stride=[1])
| | | | | | | +- GuardManager: source=L['self']._modules['norm']._parameters['weight'], accessed_by=DictGetItemGuardAccessor('weight')
| | | | | | | | +- TENSOR_MATCH: check_tensor(L['self']._modules['norm']._parameters['weight'], Parameter, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[8], stride=[1])
| | | | | | +- GuardManager: source=L['self']._modules['norm'].normalized_shape, accessed_by=DictGetItemGuardAccessor('normalized_shape')
| | | | | | | +- EQUALS_MATCH: L['self']._modules['norm'].normalized_shape == (8,)
| | | | | | | +- TYPE_MATCH: ___check_type_id(L['self']._modules['norm'].normalized_shape, 8812224)
| | | | | | | +- LENGTH_CHECK: len(L['self']._modules['norm'].normalized_shape) == 1
| | | +- GuardManager: source=L['self']._parameters, accessed_by=DictGetItemGuardAccessor('_parameters')
| | | | +- TYPE_MATCH: ___check_type_id(L['self']._parameters, 8837568)

Guard latency = 27.39 us
```

That's a long list of guards—more detail than we need to unpack right now—so we'll revisit it later. The immediate takeaway is simple: if any guard (such as `TYPE_MATCH` or `TENSOR_MATCH`) fails on a future call, the graph can't be reused. Each compiled graph is inseparable from its guard set; together they form what we'll call a **compile unit**. Let's refine our mental model with this idea in mind.

![A compile unit: a compiled graph together with its guard set](/devlogs/images/dynamo/2025-06-04-guards-2.png)

Let's extend the mental model. On every subsequent invocation, the runtime evaluates the guards to decide whether the existing compile unit can be reused. If any guard fails, **recompilation** is triggered. For example, the first compile unit is specialized for an input tensor `x` with shape `(4, 8)`.

```text
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[4, 8], stride=[8, 1])
```

Now, invoke the compiled function with a tensor of a different shape.

```python
x = torch.rand(8, 8)
opt_mod(x)
```

This call triggers a recompilation—evident in the **tlparse** output or by enabling `TORCH_LOGS=recompiles`.

```text
DEBUG:torch._dynamo.guards.__recompiles:Recompiling function forward in /home/anijain/local/pytorch/examples/guards/t0.py:15
    triggered by the following guard failure(s):
    - 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 8
```

Refining the model: each user function maps to a list of **compile units** (organized in a linked list datastructure), every unit holding a graph specialized for a particular set of input assumptions. At runtime, the system walks the list, evaluates each unit's guards in order, and dispatches the graph from the first unit whose guards pass.

![A function's compile units stored as a linked list, evaluated in order](/devlogs/images/dynamo/2025-06-04-guards-3.png)

In pathological cases, every call might trigger a recompilation (we will see a few examples later). To prevent an endless loop of work, `torch.compile` enforces a configurable `recompile_limit` (default = 8, set via `torch._dynamo.config.recompile_limit`). Once that threshold is exceeded, it abandons the function and falls back to eager execution. Previously compiled graphs remain cached, so if a later call satisfies the guards for any existing compile unit, that graph is reused—but no further recompilation is attempted.

## Why so many guards?

Let's return to that daunting list of guards we saw earlier. It can feel unsettling that such a small program spawns so many checks, but this is intentional. `torch.compile` must remain **sound**: whenever a compiled graph is no longer valid, it must trigger recompilation. To guarantee correctness, the system guards every user-controllable specialization. While this may look like over-guarding at first, it's a deliberate choice—silent errors caused by reusing an invalid graph are far harder to diagnose than an extra recompilation.

Let's revisit the original example and update the `nn.Module` in place.

```python
mod.norm.eps = 1e-02
opt_mod(x)
```

Although this scenario is rare, a user can still make such in-place changes, making the previously compiled graph unusable for future invocations. You can see the extra recompilation using tlparse or `TORCH_LOGS=recompiles`.

### Impact of Guards

Guards shape `torch.compile` performance in two main ways:

1. **Compile-time cost** — If a guard fails, `torch.compile` recompiles the function, running its full pipeline for the new inputs. Each compilation already introduces noticeable latency, so multiple recompilations can visibly lengthen the overall compilation time.

2. **Run-time overhead** — Before a graph is launched, the runtime walks through every compile unit and evaluates each of its guards. A single unit can have hundreds/thousands of guards, and a function may have several compile units, so the cumulative check can become significant. While these CPU checks run, the GPU waits, which can matter for models with very short inference times.

Let's look at some examples and mitigation techniques for both of these.

## Compile-time Cost: How to deal with recompilations?

Recompilations are costly because each pass through `torch.compile` carries significant latency. The next sections highlight common triggers and the "escape hatches" you can use to avoid them.

**Pattern 1 – Input-tensor shape changes** — A typical culprit is the input tensor's shape varying between calls. You can sidestep this by marking the shape-changing dimensions as dynamic with `mark_dynamic`, letting `torch.compile` reuse the existing compile unit instead of recompiling. For deeper guidance, see the official [documentation](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html).

**Pattern 2 - Changing constant**: Consider following example

```python
import torch

@torch.compile
def fn(x, c):
    return x + c

for i in range(1, 10):
    fn(torch.ones(i), 0.5 + i)
```

Because the guard tracks the literal value of `c`, any change triggers a recompilation. Avoid this by refactoring the model to wrap the constant in a tensor instead.

```python
for i in range(1, 10):
    fn(torch.ones(i), torch.tensor(0.5 + i))
```

**Pattern 3 - nn.Module int attribute change**: By default, `torch.compile` assumes that the integer attributes on the `nn.Module` are static. Consider the following example

```python
class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = 0

    def forward(self, x):
        self.c += 1
        return x * self.c


mod = Mod()
opt_mod = torch.compile(mod, backend="eager")

x = torch.randn(4)

for _ in range(10):
    opt_mod(x)
```

This pattern can quickly hit the `recompile_limit`, because the guard binds to the concrete integer value of `self.c`. Enable the configuration flag below to treat that attribute symbolically, allowing the existing graph to be reused instead of recompiling.

```python
torch._dynamo.config.allow_unspec_int_on_nn_module = True
```

**Pattern 4 – Exceeding the** `recompile_limit` - When a function is recompiled more than the configured `torch._dynamo.config.recompile_limit` (8 by default), `torch.compile` abandons JIT and falls back to eager execution. Although repeated recompilations typically hurt both compile-time and run-time performance, you can raise this limit if your workload genuinely warrants the extra passes and the earlier workarounds don't apply.

## Runtime cost - How to reduce guard overhead?

Guards are evaluated on every invocation of a compiled function. At runtime, the system walks the linked list of compile units, checks each unit's guards—sometimes hundreds or even thousands—and dispatches the first graph whose guards all pass. Unsurprisingly, this guard checking adds overhead.

Let's quantify that cost. The PyTorch Profiler is the standard tool for investigating performance, and it already surfaces guard time under the event **"TorchDynamo Cache Lookup."** We'll profile a different example next: a model with a nested module hierarchy that mirrors the deeply nested structures you often encounter in real-world workloads.

```python
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

class NestedModule(nn.Module):
    """
    Recursively builds a nested module. A proxy for the deeply nested modules in
    the real models.
    """
    def __init__(
        self, depth: int, width: int, in_features: int = 10, out_features: int = 10
    ):
        super().__init__()
        self.depth = depth
        self.width = width

        self.linear_a = nn.Linear(in_features, out_features)
        self.linear_b = nn.Linear(in_features, out_features)

        sub_mods = []
        if depth > 0:
            for i in range(width):
                sub_mods.append(
                    NestedModule(depth - 1, width, in_features, out_features)
                )
        else:
            for i in range(width):
                sub_mods.append(nn.Linear(in_features, out_features))
        self.sub_mods = nn.Sequential(*sub_mods)

    def forward(self, x):
        x = self.linear_a(x)
        x = x + self.sub_mods(x)
        return x + self.linear_b(x)


mod = NestedModule(depth=4, width=3, in_features=2, out_features=2).cuda()
opt_mod = torch.compile(mod)


x = torch.randn(1, 2, device="cuda")
mod(x)

opt_mod(x)
for _ in range(10):
    opt_mod(x)

with profile(activities=[ProfilerActivity.CPU]) as prof:
    opt_mod(x)

prof.export_chrome_trace("trace.json")
```

Open the generated `trace.json` in Chrome's `chrome://tracing` viewer; the timeline will clearly highlight the guard-overhead events.

![Profiler trace showing TorchDynamo guard-evaluation (cache lookup) overhead of about 0.9 ms](/devlogs/images/dynamo/2025-06-04-guards-4.png)

In this example, guard evaluation adds roughly 0.9 ms. You can measure the same metric for your own model and compare it to the full execution time of the compiled function to see whether the overhead is meaningful.

If profiling reveals that guard checks are eating into your budget, `torch.compile` offers two **escape hatches** to trim them. These options are **unsafe** and aren't enabled by default because they remove selected guards and could let an outdated graph slip through unnoticed. However, when you're confident that certain input properties will never change in your workload (which is true in many cases), disabling those guards can safely bring the overhead down.

* `skip_guard_eval_unsafe` **stance** — After a brief warm-up, this stance can be turned on to drop unnecessary guards.

* **Guard-filter API** — Lets you selectively disable, at compile time, guards that you consider safe to drop for your workload.

`skip_guard_eval_unsafe` **stance** - After an initial warm-up—say the first hundred inferences—your `torch.compile` model may stabilize: no further recompilations occur, and you're sure no one will mutate the module in place (for example, by changing `mod.norm.eps`). At that point, only a tiny subset of guards is needed to tell the existing compile units apart. This is the idea behind `skip_guard_eval_unsafe` [stance](https://github.com/pytorch/pytorch/blob/e5afbe31245287a92fe328c404b3557e5c5eca73/torch/compiler/__init__.py#L297). It finds that minimal set of differentiating guards, and prunes everything else, significantly reducing the guard overhead. Here's how you would enable it in the example above:

```python
# Warm up phase - call the model with representative inputs
for _ in range(100):
    opt_mod(x)

# Instruct the torch.compile runtime to run minimal set of guards
with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        opt_mod(x)

prof.export_chrome_trace("trace.json")
```

The profiler trace shows the guard overhead dropping from 0.9 ms to nearly zero. Because this model has only one compile unit, every guard except the one on the input tensor `x` was skipped.

![Profiler trace after skip_guard_eval_unsafe: guard overhead drops to nearly zero](/devlogs/images/dynamo/2025-06-04-guards-5.png)

Why is it "unsafe"? This mode assumes a warm-up phase during which the compiler encounters every representative input and performs any necessary (re)compilations. Once most guards are stripped away, a later change to the model or its inputs—one that would ordinarily trigger recompilation—can slip by unnoticed. The runtime would continue executing a stale graph and could silently yield incorrect results. Use this stance only when you are certain that neither the model nor its input characteristics will change after warm-up.

**Guard Filter API**: In some production settings you control the compilation stage but not the run-time environment. The `skip_guard_eval_unsafe` mode isn't ideal there, because it demands a warm-up phase followed by a run-time flag flip. The Guard-Filter API solves this: at compile time you drop specific guards explicitly, baking the reduced guard set into the compiled artifact and eliminating the need for run-time intervention.

With the Guard-Filter API you register a hook with the guard system. During tracing, that hook sees the complete list of generated guards and returns a Boolean mask—`True` to keep a guard, `False` to drop it. Several ready-made filters ship with `torch.compiler`, so you don't have to write hooks from scratch. The two we focus on below are `skip_guard_on_inbuilt_nn_modules_unsafe` and `skip_guard_on_all_nn_modules_unsafe`; the full set today also includes `keep_portable_guards_unsafe`, `keep_tensor_guards_unsafe`, `skip_guard_on_globals_unsafe`, and `skip_all_guards_unsafe`.

```python
# Remove any in-memory compiler artifacts to start afresh
torch.compiler.reset()
opt_mod = torch.compile(
    mod,
    options={"guard_filter_fn": torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe},
)
x = torch.randn(1, 2, device="cuda")
for _ in range(100):
    opt_mod(x)

with profile(activities=[ProfilerActivity.CPU]) as prof:
    opt_mod(x)

prof.export_chrome_trace("trace.json")
```

Attach the filter through the `options` keyword when invoking `torch.compile`. In most workflows, built-in modules—such as `nn.Linear` or `nn.LayerNorm`—stay unchanged after construction (in-place tweaks like `mod.layer_norm.eps = …` are uncommon). When that's the case, you can safely drop their guards with `skip_guard_on_inbuilt_nn_modules_unsafe`. In our example, this reduced guard overhead from roughly 900 µs to about 370 µs, as shown below.

![Profiler trace after skip_guard_on_inbuilt_nn_modules_unsafe: guard overhead reduced to about 370 us](/devlogs/images/dynamo/2025-06-04-guards-6.png)

```python
# Remove any in-memory compiler artifacts to start afresh
torch.compiler.reset()
opt_mod = torch.compile(
    mod,
    options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe},
)
x = torch.randn(1, 2, device="cuda")
for _ in range(100):
    opt_mod(x)

with profile(activities=[ProfilerActivity.CPU]) as prof:
    opt_mod(x)

prof.export_chrome_trace("trace.json")
```

You can take this a step further by skipping guards on **all** `nn.Module` instances—built-in *and* custom. If you're certain that none of their attributes will be mutated after compilation, dropping these guards is safe. In our example, doing so eliminates nearly every guard and drives the guard overhead down to effectively zero.

![Profiler trace after skip_guard_on_all_nn_modules_unsafe: guard overhead near zero](/devlogs/images/dynamo/2025-06-04-guards-7.png)

## Appendix: Under-the-Hood Optimizations for Fast Guard Evaluation

> **Note:** The rest of this appendix is aimed at readers interested in the internal engineering; it isn't required to understand the user-facing aspects of `torch.compile`.

### Optimizations inside a compile unit

Guards are organized in a tree strucuture. Each node executes a handful of quick checks—such as `TENSOR_MATCH` or `TYPE_MATCH`—before traversing deeper into the structure. Let's begin with the global-object pair as an example (copied from the original example).

```text
| +- GuardManager: source=G, accessed_by=GlobalsGuardAccessor
| | +- GuardManager: source=G['pair'], accessed_by=DictGetItemGuardAccessor('pair')
| | | +- TYPE_MATCH: ___check_type_id(G['pair'], 10457280)
| | | +- GuardManager: source=G['pair'].a, accessed_by=GetAttrGuardAccessor(a)
| | | | +- EQUALS_MATCH: G['pair'].a == 2
| | | +- GuardManager: source=G['pair'].b, accessed_by=GetAttrGuardAccessor(b)
| | | | +- EQUALS_MATCH: G['pair'].b == 5
```

At each node we first verify the pair's *type*, then drill down to its attributes (`a`, `b`, …) and check their constant values. In other words, every node bundles a small set of guards with the accessors needed to reach the next level.

Guard-tree evaluation short-circuits: if any guard at a node fails, its entire subtree is skipped—because every guard must pass for that compile unit to be valid.

Two engineering principles drive the design:

1. **Run guards fast** — All guard logic was moved from Python into C++, using low-level CPython APIs. Eliminating the interpreter's overhead shaved a substantial slice off guard-evaluation time.

2. **Fail fast on recompilation** — When multiple compile units exist, we want to locate the valid one quickly. Each time a guard fails, we promote that guard toward the front of its node. On the next call, the most failure-prone checks run first, producing an early exit if they fail again.

### Optimizations across compile units

![Compile units in a linked list, reordered by a most-recently-used policy](/devlogs/images/dynamo/2025-06-04-guards-8.png)

Compile units are threaded together in a linked list. To cut guard-checking overhead, we want the unit most likely to pass its guards at the front of that list. We achieve this with a simple **most-recently-used (MRU)** policy: whenever a unit's graph is dispatched, we move its node to the head of the list. Empirically, this keeps the "winning" unit near the front and minimizes unnecessary guard evaluations.

Thanks for reading—here's to faster, guard-optimized `torch.compile`. Happy compiling!
