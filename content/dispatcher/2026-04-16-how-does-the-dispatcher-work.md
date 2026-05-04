---
title: "How Does the Dispatcher Work?"
author: Aaron Orenstein (@aorenste)
date: 2026-04-16
tags: [dispatcher, dispatch_keys, backends, autocast, functionalization, torch_dispatch]
---

I wanted to write about how PT2 does autograd, but that requires understanding eager autograd, which requires understanding the dispatcher. So let's start there.

## Let's Build Ourselves A Dispatcher

Let's pretend we're building Torch. Let's start from first principles with the problems we encounter and how to solve them.

**Problem 1**: We want to be able to call operators for each backend.

**Solution**: Polymorphism! We just define a class where we have every operator defined as a virtual method. Backends just implement every operator.

```python
class Torch:
    def mm(self, a: Tensor, b: Tensor) -> Tensor: ...
    def einsum(self, equation: str, *operands: Tensor) -> Tensor: ...
    ...
```

Now I just need to implement Torch for each "real" backend (CPU, Cuda, TPU, etc).

**Problem 2**: We want to be able to have a "core" set of operators that backends MUST implement with default implementations for the others that decompose into the core operators.

**Solution**: Easy - just define the core set as abstract and the rest as implementations on the base class.

```python
class Torch:
    def mm(self, a: Tensor, b: Tensor) -> Tensor: ...
    def einsum(self, equation: str, *operands: Tensor) -> Tensor: ...

    def bmm(self, a: Tensor, b: Tensor) -> Tensor:
        out = Tensor(...)
        for i in range(a.shape()[0]):
            out[i] = self.mm(a[i], b[i])
        return out
```

**Problem 3**: We also want to be able to implement NEW operators without having to rebuild the system.

**Solution**: Instead of direct virtual methods on your class we add an indirection so when you want to run an operator you also provide the operator_name.

```python
class Torch:
    def _mm(self, a: Tensor, b: Tensor) -> Tensor: ...
    def _einsum(self, equation: str, *operands: Tensor) -> Tensor: ...

    LOOKUP = {
        "mm": _mm,
        "einsum": _einsum,
        ...
    }

    def run_op(self, operator_name, *args, **kwargs):
        return self.LOOKUP[operator_name](*args, **kwargs)
```

**Problem 4**: I want to be able to call multiple backends without having to completely swap things out!

**Solution**: Ok - so now we need to look at the args and determine which backend to call based on the device.

```python
def _lookup_backend(*args) -> Torch:
    t = args[0]
    if t.device == "cuda":
        return CudaTorch
    elif t.device == "cpu":
        return CpuTorch
    ...

class Torch:
    def call_op(self, operator_name, *args, **kwargs):
        return self.LOOKUP[operator_name](*args, **kwargs)

    @staticmethod
    def run_op(operator_name, *args, **kwargs):
        backend = _lookup_backend(*args)
        return backend.call_op(operator_name, *args, **kwargs)
```

**Problem 5:** I want to be able to dynamically add new backends.

**Solution**: We make our backends a lookup too!

```python
BACKENDS = {
    "cuda": CudaTorch,
    "cpu": CpuTorch,
    ...
}

def _lookup_backend(*args) -> Torch:
    t = args[0]
    return BACKENDS[t.device]
```

**Problem 6**: But if a backend doesn't handle an operator I still want to run it using a more generic backend (like CPU)

**Solution**: Okay - instead of a dict we turn our BACKENDS into a list and run the first one that applies.

```python
BACKENDS = [CudaTorch, CpuTorch]

def _lookup_backend(*args) -> Torch:
    for backend in BACKENDS:
        if backend._supports_args(*args):
            return backend
    raise NotImplementedError("no backend supports this operator")
```

**Problem 7**: But my backend may not know if it supports an operator and I can't tell until I'm actually trying to run it.

**Solution**: We merge _lookup_backend and run_op and we allow backends to dynamically say that they don't support individual operators.

```python
class Torch:
    @staticmethod
    def run_op(operator_name, *args, **kwargs):
        for backend in BACKENDS:
            if not backend._supports_args(*args):
                continue
            result = backend.call_op(operator_name, *args, **kwargs)
            if result is not NotImplemented:
                return result
        raise NotImplementedError("no backend supports this operator")
```

**Problem 8**: My backend looped forever because I decomposed an operator into calls which decomposed into the operator that I was already handling!

**Solution**: Don't allow reentrancy into a backend.

```python
class Torch:
    @staticmethod
    def run_op(operator_name, *args, **kwargs):
        for idx, backend in enumerate(BACKENDS):
            if backend is None or not backend._supports_args(*args):
                continue

            BACKENDS[idx] = None
            try:
                result = backend.call_op(operator_name, *args, **kwargs)
                if result is not NotImplemented:
                    return result
            finally:
                BACKENDS[idx] = backend

        raise NotImplementedError("no backend supports this operator")
```

**Problem 9**: Okay - but now my decomposition isn't working because it relies on calling operators in my own backend recursively (I promise I won't call this operator recursively!)

**Solution**: Allow opt-in reentrancy into a backend.

```python
class Torch:
    @contextmanager
    def enable_recursion(self):
        saved = BACKENDS[_current_index]
        BACKENDS[_current_index] = self
        try:
            yield
        finally:
            BACKENDS[_current_index] = saved

    @staticmethod
    def run_op(operator_name, *args, **kwargs):
        global _current_index
        saved_current_index = _current_index
        try:
            for idx, backend in enumerate(BACKENDS):
                if backend is None or not backend._supports_args(*args):
                    continue

                _current_index = idx
                BACKENDS[idx] = None
                try:
                    result = backend.call_op(
                        operator_name, *args, **kwargs)
                    if result is not NotImplemented:
                        return result
                finally:
                    BACKENDS[idx] = backend

            raise NotImplementedError(
                "no backend supports this operator")
        finally:
            _current_index = saved_current_index

class CudaTorch:
    def _bmm(self, a: Tensor, b: Tensor) -> Tensor:
        with self.enable_recursion():
            out = Tensor(...)
            for i in range(a.shape()[0]):
                out[i] = Torch.run_op("mm", a[i], b[i])
            return out
```

**TADA - We've just reinvented the dispatcher!** (Now go rewrite it in C++ for speed with support for thousands of operators, many existing backends, and add a python frontend with extensibility…)

Now once you've got this model you're not limited to backends. If a dispatch key handles all tensor types and always redispatches after doing its work, it acts as a "mode" — intercepting and transforming operator behavior before the call reaches the actual backend. Autocast, functionalization, and autograd all work this way.

What are some of the nifty built-in modes that torch uses to get the job done?

## Autocast

Autocast right-sizes tensor precision on a per-op basis. Some ops get their inputs cast to lower precision for performance (e.g., mm is faster in fp16 than fp32) and some get cast to higher precision for accuracy (e.g., sum accumulates in fp32 to avoid precision loss).

How does it do it? It registers a mode (dispatch key) that looks sort of like:

```python
def call_op(self, operator_name, *args, **kwargs):
    if operator_name == "sum":
        cast_args = []
        for t in args:
            if isinstance(t, Tensor):
                t = t.to(torch.fp32)
            cast_args.append(t)
        args = cast_args

    elif operator_name == "mm":
        cast_args = []
        for t in args:
            if isinstance(t, Tensor):
                t = t.to(torch.fp16)
            cast_args.append(t)
        args = cast_args   

    return Torch.run_op(operator_name, *args, **kwargs)
```

There's a subtlety here: when we call `t.to` we're implicitly calling `Torch.run_op` and we're re-entering the dispatch mechanism from the top - but because we disable dispatch keys as we execute them they won't re-enter autocast because we're already handling it.

## Functionalization

Functionalization turns mutable ops into immutable ops - this is useful for graph operations since it's a lot easier writing graph passes on immutable inputs without having to worry about aliasing.

The basic idea is that we convert code like:

```python
t.add_(5)
```

to

```python
t2 = t.add(5)
```

and then when we exit the functionalization mode we copy the values back. Again this is done by intercepting the ops as they get dispatched and changing the operations being performed.

## \_\_torch\_dispatch\_\_

`__torch_dispatch__` lets us extend the dispatcher dynamically at runtime. For example DTensor is an interesting case - it's an example of a User-Defined Subclass. Any user could have written this tensor subclass without any internal pytorch support.

The basic idea is to support large virtual tensors which get subdivided across ranks automatically.

How does it do it? You could write a whole post just about DTensor (and its successor SPMD types) but in essence it uses the dispatcher to convert code like this:

```python
C = A @ B
```

into code like this:

```python
A_ = A.dtensor_magic()
B_ = B.dtensor_magic()
C_ = A_ @ B_
C = C_.dtensor_magic()
```

where the "magic" parts redistribute the tensors across ranks so the local operations make sense. It makes use of the dispatcher because _how_ it needs to distribute the tensors depends on what operation is being performed.

## Some Internal Details

### native_functions.yaml

native_functions.yaml is PyTorch's central registry of all native operators. Each entry declares an operator's signature and which backends provide implementations (dispatch table). The codegen reads this file and generates:

- C++ function declarations and dispatch stubs
- Python bindings so you can call torch.mm(…) from Python
- The dispatch table entries that wire each op to its per-backend kernel

So when you add a new operator to PyTorch, you add an entry here and the build system generates all the glue code that connects the Python API to the C++ kernels through the dispatcher.

It's essentially the source of truth for "what ops exist and where do they dispatch to."

### Internal State

There are basically three places where the internal state of the dispatcher is stored:

1. **Per-tensor**: Each tensor carries a DispatchKeySet — a bitset of which keys are relevant to it (e.g., CPU, CUDA, Autograd). This is how the dispatcher knows which keys to consider for a given op based on its inputs.

2. **Thread-local state (TLS)**: Two key pieces:
   - **Excluded set**: Which dispatch keys have been "nulled out" — this is the reentrancy guard from above. When a key's handler runs, it excludes itself so redispatched ops skip it (this is the `BACKENDS[idx] = None` pattern from above.)
   - **Included set**: Which context-dependent keys are active. For example, Autocast is only in the included set when you're inside a `torch.autocast()` context. Entering the context adds it, exiting removes it.

3. **Global**: The dispatch tables themselves — the mapping from (op, dispatch key) → kernel function. These are static after registration.

The active key set for a given op call is: **(tensor keys | TLS included) - TLS excluded**.

The dispatch keys are numbered so that their order determines dispatch priority — the highest key runs first.

## References

- [Original post on dev-discuss](https://dev-discuss.pytorch.org/t/how-does-the-dispatcher-work/3358)
