---
title: "Python First Comms for Researchers"
date: 2026-05-14
author: "Tristan Rice (@d4l3k)"
tags: [distributed, torchcomms, nccl, symmetric-memory, triton, prototyping]
---

> **TL;DR** – Modifying the C++ comms layer is a big barrier when researchers want to prototype new collective features. We've added Python bindings to torchcomms ([#2080](https://github.com/meta-pytorch/torchcomms/pull/2080)) and built two pure-Python backend prototypes — one wrapping NVIDIA's new `nccl4py` bindings ([#2515](https://github.com/meta-pytorch/torchcomms/pull/2515)) and one built on `SymmetricMemory` + Triton ([#2521](https://github.com/meta-pytorch/torchcomms/pull/2521)) — both passing the core torchcomms integration test suite. Since they plug into `torch.distributed`, researchers can fork, tweak, and mix them with existing projects like TorchTitan without touching C++.

## Background / Motivation

We've been thinking about how to improve overall research and prototyping speed for comms and collective libraries. LLMs have hugely improved prototyping speed for new ideas and libraries (e.g. [torchcomms UCC+MPI](https://github.com/meta-pytorch/torchcomms/pull/2052)) but this only really works for engineers who are already comfortable with the PyTorch and torchcomms build environments.

We've gotten direct feedback from researchers that modifying the C++ comms layer for experimenting with new features is a burden — long build times and unfamiliar tooling get in the way.

Symmetric memory solves part of the problem by enabling custom kernels for specific operations, but it doesn't integrate with the standard distributed collectives, and it's effectively impossible to prototype fault tolerance features like `ncclCommShrink` / `ncclCommGrow` / `ncclCommRevoke` without touching the C++ layer. We've also seen projects like [moodist](https://github.com/facebookresearch/moodist) pop up — complete rewrites of the comms backend from folks who want to tune things beyond what NCCL allows.

Given the volume of requests from researchers, it seems reasonable to have a first-class comms backend focused on hackability in PyTorch.

We recently added bindings so you can write torchcomms backends from Python ([#2080](https://github.com/meta-pytorch/torchcomms/pull/2080)). We've put together a couple of prototypes that leverage this to see what it might look like in the future for researchers to modify comms.

## Approach 1: nccl4py

`nccl4py` is a brand new library from NVIDIA that provides first-party Python bindings for the NCCL API. This is a low-level NCCL wrapper, so direct usage lacks a lot of the features PyTorch provides on top of NCCL such as operation timeouts and tracing. However, since it binds all NCCL features into Python, it's much easier to install and prototype integration with new NCCL features.

With some LLM assistance I was able to get a `nccl4py` backend passing the core torchcomms integration test suite ([#2515](https://github.com/meta-pytorch/torchcomms/pull/2515)).

At a high level, the backend implementation looks like this:

```python
import nccl.core as nccl

class TorchCommNCCL4Py(TorchCommBackend):
    def init(self, device: torch.device, name: str, options) -> None:
        self._comm = nccl.Communicator.init(nranks=size, rank=rank, unique_id=uid)

    def broadcast(self, tensor: torch.Tensor, root: int, async_op: bool):
        self._comm.broadcast(tensor, tensor, root, ...)
```

If a user wanted to prototype using a new NCCL feature such as `revoke`, they could either extend or copy/paste this file and tweak it to add a new method without requiring any new C++ modifications or dependencies.

Here's a basic example adding NCCL's new revoke operation:

```python
class MyTorchCommNCCL4Py(TorchCommBackend):
    ...

    def revoke(self):
        self._comm.revoke()

register_backend("mynccl", MyTorchCommNCCL4Py)
comm = torchcomms.new_comm("mynccl", ...)

# queue operation
comm.broadcast(...)

# custom timeout + revoke
time.sleep(10)
comm.unsafe_get_backend().revoke()
```

This greatly improves iteration speed for testing out new NCCL backend features.

## Approach 2: Symmetric Memory + Triton

We can take the symmetric memory approach to its logical conclusion and just write all of the collectives using it. We've prototyped an implementation that does exactly that and passes the core integration test suite ([#2521](https://github.com/meta-pytorch/torchcomms/pull/2521)).

This prototype has not been tuned for performance, and with symmetric memory we end up with some circular-ish dependencies. Symmetric memory currently requires a `ProcessGroup` to bootstrap communications. We're working on making symmem work cleanly with torchcomms, but regardless we end up with a backend depending on symmem which depends on a different backend. The prototype uses Gloo for bootstrap with only NVLink operations — no NCCL/NVSHMEM.

```python
@triton.jit
def _all_reduce_kernel(buffer_ptrs_dev, out_ptr):
    ...

class TorchCommSymMem(TorchCommBackend):
    def init(self, ...):
        workspace_tensor = _SymmetricMemory.empty_strided_p2p(...)
        self.symm_mem = _SymmetricMemory.rendezvous(workspace_tensor)

    def all_reduce(self, tensor: torch.Tensor, op, async_op: bool):
        _all_reduce_kernel[grid](self.symm_mem, tensor, op, ...)
```

Users can copy the reference implementation and get started immediately on just the bits they actually want to change, without having to muck around with scaffolding.

## Why not both? nccl4py + Symmetric Memory

Given this is all Python, users can override specific operations so they can mix and match between NCCL and symmetric memory kernels. You can just call a symmem kernel instead of an NCCL operation if you want to customize the performance of a specific operation.

```python
class MyTorchCommNCCL4Py(TorchCommNCCL4Py):
    # override all_reduce
    def all_reduce(self, tensor):
        torch.ops.symm_mem.one_shot_all_reduce(
            tensor,
            "sum",
            ...,
        )
```

Since these backends plug into `torch.distributed`, you can use them with any existing projects such as TorchTitan without any code changes. This could be used by researchers to override specific operations with custom quantized kernels for low-bandwidth comms, or to customize behavior for MoE models.

## Note on Production Readiness

These are currently prototypes and missing key features such as debuggability and timeout/watchdog support. In parallel, we're looking at how to generalize those infra features so we can plug them in anywhere — for both prototypes like this and symmetric memory kernels.

If we get positive feedback, we'd like to polish up these prototypes and provide them either as reference under torchcomms or a separate Python-only annex repo that can be forked and used.

## References

- [torchcomms #2080 — Python bindings for torchcomms backends](https://github.com/meta-pytorch/torchcomms/pull/2080)
- [torchcomms #2515 — nccl4py backend prototype](https://github.com/meta-pytorch/torchcomms/pull/2515)
- [torchcomms #2521 — Pure-Python symmem backend (SymmetricMemory + Triton)](https://github.com/meta-pytorch/torchcomms/pull/2521)
- [torchcomms #2052 — UCC+MPI backend](https://github.com/meta-pytorch/torchcomms/pull/2052)
- [moodist](https://github.com/facebookresearch/moodist)
