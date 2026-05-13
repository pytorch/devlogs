---
title: "Toward Agent-Friendly Dynamo: Mirroring CPython Semantics"
date: 2026-05-13
author: "Animesh Jain (@anijain2305)"
tags: [dynamo, cpython, llm-agents, graph-breaks, tp-slots]
---

> **TL;DR** – Dynamo's ad-hoc CPython support creates fragmented graph breaks that are hard to fix — even for LLM agents. By refactoring Dynamo to mirror CPython's `tp_*` slot semantics, we make the system systematically auditable and agent-friendly, already lifting CPython test pass rates from 38% to 45% and proactively eliminating classes of graph breaks in frontier models.

## Observations

Working with frontier training frameworks has surfaced some fundamental issues in Dynamo. The issues broadly fell into four categories:

- **CPython language gaps:** For example, Dynamo supports calling a `functools.partial` object but did not support hashing it.
- **Insufficient exception messages:** One frontier framework had an unusual setup where `torch.compile` was always enabled, even during development. Users expected Dynamo to surface CPython-like errors (e.g., "list is not hashable"), but instead saw generic messages like "found unhashable object."
- **Bugs:** For instance, Dynamo assumed a property object is always a function, while one framework used a `functools.partial`.
- **PyTorch composability gaps:** Incomplete support for `autograd.Function`, tensor hooks, FSDP2 etc. This is an important category, but *we will not focus on it in this post*.

From a `torch.compile` user's perspective, these issues are indistinguishable. The common feedback was to improve error messages to enable better self-serve debugging, but that's not the root problem.

## The Deeper Issue

Dynamo loosely mirrors parts of CPython, but this mapping is not always consistent in practice. Two structural patterns stand out:

![Dynamo CPython support overview — partial support and VariableTracker overreach](/devlogs/images/dynamo/cpython-gaps-overview.png)

- **Partial support (incomplete "green circles"):** Support for CPython features is scattered. For example, we support calling a `functools.partial`, but not hashing it. These partial implementations create subtle and sometimes hard-to-debug graph breaks.
- **VariableTracker overreach (the "red circles"):** Dynamo models CPython objects through VariableTracker, which ideally should be limited to C-backed types. Over time, we've added trackers for Python-level constructs like `enum.Enum` and frozen dataclasses — often to work around missing fundamentals (e.g., metaclass handling). These trackers require significantly more investment, since there is far more behavior to model, and tend to accumulate band-aid solutions — making the system harder to maintain and extend.

The supported Dynamo surface area is "implementation-by-accident" and therefore we don't even know which parts are unsupported. If we knew, we would fix those gaps, and not write better error messages.

## Why LLM agents don't fix this (yet)

Given the rise of LLM agents in software engineering, it's natural to apply them to fixing graph breaks. In theory, this should work well: CPython is well-documented, and the task is largely mechanical — map CPython semantics into Dynamo.

In practice, LLM agents have been quite helpful, but not as effective as one might expect for what appears to be a largely mechanical task. For example, while working through an enum-related graph break in a frontier framework, Claude was able to send a PR quickly but it required too many changes and repeatedly ran into CI issues.

![Claude attempting to fix an enum-related graph break — too many changes, repeated CI failures](/devlogs/images/dynamo/claude-enum-pr.png)

This isn't a limitation of the agents themselves as much as a reflection of the current state of Dynamo. The reason is that Dynamo does not mirror CPython's data model. As a result, LLM agents end up building on top of an inconsistent and fragmented foundation, amplifying the very issues they're meant to solve.

Instead of incrementally patching gaps, we're exploring a different approach:

## Mirror CPython's structure in Dynamo using LLM agents

In CPython, every type derives from [PyTypeObject](https://docs.python.org/3/c-api/typeobj.html), which defines behavior via `tp_*` slots implemented as C function pointers. For example, [tuple](https://github.com/python/cpython/blob/main/Objects/tupleobject.c#L959) behavior is defined through these slots, with 0 indicating fallback to `PyBaseObject_Type` (not lack of support).

![CPython tuple tp_* slot definitions](/devlogs/images/dynamo/cpython-tuple-tp-slots.png)

Dynamo does not model this protocol consistently today. Equivalent `tp_*` behavior is implemented in ad hoc ways, sometimes duplicated across multiple places. This makes the system difficult to reason about, audit, and extend.

We are refactoring Dynamo to more closely follow `tp_slot` semantics. The goal is not just to fix isolated gaps, but to move from an "implementation-by-accident" model to one that can be systematically audited against CPython. Note that there are many `tp_slots` that are not relevant for Dynamo, like `tp_traverse` etc, so the goal is not 100% mirroring but close enough mirroring that facilitates LLM agent fixes.

![Dynamo refactored to mirror CPython's tp_slot structure](/devlogs/images/dynamo/dynamo-mirroring-cpython.png)

This was previously too tedious to scale but LLM agents changed that. CPython's structure, documentation, and comments map well to this task: translating C-level slot behavior into Dynamo's abstractions.

## Execution

For more details on execution, see the OSS-facing document [Bridging CPython-Dynamo Gap using Claude](https://docs.google.com/document/d/1_tddtwagLMr0wtY8kKL8mj0io6Tc3kKCoOmOmrMaSFQ/edit?tab=t.0).

In short, the effort is progressing well: several `tp_*` slots have already been implemented in Dynamo, with many more in flight. This work is being driven by a combination of Dynamo pod members and OSS contributors. If you're interested in contributing, the document includes clear instructions on how to use LLM agents for this workflow — feel free to pick up and own a [slot](https://docs.google.com/document/d/1_tddtwagLMr0wtY8kKL8mj0io6Tc3kKCoOmOmrMaSFQ/edit?tab=t.0#bookmark=id.zc0q45ph1x8v).

## Early Wins and Validation — CPython Tests

We now have ~4–6 weeks of this work landed in PyTorch, and we're already seeing an impact. Several `tp_slot` implementations have already landed, each eliminating classes of graph breaks rather than individual instances. A few concrete examples:

- **copy.deepcopy:** Instead of adding ad hoc support, we now trace into deepcopy and model `__reduce_ex__`. This follows the CPython approach — model the core primitive and let everything else compose naturally ([PR](https://github.com/pytorch/pytorch/pull/179611/)).
- **enum.Enum:** Previously handled via a VariableTracker (a "red circle"). Missing `tp_getattro` and metaclass handling caused multiple graph breaks. Implementing `tp_getattro` allowed us to remove `EnumVariable`, fixing many enum-related gaps ([PR](https://github.com/pytorch/pytorch/pull/175565), [PR](https://github.com/pytorch/pytorch/pull/179029)).
- **Frozen dataclasses:** Another "red circle" case — removing the custom tracker eliminated associated graph breaks ([PR](https://github.com/pytorch/pytorch/pull/179426)).

We're also seeing validation from CPython–Dynamo tests added by Guilherme (OpenTeams), building on an idea from Richard Zou. These run with `fullgraph=True`. Each `tp_slot` PR leads to multiple unexpected test passes, increasing the total pass rate from 38% to 45%. This is a strong signal that we're fixing *future* graph breaks preemptively by aligning with CPython, rather than waiting to encounter them in real models.

Overall, this reinforces the direction: mirroring CPython eliminates classes of graph breaks, instead of patching them one by one.

## Where this leads

Directionally, we are preparing Dynamo to be *agent-friendly by design*. If we get this right, extending CPython support in Dynamo becomes structured, repeatable — and eventually, "boring." That's the goal: to make this layer predictable enough that we can focus on harder problems.

## What's not covered

This post focuses on graph breaks arising from CPython gaps. However, there's an equally large — and largely separate — class of issues around PyTorch composability. I'm still forming clearer opinions in that area, and will share more as that work evolves. Stay tuned!
