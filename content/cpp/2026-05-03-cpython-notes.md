---
title: "CPython notes"
author: Edward Yang (@ezyang)
date: 2026-05-03
tags: [cpp, cpython, llm]
---

One of the lost arts of PyTorch development is the ability to write idiomatic
C++ code that interacts with the CPython API.  This was a very important skill
in the early days of eager PyTorch, since we spent a lot of time moving large
chunks of the framework to C++ for speed reasons, but we don't touch the C++
code that much these days and many members of the team haven't written any
amount of serious C++.

LLMs seriously lower the barrier for writing C++ and dealing with the minutiae
of manual memory management in C.  But they're not perfect.  So these devlog
is to talk about all of the things that I put into the process.  A prompt of
sorts.  It is based off of the experience driving an LLM to author this PR:
[[autograd] Allow positional arguments to be passed as kwargs for autograd
custom Function #182206](https://github.com/pytorch/pytorch/pull/182206)

## Don't forget Dynamo

When you change the semantics of parts of PyTorch implemented in C++, you have
to update the corresponding Dynamo code that simulates this.  There's no way
around this: Dynamo has to reimplement any C++ functionality since you can't
Dynamo trace into C++ code.

## Use smart pointers

Raw use of CPython API involves manual memory management.  THPObjectPtr or
py::object should be used for RAII based memory management.  Because we don't
have smart pointer wrappers for all CPython APIs, it is still necessary to do
this carefully when we interact with CPython raw APIs.  This leads to...

## Ensure ownership semantics for APIs are correct

Whenever a CPython API is used, we must verify that we have down ownership
correctly for it.  This is something that's easy to hallucinate without access
to the CPython API docs, so we need to verify this directly.

## Handle errors properly

In raw CPython API, due to C's lack of support of exceptions, errors have to
be manually propagated: when an error occurs, some error state is set, and we
must detect `NULL` return and propagate this upwards until we return to the
interpreter.  It is easy to forget to do this correctly in C++, either by
forgetting to check the error state, or by doing an operation that may
potentially clobber preexisting error state.  This is something SOTA LLMs can
do correctly, but may need some remindning to do so.

## Simple versus efficient in C

When working with CPython API, it's easy to fall into the mode that we are
writing C code only, without access to C++.  In C, there are lots of things
you might want to do that would require an annoying amount of boilerplate.
For example, if it would be algorithmically better to maintain a hash map to
answer lookup queries, an LLM writing C might just do a nested loop because
actually implementing a hash map in C is a lot of work.  You may need to
prompt the LLM to make use of the C++ features.

## Using fast or slow CPython APIs

Let's suppose that you have a code object in Python and you want to access
fields on it in C.  There are two ways you can do this: you can access the
field if as if you were accessing it from Python, e.g., the CPython equivalent
of a `getattr`.  Or, in some cases, the layout of the object is well known,
and you can convert to this struct type and then access the field directly in
the C style.  The former is safe, ABI stable and will work even if the user
duck types something unusual into the code: if you're trying to port some
existing Python code and need it to behave exactly in the same way, you need
to go the slow route to ensure all of the interposition points get triggered.
But most of the point of writing things in C is to go fast, and you will go
faster if you can enforce, e.g,. that the user gave you an actual real code
object, not some weird duck typed thing.  Empirically, LLMs seem to prefer the
former!

## Is the Python helper properly reimplemented in C

This is the inverse problem.  Suppose that you need to implement some
functionality that exists in Python, but has no C counterpart.  It's pretty
easy for the LLM to dash off a rough approximation of the function but unless
it is written in a style that is obviously equivalent (e.g., imagine just
having unrolled the Python interpreter loop into a straight line sequence of
CPython API calls that the interpreter would have done--by the way, this
*will* speed things up and it can be a useful thing to do), it probably
doesn't do the same thing as the original function.  And so there's an
important question one needs to ask here, which is in what ways is it
different, and do those differences matter?  This will not be assessed by
default, so you'll have to check.

## Fast path efficiency

You're probably writing things in C++ because you need them to be fast.  This
means that some reasoning about the cost of changes you are making are
necessary.  You had better not be introducing a virtual dispatch in a hot
loop; if there is a small inefficiency that can be eliminated in a cost free
way in the fastpath (e.g., coalescing two flags into a single flag, and
checking only that flag in the fastpath), it's a good idea to do it.  When
adding a new feature, it's useful to reason through what exactly, on an
instruction-by-instruction basis, is its overhead, and whether or not all of
that overhead is necessary or not.

## Risk assessment

This is perhaps the most important piece of it all.  Bugs in C++ are very
severe.  Buggy Python code will just result in a crash with a legible stack
trace; buggy C++ code can lead to segfaults or subtle memory corruption.  Even
when your change is exactly right, there an still be unforseen consequences:
an extra field in a struct pushing the struct from one to two cache lines that
obliterates performance, or a correct in isolation change that greatly
increases the trigger rate of a latent bug elsewhere in the codebase.  A big
part of making changes to core C++ infrastructure is having done it enough in
the past to have a sense of what is likely to be risky or not; and, perhaps
more importantly, being able to follow up when your change causes a problem
and incorporate this information into a way to make the push less risky in the
future.  Similarly, if you are just adding a change that only gets triggered
when some new conditional is set, this is substantially less risky than a
semantics preserving refactor of hot code, even though the latter
hypothetically is a no-op!
