---
title: "When is a problem blocking?"
date: 2026-10-30
author: "Edward Yang (@ezyang)"
tags: [ai-agents, code-review]
---

Claude was being overly aggressive as a code reviewer and I ended up writing this to explain to it some of the nuance of when to request changes.
In particular, when we're reviewing code, we might have to consider these situations:

1. What if there is a preexisting problem in the code?
2. What if, via local reasoning only, we think there could be a problem, but
   we can't tell if this situation can actually occur?
3. What if there is a problem, but the situation is low stakes and building
   a spaceship is counterproductive?

For all of these, the answer is "it depends," but there's a pretty decent
first pass algorithm I personally like to follow.

**1. Fix preexisting problems separately.**

If I'm convinced there's a problem, I fix the preexisting problem separately,
then do my original change. But I might not do that if the preexisting problem
is inconvenient to fix and my planned change fixes the problem: it's "easier
to do it in one go."

The big no-no is to bundle an orthogonal bug fix with a change. Bug fixes can
have unintended consequences in production; do them separately.

**2. Rely on clear invariants when you have them.**

In an ideal world, the codebase has clear invariants, and we can rely on these
invariants holding when we write code. This avoids us having to write stupidly
defensive code. Overly defensive code is a common LLM pathology: graders
rarely penalize an extra check, so historically we've needed to apply pressure
against this.

However, it's often the case that there aren't clear invariants in the code,
so it's hard to tell whether a situation can actually occur. Adding all the
checks can end up being messy or expensive. This is usually a design smell,
but it's not always obvious how to fix it, and it may be a lot of work to fix.

**3. Account for the stakes.**

We have an intuition for code that is very sensitive, where every single
possibility needs to be accounted for (no nit is too small!), versus code
that is lower stakes, usually because the outcomes when it fails are not so
bad. Some of this is common sense, but some of it is codebase history that we
need to write down.

For example, getting a serialization format wrong is very, very, very bad
because you can never fix it again. Getting APIs wrong is kind of annoying,
but much less irrecoverably bad. Causing silent data corruption is not good,
but we definitely have had plenty of those bugs in PyTorch's history.

Messing up some test harness code can be harmful (e.g., tests silently stop
running), but if it crashes, your CI will fail, so it will be very obvious.
