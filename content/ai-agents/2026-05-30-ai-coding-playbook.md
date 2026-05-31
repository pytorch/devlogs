---
title: "PyTorch's playbook for AI coding, as of May 2026"
author: Edward Yang (@ezyang)
date: 2026-05-30
tags: [ai-agents, code-review, oss, llm]
---

One of the important topics being discussed among the PyTorch team is how the
PyTorch codebase should engage with AI coding agents.  Today, many PRs to
PyTorch are AI-authored, and there have been obvious growing pains as we've
figured things out.  Based on discussions at the most recent PyTorch compiler offsite
(May 2026), I've assembled this playbook for AI coding in PyTorch.  It is half
descriptive, half prescriptive: it is trying to codify practices that are
being used among some members of the team, and bring everyone else along.
Hopefully, this post is just the beginning of our ongoing conversation about
how to engage with AI coding agents.

## Norms for AI coding

We can think of AI generated code as living in a spectrum, where on one hand
we have code that is almost exactly the same as human code, except that it was
typed by an AI, and on the other hand completely vibe-coded software which has
never been read by a human.

PyTorch is production software, used and relied upon by many people.  We have
a duty to our users to ensure that the code we ship is correct, understandable
and maintainable.  We think that SOTA coding agents can help us build better
software than we could have built purely by hand today, but they present us
with novel situations that require adapting our old rules.  We think different
norms are required depending on where code lives on the spectrum.

### As a substitution for human written code

On the most conservative end, we are adding AI coding but trying to keep as
many other aspects of the process fixed.  The human should read every line of
code.  You are responsible for every line of code.

Not everything stays the same though.  We propose these new norms:

- In an age of cheap code, we are human review bottlenecked. Authors should
  work hard to make code review easy.  Think about what information your
  intended reviewer needs and write it down (an AI written commit message is
  good for completeness, but the LLM is unlikely to know what your reviewer
  knows and doesn't know).  If a PR is big, make sure that there is a coherent
  order to engage with the change and write down the "read order."  Don't mix
  unrelated or cosmetic changes with semantic changes; ask an LLM to separate
  these out.

- It is extremely tempting to ask an AI agent to directly respond to code
  review comments.  Because we believe in human understanding on this end
  of the spectrum, we think it is important for humans to engage in dialog
  in review comments.  If a human spent time to write a question, they deserve
  a human response in return.

  On the flip side, many traditional questions one might ask in code review
  can be easily answered by an AI agent.  Code reviewers should consult AI
  first, only escalating unresolved questions to humans.  You can use
  `@claude` on GitHub, or check out a PR locally and use your coding agent
  locally on it for questions.

- It is OK to use a coding agent to autonomously fix code review comments
  (especially nits), but you are still responsible for reading and owning all
  the fixes.  This especially includes checking that the comment was actually
  fixed!

- Consider asking to directly edit someone else's code.  You should ask the
  author first before, e.g., pushing a commit, but with AI agents this is a very
  compact way to transmit small nitty feedback that would have been fed
  straight into an AI agent anyway.  It is also a good way to communicate more
  dramatic changes that would take more time to explain in text--an AI agent
  can expand text into code and help you verify that the intent of your text
  is clear.  The original author still has to read and take ownership of these
  changes.

### Mass AI PRs

Mass AI PRs are when we use agents to generate many PRs in parallel; e.g.,
using agents to burn down issues on an issue tracker.  Many bugs are not
individually important enough for a human to dedicate a few days fixing, but
in aggregate, fixing bugs is important, and AI coding agents are a big
opportunity to kill low hanging fruit (in the same way AI agents are really
good at discovering security vulnerabilities.)

The general ask here is that we should have high-level agreement that these
fixes, *in aggregate* have an ROI that justifies the human time spent on it.
While the operator of the agent swarm is responsible for doing initial
reviews, guiding it and improving it based on feedback, a mass of AI PRs will
increase reviewer burden.  The point is to have agreed that this review burden
is worth it!

### Well-encapsulated unreviewed code

As of today, we do not accept unreviewed AI generated code (aka slop) to the
main `pytorch/pytorch` repo.  However, we think the capability of SOTA models
today enables the creation of systems that otherwise could not have existed (e.g., via hill climbing.)  We have several live experiments in
unreviewed AI generated code; for now, these all live in out-of-tree
repositories.  This makes clear the experimental nature of the package; it
also makes mistakes in the code lower stakes (as we can more rapidly ship
releases).

Even unreviewed code still needs to follow some standards:

- Unreviewed code does not mean unowned code.  The human responsible for
  running the overall generation process is still accountable for this code
  (for example, we cannot accept unreviewed code from untrusted sources).

  The owner of this code still is responsible for reviewing the design of the
  overall code.  Even without having read every line, by asking factual
  questions about systems, it's still possible to form a strong mental model
  about the overall design of the code.  With SOTA models today, we find
  design guidance improves overall outcomes.  Don't just ask for feature after
  feature.

- Slop should live in well-encapsulated component boundaries, where there is a
  human-designed API boundary that separates slop from the rest of the system.
  AI slop is not acceptable for public facing UI with BC implications.  It
  should be possible to throw out all the code and rewrite it from scratch in
  a non-slop way.  It is even better if the output of the slop is verifiable,
  so it doesn't matter for overall system correctness if the slop code is
  buggy.  Although we don't currently have a precedent for this, we are
  willing to consider merging unreviewed code with a reviewed verifier to
  `pytorch/pytorch`.

- Unreviewed code should pass AI code review: e.g., guidance on security, test
  integration and global invariants.  This is a simple form of inference-time
  scaling that raises the floor for generated code.

Many of us at PyTorch have vibe-coded useful personal tools without these
standards.  We don't mean to discourage this!  However, our current opinion is
that this level of quality is not appropriate for PyTorch features proper (even
in experimental repositories).

## Tooling

We think the following tools will be helpful for a world of AI coding agents, and
we plan to implement them in the near future:

- **Risk-Aware Diff Auto Review (RADAR).**  We will make an OSS
  reimplementation of Meta's [RADAR](https://arxiv.org/abs/2605.30208) system.
  This system auto-reviews and auto-accepts PRs based on a set of criteria
  around evaluating the riskiness of a diff.  We've been using this internally
  at Meta and the RADAR approves are good.  The intention is not to relax
  PyTorch's PR landing standard; instead, it is to just remove friction around
  PRs that are already easy to review.  RADAR is predicated on trust of the
  author, so RADAR approves will be limited to PyTorch maintainers who otherwise
  already have approval rights on the files being changed.

- **Automatic AI Linting.**  We will implement an AI linter that will assess a
  diff against a well defined list of criteria, which can be defined on a
  subsystem-by-subsystem basis.  We don't call this AI code review, as it does
  not replace human code review.  Instead, the intention is to complement
  humans by leveraging the fact that AIs never get bored and will studiously
  follow instructions.  Given appropriate prompting, an AI can reliably check
  for ownership mistakes, whether or not a new public API is introduced,
  whether or not a device-to-host sync was introduced, etc.  A human reviewer
  could forget to check these things, and an AI can help avoid relying on a
  human reviewer remembering to check something.

- **Automated fbcode to OSS test case generation.**  A big annoyance for
  non-Meta contributors is when your PR gets reverted because it broke
  something Meta internal.  Previously, you get whatever minimal error
  trace was shareable and reverse engineer the failure.  We want to instead
  always have an agent draft and verify a minimal reproducer that passes
  without the diff and fails with the diff.  There will be some gating
  mechanism to ensure secret information doesn't leak, but this should setup a
  reliable pipeline of test cases to improve our OSS test coverage.

- **Codeowners.**  We have always had a notion of module maintainers (people
  who are responsible for a certain area of code), and we would like to more
  sharply define this, as ownership is a very valuable resource and we would
  like to foster it and make the best use of it.  In general, the owner is
  someone who has the overall picture of how all the code works, and therefore
  should have the privilege of getting a chance to weigh in on all changes to
  this feature.  Some of the precise mechanics should be worked out, but the
  general idea is that features affecting areas of the codebase should have an
  automatic cooldown, where the PR shouldn't be mergeable without the
  codeowner endorsing it in some way.  To ensure an AWOL codeowner doesn't
  block all changes, after some SLA, a PR can be landed with normal
  review--the cooldown is there to avoid situations where someone rushes in a
  change before the codeowner manages to take a look.  On the norms side,
  PR authors should give some deference by default to the codeowner--e.g.,
  you should convince the codeowner first that your change is a good idea!
  The `CODEOWNERS` file is probably not the best implementation of this mechanism;
  we'll figure it out (in particular it would be helpful to have non-file-based rules
  on ownership).

- **Draft, Request Changes.**  We are universally applying some new workflow
  rules for PRs to ensure that all code reviewers can easily maintain inbox
  zero on PR review requests.  Specifically, (1) if you need to put up a PR
  but do not wish for it to be reviewed, it is your responsibility to mark it
  as draft, (2) if you are a reviewer on a PR and you feel that the author
  needs to take action, it is your responsibility to request changes.
  Conversely, if you are an author of a PR that has request changes, it is
  your responsibility to clear that request when you want review.  The
  intention of these changes is to always make it clear who is responsible for
  moving a PR forward.  These tools are not new but they were used
  inconsistently by the team--we are now formalizing and requiring use of this
  UI.
