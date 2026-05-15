# Introducing debug-graph-breaks: A Skill for Torch Compile Debugging

I'm excited to share **debug-graph-breaks**, a new skill for debugging Torch Compile graph breaks, now available in the [meta-pytorch/skills](https://github.com/meta-pytorch/skills/blob/main/skills/debug-graph-breaks/SKILL.md) repository.

## Purpose

Torch Compile graph breaks prevent full graph capture and hurt performance. This skill helps you:

- **Identify** root causes of graph breaks
- **Understand** why operations break compilation
- **Get actionable fixes** with specific code changes
- **Learn** best practices for Torch Compile-friendly code

The skill is grounded in the **Graph Break Website** as its knowledge base—improvements to the website directly improve the skill's quality.

## Performance & Evaluation

Evaluated on the [OSS Model Graph Break Corpus](https://github.com/penguinwu/oss-model-graph-break-corpus)—a collection of real-world graph break scenarios from open-source models.

**Evaluation process:**
- Created a test set from the corpus covering common break patterns
- Measured accuracy of root cause identification and quality of fixes

**Results:** **425/439 (96.8% pass rate)** — the skill correctly identifies break causes and provides working fixes for the vast majority of cases.
In comparison, vanilla Claude Opus 4.6 succeeded 34% of tests, often failing due to timeouts and sub-optimal solutions.

## How It Compares to Vanilla Claude

**Example: Debugging a `print()` statement graph break**

**Vanilla Claude** typically suggests:
> "Wrap the print statement with `torch.compiler.disable` to prevent the graph break."

**debug-graph-breaks** provides the actual fix:
> "This is a logging operation that breaks the graph. Instead of disabling compile, use:
> ```python
> torch._dynamo.config.reorderable_logging_functions.add(print)
> ```
> Or use the print higher-order operator for better performance. This keeps the operation in the graph while allowing the logging to work correctly."

The skill understands Torch Compile internals and provides direct fixes rather than workarounds.

## How to Use

1. Clone [meta-pytorch/skills](https://github.com/meta-pytorch/skills)
2. Invoke the skill when you hit a graph break
3. Provide: model code, break message, and logs
4. Get: root cause analysis + specific code to fix it

## What's Next

**1. Enhanced Evaluation Infrastructure**
Migrating evaluations to the OSS repo ([PR #8](https://github.com/meta-pytorch/skills/pull/8)) to enable community contributions and local testing.

**2. Graph Break Website Improvements**
The website (the skill's knowledge base) has planned updates:
- Adding more examples
- Improving hints and explanations
- These improvements will directly benefit the skill

**3. New Skills**
- **Recompiles skill**: Currently in development to help debug recompilation issues

## Get Started

Try it on your next graph break: [meta-pytorch/skills](https://github.com/meta-pytorch/skills/blob/main/skills/debug-graph-breaks/SKILL.md)

Feedback and contributions welcome!
