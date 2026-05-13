---
title: "Your Post Title"
date: 2026-01-01
author: "Your Name (@github_handle)"
tags: [topic1, topic2]
---

<!-- Example frontmatter:
---
title: "Speeding Up Dynamic Shape Tracing by 3x"
date: 2026-03-25
author: "Laith Sakka (@laithsakka), Bob Ren (@bobrenjc93)"
tags: [dynamo, performance, tracing]
---

Author format:
- Always include your GitHub handle in parentheses: Name (@handle)
- Multiple authors: "Name1 (@handle1), Name2 (@handle2)"
- GitHub handles are used for giscus comment notifications
-->

> **TL;DR** – One or two sentence summary of the key takeaway.

## Background / Motivation

Why does this matter? What problem does it solve?

## Design / Approach

Technical details, architecture decisions, code snippets.

```python
# Example code
import torch
```

## Results / Benchmarks

Numbers, tables, charts.

**Images:** Do NOT drag-and-drop images into the GitHub editor — they'll upload to GitHub's CDN instead of the repo.

To add images:
1. Upload your images to `static/images/<topic>/` (e.g., `static/images/dynamo/my-chart.png`)
2. Reference them in your post as `![description](/devlogs/images/<topic>/my-chart.png)`

Note: images won't render when viewing the `.md` file on GitHub — they only render on the Hugo site.

| Configuration | Throughput | Latency |
|---------------|-----------|---------|
| Baseline      | …         | …       |
| New approach  | …         | …       |

## Open questions / Future work

What remains to be done? What trade-offs were made?

## References

- [Link to relevant PR / diff](#)
- [Link to related discussion](#)
