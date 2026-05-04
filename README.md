# PyTorch DevLog

Developer technical notes — durable, AI-accessible, and open to the OSS community.

**[Browse the devlog →](https://docs.pytorch.org/devlogs/)**

PyTorch developers regularly produce deep technical content about design decisions, performance analyses, and feature implementations. This repo makes that knowledge **permanent, searchable, machine-indexable, and available to OSS users and AI agents** — so it doesn't get buried in chat threads or ephemeral posts.

## Why

- **AI-accessible**: LLM tools that index PyTorch automatically gain access to this technical context
- **Open to OSS**: Contributors get visibility into internal technical discussions
- **Durable**: Versioned Markdown files remain discoverable indefinitely

## Topics

| Directory | Scope |
|-----------|-------|
| [`dynamic_shapes/`](./content/dynamic_shapes/) | Unbacked shapes, guards, symbol semantics |
| [`dispatcher/`](./content/dispatcher/) | Dispatch keys, operator registry, extensibility |
| [`dynamo/`](./content/dynamo/) | TorchDynamo graph capture |
| [`inductor/`](./content/inductor/) | TorchInductor codegen |
| [`distributed/`](./content/distributed/) | FSDP, DTensor, c10d |
| [`export/`](./content/export/) | torch.export, AOTInductor |

## Contributing

1. Pick or create a topic directory under `content/` — each folder automatically becomes a topic on the site
2. Add an `_index.md` with a `title` and `description` if creating a new topic
3. Write a Markdown file with a date-prefixed filename: `YYYY-MM-DD-title.md`
4. Add images under `content/<topic>/images/`
5. Open a PR

See the [post template](./template.md) for the recommended front matter and structure.