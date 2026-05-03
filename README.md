# PyTorch DevLog

Developer technical notes — durable, AI-accessible, and open to the OSS community.

PyTorch developers regularly produce deep technical content about design decisions, performance analyses, and feature implementations. This repo makes that knowledge **permanent, searchable, machine-indexable, and available to OSS users and AI agents** — so it doesn't get buried in chat threads or ephemeral posts.

## Why

- **AI-accessible**: LLM tools that index PyTorch automatically gain access to this technical context
- **Open to OSS**: Contributors get visibility into internal technical discussions
- **Durable**: Versioned Markdown files remain discoverable indefinitely

## Topics

| Directory | Scope |
|-----------|-------|
| [`dynamic_shapes/`](./dynamic_shapes/) | Unbacked shapes, guards, symbol semantics |
| [`dynamo/`](./dynamo/) | TorchDynamo graph capture |
| [`inductor/`](./inductor/) | TorchInductor codegen |
| [`distributed/`](./distributed/) | FSDP, DTensor, c10d |
| [`export/`](./export/) | torch.export, AOTInductor |

## Contributing

1. Pick (or create) a topic directory
2. Write a Markdown file with a date-prefixed filename: `YYYY-MM-DD-title.md`
3. Add images under `<topic>/images/`
4. Open a PR

See the [post template](./_template.md) for the recommended front matter and structure.

## License

Content in this repository is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
