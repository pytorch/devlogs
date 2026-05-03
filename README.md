# PyTorch DevLog

Developer technical notes — durable, AI-accessible, and open to the OSS community.

PyTorch developers regularly produce deep technical content about design decisions, performance analyses, and feature implementations. This repo makes that knowledge **permanent, searchable, and machine-indexable** — so it doesn't get buried in chat threads or ephemeral posts.

## Why

- **AI-accessible**: LLM tools that index PyTorch automatically gain access to this technical context
- **Open to OSS**: Contributors get visibility into internal technical discussions
- **Durable**: Versioned Markdown files remain discoverable indefinitely

## Recent Posts

| Date | Title | Topic |
|------|-------|-------|
| 2026-03-25 | [Unbacked Dynamic Shapes Shouldn't Be Slower — Now They Aren't](./dynamic_shapes/2026-03-25-unbacked-perf-parity.md) | Dynamic Shapes |
| 2026-02-27 | [Reducing Compile-Time Overhead in Unbacked-Symbol-Heavy torch.export Traces](./dynamic_shapes/2026-02-27-compile-time-unbacked-export.md) | Dynamic Shapes |
| 2026-01-20 | [Backed to Unbacked: From Guardable to Guardless Shapes](./dynamic_shapes/2026-01-20-backed-to-unbacked.md) | Dynamic Shapes |
| 2025-10-29 | [Slaying Framework Data-Dependent Errors Dragon](./dynamic_shapes/2025-10-29-slaying-framework-ddes.md) | Dynamic Shapes |
| 2025-07-08 | [Guard-Free Dynamic Shapes](./dynamic_shapes/2025-07-08-guard-free-dynamic-shapes.md) | Dynamic Shapes |

## Topics

| Directory | Scope |
|-----------|-------|
| [`dynamic_shapes/`](./dynamic_shapes/) | Unbacked shapes, guards, symbol semantics |
| [`dynamo/`](./dynamo/) | TorchDynamo graph capture |
| [`inductor/`](./inductor/) | TorchInductor codegen |

## Contributing

1. Pick (or create) a topic directory
2. Write a Markdown file with a date-prefixed filename: `YYYY-MM-DD-title.md`
3. Add images under `<topic>/images/`
4. Open a PR

See the [post template](./_template.md) for the recommended front matter and structure.

## License

Content in this repository is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
