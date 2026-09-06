# AGENTS.md

Instructions for AI agents working in this repository.

## Project overview

This is `journal-rag`, an MCP server + CLI that provides hybrid BM25 + vector semantic search over markdown journal files. It's designed to be installed once and used across multiple consuming repos, each with their own `journal-rag.config.json`.

## Architecture

```
src/
  server.ts     — Lightweight MCP stdio proxy entry point
  daemon.ts     — Shared per-user IPC daemon entry point
  daemon-client.ts / daemon-protocol.ts — Daemon startup, transport, and protocol
  workspace-runtime.ts — Deduplicated workspace state and tool operations
  logger.ts     — Cross-platform daily JSONL diagnostics
  indexing-lock.ts — Cross-process vector-build lock
  cli.ts        — CLI entry point (journal search/list/get/regex/index)
  index.ts      — Markdown file discovery, heading-based chunking, BM25 index cache
  embeddings.ts — Local vector embeddings via @huggingface/transformers, vector cache
  search.ts     — BM25, substring, regex, and hybrid (RRF) search implementations
  config.ts     — Config file discovery and parsing
  types.ts      — Shared TypeScript interfaces
```

## Key design decisions

- **Hybrid retrieval**: `search_journal` fuses BM25 keyword scores with vector cosine similarity using Reciprocal Rank Fusion (RRF, k=60). This avoids score normalization issues.
- **Shared daemon**: Every editor needs its own lightweight stdio MCP proxy, but all proxies connect to one per-user daemon over a Windows named pipe or Unix socket. No OS service installation is required.
- **Workspace deduplication**: Daemon runtimes are keyed by the canonical path to `journal-rag.config.json`, so Cursor and ChatGPT share one runtime when they open the same workspace.
- **Shared local embeddings**: The daemon loads embeddings through `@huggingface/transformers`. macOS uses the cache-free `Xenova/all-MiniLM-L6-v2` encoder; other platforms use `onnx-community/Qwen3-Embedding-0.6B-ONNX`. Model loads and inference calls are serialized; model promises are deduplicated by model ID and device.
- **Platform acceleration**: Use DirectML on Windows, Core ML on macOS, and CPU elsewhere. DirectML must use `enableMemPattern: false` and sequential execution. If an accelerated provider fails during load or inference, log the original error and use a CPU session for that model for the rest of the daemon lifetime.
- **Conservative indexing**: Embed one journal chunk per batch. This avoids DirectML device hangs on 6 GB GPUs. A `proper-lockfile` lock serializes vector builds across processes and recovers stale locks.
- **Incremental vector cache**: Stored at `.journal-rag/vectors.json` alongside the BM25 cache. Only new or missing chunks are embedded, deleted chunks are pruned, and partial results are checkpointed every 100 chunks.
- **Graceful degradation**: If the vector index fails to build (e.g., model download issue), the server falls back to BM25-only search without crashing.
- **Heading-based chunking**: Markdown files are split at `##` and `###` boundaries. Each chunk carries its heading path for citation.

## Daemon lifecycle

- `daemon-client.ts` starts `dist/daemon.js` automatically under a startup lock when the endpoint is unavailable.
- `DAEMON_PROTOCOL_VERSION` is part of the endpoint name. Bump it for breaking wire-protocol changes.
- Rebuilding does not replace an already-running daemon. Stop the old daemon before testing new daemon code; restart MCP clients separately when proxy code changes.
- The CLI still runs locally; the global indexing lock prevents it from colliding with daemon builds.
- `journal-daemon` is exposed as a binary for diagnostics, but normal use is automatic.

## Build and test

```bash
npm install        # install deps + build (via prepare script)
npm run build      # tsc compile
npm run typecheck  # type-check without emitting
```

No test suite currently exists.

For daemon smoke testing, build first, then call `ensure_workspace` concurrently from separate Node processes. Both responses should contain the same `daemonPid` and `runtimeId` for differently-cased or equivalent paths to the same config. Also test a raw MCP initialize + tool call through `dist/server.js`.

## Cache files

Both caches live in the directory specified by `cachePath` in config (default `.journal-rag/`):

| File | Contents |
|------|----------|
| `index.json` | BM25 chunk index with file fingerprints for invalidation |
| `vectors.json` | Embedding vectors keyed by `file\0heading`, model name, version |

Cache version constants are in `index.ts` (`CACHE_VERSION`) and `embeddings.ts` (`VECTOR_CACHE_VERSION`). Bump these when changing the cache schema.

The final vector cache is written after pruning. During long rebuilds, partial caches are checkpointed every 100 embedded chunks so interrupted builds resume instead of restarting.

## Diagnostics

Daily JSONL logs are retained for 14 days:

- Windows: `%LOCALAPPDATA%\journal-rag\logs`
- macOS: `~/Library/Logs/journal-rag`
- Linux: `$XDG_STATE_HOME/journal-rag/logs` or `~/.local/state/journal-rag/logs`

Override the location with `JOURNAL_RAG_LOG_DIR`. Logs include process IDs, config paths, request methods, indexing progress, selected device, model-load timings, memory after model load, and stack traces. Never log query text or journal content.

For DirectML error `0x887A0006` / `DXGI_ERROR_DEVICE_HUNG`, inspect the matching NVIDIA/Windows event, stop the damaged daemon, and retry only after rebuilding. A hung DirectML session is not reusable. If the required session options and batch size 1 still fail, use a WebGPU or CPU fallback rather than increasing batch size.

## Adding a new search strategy

1. Implement your search function in `search.ts` (or a new module).
2. If it needs pre-computed data, add a build step in the relevant index module.
3. Wire it into `searchHybrid()` as another RRF input, or expose it through the daemon protocol and workspace runtime.

## Common tasks

- **Change the default embedding model**: Update the platform-specific model constants and model-specific pooling/instruction behavior in `embeddings.ts`. Keep the macOS export cache-free because Core ML rejects zero-length key/value-cache tensors.
- **Adjust RRF fusion weight**: Modify `RRF_K` in `search.ts` (higher values flatten rank differences).
- **Add a config option**: Add the field to `JournalRagConfig` in `types.ts`, parse it in `config.ts`.
- **Expose a new MCP tool**: Add the method to `DaemonMethod`, dispatch it in `workspace-runtime.ts`, and register the proxy tool in `server.ts`.
- **Build/relink for local testing**: Run `npm run build` and `npm link` from this repository, verify the global junction targets this exact checkout, then restart existing MCP proxy/daemon processes.
