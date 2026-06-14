# AGENTS.md

Instructions for AI agents working in this repository.

## Project overview

This is `journal-rag`, an MCP server + CLI that provides hybrid BM25 + vector semantic search over markdown journal files. It's designed to be installed once and used across multiple consuming repos, each with their own `journal-rag.config.json`.

## Architecture

```
src/
  server.ts     — MCP stdio server entry point
  cli.ts        — CLI entry point (journal search/list/get/regex/index)
  index.ts      — Markdown file discovery, heading-based chunking, BM25 index cache
  embeddings.ts — Local vector embeddings via @huggingface/transformers, vector cache
  search.ts     — BM25, substring, regex, and hybrid (RRF) search implementations
  config.ts     — Config file discovery and parsing
  types.ts      — Shared TypeScript interfaces
```

## Key design decisions

- **Hybrid retrieval**: `search_journal` fuses BM25 keyword scores with vector cosine similarity using Reciprocal Rank Fusion (RRF, k=60). This avoids score normalization issues.
- **Local embeddings**: Uses `@huggingface/transformers` with `Xenova/all-MiniLM-L6-v2` by default. No API keys or network calls after initial model download.
- **Incremental vector cache**: Stored at `.journal-rag/vectors.json` alongside the BM25 cache. Only embeds chunks that are new or missing from the cache. Prunes deleted chunks automatically.
- **Graceful degradation**: If the vector index fails to build (e.g., model download issue), the server falls back to BM25-only search without crashing.
- **Heading-based chunking**: Markdown files are split at `##` and `###` boundaries. Each chunk carries its heading path for citation.

## Build and test

```bash
npm install        # install deps + build (via prepare script)
npm run build      # tsc compile
npm run typecheck  # type-check without emitting
```

No test suite currently exists.

## Cache files

Both caches live in the directory specified by `cachePath` in config (default `.journal-rag/`):

| File | Contents |
|------|----------|
| `index.json` | BM25 chunk index with file fingerprints for invalidation |
| `vectors.json` | Embedding vectors keyed by `file\0heading`, model name, version |

Cache version constants are in `index.ts` (`CACHE_VERSION`) and `embeddings.ts` (`VECTOR_CACHE_VERSION`). Bump these when changing the cache schema.

## Adding a new search strategy

1. Implement your search function in `search.ts` (or a new module).
2. If it needs pre-computed data, add a build step in the relevant index module.
3. Wire it into `searchHybrid()` as another RRF input, or expose it as a separate MCP tool in `server.ts`.

## Common tasks

- **Change the default embedding model**: Update `DEFAULT_MODEL` in `embeddings.ts`.
- **Adjust RRF fusion weight**: Modify `RRF_K` in `search.ts` (higher values flatten rank differences).
- **Add a config option**: Add the field to `JournalRagConfig` in `types.ts`, parse it in `config.ts`.
- **Expose a new MCP tool**: Register it in `server.ts` using `server.registerTool()` with a zod schema.
