# journal-rag

Source-control-friendly **keyword** retrieval over team markdown journals. No vector DB, no embeddings: heading-chunked **BM25** + regex, index built in memory on startup (optional gitignored JSON cache).

Each consuming repo commits `journal-rag.config.json` and markdown under `docs/journal/` (or other configured folders). This package is the shared engine.

## Per-repo config

Create `journal-rag.config.json` at the repo root:

```json
{
  "sources": ["docs/journal"],
  "cachePath": ".journal-rag/index.json"
}
```

Add to `.gitignore`:

```
.journal-rag/
```

## Build & install (once per machine)

```bash
cd c:/repos/journal-rag
npm install          # runs prepare → build
npm link             # puts journal + journal-mcp on your PATH
```

`npm link` registers two global commands:

| Command | What it runs |
|---------|----------------|
| `journal` | CLI (`search`, `list`, `get`, …) |
| `journal-mcp` | MCP stdio server (for editor config) |

Re-run `npm run build` (or `npm link` again) after pulling server changes.
Alternative to link: `npm install -g .` from this repo (same effect).

## CLI (any teammate)

From a repo root with config:

```bash
node c:/repos/journal-rag/dist/cli.js search "HttpFacade singleton"
node c:/repos/journal-rag/dist/cli.js list --filter dialog
node c:/repos/journal-rag/dist/cli.js get docs/journal/2026-04-21_vapp-http-facade-and-singleton-sweep.md
node c:/repos/journal-rag/dist/cli.js index --rebuild
```

After `npm link` in this repo, `journal search "..."` works globally.

Set `JOURNAL_RAG_WORKSPACE` to an absolute repo root only when you must run the CLI from a subdirectory.

## MCP tools

| Tool | Purpose |
|------|---------|
| `search_journal` | BM25 + substring fallback (`query`, `k`) |
| `get_entry` | Full file by path or filename |
| `list_entries` | Browse metadata (`filter` optional) |
| `search_regex` | Exact / path / symbol lookup |

## Editor setup

Use **stdio** — spawn Node with `dist/server.js`.

### Put MCP config in the workspace, not your user profile

The server resolves `journal-rag.config.json` by walking up from its **working directory**. That file lives at each **consuming repo's root** (next to `docs/journal/`), not in `journal-rag` itself.

If you add the server to a **global / user-level** editor profile, the spawn cwd is usually wrong (home dir, editor install dir, last random folder, etc.) and the server cannot find config — even if you hardcode `"cwd": "C:/repos/my-repo"`, that breaks the moment you open a second repo workspace.

**Do this instead:** commit workspace-level MCP config **inside each repo** that has journals. Teammates run `npm link` once (see above) so `journal-mcp` is on PATH — no machine-specific paths in the committed JSON.

### Cursor

`.cursor/mcp.json` at the **repo root** (e.g. `my-repo/.cursor/mcp.json`) — safe to commit:

```json
{
  "mcpServers": {
    "journal": {
      "command": "journal-mcp",
      "cwd": "${workspaceFolder}",
      "env": {
        "JOURNAL_RAG_WORKSPACE": "${workspaceFolder}"
      }
    }
  }
}
```

`${workspaceFolder}` resolves to the repo you opened. `journal-mcp` comes from `npm link` in the `journal-rag` repo.

### VS Code (Copilot agent mode)

Same idea: `.vscode/mcp.json` in the repo, not User settings:

```json
{
  "servers": {
    "journal": {
      "type": "stdio",
      "command": "journal-mcp",
      "cwd": "${workspaceFolder}"
    }
  }
}
```

### JetBrains AI Assistant / Junie

Configure MCP at **project** scope (`.idea` / project settings), not the IDE default profile. Open the repo as the project root. Command: `journal-mcp` (after `npm link`).

### If `journal-mcp` is not found

Ensure npm's global bin dir is on your PATH (`npm bin -g`). On Windows that is usually `%APPDATA%\\npm`. Then re-run `npm link` from `journal-rag`. Fallback for a single machine only: `"command": "node", "args": ["<absolute-path>/journal-rag/dist/server.js"]`.

### Fallback

If an editor cannot set cwd per workspace, set env `JOURNAL_RAG_WORKSPACE` to the absolute path of the consuming repo root in that workspace's MCP config.

## Design notes

- Corpus is small (~tens of files); BM25 over heading chunks matches how journals are written.
- Index cache is optional and gitignored; markdown in git is the source of truth.
- Hybrid embeddings can be added later without changing the MCP tool surface.
