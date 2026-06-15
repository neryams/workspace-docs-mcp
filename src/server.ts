#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { findWorkspaceRoot, loadConfig, resolveCachePath } from "./config.js";
import { buildVectorIndex, type VectorIndex } from "./embeddings.js";
import { loadJournalIndex, readEntry, writeEntry } from "./index.js";
import { searchHybrid, searchJournal, searchRegex } from "./search.js";

async function main(): Promise<void> {
  console.error("[journal-rag] Starting...");
  const workspaceRoot = findWorkspaceRoot();
  console.error(`[journal-rag] Workspace root: ${workspaceRoot}`);
  const config = loadConfig(workspaceRoot);
  console.error(`[journal-rag] Config loaded (sources: ${config.sources.join(", ")}, model: ${config.embeddingModel ?? "default"})`);
  let index = loadJournalIndex(workspaceRoot, config);
  console.error(`[journal-rag] Journal index loaded: ${index.entries.length} entries, ${index.chunks.length} chunks`);
  const cachePath = resolveCachePath(workspaceRoot, config);

  let vectorIndex: VectorIndex | null = null;

  const server = new McpServer({
    name: "journal-rag",
    version: "0.1.0",
  });

  server.registerTool(
    "search_journal",
    {
      description:
        "Hybrid BM25 + vector semantic search over team markdown journals (heading-chunked). Returns ranked hits with file#heading citations.",
      inputSchema: z.object({
        query: z.string().describe("Search terms or phrase"),
        k: z.number().int().min(1).max(50).optional().default(8),
      }),
    },
    async ({ query, k }) => {
      const limit = k ?? 8;
      const hits = vectorIndex
        ? await searchHybrid(index.chunks, vectorIndex, query, limit)
        : searchJournal(index.chunks, query, limit);
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({ query, hits }, null, 2),
          },
        ],
      };
    },
  );

  server.registerTool(
    "get_entry",
    {
      description: "Return the full markdown of one journal file by relative path or filename.",
      inputSchema: z.object({
        path: z.string().describe("Relative path (e.g. docs/journal/2026-04-21_topic.md) or filename"),
      }),
    },
    async ({ path }) => {
      const content = readEntry(workspaceRoot, config, path);
      return {
        content: [{ type: "text" as const, text: content }],
      };
    },
  );

  server.registerTool(
    "list_entries",
    {
      description:
        "List journal files with title, date, and section headings. Optional filter substring on path/title.",
      inputSchema: z.object({
        filter: z.string().optional().describe("Case-insensitive substring filter"),
      }),
    },
    async ({ filter }) => {
      const needle = filter?.toLowerCase();
      const entries = needle
        ? index.entries.filter(
            (e) =>
              e.file.toLowerCase().includes(needle) ||
              e.title.toLowerCase().includes(needle),
          )
        : index.entries;
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({ count: entries.length, entries }, null, 2),
          },
        ],
      };
    },
  );

  server.registerTool(
    "search_regex",
    {
      description:
        "Regex search across journal chunks (escape hatch for symbols, paths, exact identifiers).",
      inputSchema: z.object({
        pattern: z.string().describe("JavaScript regex pattern (case-insensitive)"),
        k: z.number().int().min(1).max(50).optional().default(20),
      }),
    },
    async ({ pattern, k }) => {
      const hits = searchRegex(index.chunks, pattern, k ?? 20);
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({ pattern, hits }, null, 2),
          },
        ],
      };
    },
  );

  server.registerTool(
    "write_entry",
    {
      description:
        "Create a new markdown journal entry. Filename is generated from today's date and the title slug. Returns the file path. The vector index is incrementally updated (only new chunks are embedded).",
      inputSchema: z.object({
        title: z.string().describe("Entry title (used for H1 heading and filename slug)"),
        content: z.string().describe("Full markdown content of the journal entry (should start with # Title)"),
      }),
    },
    async ({ title, content }) => {
      const { file } = writeEntry(workspaceRoot, config, title, content);

      index = loadJournalIndex(workspaceRoot, config, { forceRebuild: true });

      if (vectorIndex) {
        try {
          vectorIndex = await buildVectorIndex(index.chunks, cachePath, {
            model: config.embeddingModel,
          });
        } catch (err) {
          console.error("[journal-rag] Incremental vector index update failed:", err);
        }
      }

      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({ created: file }, null, 2),
          },
        ],
      };
    },
  );

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("[journal-rag] MCP server running on stdio");

  // Build vector index in the background after transport is connected,
  // so the server responds to the MCP handshake immediately.
  (async () => {
    try {
      const t0 = Date.now();
      console.error("[journal-rag] Building vector index...");
      vectorIndex = await buildVectorIndex(index.chunks, cachePath, {
        model: config.embeddingModel,
      });
      console.error(`[journal-rag] Vector index ready (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
    } catch (err) {
      console.error("[journal-rag] Vector index build failed, falling back to BM25-only:", err);
    }
  })();
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
