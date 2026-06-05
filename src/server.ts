#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { findWorkspaceRoot, loadConfig } from "./config.js";
import { loadJournalIndex, readEntry } from "./index.js";
import { searchJournal, searchRegex } from "./search.js";

async function main(): Promise<void> {
  const workspaceRoot = findWorkspaceRoot();
  const config = loadConfig(workspaceRoot);
  const index = loadJournalIndex(workspaceRoot, config);

  const server = new McpServer({
    name: "journal-rag",
    version: "0.1.0",
  });

  server.registerTool(
    "search_journal",
    {
      description:
        "BM25 keyword search over team markdown journals (heading-chunked). Returns ranked hits with file#heading citations.",
      inputSchema: z.object({
        query: z.string().describe("Search terms or phrase"),
        k: z.number().int().min(1).max(50).optional().default(8),
      }),
    },
    async ({ query, k }) => {
      const hits = searchJournal(index.chunks, query, k ?? 8);
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

  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
