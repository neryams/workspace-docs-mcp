#!/usr/bin/env node
import { join } from "node:path";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { callDaemon } from "./daemon-client.js";
import { CONFIG_FILENAME, findWorkspaceRoot } from "./config.js";
import { LOG_DIR, writeError, writeLog } from "./logger.js";

function jsonText(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

async function main(): Promise<void> {
  console.error("[journal-rag] Starting...");
  const workspaceRoot = findWorkspaceRoot();
  const configPath = join(workspaceRoot, CONFIG_FILENAME);
  console.error(`[journal-rag] Workspace root: ${workspaceRoot}`);
  console.error(`[journal-rag] Logs: ${LOG_DIR}`);
  writeLog("mcp", "info", "MCP proxy starting", {
    workspaceRoot,
    configPath,
  });

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
      const result = await callDaemon("search_journal", configPath, {
        query,
        k: k ?? 8,
      });
      return {
        content: [
          {
            type: "text" as const,
            text: jsonText(result),
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
      const content = await callDaemon("get_entry", configPath, { path });
      return {
        content: [{ type: "text" as const, text: String(content) }],
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
      const result = await callDaemon("list_entries", configPath, { filter });
      return {
        content: [
          {
            type: "text" as const,
            text: jsonText(result),
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
      const result = await callDaemon("search_regex", configPath, {
        pattern,
        k: k ?? 20,
      });
      return {
        content: [
          {
            type: "text" as const,
            text: jsonText(result),
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
      const result = await callDaemon("write_entry", configPath, {
        title,
        content,
      });
      return {
        content: [
          {
            type: "text" as const,
            text: jsonText(result),
          },
        ],
      };
    },
  );

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("[journal-rag] MCP server running on stdio");
  writeLog("mcp", "info", "MCP proxy connected", { configPath });

  void callDaemon("ensure_workspace", configPath).catch((error: unknown) => {
    console.error("[journal-rag] Daemon initialization failed:", error);
    writeError("mcp", "Daemon initialization failed", error, { configPath });
  });
}

main().catch((err: unknown) => {
  console.error(err);
  writeError("mcp", "MCP proxy fatal error", err);
  process.exit(1);
});
