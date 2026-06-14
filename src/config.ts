import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { JournalRagConfig } from "./types.js";

export const CONFIG_FILENAME = "journal-rag.config.json";

export function findWorkspaceRoot(startDir = process.cwd()): string {
  const envRoot = process.env.JOURNAL_RAG_WORKSPACE?.trim();
  if (envRoot) {
    const resolved = resolve(envRoot);
    if (existsSync(join(resolved, CONFIG_FILENAME))) {
      return resolved;
    }
    throw new Error(
      `JOURNAL_RAG_WORKSPACE=${envRoot} has no ${CONFIG_FILENAME}`,
    );
  }

  let dir = resolve(startDir);
  for (;;) {
    if (existsSync(join(dir, CONFIG_FILENAME))) {
      return dir;
    }
    const parent = dirname(dir);
    if (parent === dir) {
      throw new Error(
        `No ${CONFIG_FILENAME} found walking up from ${startDir}. Set JOURNAL_RAG_WORKSPACE or run from a repo root.`,
      );
    }
    dir = parent;
  }
}

export function loadConfig(workspaceRoot: string): JournalRagConfig {
  const raw = readFileSync(join(workspaceRoot, CONFIG_FILENAME), "utf8");
  const parsed = JSON.parse(raw) as Partial<JournalRagConfig>;
  if (!parsed.sources?.length) {
    throw new Error(`${CONFIG_FILENAME} must define a non-empty "sources" array`);
  }
  return {
    sources: parsed.sources,
    cachePath: parsed.cachePath ?? ".journal-rag/index.json",
    embeddingModel: parsed.embeddingModel,
  };
}

export function resolveSourceDirs(
  workspaceRoot: string,
  config: JournalRagConfig,
): string[] {
  return config.sources.map((source) => resolve(workspaceRoot, source));
}

export function resolveCachePath(
  workspaceRoot: string,
  config: JournalRagConfig,
): string {
  return resolve(workspaceRoot, config.cachePath ?? ".journal-rag/index.json");
}
