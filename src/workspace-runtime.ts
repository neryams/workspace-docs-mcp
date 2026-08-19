import { randomUUID } from "node:crypto";
import { realpathSync } from "node:fs";
import { basename, dirname } from "node:path";
import {
  loadConfig,
  resolveCachePath,
  CONFIG_FILENAME,
} from "./config.js";
import { buildVectorIndex, type VectorIndex } from "./embeddings.js";
import {
  loadJournalIndex,
  readEntry,
  writeEntry,
} from "./index.js";
import {
  searchHybrid,
  searchJournal,
  searchRegex,
} from "./search.js";
import type {
  JournalIndex,
  JournalRagConfig,
} from "./types.js";
import type { DaemonMethod } from "./daemon-protocol.js";
import { writeError, writeLog } from "./logger.js";

class WorkspaceRuntime {
  readonly id = randomUUID();
  readonly workspaceRoot: string;
  readonly configPath: string;

  private readonly config: JournalRagConfig;
  private readonly cachePath: string;
  private index: JournalIndex;
  private vectorIndex: VectorIndex | null = null;
  private vectorBuild: Promise<void> | null = null;

  constructor(configPath: string) {
    this.configPath = configPath;
    this.workspaceRoot = dirname(configPath);
    this.config = loadConfig(this.workspaceRoot);
    this.cachePath = resolveCachePath(this.workspaceRoot, this.config);
    this.index = loadJournalIndex(this.workspaceRoot, this.config);
    writeLog("daemon", "info", "Workspace runtime created", {
      runtimeId: this.id,
      configPath: this.configPath,
      workspaceRoot: this.workspaceRoot,
      entries: this.index.entries.length,
      chunks: this.index.chunks.length,
      model: this.config.embeddingModel ?? "default",
    });
    void this.startVectorBuild();
  }

  private startVectorBuild(): Promise<void> {
    if (this.vectorBuild) {
      return this.vectorBuild;
    }

    const sourceIndex = this.index;
    const startedAt = Date.now();
    writeLog("daemon", "info", "Vector index build started", {
      runtimeId: this.id,
      configPath: this.configPath,
      chunks: sourceIndex.chunks.length,
    });
    this.vectorBuild = (async () => {
      try {
        const built = await buildVectorIndex(
          sourceIndex.chunks,
          this.cachePath,
          { model: this.config.embeddingModel },
        );
        if (this.index === sourceIndex) {
          this.vectorIndex = built;
          writeLog("daemon", "info", "Vector index build completed", {
            runtimeId: this.id,
            configPath: this.configPath,
            chunks: sourceIndex.chunks.length,
            durationMs: Date.now() - startedAt,
          });
        }
      } catch (error) {
        console.error(
          `[journal-rag-daemon] Vector index build failed for ${this.configPath}:`,
          error,
        );
        writeError("daemon", "Vector index build failed", error, {
          runtimeId: this.id,
          configPath: this.configPath,
          chunks: sourceIndex.chunks.length,
          durationMs: Date.now() - startedAt,
        });
      } finally {
        this.vectorBuild = null;
        if (this.index !== sourceIndex) {
          void this.startVectorBuild();
        }
      }
    })();
    return this.vectorBuild;
  }

  ensureWorkspace(): Record<string, unknown> {
    return {
      runtimeId: this.id,
      daemonPid: process.pid,
      configPath: this.configPath,
      workspaceRoot: this.workspaceRoot,
      entries: this.index.entries.length,
      chunks: this.index.chunks.length,
    };
  }

  async searchJournal(query: string, k: number): Promise<Record<string, unknown>> {
    const hits = this.vectorIndex
      ? await searchHybrid(this.index.chunks, this.vectorIndex, query, k)
      : searchJournal(this.index.chunks, query, k);
    return { query, hits };
  }

  getEntry(path: string): string {
    return readEntry(this.workspaceRoot, this.config, path);
  }

  listEntries(filter?: string): Record<string, unknown> {
    const needle = filter?.toLowerCase();
    const entries = needle
      ? this.index.entries.filter(
          (entry) =>
            entry.file.toLowerCase().includes(needle) ||
            entry.title.toLowerCase().includes(needle),
        )
      : this.index.entries;
    return { count: entries.length, entries };
  }

  searchRegex(pattern: string, k: number): Record<string, unknown> {
    return {
      pattern,
      hits: searchRegex(this.index.chunks, pattern, k),
    };
  }

  writeEntry(title: string, content: string): Record<string, unknown> {
    const { file } = writeEntry(
      this.workspaceRoot,
      this.config,
      title,
      content,
    );
    this.index = loadJournalIndex(this.workspaceRoot, this.config, {
      forceRebuild: true,
    });
    this.vectorIndex = null;
    void this.startVectorBuild();
    return { created: file };
  }
}

const runtimes = new Map<string, Promise<WorkspaceRuntime>>();

function canonicalConfigPath(configPath: string): string {
  const canonical = realpathSync.native(configPath);
  if (basename(canonical).toLowerCase() !== CONFIG_FILENAME.toLowerCase()) {
    throw new Error(`Expected a path to ${CONFIG_FILENAME}: ${configPath}`);
  }
  return canonical;
}

async function getRuntime(configPath: string): Promise<WorkspaceRuntime> {
  const canonical = canonicalConfigPath(configPath);
  const key = process.platform === "win32"
    ? canonical.toLowerCase()
    : canonical;
  let runtime = runtimes.get(key);
  if (!runtime) {
    runtime = Promise.resolve(new WorkspaceRuntime(canonical));
    runtimes.set(key, runtime);
    runtime.catch(() => runtimes.delete(key));
  }
  return runtime;
}

function requiredString(
  params: Record<string, unknown>,
  name: string,
): string {
  const value = params[name];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Expected non-empty string parameter: ${name}`);
  }
  return value;
}

function optionalString(
  params: Record<string, unknown>,
  name: string,
): string | undefined {
  const value = params[name];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new Error(`Expected string parameter: ${name}`);
  }
  return value;
}

function boundedInteger(
  params: Record<string, unknown>,
  name: string,
  fallback: number,
  max: number,
): number {
  const value = params[name] ?? fallback;
  if (!Number.isInteger(value) || (value as number) < 1 || (value as number) > max) {
    throw new Error(`Expected ${name} to be an integer from 1 to ${max}`);
  }
  return value as number;
}

export async function dispatchWorkspaceRequest(
  method: DaemonMethod,
  configPath: string,
  params: Record<string, unknown>,
): Promise<unknown> {
  const runtime = await getRuntime(configPath);
  switch (method) {
    case "ensure_workspace":
      return runtime.ensureWorkspace();
    case "search_journal":
      return runtime.searchJournal(
        requiredString(params, "query"),
        boundedInteger(params, "k", 8, 50),
      );
    case "get_entry":
      return runtime.getEntry(requiredString(params, "path"));
    case "list_entries":
      return runtime.listEntries(optionalString(params, "filter"));
    case "search_regex":
      return runtime.searchRegex(
        requiredString(params, "pattern"),
        boundedInteger(params, "k", 20, 50),
      );
    case "write_entry":
      return runtime.writeEntry(
        requiredString(params, "title"),
        requiredString(params, "content"),
      );
  }
}
