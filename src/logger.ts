import {
  appendFileSync,
  mkdirSync,
  readdirSync,
  statSync,
  unlinkSync,
} from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const LOG_RETENTION_DAYS = 14;

export type LogComponent = "daemon" | "mcp";
export type LogLevel = "info" | "warn" | "error";

function defaultLogDir(): string {
  const override = process.env.JOURNAL_RAG_LOG_DIR?.trim();
  if (override) {
    return override;
  }
  if (process.platform === "win32") {
    return join(
      process.env.LOCALAPPDATA || join(homedir(), "AppData", "Local"),
      "journal-rag",
      "logs",
    );
  }
  if (process.platform === "darwin") {
    return join(homedir(), "Library", "Logs", "journal-rag");
  }
  return join(
    process.env.XDG_STATE_HOME || join(homedir(), ".local", "state"),
    "journal-rag",
    "logs",
  );
}

export const LOG_DIR = defaultLogDir();

function dateStamp(date = new Date()): string {
  return date.toISOString().slice(0, 10);
}

export function logPath(component: LogComponent): string {
  return join(LOG_DIR, `${component}-${dateStamp()}.jsonl`);
}

function serializeError(error: unknown): Record<string, unknown> {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
      cause:
        error.cause instanceof Error
          ? {
              name: error.cause.name,
              message: error.cause.message,
              stack: error.cause.stack,
            }
          : error.cause === undefined
            ? undefined
            : String(error.cause),
    };
  }
  return { value: String(error) };
}

let initialized = false;

function initializeLogDir(): void {
  if (initialized) {
    return;
  }
  initialized = true;
  mkdirSync(LOG_DIR, { recursive: true });

  const cutoff = Date.now() - LOG_RETENTION_DAYS * 24 * 60 * 60 * 1_000;
  for (const name of readdirSync(LOG_DIR)) {
    if (!/^(daemon|mcp)-\d{4}-\d{2}-\d{2}\.jsonl$/.test(name)) {
      continue;
    }
    const path = join(LOG_DIR, name);
    try {
      if (statSync(path).mtimeMs < cutoff) {
        unlinkSync(path);
      }
    } catch {
      // Logging and cleanup must never prevent the server from starting.
    }
  }
}

export function writeLog(
  component: LogComponent,
  level: LogLevel,
  message: string,
  details?: Record<string, unknown>,
): void {
  try {
    initializeLogDir();
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      component,
      pid: process.pid,
      message,
      ...details,
    };
    appendFileSync(logPath(component), `${JSON.stringify(entry)}\n`, "utf8");
  } catch {
    // Diagnostics are best-effort and must not break MCP operations.
  }
}

export function writeError(
  component: LogComponent,
  message: string,
  error: unknown,
  details?: Record<string, unknown>,
): void {
  writeLog(component, "error", message, {
    ...details,
    error: serializeError(error),
  });
}
