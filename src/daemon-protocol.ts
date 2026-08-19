import { createHash, randomUUID } from "node:crypto";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

export const DAEMON_PROTOCOL_VERSION = 1;

export type DaemonMethod =
  | "ensure_workspace"
  | "search_journal"
  | "get_entry"
  | "list_entries"
  | "search_regex"
  | "write_entry";

export interface DaemonRequest {
  id: string;
  method: DaemonMethod;
  configPath: string;
  params: Record<string, unknown>;
}

export interface DaemonResponse {
  id: string;
  result?: unknown;
  error?: string;
}

function userKey(): string {
  return createHash("sha256").update(homedir()).digest("hex").slice(0, 12);
}

export function daemonEndpoint(): string {
  const name = `journal-rag-v${DAEMON_PROTOCOL_VERSION}-${userKey()}`;
  return process.platform === "win32"
    ? `\\\\.\\pipe\\${name}`
    : join(tmpdir(), `${name}.sock`);
}

export function daemonStartupLockTarget(): string {
  return join(tmpdir(), `journal-rag-${userKey()}-daemon-start`);
}

export function createDaemonRequest(
  method: DaemonMethod,
  configPath: string,
  params: Record<string, unknown> = {},
): DaemonRequest {
  return {
    id: randomUUID(),
    method,
    configPath,
    params,
  };
}
