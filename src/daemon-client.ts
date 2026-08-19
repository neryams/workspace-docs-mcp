import { existsSync } from "node:fs";
import { connect } from "node:net";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { lock } from "proper-lockfile";
import {
  createDaemonRequest,
  daemonEndpoint,
  daemonStartupLockTarget,
  type DaemonMethod,
  type DaemonResponse,
} from "./daemon-protocol.js";
import { writeError, writeLog } from "./logger.js";

const STARTUP_TIMEOUT_MS = 30_000;
const REQUEST_TIMEOUT_MS = 10 * 60_000;
const MAX_RESPONSE_BYTES = 64 * 1024 * 1024;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function canConnect(): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = connect(daemonEndpoint());
    const finish = (connected: boolean) => {
      socket.destroy();
      resolve(connected);
    };
    socket.once("connect", () => finish(true));
    socket.once("error", () => finish(false));
  });
}

function spawnDaemon(): void {
  const compiledPath = fileURLToPath(new URL("./daemon.js", import.meta.url));
  const sourcePath = fileURLToPath(new URL("./daemon.ts", import.meta.url));
  const args = existsSync(compiledPath)
    ? [compiledPath]
    : ["--experimental-strip-types", sourcePath];

  const child = spawn(process.execPath, args, {
    detached: true,
    stdio: "ignore",
    windowsHide: true,
  });
  writeLog("mcp", "info", "Spawned daemon process", {
    daemonPid: child.pid,
    executable: process.execPath,
    script: args.at(-1),
  });
  child.once("error", (error) => {
    writeError("mcp", "Failed to spawn daemon process", error);
  });
  child.unref();
}

async function ensureDaemon(): Promise<void> {
  if (await canConnect()) {
    return;
  }

  const release = await lock(daemonStartupLockTarget(), {
    realpath: false,
    stale: 30_000,
    update: 10_000,
    retries: {
      forever: true,
      factor: 1,
      minTimeout: 250,
      maxTimeout: 250,
      randomize: true,
    },
  });

  try {
    if (await canConnect()) {
      return;
    }

    spawnDaemon();
    const deadline = Date.now() + STARTUP_TIMEOUT_MS;
    while (Date.now() < deadline) {
      await delay(200);
      if (await canConnect()) {
        return;
      }
    }
    writeLog("mcp", "error", "Timed out waiting for daemon startup", {
      endpoint: daemonEndpoint(),
    });
    throw new Error("Timed out waiting for journal-rag daemon to start");
  } finally {
    await release();
  }
}

function sendRequest(
  method: DaemonMethod,
  configPath: string,
  params: Record<string, unknown>,
): Promise<unknown> {
  const request = createDaemonRequest(method, configPath, params);

  return new Promise((resolve, reject) => {
    const socket = connect(daemonEndpoint());
    let responseText = "";
    let responseBytes = 0;
    let settled = false;

    const finish = (error?: Error, result?: unknown) => {
      if (settled) {
        return;
      }
      settled = true;
      socket.destroy();
      if (error) {
        reject(error);
      } else {
        resolve(result);
      }
    };

    socket.setTimeout(REQUEST_TIMEOUT_MS);
    socket.once("timeout", () =>
      finish(new Error(`Daemon request timed out: ${method}`)),
    );
    socket.once("error", (error) => finish(error));
    socket.on("data", (data: Buffer) => {
      responseBytes += data.length;
      if (responseBytes > MAX_RESPONSE_BYTES) {
        finish(new Error("Daemon response exceeded size limit"));
        return;
      }
      responseText += data.toString("utf8");
      const newline = responseText.indexOf("\n");
      if (newline < 0) {
        return;
      }
      try {
        const response = JSON.parse(
          responseText.slice(0, newline),
        ) as DaemonResponse;
        if (response.id !== request.id) {
          finish(new Error("Daemon response ID did not match request"));
        } else if (response.error) {
          finish(new Error(response.error));
        } else {
          finish(undefined, response.result);
        }
      } catch (error) {
        finish(
          error instanceof Error
            ? error
            : new Error("Invalid daemon response"),
        );
      }
    });
    socket.once("connect", () => {
      socket.write(`${JSON.stringify(request)}\n`);
    });
  });
}

export async function callDaemon(
  method: DaemonMethod,
  configPath: string,
  params: Record<string, unknown> = {},
): Promise<unknown> {
  await ensureDaemon();
  try {
    return await sendRequest(method, configPath, params);
  } catch (error) {
    if (!(error instanceof Error) || !("code" in error)) {
      writeError("mcp", `Daemon request failed: ${method}`, error, {
        configPath,
      });
      throw error;
    }
    writeError("mcp", `Daemon connection failed; retrying: ${method}`, error, {
      configPath,
    });
    await ensureDaemon();
    return sendRequest(method, configPath, params);
  }
}
