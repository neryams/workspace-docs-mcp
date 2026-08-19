#!/usr/bin/env node
import {
  chmodSync,
  existsSync,
  unlinkSync,
} from "node:fs";
import {
  connect,
  createServer,
  type Socket,
} from "node:net";
import {
  daemonEndpoint,
  type DaemonRequest,
  type DaemonResponse,
} from "./daemon-protocol.js";
import { LOG_DIR, writeError, writeLog } from "./logger.js";
import { dispatchWorkspaceRequest } from "./workspace-runtime.js";

const MAX_REQUEST_BYTES = 64 * 1024 * 1024;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function probeEndpoint(endpoint: string): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = connect(endpoint);
    const finish = (connected: boolean) => {
      socket.destroy();
      resolve(connected);
    };
    socket.once("connect", () => finish(true));
    socket.once("error", () => finish(false));
  });
}

async function prepareEndpoint(endpoint: string): Promise<boolean> {
  if (process.platform === "win32" || !existsSync(endpoint)) {
    return true;
  }

  for (let attempt = 0; attempt < 5; attempt++) {
    if (await probeEndpoint(endpoint)) {
      return false;
    }
    await delay(100);
  }

  unlinkSync(endpoint);
  return true;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function handleRequest(request: DaemonRequest): Promise<DaemonResponse> {
  try {
    if (
      typeof request.id !== "string" ||
      typeof request.method !== "string" ||
      typeof request.configPath !== "string" ||
      !request.params ||
      typeof request.params !== "object"
    ) {
      throw new Error("Invalid daemon request");
    }
    writeLog("daemon", "info", "Handling request", {
      requestId: request.id,
      method: request.method,
      configPath: request.configPath,
    });
    const result = await dispatchWorkspaceRequest(
      request.method,
      request.configPath,
      request.params,
    );
    return { id: request.id, result };
  } catch (error) {
    writeError("daemon", "Request failed", error, {
      requestId: request?.id,
      method: request?.method,
      configPath: request?.configPath,
    });
    return {
      id: typeof request?.id === "string" ? request.id : "",
      error: errorMessage(error),
    };
  }
}

function handleConnection(socket: Socket): void {
  let requestText = "";
  let requestBytes = 0;
  let handled = false;

  const fail = (message: string) => {
    if (handled) {
      return;
    }
    handled = true;
    socket.end(`${JSON.stringify({ id: "", error: message })}\n`);
  };

  socket.on("data", (data: Buffer) => {
    if (handled) {
      return;
    }
    requestBytes += data.length;
    if (requestBytes > MAX_REQUEST_BYTES) {
      fail("Daemon request exceeded size limit");
      return;
    }
    requestText += data.toString("utf8");
    const newline = requestText.indexOf("\n");
    if (newline < 0) {
      return;
    }
    handled = true;
    void (async () => {
      try {
        const request = JSON.parse(
          requestText.slice(0, newline),
        ) as DaemonRequest;
        const response = await handleRequest(request);
        socket.end(`${JSON.stringify(response)}\n`);
      } catch (error) {
        socket.end(
          `${JSON.stringify({ id: "", error: errorMessage(error) })}\n`,
        );
      }
    })();
  });
  socket.once("error", () => {
    socket.destroy();
  });
}

async function main(): Promise<void> {
  const endpoint = daemonEndpoint();
  if (!(await prepareEndpoint(endpoint))) {
    return;
  }

  const server = createServer(handleConnection);
  server.on("error", (error: NodeJS.ErrnoException) => {
    if (error.code === "EADDRINUSE") {
      writeLog("daemon", "info", "Daemon already running", { endpoint });
      process.exit(0);
    }
    console.error("[journal-rag-daemon] Server error:", error);
    writeError("daemon", "Daemon server error", error, { endpoint });
    process.exit(1);
  });

  const cleanup = () => {
    server.close();
    if (process.platform !== "win32" && existsSync(endpoint)) {
      unlinkSync(endpoint);
    }
  };
  process.once("SIGINT", cleanup);
  process.once("SIGTERM", cleanup);
  process.once("exit", () => {
    if (process.platform !== "win32" && existsSync(endpoint)) {
      unlinkSync(endpoint);
    }
  });

  server.listen(endpoint, () => {
    if (process.platform !== "win32") {
      chmodSync(endpoint, 0o600);
    }
    console.error(`[journal-rag-daemon] Listening on ${endpoint}`);
    console.error(`[journal-rag-daemon] Logs: ${LOG_DIR}`);
    writeLog("daemon", "info", "Daemon listening", {
      endpoint,
      logDir: LOG_DIR,
    });
  });
}

main().catch((error: unknown) => {
  console.error("[journal-rag-daemon] Fatal error:", error);
  writeError("daemon", "Daemon fatal error", error);
  process.exit(1);
});
