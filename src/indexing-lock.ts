import { tmpdir } from "node:os";
import { join } from "node:path";
import { lock } from "proper-lockfile";
import { writeLog } from "./logger.js";

const LOCK_TARGET = join(tmpdir(), "journal-rag-vector-index");

export async function withIndexingLock<T>(
  task: () => Promise<T>,
): Promise<T> {
  const waitStartedAt = Date.now();
  const release = await lock(LOCK_TARGET, {
    realpath: false,
    stale: 60_000,
    update: 20_000,
    retries: {
      forever: true,
      factor: 1,
      minTimeout: 1_000,
      maxTimeout: 1_000,
      randomize: true,
    },
  });
  const waitMs = Date.now() - waitStartedAt;
  if (waitMs >= 100) {
    writeLog("daemon", "info", "Acquired vector indexing lock", {
      waitMs,
      lockTarget: LOCK_TARGET,
    });
  }

  try {
    return await task();
  } finally {
    await release();
  }
}
