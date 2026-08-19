import { tmpdir } from "node:os";
import { join } from "node:path";
import { lock } from "proper-lockfile";

const LOCK_TARGET = join(tmpdir(), "journal-rag-vector-index");

export async function withIndexingLock<T>(
  task: () => Promise<T>,
): Promise<T> {
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

  try {
    return await task();
  } finally {
    await release();
  }
}
