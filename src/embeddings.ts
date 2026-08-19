import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { withIndexingLock } from "./indexing-lock.js";
import { writeError, writeLog } from "./logger.js";
import type { Chunk } from "./types.js";

const VECTOR_CACHE_VERSION = 2;
const DEFAULT_MODEL = "onnx-community/Qwen3-Embedding-0.6B-ONNX";
const EMBEDDING_BATCH_SIZE = 1;
const VECTOR_CHECKPOINT_INTERVAL = 100;
const QUERY_INSTRUCT =
  "Given a search query, retrieve relevant journal entries that answer the query";

interface VectorCacheEntry {
  key: string;
  embedding: number[];
}

interface VectorCacheFile {
  version: number;
  model: string;
  entries: VectorCacheEntry[];
}

export interface VectorIndex {
  model: string;
  keys: string[];
  embeddings: Float32Array[];
}

function chunkKey(chunk: Chunk): string {
  return `${chunk.file}\0${chunk.heading}`;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

type EmbedOptions = { pooling: string; normalize: boolean };
type Embedder = (
  texts: string[],
  options?: EmbedOptions,
) => Promise<{ tolist: () => number[][] }>;

const pipelineInstances = new Map<string, Promise<Embedder>>();
let inferenceTail: Promise<void> = Promise.resolve();

async function serializeInference<T>(task: () => Promise<T>): Promise<T> {
  const previous = inferenceTail;
  let release!: () => void;
  inferenceTail = new Promise<void>((resolve) => {
    release = resolve;
  });
  await previous;
  try {
    return await task();
  } finally {
    release();
  }
}

function embeddingDevice(): "dml" | "coreml" | "cpu" {
  if (process.platform === "win32") {
    return "dml";
  }
  if (process.platform === "darwin") {
    return "coreml";
  }
  return "cpu";
}

async function getEmbedder(model: string): Promise<Embedder> {
  const existing = pipelineInstances.get(model);
  if (existing) {
    return existing;
  }

  const loading: Promise<Embedder> = serializeInference<Embedder>(async () => {
    const device = embeddingDevice();
    const startedAt = Date.now();
    writeLog("daemon", "info", "Embedding model load started", {
      model,
      device,
    });
    const { pipeline } = await import("@huggingface/transformers");
    const extractor = await pipeline("feature-extraction", model, {
      dtype: "q8",
      device,
      session_options:
        device === "dml"
          ? {
              enableMemPattern: false,
              executionMode: "sequential",
            }
          : undefined,
    });
    const memory = process.memoryUsage();
    writeLog("daemon", "info", "Embedding model load completed", {
      model,
      device,
      durationMs: Date.now() - startedAt,
      rssMiB: Math.round(memory.rss / 1024 / 1024),
      heapUsedMiB: Math.round(memory.heapUsed / 1024 / 1024),
      externalMiB: Math.round(memory.external / 1024 / 1024),
    });
    const embedder: Embedder = async (
      texts: string[],
      options?: EmbedOptions,
    ) => {
      return serializeInference(async () => {
        const result = await extractor(
          texts,
          options as Record<string, unknown>,
        );
        return result as unknown as { tolist: () => number[][] };
      });
    };
    return embedder;
  });
  pipelineInstances.set(model, loading);
  try {
    return await loading;
  } catch (error) {
    pipelineInstances.delete(model);
    writeError("daemon", "Embedding model load failed", error, {
      model,
      device: embeddingDevice(),
    });
    throw error;
  }
}

async function embed(
  texts: string[],
  model: string,
  options?: { queryInstruct?: string },
): Promise<number[][]> {
  const extractor = await getEmbedder(model);
  const input = options?.queryInstruct
    ? texts.map((t) => `Instruct: ${options.queryInstruct}\nQuery:${t}`)
    : texts;
  const output = await extractor(input, {
    pooling: "last_token",
    normalize: true,
  });
  return output.tolist();
}

function vectorCachePath(bm25CachePath: string): string {
  const cacheDir = dirname(bm25CachePath);
  return join(cacheDir, "vectors.json");
}

function readVectorCache(path: string, model: string): Map<string, number[]> | null {
  if (!existsSync(path)) {
    return null;
  }
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as VectorCacheFile;
    if (parsed.version !== VECTOR_CACHE_VERSION || parsed.model !== model) {
      return null;
    }
    const map = new Map<string, number[]>();
    for (const entry of parsed.entries) {
      map.set(entry.key, entry.embedding);
    }
    return map;
  } catch {
    return null;
  }
}

function writeVectorCache(
  path: string,
  model: string,
  embeddings: Map<string, number[]>,
): void {
  mkdirSync(dirname(path), { recursive: true });
  const entries: VectorCacheEntry[] = Array.from(
    embeddings,
    ([key, embedding]) => ({
      key,
      embedding,
    }),
  );
  const payload: VectorCacheFile = {
    version: VECTOR_CACHE_VERSION,
    model,
    entries,
  };
  writeFileSync(path, JSON.stringify(payload), "utf8");
}

function checkpointVectorCache(
  path: string,
  model: string,
  embeddings: Map<string, number[]>,
): void {
  try {
    writeVectorCache(path, model, embeddings);
  } catch (error) {
    writeError("daemon", "Vector cache checkpoint failed", error, {
      path,
      model,
      entries: embeddings.size,
    });
  }
}

interface BuildVectorIndexOptions {
  forceRebuild?: boolean;
  model?: string;
}

async function buildVectorIndexUnlocked(
  chunks: Chunk[],
  bm25CachePath: string,
  options?: BuildVectorIndexOptions,
): Promise<VectorIndex> {
  const model = options?.model ?? DEFAULT_MODEL;
  const cachePath = vectorCachePath(bm25CachePath);

  const keys = chunks.map(chunkKey);
  const keySet = new Set(keys);

  let cached: Map<string, number[]> | null = null;
  if (!options?.forceRebuild) {
    cached = readVectorCache(cachePath, model);
  }

  const missingIndices: number[] = [];
  if (cached) {
    for (let i = 0; i < keys.length; i++) {
      if (!cached.has(keys[i])) {
        missingIndices.push(i);
      }
    }
  } else {
    for (let i = 0; i < keys.length; i++) {
      missingIndices.push(i);
    }
  }

  let newEmbeddings: Map<string, number[]> = cached ?? new Map();

  if (missingIndices.length > 0) {
    for (
      let batchStart = 0;
      batchStart < missingIndices.length;
      batchStart += EMBEDDING_BATCH_SIZE
    ) {
      const batchIndices = missingIndices.slice(
        batchStart,
        batchStart + EMBEDDING_BATCH_SIZE,
      );
      const texts = batchIndices.map((i) => {
        const chunk = chunks[i];
        return `${chunk.heading}\n${chunk.text}`.slice(0, 2048);
      });
      let embeddings: number[][];
      try {
        embeddings = await embed(texts, model);
      } catch (error) {
        checkpointVectorCache(cachePath, model, newEmbeddings);
        writeError("daemon", "Embedding batch failed", error, {
          model,
          batchStart,
          batchSize: batchIndices.length,
          totalMissing: missingIndices.length,
        });
        throw error;
      }
      for (let j = 0; j < batchIndices.length; j++) {
        newEmbeddings.set(keys[batchIndices[j]], embeddings[j]);
      }
      const completed = batchStart + batchIndices.length;
      if (
        completed === missingIndices.length ||
        completed % VECTOR_CHECKPOINT_INTERVAL === 0
      ) {
        checkpointVectorCache(cachePath, model, newEmbeddings);
      }
      if (completed === missingIndices.length || completed % 50 === 0) {
        writeLog("daemon", "info", "Embedding progress", {
          model,
          completed,
          totalMissing: missingIndices.length,
        });
      }
    }
  }

  // Prune entries no longer in the chunk set
  for (const key of newEmbeddings.keys()) {
    if (!keySet.has(key)) {
      newEmbeddings.delete(key);
    }
  }

  const embeddings: Float32Array[] = keys.map((key) => {
    const vec = newEmbeddings.get(key)!;
    return new Float32Array(vec);
  });

  const index: VectorIndex = { model, keys, embeddings };
  writeVectorCache(cachePath, model, newEmbeddings);
  return index;
}

export async function buildVectorIndex(
  chunks: Chunk[],
  bm25CachePath: string,
  options?: BuildVectorIndexOptions,
): Promise<VectorIndex> {
  return withIndexingLock(() =>
    buildVectorIndexUnlocked(chunks, bm25CachePath, options),
  );
}

export async function searchVector(
  chunks: Chunk[],
  vectorIndex: VectorIndex,
  query: string,
  k: number,
): Promise<{ chunkIndex: number; score: number }[]> {
  const queryEmbedding = await embed([query], vectorIndex.model, {
    queryInstruct: QUERY_INSTRUCT,
  });
  const queryVec = new Float32Array(queryEmbedding[0]);

  const scored: { chunkIndex: number; score: number }[] = [];
  for (let i = 0; i < vectorIndex.embeddings.length; i++) {
    const score = cosineSimilarity(queryVec, vectorIndex.embeddings[i]);
    scored.push({ chunkIndex: i, score });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}
