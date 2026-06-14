import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { Chunk } from "./types.js";

const VECTOR_CACHE_VERSION = 1;
const DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2";

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

let pipelineInstance: ((texts: string[], options?: { pooling: "mean" | "cls" | "none"; normalize: boolean }) => Promise<{ tolist: () => number[][] }>) | null = null;

async function getEmbedder(model: string) {
  if (pipelineInstance) {
    return pipelineInstance;
  }
  const { pipeline } = await import("@huggingface/transformers");
  const extractor = await pipeline("feature-extraction", model, {
    dtype: "fp32",
  });
  pipelineInstance = async (texts: string[], options?: { pooling: "mean" | "cls" | "none"; normalize: boolean }) => {
    const result = await extractor(texts, options);
    return result as unknown as { tolist: () => number[][] };
  };
  return pipelineInstance;
}

async function embed(texts: string[], model: string): Promise<number[][]> {
  const extractor = await getEmbedder(model);
  const output = await extractor(texts, { pooling: "mean", normalize: true });
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
  index: VectorIndex,
): void {
  mkdirSync(dirname(path), { recursive: true });
  const entries: VectorCacheEntry[] = index.keys.map((key, i) => ({
    key,
    embedding: Array.from(index.embeddings[i]),
  }));
  const payload: VectorCacheFile = {
    version: VECTOR_CACHE_VERSION,
    model,
    entries,
  };
  writeFileSync(path, JSON.stringify(payload), "utf8");
}

export async function buildVectorIndex(
  chunks: Chunk[],
  bm25CachePath: string,
  options?: { forceRebuild?: boolean; model?: string },
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
    const BATCH_SIZE = 32;
    for (let batchStart = 0; batchStart < missingIndices.length; batchStart += BATCH_SIZE) {
      const batchIndices = missingIndices.slice(batchStart, batchStart + BATCH_SIZE);
      const texts = batchIndices.map((i) => {
        const chunk = chunks[i];
        return `${chunk.heading}\n${chunk.text}`.slice(0, 512);
      });
      const embeddings = await embed(texts, model);
      for (let j = 0; j < batchIndices.length; j++) {
        newEmbeddings.set(keys[batchIndices[j]], embeddings[j]);
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
  writeVectorCache(cachePath, model, index);
  return index;
}

export async function searchVector(
  chunks: Chunk[],
  vectorIndex: VectorIndex,
  query: string,
  k: number,
): Promise<{ chunkIndex: number; score: number }[]> {
  const queryEmbedding = await embed([query], vectorIndex.model);
  const queryVec = new Float32Array(queryEmbedding[0]);

  const scored: { chunkIndex: number; score: number }[] = [];
  for (let i = 0; i < vectorIndex.embeddings.length; i++) {
    const score = cosineSimilarity(queryVec, vectorIndex.embeddings[i]);
    scored.push({ chunkIndex: i, score });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}
