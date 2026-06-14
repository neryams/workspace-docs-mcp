import type { VectorIndex } from "./embeddings.js";
import { searchVector } from "./embeddings.js";
import type { Chunk, SearchHit } from "./types.js";

const K1 = 1.2;
const B = 0.75;
const RRF_K = 60;

export function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9_]+/g) ?? [];
}

function makeSnippet(text: string, terms: string[], maxLen = 240): string {
  const lower = text.toLowerCase();
  let bestIdx = 0;
  for (const term of terms) {
    const idx = lower.indexOf(term);
    if (idx >= 0) {
      bestIdx = idx;
      break;
    }
  }
  const start = Math.max(0, bestIdx - 60);
  const slice = text.slice(start, start + maxLen).replace(/\s+/g, " ").trim();
  const prefix = start > 0 ? "…" : "";
  const suffix = start + maxLen < text.length ? "…" : "";
  return `${prefix}${slice}${suffix}`;
}

function citation(file: string, heading: string, headingPath: string[]): string {
  if (headingPath.length) {
    return `${file}#${headingPath.join(" > ")}`;
  }
  if (heading && heading !== "(preamble)") {
    return `${file}#${heading}`;
  }
  return file;
}

export function searchBm25(chunks: Chunk[], query: string, k: number): SearchHit[] {
  const queryTerms = [...new Set(tokenize(query))];
  if (!queryTerms.length || !chunks.length) {
    return [];
  }

  const docTokens = chunks.map((chunk) => tokenize(`${chunk.heading} ${chunk.text}`));
  const docLengths = docTokens.map((t) => t.length);
  const avgDl =
    docLengths.reduce((sum, len) => sum + len, 0) / Math.max(docLengths.length, 1);
  const N = chunks.length;

  const df = new Map<string, number>();
  for (const terms of docTokens) {
    const seen = new Set(terms);
    for (const term of seen) {
      df.set(term, (df.get(term) ?? 0) + 1);
    }
  }

  const scores = chunks.map((chunk, i) => {
    const terms = docTokens[i];
    const termFreq = new Map<string, number>();
    for (const term of terms) {
      termFreq.set(term, (termFreq.get(term) ?? 0) + 1);
    }

    let score = 0;
    const dl = docLengths[i];
    for (const term of queryTerms) {
      const f = termFreq.get(term) ?? 0;
      if (f === 0) {
        continue;
      }
      const docFreq = df.get(term) ?? 0;
      const idf = Math.log((N - docFreq + 0.5) / (docFreq + 0.5) + 1);
      const denom = f + K1 * (1 - B + (B * dl) / avgDl);
      score += idf * ((f * (K1 + 1)) / denom);
    }

    return {
      file: chunk.file,
      heading: chunk.heading,
      headingPath: chunk.headingPath,
      snippet: makeSnippet(chunk.text, queryTerms),
      score,
      citation: citation(chunk.file, chunk.heading, chunk.headingPath),
    };
  });

  return scores
    .filter((hit) => hit.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

export function searchSubstring(chunks: Chunk[], query: string, k: number): SearchHit[] {
  const needle = query.toLowerCase();
  const hits: SearchHit[] = [];

  for (const chunk of chunks) {
    const haystack = `${chunk.heading}\n${chunk.text}`.toLowerCase();
    if (!haystack.includes(needle)) {
      continue;
    }
    hits.push({
      file: chunk.file,
      heading: chunk.heading,
      headingPath: chunk.headingPath,
      snippet: makeSnippet(chunk.text, tokenize(query)),
      score: 0.01,
      citation: citation(chunk.file, chunk.heading, chunk.headingPath),
    });
  }

  return hits.slice(0, k);
}

export function searchJournal(
  chunks: Chunk[],
  query: string,
  k: number,
): SearchHit[] {
  const bm25 = searchBm25(chunks, query, k);
  if (bm25.length >= k) {
    return bm25;
  }

  const seen = new Set(bm25.map((h) => `${h.file}\0${h.heading}`));
  const fallback = searchSubstring(chunks, query, k);
  for (const hit of fallback) {
    const key = `${hit.file}\0${hit.heading}`;
    if (!seen.has(key)) {
      bm25.push(hit);
      seen.add(key);
    }
    if (bm25.length >= k) {
      break;
    }
  }
  return bm25;
}

export function searchRegex(
  chunks: Chunk[],
  pattern: string,
  k: number,
): SearchHit[] {
  const re = new RegExp(pattern, "im");
  const hits: SearchHit[] = [];

  for (const chunk of chunks) {
    const match = re.exec(`${chunk.heading}\n${chunk.text}`);
    if (!match) {
      continue;
    }
    hits.push({
      file: chunk.file,
      heading: chunk.heading,
      headingPath: chunk.headingPath,
      snippet: makeSnippet(chunk.text, tokenize(match[0])),
      score: 1,
      citation: citation(chunk.file, chunk.heading, chunk.headingPath),
    });
    if (hits.length >= k) {
      break;
    }
  }

  return hits;
}

export async function searchHybrid(
  chunks: Chunk[],
  vectorIndex: VectorIndex,
  query: string,
  k: number,
): Promise<SearchHit[]> {
  const expandedK = Math.min(chunks.length, k * 3);

  const bm25Hits = searchBm25(chunks, query, expandedK);
  const vectorHits = await searchVector(chunks, vectorIndex, query, expandedK);

  const rrfScores = new Map<string, { score: number; chunkIdx: number }>();

  for (let rank = 0; rank < bm25Hits.length; rank++) {
    const hit = bm25Hits[rank];
    const key = `${hit.file}\0${hit.heading}`;
    const prev = rrfScores.get(key);
    const rrfScore = 1 / (RRF_K + rank + 1);
    if (prev) {
      prev.score += rrfScore;
    } else {
      const chunkIdx = chunks.findIndex(
        (c) => c.file === hit.file && c.heading === hit.heading,
      );
      rrfScores.set(key, { score: rrfScore, chunkIdx });
    }
  }

  for (let rank = 0; rank < vectorHits.length; rank++) {
    const { chunkIndex } = vectorHits[rank];
    const chunk = chunks[chunkIndex];
    const key = `${chunk.file}\0${chunk.heading}`;
    const prev = rrfScores.get(key);
    const rrfScore = 1 / (RRF_K + rank + 1);
    if (prev) {
      prev.score += rrfScore;
    } else {
      rrfScores.set(key, { score: rrfScore, chunkIdx: chunkIndex });
    }
  }

  const queryTerms = tokenize(query);
  const results = [...rrfScores.entries()]
    .sort((a, b) => b[1].score - a[1].score)
    .slice(0, k)
    .map(([_, { score, chunkIdx }]) => {
      const chunk = chunks[chunkIdx];
      return {
        file: chunk.file,
        heading: chunk.heading,
        headingPath: chunk.headingPath,
        snippet: makeSnippet(chunk.text, queryTerms),
        score,
        citation: citation(chunk.file, chunk.heading, chunk.headingPath),
      };
    });

  return results;
}
