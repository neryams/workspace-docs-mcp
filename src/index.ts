import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, relative, resolve } from "node:path";
import { resolveCachePath, resolveSourceDirs } from "./config.js";
import type {
  Chunk,
  EntryMeta,
  JournalIndex,
  JournalRagConfig,
} from "./types.js";

const CACHE_VERSION = 1;

interface SourceFingerprint {
  mtimeMs: number;
  size: number;
}

interface CacheFile {
  version: number;
  workspaceRoot: string;
  repo: string;
  fingerprints: Record<string, SourceFingerprint>;
  chunks: Chunk[];
  entries: EntryMeta[];
}

function walkMarkdownFiles(dir: string): string[] {
  const files: string[] = [];
  if (!existsSync(dir)) {
    return files;
  }
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkMarkdownFiles(full));
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      files.push(full);
    }
  }
  return files.sort();
}

function dateFromFilename(filename: string): string | null {
  const match = basename(filename).match(/^(\d{4}-\d{2}-\d{2})/);
  return match ? match[1] : null;
}

function splitIntoChunks(
  content: string,
  fileRel: string,
  repo: string,
): Chunk[] {
  const lines = content.split(/\r?\n/);
  let h1Title: string | null = null;
  const headingStack: string[] = [];
  let currentHeading = "(preamble)";
  let buffer: string[] = [];
  const chunks: Chunk[] = [];
  const date = dateFromFilename(fileRel);

  const flush = () => {
    const text = buffer.join("\n").trim();
    if (!text && currentHeading === "(preamble)") {
      return;
    }
    chunks.push({
      repo,
      file: fileRel,
      heading: currentHeading,
      headingPath: [...headingStack],
      text,
      dateFromFilename: date,
      h1Title,
    });
    buffer = [];
  };

  for (const line of lines) {
    const h1 = line.match(/^#\s+(.+)/);
    const h2 = line.match(/^##\s+(.+)/);
    const h3 = line.match(/^###\s+(.+)/);

    if (h3) {
      flush();
      headingStack.length = Math.min(headingStack.length, 1);
      headingStack[1] = h3[1].trim();
      if (headingStack.length < 2) {
        headingStack.push(h3[1].trim());
      }
      currentHeading = h3[1].trim();
      buffer.push(line);
      continue;
    }

    if (h2) {
      flush();
      headingStack.length = 0;
      if (h1Title) {
        headingStack.push(h1Title);
      }
      headingStack.push(h2[1].trim());
      currentHeading = h2[1].trim();
      buffer.push(line);
      continue;
    }

    if (h1 && !h2 && !h3) {
      h1Title = h1[1].trim();
      buffer.push(line);
      continue;
    }

    buffer.push(line);
  }

  flush();
  return chunks;
}

function buildFingerprints(files: string[]): Record<string, SourceFingerprint> {
  const fingerprints: Record<string, SourceFingerprint> = {};
  for (const file of files) {
    const stat = statSync(file);
    fingerprints[file] = { mtimeMs: stat.mtimeMs, size: stat.size };
  }
  return fingerprints;
}

function fingerprintsMatch(
  a: Record<string, SourceFingerprint>,
  b: Record<string, SourceFingerprint>,
): boolean {
  const aKeys = Object.keys(a).sort();
  const bKeys = Object.keys(b).sort();
  if (aKeys.length !== bKeys.length) {
    return false;
  }
  return aKeys.every((key, i) => {
    if (key !== bKeys[i]) {
      return false;
    }
    return a[key].mtimeMs === b[key].mtimeMs && a[key].size === b[key].size;
  });
}

function indexFromSources(
  workspaceRoot: string,
  sourceDirs: string[],
): { chunks: Chunk[]; entries: EntryMeta[]; fingerprints: Record<string, SourceFingerprint> } {
  const repo = basename(workspaceRoot);
  const chunks: Chunk[] = [];
  const entries: EntryMeta[] = [];
  const allFiles: string[] = [];

  for (const sourceDir of sourceDirs) {
    allFiles.push(...walkMarkdownFiles(sourceDir));
  }

  for (const absPath of allFiles) {
    const fileRel = relative(workspaceRoot, absPath).replace(/\\/g, "/");
    const content = readFileSync(absPath, "utf8");
    const fileChunks = splitIntoChunks(content, fileRel, repo);
    chunks.push(...fileChunks);

    const headings = fileChunks
      .map((c) => c.heading)
      .filter((h) => h !== "(preamble)");
    const firstChunk = fileChunks[0];
    entries.push({
      file: fileRel,
      title: firstChunk?.h1Title ?? basename(fileRel, ".md"),
      date: dateFromFilename(fileRel),
      headings,
    });
  }

  entries.sort((a, b) => {
    const dateCmp = (b.date ?? "").localeCompare(a.date ?? "");
    if (dateCmp !== 0) {
      return dateCmp;
    }
    return a.file.localeCompare(b.file);
  });

  chunks.sort((a, b) => {
    const fileCmp = a.file.localeCompare(b.file);
    if (fileCmp !== 0) {
      return fileCmp;
    }
    return a.heading.localeCompare(b.heading);
  });

  return {
    chunks,
    entries,
    fingerprints: buildFingerprints(allFiles),
  };
}

function readCache(cachePath: string): CacheFile | null {
  if (!existsSync(cachePath)) {
    return null;
  }
  try {
    const parsed = JSON.parse(readFileSync(cachePath, "utf8")) as CacheFile;
    if (parsed.version !== CACHE_VERSION) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function writeCache(
  cachePath: string,
  payload: CacheFile,
): void {
  mkdirSync(resolve(cachePath, ".."), { recursive: true });
  writeFileSync(cachePath, JSON.stringify(payload, null, 2), "utf8");
}

export function loadJournalIndex(
  workspaceRoot: string,
  config: JournalRagConfig,
  options?: { forceRebuild?: boolean },
): JournalIndex {
  const sourceDirs = resolveSourceDirs(workspaceRoot, config);
  const cachePath = resolveCachePath(workspaceRoot, config);
  const repo = basename(workspaceRoot);

  if (!options?.forceRebuild) {
    const cached = readCache(cachePath);
    if (cached && cached.workspaceRoot === workspaceRoot) {
      const { fingerprints } = indexFromSources(workspaceRoot, sourceDirs);
      if (fingerprintsMatch(cached.fingerprints, fingerprints)) {
        return {
          workspaceRoot,
          repo,
          chunks: cached.chunks,
          entries: cached.entries,
        };
      }
    }
  }

  const built = indexFromSources(workspaceRoot, sourceDirs);
  writeCache(cachePath, {
    version: CACHE_VERSION,
    workspaceRoot,
    repo,
    fingerprints: built.fingerprints,
    chunks: built.chunks,
    entries: built.entries,
  });

  return {
    workspaceRoot,
    repo,
    chunks: built.chunks,
    entries: built.entries,
  };
}

export function readEntry(
  workspaceRoot: string,
  config: JournalRagConfig,
  pathArg: string,
): string {
  const normalized = pathArg.replace(/\\/g, "/");
  const candidates = [
    resolve(workspaceRoot, normalized),
    ...resolveSourceDirs(workspaceRoot, config).map((dir) =>
      resolve(dir, normalized),
    ),
    ...resolveSourceDirs(workspaceRoot, config).map((dir) =>
      resolve(dir, basename(normalized)),
    ),
  ];

  for (const candidate of candidates) {
    if (existsSync(candidate) && candidate.endsWith(".md")) {
      return readFileSync(candidate, "utf8");
    }
  }

  throw new Error(`Journal entry not found: ${pathArg}`);
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

export function writeEntry(
  workspaceRoot: string,
  config: JournalRagConfig,
  title: string,
  content: string,
): { file: string; absPath: string } {
  const sourceDir = resolveSourceDirs(workspaceRoot, config)[0];
  mkdirSync(sourceDir, { recursive: true });

  const date = new Date().toISOString().slice(0, 10);
  const slug = slugify(title);
  const filename = `${date}_${slug}.md`;
  const absPath = join(sourceDir, filename);

  if (existsSync(absPath)) {
    throw new Error(`Journal entry already exists: ${filename}`);
  }

  writeFileSync(absPath, content, "utf8");
  const fileRel = relative(workspaceRoot, absPath).replace(/\\/g, "/");
  return { file: fileRel, absPath };
}
