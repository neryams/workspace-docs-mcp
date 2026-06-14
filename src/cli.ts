#!/usr/bin/env node
import { findWorkspaceRoot, loadConfig, resolveCachePath } from "./config.js";
import { buildVectorIndex } from "./embeddings.js";
import { loadJournalIndex, readEntry } from "./index.js";
import { searchHybrid, searchJournal, searchRegex } from "./search.js";

function usage(): void {
  console.error(`Usage:
  journal search <query> [--k N]
  journal list [--filter text]
  journal get <path>
  journal regex <pattern> [--k N]
  journal index [--rebuild]

Run from a repo root that contains journal-rag.config.json, or set JOURNAL_RAG_WORKSPACE.`);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const command = args[0];

  if (!command || command === "--help" || command === "-h") {
    usage();
    process.exit(command ? 0 : 1);
  }

  const workspaceRoot = findWorkspaceRoot();
  const config = loadConfig(workspaceRoot);
  const forceRebuild = args.includes("--rebuild");

  const readFlag = (name: string, fallback: number): number => {
    const idx = args.indexOf(name);
    if (idx >= 0 && args[idx + 1]) {
      return Number.parseInt(args[idx + 1], 10);
    }
    return fallback;
  };

  if (command === "index") {
    const index = loadJournalIndex(workspaceRoot, config, { forceRebuild });
    console.log(
      JSON.stringify(
        {
          workspaceRoot,
          entries: index.entries.length,
          chunks: index.chunks.length,
          rebuilt: forceRebuild,
        },
        null,
        2,
      ),
    );
    return;
  }

  const index = loadJournalIndex(workspaceRoot, config, { forceRebuild });

  if (command === "search") {
    const query = args[1];
    if (!query) {
      usage();
      process.exit(1);
    }
    const k = readFlag("--k", 8);
    const bm25Only = args.includes("--bm25");
    if (bm25Only) {
      const hits = searchJournal(index.chunks, query, k);
      console.log(JSON.stringify({ query, hits }, null, 2));
    } else {
      const cachePath = resolveCachePath(workspaceRoot, config);
      const vectorIndex = await buildVectorIndex(index.chunks, cachePath, {
        model: config.embeddingModel,
      });
      const hits = await searchHybrid(index.chunks, vectorIndex, query, k);
      console.log(JSON.stringify({ query, hits }, null, 2));
    }
    return;
  }

  if (command === "regex") {
    const pattern = args[1];
    if (!pattern) {
      usage();
      process.exit(1);
    }
    const k = readFlag("--k", 20);
    const hits = searchRegex(index.chunks, pattern, k);
    console.log(JSON.stringify({ pattern, hits }, null, 2));
    return;
  }

  if (command === "list") {
    const filterIdx = args.indexOf("--filter");
    const filter = filterIdx >= 0 ? args[filterIdx + 1] : undefined;
    const needle = filter?.toLowerCase();
    const entries = needle
      ? index.entries.filter(
          (e) =>
            e.file.toLowerCase().includes(needle) ||
            e.title.toLowerCase().includes(needle),
        )
      : index.entries;
    console.log(JSON.stringify({ count: entries.length, entries }, null, 2));
    return;
  }

  if (command === "get") {
    const pathArg = args[1];
    if (!pathArg) {
      usage();
      process.exit(1);
    }
    process.stdout.write(readEntry(workspaceRoot, config, pathArg));
    return;
  }

  usage();
  process.exit(1);
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
