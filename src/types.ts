export interface JournalRagConfig {
  sources: string[];
  cachePath?: string;
  embeddingModel?: string;
}

export interface Chunk {
  repo: string;
  file: string;
  heading: string;
  headingPath: string[];
  text: string;
  dateFromFilename: string | null;
  h1Title: string | null;
}

export interface SearchHit {
  file: string;
  heading: string;
  headingPath: string[];
  snippet: string;
  score: number;
  citation: string;
}

export interface EntryMeta {
  file: string;
  title: string;
  date: string | null;
  headings: string[];
}

export interface JournalIndex {
  workspaceRoot: string;
  repo: string;
  chunks: Chunk[];
  entries: EntryMeta[];
}
