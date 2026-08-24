import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const CONFIG = {
  // Retrieval
  MIN_RELEVANCE: Number(process.env.MIN_RELEVANCE ?? 0.32),
  MAX_FROM_SOURCE: Number(process.env.MAX_FROM_SOURCE ?? 4),
  RAG_DEBUG: process.env.RAG_DEBUG === "true",

  // Chunking
  TARGET_MAX_TOKENS: Number(process.env.TARGET_MAX_TOKENS ?? 220),
  TARGET_MIN_TOKENS: Number(process.env.TARGET_MIN_TOKENS ?? 80),
  OVERLAP_TOKENS: Number(process.env.OVERLAP_TOKENS ?? 40),

  // Embedding
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || "Xenova/bge-small-en-v1.5",
  QUERY_PREFIX: "Represent this sentence for searching relevant passages: ",

  // Crawling
  MAX_PAGES_PER_SITE: Number(process.env.MAX_PAGES_PER_SITE ?? 500),
  MAX_DEPTH: Number(process.env.MAX_CRAWL_DEPTH ?? 10),
  REQUEST_DELAY_MS: Number(process.env.REQUEST_DELAY_MS ?? 350),

  __dirname,
};