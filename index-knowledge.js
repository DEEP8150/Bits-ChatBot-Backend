import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline, AutoTokenizer } from "@xenova/transformers";
import { readStore, writeStore } from "./knowledge-store.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SOURCE_FILE = path.join(__dirname, "site-content.json");
const COLLECTION = "bits_swd_knowledge";
// bge-small-en-v1.5: 512-token window (vs 256 for MiniLM), stronger on
// retrieval benchmarks, still small/fast/free/local.
const EMBEDDING_MODEL = "Xenova/bge-small-en-v1.5";

// Target sizes are in TOKENS (of the embedding model's own tokenizer), not
// words — this is what actually determines whether content gets truncated
// before it's embedded. Kept well under the 512-token limit to leave margin
// and to keep each chunk topically focused (better retrieval precision than
// a small number of huge chunks).
const TARGET_MAX_TOKENS = 220;
const TARGET_MIN_TOKENS = 80;
const OVERLAP_TOKENS = 40;

function splitIntoParagraphs(text) {
  return text
    .replace(/\r\n/g, "\n")
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function splitLongParagraphIntoSentences(paragraph) {
  return paragraph
    .split(/(?<=[.?!])\s+(?=[A-Z0-9])/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function buildChunks(record, tokenizer) {
  const tokenCount = (text) => tokenizer.encode(text).length;
  const paragraphs = splitIntoParagraphs(record.text);
  const chunks = [];
  let current = [];
  let currentTokens = 0;

  const flush = () => {
    const text = current.join("\n\n").trim();
    if (text) chunks.push(text);
  };

  const takeOverlapTail = (text) => {
    const words = text.split(/\s+/);
    let tail = "";
    for (let i = words.length - 1; i >= 0; i -= 1) {
      const candidate = tail ? `${words[i]} ${tail}` : words[i];
      if (tokenCount(candidate) > OVERLAP_TOKENS) break;
      tail = candidate;
    }
    return tail;
  };

  for (const paragraph of paragraphs) {
    let pieces = [paragraph];
    // If a single paragraph alone is already too big (common in PDF sections
    // with no internal breaks), split it into sentence groups so nothing
    // gets silently truncated by the embedder.
    if (tokenCount(paragraph) > TARGET_MAX_TOKENS) {
      const sentences = splitLongParagraphIntoSentences(paragraph);
      pieces = [];
      let sub = [];
      let subTokens = 0;
      for (const sentence of sentences) {
        const st = tokenCount(sentence);
        if (sub.length && subTokens + st > TARGET_MAX_TOKENS) {
          pieces.push(sub.join(" "));
          sub = [];
          subTokens = 0;
        }
        sub.push(sentence);
        subTokens += st;
      }
      if (sub.length) pieces.push(sub.join(" "));
    }

    for (const piece of pieces) {
      const pieceTokens = tokenCount(piece);

      if (current.length && currentTokens + pieceTokens > TARGET_MAX_TOKENS) {
        flush();
        const tail = takeOverlapTail(current.join(" "));
        current = tail ? [tail] : [];
        currentTokens = tail ? tokenCount(tail) : 0;
      }

      current.push(piece);
      currentTokens += pieceTokens;

      if (currentTokens >= TARGET_MIN_TOKENS && currentTokens >= TARGET_MAX_TOKENS * 0.85) {
        flush();
        const tail = takeOverlapTail(current.join(" "));
        current = tail ? [tail] : [];
        currentTokens = tail ? tokenCount(tail) : 0;
      }
    }
  }

  if (current.length) flush();
  return chunks;
}

function chunkId(recordId, index) {
  return `${recordId}::chunk_${String(index + 1).padStart(3, "0")}`;
}

async function main() {
  const raw = await readFile(SOURCE_FILE, "utf8");
  const records = JSON.parse(raw);

  const embedder = await pipeline("feature-extraction", EMBEDDING_MODEL, { quantized: true });
  const tokenizer = await AutoTokenizer.from_pretrained(EMBEDDING_MODEL);

  const store = await readStore();
  store.collection = COLLECTION;
  store.model = EMBEDDING_MODEL;
  store.records = [];

  let totalChunks = 0;
  let oversizedChunks = 0;
  const siteCounts = {};

  for (const record of records) {
    const chunks = buildChunks(record, tokenizer);
    siteCounts[record.site] = (siteCounts[record.site] || 0) + chunks.length;

    for (let i = 0; i < chunks.length; i += 1) {
      const text = chunks[i];
      const tokenLength = tokenizer.encode(text).length;
      if (tokenLength > 512) oversizedChunks += 1; // sanity check, should never fire

      // Prefix title/site into the EMBEDDING input only (not the stored/
      // displayed text) — helps match queries like "Krishna Bhawan warden"
      // against pages whose body text doesn't repeat the page name often.
      const embeddingInput = `${record.title} — ${record.site}\n${text}`;
      const output = await embedder(embeddingInput, { pooling: "mean", normalize: true });
      const embedding = Array.from(output.data ?? output[0] ?? []);

      store.records.push({
        id: chunkId(record.id, i),
        recordId: record.id,
        site: record.site,
        type: record.type,
        title: record.title,
        sourceUrl: record.sourceUrl,
        discoveredFrom: record.discoveredFrom,
        chunkIndex: i,
        tokenLength,
        text,
        embedding,
      });
      totalChunks += 1;
    }
  }

  await writeStore(store);
  console.log(`Loaded ${records.length} source records`);
  console.log(`Generated ${totalChunks} chunks`);
  console.log(`Generated embeddings with ${EMBEDDING_MODEL}`);
  if (oversizedChunks > 0) {
    console.warn(`WARNING: ${oversizedChunks} chunks exceeded 512 tokens — check buildChunks logic`);
  }
  console.log(`Stored ${totalChunks} chunks in local store: ${COLLECTION}`);
  for (const [site, count] of Object.entries(siteCounts)) {
    console.log(`${site} chunks: ${count}`);
  }
}

main().catch((err) => {
  console.error("Indexing failed:", err);
  process.exit(1);
});