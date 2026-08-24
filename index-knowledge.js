import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline, AutoTokenizer } from "@xenova/transformers";
import { readStore, writeStore } from "./knowledge-store.js";
import { CONFIG } from "./config.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SOURCE_FILE = path.join(__dirname, "site-content.json");
const COLLECTION = "bits_swd_knowledge";
const EMBEDDING_MODEL = CONFIG.EMBEDDING_MODEL;

const TARGET_MAX_TOKENS = CONFIG.TARGET_MAX_TOKENS;
const TARGET_MIN_TOKENS = CONFIG.TARGET_MIN_TOKENS;
const OVERLAP_TOKENS = CONFIG.OVERLAP_TOKENS;

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

function buildPrefixedChunkText(recordTitle, bodyText, heading = null) {
  const body = String(bodyText || "").trim();
  if (!body) return "";

  if (heading && String(heading).trim()) {
    return `${recordTitle} — ${String(heading).trim()}\n${body}`;
  }

  return `${recordTitle}\n${body}`;
}

// Original greedy, token-aware chunking algorithm - unchanged logic, just
// operating on a plain text blob so it can be reused per-page / per-section.
function chunkPlainText(text, tokenizer) {
  const tokenCount = (t) => tokenizer.encode(t).length;
  const paragraphs = splitIntoParagraphs(text);
  const chunks = [];
  let current = [];
  let currentTokens = 0;

  const flush = () => {
    const chunkText = current.join("\n\n").trim();
    if (chunkText) chunks.push(chunkText);
  };

  const takeOverlapTail = (t) => {
    const words = t.split(/\s+/);
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

// Decides chunk boundaries based on what structure the record has:
// PDF -> never cross a page boundary. HTML -> prefer section boundaries.
// Falls back to whole-record text if neither is available.
function buildChunksForRecord(record, tokenizer) {
  if (record.type === "pdf" && Array.isArray(record.pages) && record.pages.length) {
    const results = [];
    for (const page of record.pages) {
      if (!page.text || !page.text.trim()) continue;
      const pageHeading = page.heading || page.label || null;
      const sourceText = pageHeading ? buildPrefixedChunkText(record.title, page.text, pageHeading) : page.text;
      for (const text of chunkPlainText(sourceText, tokenizer)) {
        results.push({ text, pageNumber: page.pageNumber, section: null });
      }
    }
    if (results.length) return results;
  }

  if (Array.isArray(record.sections) && record.sections.length) {
    const results = [];
    for (const section of record.sections) {
      if (!section.text || !section.text.trim()) continue;
      const sourceText = buildPrefixedChunkText(record.title, section.text, section.heading);
      for (const text of chunkPlainText(sourceText, tokenizer)) {
        results.push({ text, pageNumber: null, section: section.heading });
      }
    }
    if (results.length) return results;
  }

  return chunkPlainText(record.text, tokenizer).map((text) => ({ text, pageNumber: null, section: null }));
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
  let shortChunks = 0;
  let totalTokens = 0;
  let htmlChunks = 0;
  let pdfChunks = 0;
  let chunksWithSection = 0;
  const siteCounts = {};

  for (const record of records) {
    const pieces = buildChunksForRecord(record, tokenizer);
    siteCounts[record.site] = (siteCounts[record.site] || 0) + pieces.length;

    for (let i = 0; i < pieces.length; i += 1) {
      const { text, pageNumber, section } = pieces[i];
      const tokenLength = tokenizer.encode(text).length;
      if (tokenLength > 512) {
        oversizedChunks += 1;
        console.warn(
          `WARNING: final chunk exceeded 512 tokens after prefixing: record=${record.id} chunk=${i + 1} tokens=${tokenLength}`
        );
      }
      if (tokenLength < 20) shortChunks += 1;
      totalTokens += tokenLength;

      const embeddingInput = `${record.title}${section ? ` — ${section}` : ""} — ${record.site}\n${text}`;
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
        pageNumber,
        section,
        text,
        embedding,
      });
      totalChunks += 1;
      if (record.type === "pdf") pdfChunks += 1;
      else htmlChunks += 1;
      if (section) chunksWithSection += 1;
    }
  }

  await writeStore(store);

  console.log(`Loaded ${records.length} source records`);
  console.log(`Generated ${totalChunks} chunks`);
  console.log(`Generated embeddings with ${EMBEDDING_MODEL}`);
  if (oversizedChunks > 0) {
    console.warn(`WARNING: ${oversizedChunks} chunks exceeded 512 tokens after prefixing`);
  }
  console.log(`Stored ${totalChunks} chunks in local store: ${COLLECTION}`);
  for (const [site, count] of Object.entries(siteCounts)) {
    console.log(`${site} chunks: ${count}`);
  }

  console.log(`\n=== Chunk quality summary ===`);
  console.log(`HTML chunks: ${htmlChunks}`);
  console.log(`PDF chunks: ${pdfChunks}`);
  console.log(`Average chunk tokens: ${(totalTokens / Math.max(totalChunks, 1)).toFixed(1)}`);
  console.log(
    `Chunks with section metadata: ${chunksWithSection} (${((chunksWithSection / Math.max(totalChunks, 1)) * 100).toFixed(1)}%)`
  );
  console.log(`Suspiciously short chunks (<20 tokens): ${shortChunks}`);
}

main().catch((err) => {
  console.error("Indexing failed:", err);
  process.exit(1);
});
