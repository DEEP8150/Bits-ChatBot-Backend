import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "@xenova/transformers";
import { readStore, writeStore } from "./knowledge-store.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SOURCE_FILE = path.join(__dirname, "site-content.json");
const COLLECTION = "bits_swd_knowledge";
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";

function wordCount(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function splitIntoParagraphs(text) {
  return text
    .replace(/\r\n/g, "\n")
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function buildChunks(record) {
  const paragraphs = splitIntoParagraphs(record.text);
  const chunks = [];
  let current = [];
  let words = 0;
  const targetMin = 300;
  const targetMax = 500;
  const overlap = 75;

  const flush = () => {
    const text = current.join("\n\n").trim();
    if (!text) return;
    chunks.push(text);
  };

  for (const paragraph of paragraphs) {
    const wc = wordCount(paragraph);
    if (current.length && words + wc > targetMax) {
      flush();
      const tail = chunks.length ? chunks[chunks.length - 1].split(/\s+/).slice(-overlap).join(" ") : "";
      current = tail ? [tail] : [];
      words = tail ? wordCount(tail) : 0;
    }
    current.push(paragraph);
    words += wc;
    if (words >= targetMin) {
      flush();
      const tail = current.join(" ").split(/\s+/).slice(-overlap).join(" ");
      current = tail ? [tail] : [];
      words = tail ? wordCount(tail) : 0;
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

  const store = await readStore();
  store.collection = COLLECTION;
  store.model = EMBEDDING_MODEL;
  store.records = [];

  let totalChunks = 0;
  const siteCounts = {};

  for (const record of records) {
    const chunks = buildChunks(record);
    siteCounts[record.site] = (siteCounts[record.site] || 0) + chunks.length;
    for (let i = 0; i < chunks.length; i += 1) {
      const text = chunks[i];
      const output = await embedder(text, { pooling: "mean", normalize: true });
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
  console.log(`Stored ${totalChunks} chunks in Chroma-like local store: ${COLLECTION}`);
  for (const [site, count] of Object.entries(siteCounts)) {
    console.log(`${site} chunks: ${count}`);
  }
}

main().catch((err) => {
  console.error("Indexing failed:", err);
  process.exit(1);
});
