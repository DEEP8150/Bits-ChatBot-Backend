import { pipeline } from "@xenova/transformers";
import { cosineSimilarity, readStore } from "./knowledge-store.js";

const EMBEDDING_MODEL = "Xenova/bge-small-en-v1.5";
// BGE models are trained asymmetrically: queries should be prefixed with
// this instruction, passages should NOT be (our stored chunk embeddings
// aren't prefixed — see index-knowledge.js). This measurably improves
// retrieval for BGE-family models; it's free, it's just prepended text.
const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
const DEFAULT_MIN_SIMILARITY = 0.32;
const TITLE_BOOST = 0.08;
const SITE_BOOST = 0.03;
const MAX_FROM_SOURCE = 4;
let embedderPromise;

async function getEmbedder() {
  if (!embedderPromise) {
    embedderPromise = pipeline("feature-extraction", EMBEDDING_MODEL, { quantized: true });
  }
  return embedderPromise;
}

export async function embedQuery(text) {
  const embedder = await getEmbedder();
  const output = await embedder(`${QUERY_PREFIX}${text}`, { pooling: "mean", normalize: true });
  return Array.from(output.data ?? output[0] ?? []);
}

export async function retrieveChunks(question, { topK = 8, minSimilarity = DEFAULT_MIN_SIMILARITY } = {}) {
  const normalizedQuestion = question.toLowerCase();
  const store = await readStore();
  const queryEmbedding = await embedQuery(question);
  const scored = store.records
    .map((record) => ({
      ...record,
      similarity: cosineSimilarity(queryEmbedding, record.embedding),
    }))
    .map((record) => {
      let boost = 0;
      const title = `${record.title || ""} ${record.sourceUrl || ""}`.toLowerCase();
      const siteName = (record.site || "").toLowerCase();

      for (const token of normalizedQuestion.split(/\s+/).filter(Boolean)) {
        if (token.length < 4) continue;
        if (title.includes(token)) boost += TITLE_BOOST;
      }

      if (
        (normalizedQuestion.includes("scholar") && title.includes("scholar")) ||
        (normalizedQuestion.includes("hostel") && title.includes("hostel")) ||
        (normalizedQuestion.includes("ragging") && title.includes("ragging")) ||
        (normalizedQuestion.includes("admission") && title.includes("admission")) ||
        (normalizedQuestion.includes("welfare") && siteName.includes("swd"))
      ) {
        boost += SITE_BOOST;
      }

      const finalScore = record.similarity + Math.min(boost, 0.15);
      return {
        ...record,
        finalScore,
      };
    })
    .filter((record) => record.finalScore >= minSimilarity)
    .sort((a, b) => b.finalScore - a.finalScore);

  const selected = [];
  const seenSources = new Map();
  const seenTexts = new Set();

  for (const item of scored) {
    if (selected.length >= topK) break;
    const sourceKey = `${item.site}::${item.sourceUrl}`;
    const textKey = item.text.slice(0, 160).toLowerCase();
    const sourceCount = seenSources.get(sourceKey) || 0;
    if (sourceCount >= MAX_FROM_SOURCE) continue;
    if (seenTexts.has(textKey)) continue;
    seenSources.set(sourceKey, sourceCount + 1);
    seenTexts.add(textKey);
    selected.push(item);
  }

  return selected;
}

export function formatRetrievedContext(chunks) {
  if (!chunks.length) return "";
  return chunks
    .map(
      (chunk, index) =>
        `SOURCE ${index + 1}\nSite: ${chunk.site}\nTitle: ${chunk.title}\nURL: ${chunk.sourceUrl}\nType: ${chunk.type}\nContent:\n${chunk.text}`
    )
    .join("\n\n");
}