import { pipeline } from "@xenova/transformers";
import MiniSearch from "minisearch";
import { cosineSimilarity, readStore } from "./knowledge-store.js";
import { CONFIG } from "./config.js";

let embedderPromise;
let cachedIndex = null;
let cachedRecordCount = -1;

async function getEmbedder() {
  if (!embedderPromise) {
    embedderPromise = pipeline("feature-extraction", CONFIG.EMBEDDING_MODEL, { quantized: true });
  }
  return embedderPromise;
}

export async function embedQuery(text) {
  const embedder = await getEmbedder();
  const output = await embedder(`${CONFIG.QUERY_PREFIX}${text}`, { pooling: "mean", normalize: true });
  return Array.from(output.data ?? output[0] ?? []);
}

// Rebuilds the lexical index only when the store's record count changes,
// so we don't re-index on every single query.
function getLexicalIndex(store) {
  if (cachedIndex && cachedRecordCount === store.records.length) return cachedIndex;

  const mini = new MiniSearch({
    idField: "id",
    fields: ["title", "text", "section"],
    storeFields: ["id"],
    searchOptions: { boost: { title: 2, section: 1.5 }, prefix: true, fuzzy: 0.1 },
  });

  mini.addAll(
    store.records.map((r) => ({
      id: r.id,
      title: r.title || "",
      text: r.text || "",
      section: r.section || "",
    }))
  );

  cachedIndex = mini;
  cachedRecordCount = store.records.length;
  return mini;
}

function rrfFuse(rankedIdLists, k = 60) {
  const scores = new Map();
  for (const ids of rankedIdLists) {
    ids.forEach((id, rank) => {
      scores.set(id, (scores.get(id) || 0) + 1 / (k + rank + 1));
    });
  }
  return scores;
}

export async function retrieveChunks(question, { topK = 8, minSimilarity = CONFIG.MIN_RELEVANCE } = {}) {
  const store = await readStore();
  if (!store.records.length) return [];

  const [queryEmbedding, lexicalIndex] = await Promise.all([embedQuery(question), getLexicalIndex(store)]);

  const denseScored = store.records
    .map((record) => ({ id: record.id, similarity: cosineSimilarity(queryEmbedding, record.embedding) }))
    .sort((a, b) => b.similarity - a.similarity);
  const denseRankedIds = denseScored.map((r) => r.id);
  const denseScoreMap = new Map(denseScored.map((r) => [r.id, r.similarity]));

  const lexicalResults = lexicalIndex.search(question);
  const lexicalRankedIds = lexicalResults.map((r) => r.id);
  const lexicalScoreMap = new Map(lexicalResults.map((r) => [r.id, r.score]));

  const fused = rrfFuse([denseRankedIds, lexicalRankedIds]);
  const recordById = new Map(store.records.map((r) => [r.id, r]));

  const combined = [...fused.entries()]
    .map(([id, rrfScore]) => {
      const record = recordById.get(id);
      if (!record) return null;
      return {
        ...record,
        similarity: denseScoreMap.get(id) ?? 0,
        lexicalScore: lexicalScoreMap.get(id) ?? 0,
        rrfScore,
        finalScore: rrfScore,
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.rrfScore - a.rrfScore);

  // A chunk qualifies if dense similarity clears the threshold or it has a
  // strong lexical hit. We keep lexical-only matches, but still require the
  // term to appear in the chunk rather than allowing any weak fuzzy match.
  const qualifying = combined.filter((r) => r.similarity >= minSimilarity || r.lexicalScore >= 0.85);

  if (CONFIG.RAG_DEBUG) {
    console.log(`\n[RAG_DEBUG] Query: "${question}"`);
    console.log(
      `[RAG_DEBUG] Top dense:`,
      denseScored.slice(0, 5).map((r) => `${r.id}=${r.similarity.toFixed(3)}`)
    );
    console.log(
      `[RAG_DEBUG] Top lexical:`,
      lexicalResults.slice(0, 5).map((r) => `${r.id}=${r.score.toFixed(3)}`)
    );
    console.log(
      `[RAG_DEBUG] Top fused:`,
      combined
        .slice(0, 8)
        .map((r) => `${r.id}=${r.rrfScore.toFixed(4)} (dense=${r.similarity.toFixed(3)}, lex=${r.lexicalScore.toFixed(3)})`)
    );
  }

  const selected = [];
  const seenSources = new Map();
  const seenTexts = new Set();

  for (const item of qualifying) {
    if (selected.length >= topK) break;
    const sourceKey = `${item.site}::${item.sourceUrl}`;
    const textKey = item.text.slice(0, 160).toLowerCase();
    const sourceCount = seenSources.get(sourceKey) || 0;
    if (sourceCount >= CONFIG.MAX_FROM_SOURCE) continue;
    if (seenTexts.has(textKey)) continue;
    seenSources.set(sourceKey, sourceCount + 1);
    seenTexts.add(textKey);
    selected.push(item);
  }

  selected.confidence = selected.length
    ? Math.max(...selected.map((item) => Math.max(item.similarity || 0, item.lexicalScore || 0)))
    : 0;

  return selected;
}

export function formatRetrievedContext(chunks) {
  if (!chunks.length) return "";
  return chunks
    .map((chunk, index) => {
      const sectionInfo = chunk.section ? `\nSection: ${chunk.section}` : "";
      const pageInfo = chunk.pageNumber ? `\nPage: ${chunk.pageNumber}` : "";
      return `SOURCE ${index + 1}\nSite: ${chunk.site}\nTitle: ${chunk.title}${sectionInfo}${pageInfo}\nURL: ${chunk.sourceUrl}\nType: ${chunk.type}\nContent:\n${chunk.text}`;
    })
    .join("\n\n");
}
