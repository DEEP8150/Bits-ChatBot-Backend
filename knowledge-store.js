import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const STORE_DIR = path.join(__dirname, "chroma-store");
export const STORE_FILE = path.join(STORE_DIR, "bits_swd_knowledge.json");

export async function readStore() {
  try {
    const raw = await readFile(STORE_FILE, "utf8");
    return JSON.parse(raw);
  } catch {
    return { collection: "bits_swd_knowledge", records: [] };
  }
}

export async function writeStore(store) {
  await mkdir(STORE_DIR, { recursive: true });
  await writeFile(STORE_FILE, JSON.stringify(store, null, 2), "utf8");
}

export function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}
