import { runExtraction } from "./extractor-core.js";

runExtraction().catch((err) => {
  console.error("Extraction failed:", err);
  process.exit(1);
});
