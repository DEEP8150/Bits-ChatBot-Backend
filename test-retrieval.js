import { retrieveChunks } from "./retrieval.js";

const questions = [
  "What scholarships are available?",
  "Tell me about hostel rules.",
  "What is the anti-ragging policy?",
  "How do I contact SWD?",
  "What student welfare services are available?",
  "What is BITSAT?",
  "What are the admission reporting instructions?",
  "What is the weather today?",
  "Who is the Prime Minister of India?",
  "hello",
];

for (const question of questions) {
  const normalized = question.trim().toLowerCase();
  const isGreeting = /^(hi|hello|hey|thanks|thank you)\b/.test(normalized);
  if (isGreeting) {
    console.log(`\nQ: ${question}`);
    console.log("Greeting detected, retrieval skipped");
    continue;
  }

  const chunks = await retrieveChunks(question, { topK: 6, minSimilarity: 0.32 });
  console.log(`\nQ: ${question}`);
  if (!chunks.length) {
    console.log("No chunks exceeded threshold");
    continue;
  }

  chunks.forEach((chunk, index) => {
    const preview = chunk.text.slice(0, 140).replace(/\s+/g, " ");
    console.log(
      `${index + 1}. similarity=${chunk.finalScore.toFixed(3)} site=${chunk.site} title=${chunk.title} url=${chunk.sourceUrl}`
    );
    console.log(`   preview: ${preview}`);
  });
}
