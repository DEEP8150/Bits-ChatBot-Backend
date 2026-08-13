import express from "express";
import cors from "cors";
import axios from "axios";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { formatRetrievedContext, retrieveChunks } from "./retrieval.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = Number(process.env.PORT || 4000);
const LLAMACPP_URL = process.env.LLAMACPP_URL || "http://127.0.0.1:8080/v1/chat/completions";
const LLAMACPP_MODEL = process.env.LLAMACPP_MODEL || "default";
const MAX_HISTORY_MESSAGES = 8;
const MAX_CONTEXT_CHUNKS = 6;
const MIN_RELEVANCE = 0.32;

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

async function loadSystemPrompt() {
  return readFile(path.join(__dirname, "system-prompt.txt"), "utf8");
}

function normalizeHistory(history = []) {
  return history
    .slice(-MAX_HISTORY_MESSAGES)
    .map((item) => {
      if (!item) return null;
      const role = item.role || item.sender || item.type || "user";
      const content = item.content || item.message || item.text || "";
      if (!content) return null;
      return { role, content };
    })
    .filter(Boolean);
}

function buildHistoryText(history = []) {
  return normalizeHistory(history)
    .map((item) => `${item.role.toUpperCase()}: ${item.content}`)
    .join("\n");
}

function compressForBudget(text, maxChars) {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars - 200)}\n[truncated for context budget]`;
}

function isGreeting(question) {
  return /^(hi|hello|hey|thanks|thank you)\b/i.test(question.trim());
}

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.post("/api/chat", async (req, res) => {
  const { question = "", history = [] } = req.body || {};
  if (!question.trim()) {
    return res.status(400).json({ error: "question is required" });
  }

  if (isGreeting(question)) {
    return res.json({
      answer: "Hello! I can help with BITS Pilani student welfare, scholarships, hostels, anti-ragging information, admissions, and related services.",
      sources: [],
    });
  }

  const retrieved = await retrieveChunks(question, { topK: MAX_CONTEXT_CHUNKS + 2, minSimilarity: MIN_RELEVANCE });
  const selected = retrieved.slice(0, MAX_CONTEXT_CHUNKS);
  const context = formatRetrievedContext(selected);
  const systemPrompt = await loadSystemPrompt();
  const historyText = buildHistoryText(history);

  console.log(`Question:\n"${question}"`);
  console.log("Retrieved:");
  if (!selected.length) {
    console.log("No chunks passed relevance filtering");
  } else {
    selected.forEach((chunk, index) => {
      console.log(
        `${index + 1}. similarity=${chunk.finalScore.toFixed(3)} site=${chunk.site} title=${chunk.title} url=${chunk.sourceUrl}`
      );
    });
  }

  const promptChars = systemPrompt.length + context.length + question.length + (historyText ? historyText.length : 0);
  console.log(`Prompt estimate: chunks=${selected.length}, contextChars=${context.length}, historyChars=${historyText.length}, totalChars=${promptChars}, approxTokens=${Math.ceil(promptChars / 4)}`);

  const messages = [
    {
      role: "system",
      content: systemPrompt.replace("{{RETRIEVED_CONTEXT}}", compressForBudget(context, 12000)),
    },
  ];

  if (historyText) {
    messages.push({
      role: "system",
      content: `Conversation history so far:\n${compressForBudget(historyText, 4000)}`,
    });
  }

  messages.push({ role: "user", content: question });

  try {
    if (!selected.length) {
      return res.json({
        answer: "I couldn't find that in the available SWD information.",
        sources: [],
      });
    }

    const response = await axios.post(
      LLAMACPP_URL,
      {
        model: LLAMACPP_MODEL,
        messages,
        temperature: 0.2,
      },
      { timeout: 60000 }
    );

    const answer =
      response.data?.choices?.[0]?.message?.content ||
      response.data?.choices?.[0]?.text ||
      response.data?.response ||
      "";

    return res.json({
      answer,
      sources: selected.map((chunk) => ({
        title: chunk.title,
        url: chunk.sourceUrl,
        site: chunk.site,
      })),
    });
  } catch (err) {
    return res.status(502).json({
      error: "Failed to generate response",
      details: err.message,
    });
  }
});

const server = app.listen(PORT, () => {
  console.log(`SWD chatbot backend listening on http://localhost:${PORT}`);
});

server.on("error", (err) => {
  if (err.code === "EADDRINUSE") {
    console.error(`Port ${PORT} is already in use. Stop the existing process on that port or set PORT to a different value.`);
    process.exit(1);
  }
  throw err;
});
