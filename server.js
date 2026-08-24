import express from "express";
import cors from "cors";
import axios from "axios";
import { readFile, writeFile, unlink, mkdir, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { formatRetrievedContext, retrieveChunks } from "./retrieval.js";
import { CONFIG } from "./config.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const execFileAsync = promisify(execFile);
const PORT = Number(process.env.PORT || 4000);
const LLAMACPP_URL = process.env.LLAMACPP_URL || "http://127.0.0.1:8080/v1/chat/completions";
const LLAMACPP_MODEL = process.env.LLAMACPP_MODEL || "default";
const PIPER_SERVER_URL = process.env.PIPER_SERVER_URL || "http://localhost:5001";
const RHUBARB_BIN_PATH = path.resolve(__dirname, process.env.RHUBARB_BIN_PATH || "../rhubarb/rhubarb.exe");
const TTS_TEMP_DIR = path.join(__dirname, "tts", "temp");
const TEMP_CLEANUP_MS = 10 * 60 * 1000;
const MAX_HISTORY_MESSAGES = 8;
const MAX_CONTEXT_CHUNKS = 6;
const MIN_RELEVANCE = 0.32;
const MIN_ANSWER_CONFIDENCE = 0.45;

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use("/tts/temp", express.static(TTS_TEMP_DIR, { maxAge: 0, etag: false }));

const pendingCleanupTimers = new Map();

async function ensureTempDir() {
  await mkdir(TTS_TEMP_DIR, { recursive: true });
}

async function cleanupTempFiles(basePath) {
  const targets = [basePath, basePath.replace(/\.wav$/i, ".json")];
  await Promise.all(
    targets.map(async (filePath) => {
      try {
        await unlink(filePath);
      } catch (error) {
        if (error?.code !== "ENOENT") {
          console.warn(`Failed to remove temp file ${filePath}:`, error.message);
        }
      }
    })
  );
}

function scheduleTempCleanup(basePath) {
  const existing = pendingCleanupTimers.get(basePath);
  if (existing) clearTimeout(existing);

  const timer = setTimeout(async () => {
    pendingCleanupTimers.delete(basePath);
    await cleanupTempFiles(basePath);
  }, TEMP_CLEANUP_MS);

  pendingCleanupTimers.set(basePath, timer);
}

async function cleanupOldTempFiles() {
  try {
    await ensureTempDir();
    const files = await readdir(TTS_TEMP_DIR);
    const now = Date.now();
    await Promise.all(
      files
        .filter((file) => file.endsWith(".wav") || file.endsWith(".json"))
        .map(async (file) => {
          const filePath = path.join(TTS_TEMP_DIR, file);
          try {
            const fileStat = await stat(filePath);
            if (now - fileStat.mtimeMs > TEMP_CLEANUP_MS) {
              await cleanupTempFiles(filePath.replace(/\.json$/i, ".wav"));
            }
          } catch {
            /* ignore cleanup failures */
          }
        })
    );
  } catch {
    /* ignore cleanup failures */
  }
}

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

async function synthesizeSpeech(text) {
  if (!text?.trim()) {
    throw new Error("text is required");
  }

  await ensureTempDir();
  const id = `response_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  const wavPath = path.join(TTS_TEMP_DIR, `${id}.wav`);
  const jsonPath = path.join(TTS_TEMP_DIR, `${id}.json`);

  console.log("[speak] Requesting Piper audio...");
  const piperUrl = `${PIPER_SERVER_URL.replace(/\/$/, "")}/synthesize`;

  const piperResponse = await fetch(piperUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!piperResponse.ok) {
    const errorText = await piperResponse.text().catch(() => "");
    throw new Error(`Piper request failed: ${piperResponse.status} ${errorText || piperResponse.statusText}`);
  }

  const audioBuffer = Buffer.from(await piperResponse.arrayBuffer());
  await writeFile(wavPath, audioBuffer);

  console.log("[speak] Running Rhubarb...");
  try {
    await execFileAsync(RHUBARB_BIN_PATH, [wavPath, "-o", jsonPath, "-f", "json"], { windowsHide: true });
  } catch (error) {
    throw new Error(`Rhubarb failed: ${error.stderr || error.message}`);
  }

  const jsonText = await readFile(jsonPath, "utf8");
  const parsed = JSON.parse(jsonText);
  const cues = parsed?.mouthCues || [];
  scheduleTempCleanup(wavPath);

  return {
    audioUrl: `/tts/temp/${path.basename(wavPath)}`,
    cues,
  };
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
  const confidence = retrieved.confidence || 0;
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

  if (!selected.length || confidence < MIN_ANSWER_CONFIDENCE) {
    return res.json({
      answer:
        "I couldn't find that in the available SWD information. Please contact the Student Welfare Division for confirmation.",
      sources: selected.map((chunk) => ({
        title: chunk.title,
        url: chunk.sourceUrl,
        site: chunk.site,
      })),
    });
  }

  const promptChars = systemPrompt.length + context.length + question.length + (historyText ? historyText.length : 0);
  console.log(`Prompt estimate: chunks=${selected.length}, contextChars=${context.length}, historyChars=${historyText.length}, totalChars=${promptChars}, approxTokens=${Math.ceil(promptChars / 4)}`);

  const messages = [
    {
      role: "system",
      content: systemPrompt
        .replace("{{RETRIEVED_CONTEXT}}", compressForBudget(context, 12000))
        .replace("{{CHAT_HISTORY}}", historyText ? compressForBudget(historyText, 4000) : "No previous messages."),
    },
  ];

  messages.push({ role: "user", content: question });

  try {
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

    const result = await synthesizeSpeech(answer);

    return res.json({
      answer,
      result,
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

app.post("/api/speak", async (req, res) => {
  const text = String(req.body?.text || "");
  if (!text.trim()) {
    return res.status(400).json({ error: "text is required" });
  }

  try {
    const result = await synthesizeSpeech(text);
    return res.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to synthesize speech";
    console.error("[speak] Error:", message);
    const status = /Piper/i.test(message) ? 502 : /Rhubarb/i.test(message) ? 502 : 500;
    return res.status(status).json({ error: message });
  } finally {
    cleanupOldTempFiles();
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
