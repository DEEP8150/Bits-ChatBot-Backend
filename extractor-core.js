import axios from "axios";
import * as cheerio from "cheerio";
import https from "node:https";
import { mkdir, writeFile } from "node:fs/promises";
import { chromium } from "playwright";
import { PDFParse } from "pdf-parse";

export const SITES = [
  { start: "https://swd.bits-pilani.ac.in/index.aspx", label: "SWD" },
  { start: "https://admissions.bits-pilani.ac.in/index.html", label: "Admissions" },
];

export const MAX_PAGES_PER_SITE = 40;
export const MAX_DEPTH = 2;
export const REQUEST_DELAY_MS = 350;

const HEADERS = {
  "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
  Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
  "Accept-Language": "en-US,en;q=0.9",
};

const insecureAgent = new https.Agent({ rejectUnauthorized: false });
let browserPromise = null;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalize(text) {
  return text.replace(/\s+/g, " ").trim();
}

function wordCount(text) {
  return normalize(text).split(/\s+/).filter(Boolean).length;
}

export function isPdfUrl(url) {
  try {
    return new URL(url).pathname.toLowerCase().endsWith(".pdf");
  } catch {
    return /\.pdf(\?|#|$)/i.test(url);
  }
}

function isLikelyBoilerplate(text) {
  const lower = text.toLowerCase();
  return (
    lower.includes("cookie") ||
    lower.includes("privacy policy") ||
    lower.includes("google tag manager") ||
    lower.includes("captcha") ||
    lower.includes("tracking") ||
    lower.includes("analytics")
  );
}

export function isUsefulContent(text) {
  const normalized = normalize(text || "");
  if (normalized.length < 200) return false;
  if (wordCount(normalized) < 40) return false;
  const letters = (normalized.match(/[A-Za-z]/g) || []).length;
  const ratio = letters / Math.max(normalized.length, 1);
  if (ratio < 0.45) return false;
  if (isLikelyBoilerplate(normalized) && wordCount(normalized) < 120) return false;
  return true;
}

function cleanDom($) {
  $("script, style, noscript, iframe, svg, canvas, input, button, select, textarea").remove();
  $("[aria-label*='cookie' i], [class*='cookie' i], [id*='cookie' i]").remove();
  $("[class*='gtm' i], [id*='gtm' i], [class*='analytics' i], [id*='analytics' i]").remove();
  $("[class*='tracking' i], [id*='tracking' i]").remove();
}

export function extractCleanText($) {
  cleanDom($);
  const root = $("main").length ? $("main") : $("body");
  return normalize(root.text() || $("body").text() || "");
}

export function extractPdfLinks($, baseUrl) {
  const links = new Set();
  $("a[href]").each((_, el) => {
    const href = $(el).attr("href");
    if (!href || !isPdfUrl(href)) return;
    try {
      links.add(new URL(href, baseUrl).toString());
    } catch {
      /* ignore malformed links */
    }
  });
  return [...links];
}

export function extractPageLinks($, baseUrl) {
  const base = new URL(baseUrl);
  const links = new Set();

  $("a[href]").each((_, el) => {
    const href = $(el).attr("href");
    if (!href) return;
    if (/^(mailto:|tel:|javascript:|#)/i.test(href)) return;
    if (isPdfUrl(href)) return;

    try {
      const resolved = new URL(href, baseUrl);
      resolved.hash = "";
      if (resolved.hostname !== base.hostname) return;
      if (/\.(jpg|jpeg|png|gif|svg|css|js|zip|ico)$/i.test(resolved.pathname)) return;
      links.add(resolved.toString());
    } catch {
      /* ignore malformed links */
    }
  });

  return [...links];
}

async function fetchHtml(url) {
  const { data: html } = await axios.get(url, {
    timeout: 20000,
    headers: HEADERS,
    httpsAgent: insecureAgent,
  });
  return html;
}

async function getBrowser() {
  if (!browserPromise) {
    browserPromise = chromium.launch({ headless: true });
  }
  return browserPromise;
}

async function extractWithPlaywright(url) {
  const browser = await getBrowser();
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  const page = await context.newPage();
  try {
    await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    await page.waitForTimeout(1000);
    const html = await page.content();
    const $ = cheerio.load(html);
    return {
      title: normalize(await page.title()) || url,
      text: extractCleanText($),
    };
  } finally {
    await page.close().catch(() => {});
    await context.close().catch(() => {});
  }
}

function stableId(site, sourceUrl) {
  const raw = `${site}::${sourceUrl}`;
  let hash = 2166136261;
  for (let i = 0; i < raw.length; i += 1) {
    hash ^= raw.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return `rec_${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function pageTextToDebug(record) {
  return `# SITE: ${record.site}\n# TYPE: ${record.type}\n# PAGE: ${record.title}\nSource: ${record.sourceUrl}\n\n${record.text}\n`;
}

async function extractPage(url) {
  const html = await fetchHtml(url);
  const $ = cheerio.load(html);
  const title = normalize($("title").first().text()) || url;
  return { html, $, title, text: extractCleanText($) };
}

async function crawlSite(site, logger = console) {
  const visited = new Set();
  const queue = [{ url: site.start, depth: 0 }];
  const records = [];
  const pdfCandidates = new Map();
  const stats = {
    htmlDiscovered: 0,
    htmlExtracted: 0,
    htmlSkipped: 0,
    axiosSuccesses: 0,
    playwrightSuccesses: 0,
    playwrightFailures: 0,
    pdfsDiscovered: 0,
    pdfsExtracted: 0,
    pdfsSkipped: 0,
  };

  while (queue.length > 0 && visited.size < MAX_PAGES_PER_SITE) {
    const { url, depth } = queue.shift();
    if (visited.has(url)) continue;
    visited.add(url);
    stats.htmlDiscovered += 1;

    logger.log(`[${site.label}] depth=${depth}`);
    logger.log(`Fetching: ${url}`);

    let page;
    try {
      page = await extractPage(url);
    } catch (err) {
      stats.htmlSkipped += 1;
      logger.warn(`Skipped page: ${err.message}`);
      continue;
    }

    logger.log(`HTTP HTML: ${page.html.length.toLocaleString()} chars`);
    logger.log(`Cheerio extracted: ${wordCount(page.text).toLocaleString()} words`);

    let finalText = page.text;
    let finalTitle = page.title;
    if (isUsefulContent(page.text)) {
      stats.axiosSuccesses += 1;
    } else {
      logger.log(`Falling back to Playwright...`);
      try {
        const fallback = await extractWithPlaywright(url);
        finalText = fallback.text;
        finalTitle = fallback.title || finalTitle;
        if (isUsefulContent(finalText)) {
          stats.playwrightSuccesses += 1;
          logger.log(`Playwright extracted: ${wordCount(finalText).toLocaleString()} words`);
        } else {
          stats.playwrightFailures += 1;
          logger.log(`Playwright extracted: ${wordCount(finalText).toLocaleString()} words`);
        }
      } catch (err) {
        stats.playwrightFailures += 1;
        logger.warn(`Playwright failed: ${err.message}`);
      }
    }

    if (isUsefulContent(finalText)) {
      records.push({
        id: stableId(site.label, url),
        type: "page",
        site: site.label,
        title: finalTitle,
        sourceUrl: url,
        text: normalize(finalText),
      });
      stats.htmlExtracted += 1;
      logger.log(`Saved page ✓`);
    } else {
      stats.htmlSkipped += 1;
      logger.warn(`Skipped page: no useful content`);
    }

    for (const pdfUrl of extractPdfLinks(page.$, url)) {
      if (!pdfCandidates.has(pdfUrl)) {
        pdfCandidates.set(pdfUrl, { url: pdfUrl, site: site.label, discoveredFrom: url });
        stats.pdfsDiscovered += 1;
      }
    }

    if (depth < MAX_DEPTH) {
      for (const link of extractPageLinks(page.$, url)) {
        if (!visited.has(link)) queue.push({ url: link, depth: depth + 1 });
      }
    }

    await sleep(REQUEST_DELAY_MS);
  }

  return { records, pdfCandidates, stats };
}

async function extractPdf(url) {
  let parser;
  try {
    const { data: buffer } = await axios.get(url, {
      responseType: "arraybuffer",
      timeout: 30000,
      headers: HEADERS,
      httpsAgent: insecureAgent,
    });
    parser = new PDFParse({ data: buffer });
    const result = await parser.getText();
    return normalize(result.text);
  } catch {
    return null;
  } finally {
    if (parser) await parser.destroy().catch(() => {});
  }
}

export async function runExtraction({ logger = console } = {}) {
  const allRecords = [];
  const summary = {
    htmlDiscovered: 0,
    htmlExtracted: 0,
    htmlSkipped: 0,
    axiosSuccesses: 0,
    playwrightSuccesses: 0,
    playwrightFailures: 0,
    pdfsDiscovered: 0,
    pdfsExtracted: 0,
    pdfsSkipped: 0,
    sites: {},
  };
  const pdfEntries = [];

  try {
    for (const site of SITES) {
      const { records, pdfCandidates, stats } = await crawlSite(site, logger);
      allRecords.push(...records);
      pdfEntries.push(...pdfCandidates.values());
      summary.sites[site.label] = records.length;
      for (const [key, value] of Object.entries(stats)) {
        if (key === "sites") continue;
        summary[key] += value;
      }
    }

    for (const pdf of pdfEntries) {
      logger.log(`[${pdf.site}] PDF`);
      logger.log(`Fetching: ${pdf.url}`);
      const text = await extractPdf(pdf.url);
      if (text && isUsefulContent(text)) {
        allRecords.push({
          id: stableId(pdf.site, pdf.url),
          type: "pdf",
          site: pdf.site,
          title: decodeURIComponent(pdf.url.split("/").pop() || pdf.url),
          sourceUrl: pdf.url,
          discoveredFrom: pdf.discoveredFrom,
          text: normalize(text),
        });
        summary.pdfsExtracted += 1;
        logger.log(`Extracted: ${wordCount(text).toLocaleString()} words`);
        logger.log(`Saved PDF ✓`);
      } else {
        summary.pdfsSkipped += 1;
        logger.warn(`Skipped PDF`);
      }
      await sleep(REQUEST_DELAY_MS);
    }

    await mkdir(".", { recursive: true });
    await writeFile("site-content.json", JSON.stringify(allRecords, null, 2), "utf8");
    await writeFile("site-content.txt", allRecords.map(pageTextToDebug).join("\n"), "utf8");

    logger.log(`\nExtraction complete`);
    logger.log(`HTML pages discovered: ${summary.htmlDiscovered}`);
    logger.log(`HTML pages successfully extracted: ${summary.htmlExtracted}`);
    logger.log(`HTML pages skipped: ${summary.htmlSkipped}`);
    logger.log(`Axios/Cheerio successes: ${summary.axiosSuccesses}`);
    logger.log(`Playwright fallback successes: ${summary.playwrightSuccesses}`);
    logger.log(`Playwright failures: ${summary.playwrightFailures}`);
    logger.log(`PDFs discovered: ${summary.pdfsDiscovered}`);
    logger.log(`PDFs extracted: ${summary.pdfsExtracted}`);
    logger.log(`PDFs skipped: ${summary.pdfsSkipped}`);
    logger.log(`SWD records: ${summary.sites.SWD || 0}`);
    logger.log(`Admissions records: ${summary.sites.Admissions || 0}`);
    logger.log(`Total records: ${allRecords.length}`);
    logger.log(`site-content.json written`);
    logger.log(`site-content.txt written`);

    return { records: allRecords, summary };
  } finally {
    const browser = browserPromise ? await browserPromise.catch(() => null) : null;
    await browser?.close().catch(() => {});
    browserPromise = null;
  }
}
