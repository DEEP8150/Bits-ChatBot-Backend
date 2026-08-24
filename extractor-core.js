import axios from "axios";
import * as cheerio from "cheerio";
import https from "node:https";
import { mkdir, writeFile } from "node:fs/promises";
import { chromium } from "playwright";
import { CONFIG } from "./config.js";

export const SITES = [
  { start: "https://swd.bits-pilani.ac.in/index.aspx", label: "SWD" },
  { start: "https://admissions.bits-pilani.ac.in/index.html", label: "Admissions" },
];

export const MAX_PAGES_PER_SITE = CONFIG.MAX_PAGES_PER_SITE;
export const MAX_DEPTH = CONFIG.MAX_DEPTH;
export const REQUEST_DELAY_MS = CONFIG.REQUEST_DELAY_MS;

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

// --- Section-aware extraction -------------------------------------------
// Walks block-level elements in document order and groups text under the
// nearest preceding heading. Falls back to one section with no heading if
// the page has no headings at all.
export function extractSections($) {
  cleanDom($);
  const root = $("main").length ? $("main") : $("body");
  const sections = [];
  let current = { heading: null, parts: [] };

  root.find("h1, h2, h3, h4, h5, h6, p, li, td, blockquote").each((_, el) => {
    const $el = $(el);
    const tag = el.tagName ? el.tagName.toLowerCase() : "";
    const text = normalize($el.text());
    if (!text) return;

    if (/^h[1-6]$/.test(tag)) {
      if (current.parts.length) sections.push({ heading: current.heading, text: current.parts.join(" ") });
      current = { heading: text, parts: [] };
    } else {
      current.parts.push(text);
    }
  });

  if (current.parts.length) sections.push({ heading: current.heading, text: current.parts.join(" ") });

  if (!sections.length) {
    const fullText = normalize(root.text());
    if (fullText) sections.push({ heading: null, text: fullText });
  }

  return sections;
}

export function extractCleanText($) {
  return extractSections($).map((s) => s.text).join("\n\n");
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

// --- Sitemap / robots discovery ------------------------------------------
async function discoverSitemapUrls(site, logger = console) {
  const base = new URL(site.start);
  const candidates = [`${base.origin}/sitemap.xml`, `${base.origin}/sitemap_index.xml`];
  const discovered = new Set();

  try {
    const { data: robots } = await axios.get(`${base.origin}/robots.txt`, {
      timeout: 10000,
      headers: HEADERS,
      httpsAgent: insecureAgent,
    });
    for (const match of robots.matchAll(/^Sitemap:\s*(\S+)/gim)) {
      candidates.push(match[1]);
    }
  } catch {
    /* no robots.txt, ignore */
  }

  for (const sitemapUrl of candidates) {
    try {
      const { data: xml } = await axios.get(sitemapUrl, {
        timeout: 15000,
        headers: HEADERS,
        httpsAgent: insecureAgent,
      });
      const $ = cheerio.load(xml, { xmlMode: true });

      const nestedSitemaps = [];
      $("sitemap > loc").each((_, el) => nestedSitemaps.push($(el).text().trim()));
      $("url > loc").each((_, el) => discovered.add($(el).text().trim()));

      for (const nested of nestedSitemaps) {
        try {
          const { data: nestedXml } = await axios.get(nested, {
            timeout: 15000,
            headers: HEADERS,
            httpsAgent: insecureAgent,
          });
          const $$ = cheerio.load(nestedXml, { xmlMode: true });
          $$("url > loc").each((_, el) => discovered.add($$(el).text().trim()));
        } catch {
          /* skip broken nested sitemap */
        }
      }
    } catch {
      /* sitemap not found at this candidate url, try next */
    }
  }

  const filtered = [...discovered].filter((url) => {
    try {
      return new URL(url).hostname === base.hostname;
    } catch {
      return false;
    }
  });

  logger.log(`[${site.label}] Sitemap discovery found ${filtered.length} URLs`);
  return filtered;
}

async function getBrowser() {
  if (!browserPromise) {
    browserPromise = chromium.launch({ headless: true });
  }
  return browserPromise;
}

// --- Hidden content expansion (accordions/tabs/show-more) ----------------
const DANGEROUS_TEXT_PATTERN = /log ?out|sign ?out|delete|remove|submit|pay|checkout|download|unsubscribe/i;
const EXPAND_TEXT_PATTERN = /show more|load more|read more|view more|see more|expand/i;

async function expandHiddenContent(page) {
  try {
    const detailsHandles = await page.$$("details:not([open])");
    for (const handle of detailsHandles) {
      try {
        await handle.evaluate((el) => el.setAttribute("open", "true"));
      } catch {
        /* ignore individual failures */
      }
    }

    const toggles = await page.$$('[aria-expanded="false"]');
    for (const toggle of toggles) {
      try {
        const text = (await toggle.innerText().catch(() => "")) || "";
        if (DANGEROUS_TEXT_PATTERN.test(text)) continue;
        await toggle.click({ timeout: 2000 }).catch(() => {});
        await page.waitForTimeout(150);
      } catch {
        /* ignore individual failures */
      }
    }

    for (let attempt = 0; attempt < 3; attempt += 1) {
      const buttons = await page.$$("button, a[role=button], [role=tab]");
      let clickedAny = false;
      for (const button of buttons) {
        try {
          const text = (await button.innerText().catch(() => "")) || "";
          if (!EXPAND_TEXT_PATTERN.test(text)) continue;
          if (DANGEROUS_TEXT_PATTERN.test(text)) continue;
          const typeAttr = await button.getAttribute("type").catch(() => null);
          if (typeAttr === "submit") continue;
          await button.click({ timeout: 2000 }).catch(() => {});
          clickedAny = true;
          await page.waitForTimeout(200);
        } catch {
          /* ignore individual failures */
        }
      }
      if (!clickedAny) break;
    }
  } catch (err) {
    console.warn(`Hidden-content expansion issue: ${err.message}`);
  }
}

async function extractWithPlaywright(url) {
  const browser = await getBrowser();
  // ignoreHTTPSErrors + matching User-Agent — same settings that already
  // worked in the original crawler. This is the config that was likely
  // missing when the Crawlee migration broke navigation.
  const context = await browser.newContext({
    ignoreHTTPSErrors: true,
    userAgent: HEADERS["User-Agent"],
  });
  const page = await context.newPage();
  try {
    await page.setExtraHTTPHeaders({ "Accept-Language": HEADERS["Accept-Language"] });
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.waitForSelector("main, article, [role='main'], body", { timeout: 8000 }).catch(() => {});
    await expandHiddenContent(page);
    await page.waitForTimeout(500);
    const html = await page.content();
    const $ = cheerio.load(html);
    const sections = extractSections($);
    return {
      title: normalize(await page.title()) || url,
      text: sections.map((s) => s.text).join("\n\n"),
      sections,
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
  const sections = extractSections($);
  return { html, $, title, text: sections.map((s) => s.text).join("\n\n"), sections };
}

async function crawlSite(site, logger = console) {
  const visited = new Set();
  const queue = [{ url: site.start, depth: 0 }];
  const records = [];
  const pdfCandidates = new Map();
  const rejectedPages = [];
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
    sitemapUrls: 0,
  };

  const sitemapUrls = await discoverSitemapUrls(site, logger);
  stats.sitemapUrls = sitemapUrls.length;
  for (const url of sitemapUrls) {
    if (isPdfUrl(url)) {
      if (!pdfCandidates.has(url)) {
        pdfCandidates.set(url, { url, site: site.label, discoveredFrom: "sitemap" });
        stats.pdfsDiscovered += 1;
      }
    } else {
      queue.push({ url, depth: 0 });
    }
  }

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
      rejectedPages.push({ url, reason: `fetch failed: ${err.message}` });
      logger.warn(`Skipped page: ${err.message}`);
      continue;
    }

    logger.log(`HTTP HTML: ${page.html.length.toLocaleString()} chars`);
    logger.log(`Cheerio extracted: ${wordCount(page.text).toLocaleString()} words`);

    let finalText = page.text;
    let finalTitle = page.title;
    let finalSections = page.sections;

    if (isUsefulContent(page.text)) {
      stats.axiosSuccesses += 1;
    } else {
      logger.log(`Falling back to Playwright...`);
      try {
        const fallback = await extractWithPlaywright(url);
        finalText = fallback.text;
        finalTitle = fallback.title || finalTitle;
        finalSections = fallback.sections;
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
        sections: finalSections,
      });
      stats.htmlExtracted += 1;
      logger.log(`Saved page ✓`);
    } else {
      stats.htmlSkipped += 1;
      rejectedPages.push({ url, reason: "no useful content", wordCount: wordCount(finalText) });
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

  return { records, pdfCandidates, stats, rejectedPages };
}

// --- PDF extraction with page-level text via pdfjs-dist -------------------
async function extractPdf(url) {
  try {
    const { data: buffer } = await axios.get(url, {
      responseType: "arraybuffer",
      timeout: 30000,
      headers: HEADERS,
      httpsAgent: insecureAgent,
    });

    const pdfjsLib = await import("pdfjs-dist/legacy/build/pdf.mjs");
    const loadingTask = pdfjsLib.getDocument({ data: new Uint8Array(buffer), disableWorker: true });
    const pdf = await loadingTask.promise;

    const pages = [];
    for (let i = 1; i <= pdf.numPages; i += 1) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      const text = normalize(content.items.map((item) => item.str).join(" "));
      pages.push({ pageNumber: i, text });
    }

    return { text: normalize(pages.map((p) => p.text).join("\n\n")), pages };
  } catch (err) {
    console.warn(`PDF extraction failed for ${url}: ${err.message}`);
    return null;
  }
}

export async function runExtraction({ logger = console } = {}) {
  const allRecords = [];
  const allRejectedPages = [];
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
    sitemapUrls: 0,
    sites: {},
  };
  const pdfEntries = [];

  try {
    for (const site of SITES) {
      const { records, pdfCandidates, stats, rejectedPages } = await crawlSite(site, logger);
      allRecords.push(...records);
      allRejectedPages.push(...rejectedPages.map((r) => ({ ...r, site: site.label })));
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
      const result = await extractPdf(pdf.url);
      if (result && isUsefulContent(result.text)) {
        allRecords.push({
          id: stableId(pdf.site, pdf.url),
          type: "pdf",
          site: pdf.site,
          title: decodeURIComponent(pdf.url.split("/").pop() || pdf.url),
          sourceUrl: pdf.url,
          discoveredFrom: pdf.discoveredFrom,
          text: result.text,
          pages: result.pages,
        });
        summary.pdfsExtracted += 1;
        logger.log(`Extracted: ${wordCount(result.text).toLocaleString()} words across ${result.pages.length} pages`);
        logger.log(`Saved PDF ✓`);
      } else {
        summary.pdfsSkipped += 1;
        allRejectedPages.push({ url: pdf.url, site: pdf.site, reason: "PDF unparseable or empty" });
        logger.warn(`Skipped PDF`);
      }
      await sleep(REQUEST_DELAY_MS);
    }

    await mkdir(".", { recursive: true });
    await writeFile("site-content.json", JSON.stringify(allRecords, null, 2), "utf8");
    await writeFile("site-content.txt", allRecords.map(pageTextToDebug).join("\n"), "utf8");
    await writeFile("rejected-pages.json", JSON.stringify(allRejectedPages, null, 2), "utf8");

    logger.log(`\nExtraction complete`);
    logger.log(`Sitemap URLs discovered: ${summary.sitemapUrls}`);
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
    logger.log(`Rejected pages logged: ${allRejectedPages.length} (see rejected-pages.json)`);
    logger.log(`site-content.json written`);
    logger.log(`site-content.txt written`);

    return { records: allRecords, summary, rejectedPages: allRejectedPages };
  } finally {
    const browser = browserPromise ? await browserPromise.catch(() => null) : null;
    await browser?.close().catch(() => {});
    browserPromise = null;
  }
}