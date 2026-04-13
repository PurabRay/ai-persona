// ingest.js — run once: node ingest.js
// Loads resume.txt + GitHub READMEs only (no source code) into Pinecone for RAG

import "dotenv/config";
import fs from "fs/promises";
import { Pinecone } from "@pinecone-database/pinecone";
import { Octokit } from "octokit";

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

const EMBED_MODEL = "gemini-embedding-001";
const EMBED_DIM = 3072;

// These 10 repos are always ingested first and guaranteed to be included
const PRIORITY_REPOS = [
  "IDP",
  "smart-meeting-scheduler",
  "AI-Resume-Parser",
  "Course-recommender",
  "spreadsheet-app",
  "graph_assignment",
  "pocket-bazaar",
  "LinkedInScraper",
  "Fitpro",
  "pocket-expense",
];

function chunkText(text, source) {
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i = 0; i < words.length; i += 400) {
    chunks.push({ text: words.slice(i, i + 450).join(" "), source });
  }
  return chunks;
}

async function embedOne(text) {
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${EMBED_MODEL}:embedContent?key=${process.env.GEMINI_API_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: `models/${EMBED_MODEL}`,
        content: { parts: [{ text }] },
      }),
    }
  );
  const data = await res.json();
  if (!data.embedding) throw new Error(`Embed failed: ${JSON.stringify(data)}`);
  return data.embedding.values;
}

async function upsert(index, chunks) {
  for (let i = 0; i < chunks.length; i += 20) {
    const batch = chunks.slice(i, i + 20);
    const vectors = [];
    for (const chunk of batch) {
      const values = await embedOne(chunk.text);
      vectors.push({
        id: `${chunk.source}-${i + vectors.length}`,
        values,
        metadata: { text: chunk.text, source: chunk.source },
      });
      // Stay under Gemini free tier rate limit (5 RPM)
      await new Promise((r) => setTimeout(r, 13000));
    }
    await index.upsert(vectors);
    console.log(`  upserted ${i + batch.length}/${chunks.length}`);
  }
}

async function ingestRepo(repo, chunks, label = "") {
  try {
    let content = `Repo: ${repo.name}\nLanguage: ${repo.language || "N/A"}\nDescription: ${repo.description || "No description"}\n\n`;

    // README only — no source code
    try {
      const { data } = await octokit.rest.repos.getReadme({
        owner: repo.owner.login,
        repo: repo.name,
      });
      const readme = Buffer.from(data.content, "base64").toString("utf8");
      content += `README:\n${readme}\n`;
    } catch {
      console.log(`  ⚠️  No README found for ${repo.name}`);
    }

    chunks.push(...chunkText(content, `github:${repo.name}`));
    console.log(`  ✓ ${label || repo.name}`);
  } catch (e) {
    console.log(`  ⚠️  Skipped ${repo.name}: ${e.message}`);
  }
}

async function main() {
  // Recreate index fresh
  const existing = await pinecone.listIndexes();
  if (existing.indexes?.find((x) => x.name === process.env.PINECONE_INDEX)) {
    console.log("Deleting existing Pinecone index for clean re-ingest...");
    await pinecone.deleteIndex(process.env.PINECONE_INDEX);
    await new Promise((r) => setTimeout(r, 10000));
  }

  console.log("Creating Pinecone index (dim=3072 for Gemini embeddings)...");
  await pinecone.createIndex({
    name: process.env.PINECONE_INDEX,
    dimension: EMBED_DIM,
    metric: "cosine",
    spec: { serverless: { cloud: "aws", region: "us-east-1" } },
  });
  console.log("Waiting 15s for index to be ready...");
  await new Promise((r) => setTimeout(r, 15000));

  const index = pinecone.index(process.env.PINECONE_INDEX);
  const chunks = [];

  // ── Resume ────────────────────────────────────────────────────────────────
  console.log("\nLoading resume.txt...");
  const resume = await fs.readFile("resume.txt", "utf8");
  chunks.push(...chunkText(resume, "resume"));
  console.log("  ✓ resume.txt");

  // ── Priority repos — always ingested, guaranteed ──────────────────────────
  console.log(`\nIngesting ${PRIORITY_REPOS.length} priority repos (README only)...`);
  for (const repoName of PRIORITY_REPOS) {
    try {
      const { data: repo } = await octokit.rest.repos.get({
        owner: process.env.GITHUB_USERNAME,
        repo: repoName,
      });
      await ingestRepo(repo, chunks, `${repoName} [PRIORITY]`);
    } catch (e) {
      console.log(`  ⚠️  Could not fetch priority repo ${repoName}: ${e.message}`);
    }
    // Small delay between requests
    await new Promise((r) => setTimeout(r, 2000));
  }

  // ── Upsert all chunks ─────────────────────────────────────────────────────
  console.log(`\nUpserting ${chunks.length} chunks into Pinecone...`);
  console.log("Note: pacing requests to stay within Gemini free tier (5 RPM).");
  console.log(`Estimated time: ~${Math.ceil((chunks.length * 13) / 60)} minutes\n`);
  await upsert(index, chunks);
  console.log("\nDone! Your persona is ready.");
}

main().catch(console.error);