// ingest.js — run once: node ingest.js
// Loads resume.txt + your GitHub repos into Pinecone for RAG
// Uses Gemini text-embedding-004 (free tier, no credit card needed)

import "dotenv/config";
import fs from "fs/promises";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Octokit } from "octokit";

const genai = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const octokit = new Octokit();

// Gemini text-embedding-004 → 768 dimensions (free tier)
const EMBED_MODEL = "text-embedding-004";
const EMBED_DIM = 768;

function chunkText(text, source) {
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i = 0; i < words.length; i += 400) {
    chunks.push({ text: words.slice(i, i + 450).join(" "), source });
  }
  return chunks;
}

// Embed one text at a time with a small delay to respect Gemini free tier (5 RPM)
async function embedOne(text) {
  const model = genai.getGenerativeModel({ model: EMBED_MODEL });
  const result = await model.embedContent(text);
  return result.embedding.values;
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
      // Stay under the free tier rate limit (5 RPM = 1 request per 12s to be safe)
      await new Promise((r) => setTimeout(r, 13000));
    }
    await index.upsert(vectors);
    console.log(`  upserted ${i + batch.length}/${chunks.length}`);
  }
}

async function main() {
  // Create Pinecone index if it doesn't exist — must be dim 768 for Gemini embeddings
  const existing = await pinecone.listIndexes();
  if (!existing.indexes?.find((x) => x.name === process.env.PINECONE_INDEX)) {
    console.log("Creating Pinecone index (dim=768 for Gemini embeddings)...");
    await pinecone.createIndex({
      name: process.env.PINECONE_INDEX,
      dimension: EMBED_DIM,
      metric: "cosine",
      spec: { serverless: { cloud: "aws", region: "us-east-1" } },
    });
    console.log("Waiting 15s for index to be ready...");
    await new Promise((r) => setTimeout(r, 15000));
  }

  const index = pinecone.index(process.env.PINECONE_INDEX);
  const chunks = [];

  // Load resume
  console.log("Loading resume.txt...");
  const resume = await fs.readFile("resume.txt", "utf8");
  chunks.push(...chunkText(resume, "resume"));

  // Load GitHub repos
  console.log(`Loading GitHub repos for ${process.env.GITHUB_USERNAME}...`);
  const { data: repos } = await octokit.rest.repos.listForUser({
    username: process.env.GITHUB_USERNAME,
    sort: "updated",
    per_page: 15,
  });

  for (const repo of repos.filter((r) => !r.fork)) {
    try {
      const { data } = await octokit.rest.repos.getReadme({
        owner: repo.owner.login,
        repo: repo.name,
      });
      const readme = Buffer.from(data.content, "base64").toString("utf8");
      const content = `Repo: ${repo.name}\nDescription: ${repo.description}\nLanguage: ${repo.language}\n\n${readme}`;
      chunks.push(...chunkText(content, `github:${repo.name}`));
      console.log(`  + ${repo.name}`);
    } catch {
      /* no readme — skip */
    }
  }

  console.log(`\nUpserting ${chunks.length} chunks...`);
  console.log("Note: pacing requests to stay within Gemini free tier (5 RPM).");
  console.log(`Estimated time: ~${Math.ceil((chunks.length * 13) / 60)} minutes\n`);
  await upsert(index, chunks);
  console.log("Done! Your persona is ready.");
}

main().catch(console.error);
