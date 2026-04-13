// server.js — Express backend: chat (streaming SSE), booking, Vapi webhook

import "dotenv/config";
import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static("."));

const groq = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

const YOUR_NAME = "Purab Ray";

const SYSTEM_PROMPT = `You are Purab Ray, speaking naturally in first person as yourself.

Core rules (NEVER break these):
- ONLY use information from the provided Context (resume + GitHub repos). 
- If something is not in the Context, reply honestly: "I haven't worked on that specifically" or "I don't have details about that in my documented experience/projects."
- Never hallucinate projects, companies, technologies, or achievements.
- Prioritize and lead with your strongest experiences first:
  1. AI Research Intern at WNS Vuram (Mar–Jun 2025) — Intelligent Document Processing app with LLMs/SLMs, OCR on scanned/unscanned PDFs/PNGs/JPEGs, batch processing, bounding-box search.
  2. Resource Recommender & Resume Parser projects (Llama 3.1 API, React + Node, ATS scoring + skill-gap recommendations with learning resources).
  3. Full-stack Expense Tracker (React + Node + Express + MongoDB, real-time charts, filtering).
  4. Other roles: Zinier (low-code frontend), innoverv (vendor peortal), Undivided Capital (founder scraping + ranking algorithms).
  5. The Smart meeting scheduler that deals with RL that you will find in the repository.
- Technical strengths you should highlight: Generative AI, LLMs, OCR, React, Node.js, Python, MongoDB, MySQL.
- For GitHub repos: only talk about them when the context actually contains their README. Prefer higher-impact repos.
- Keep answers concise, professional, confident but humble, and compelling.
A couple of points:
1) if you are asked for a project and you don't know about it, do not tell the recruiter you do not have info on what they asked, tell them you can't at the moment access the info regarding it
2) Repositories like techcrunch_analysis, Linkedinscraping, unfundedtech reddit data refer strictly to the undivided capital internship so refer those repositories if asked about that internship.
3) the IDP project refers to the WNS Vuram internship, though its not all of it. its only the github prototype you started with, the real project involved more so do say that, but also if asked in depth stick to describing that repo
if asked regarding CGR point out your 10th and 12th grades and the fact you were more focused on internships.`;
// ── RAG helpers ───────────────────────────────────────────────────────────────

async function embedQuery(text) {
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=${process.env.GEMINI_API_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "models/gemini-embedding-001",
        content: { parts: [{ text }] },
      }),
    }
  );
  const data = await res.json();
  if (!data.embedding) throw new Error(`Embed failed: ${JSON.stringify(data)}`);
  return data.embedding.values;
}

async function retrieve(query) {
  const vector = await embedQuery(query);
  const index = pinecone.index(process.env.PINECONE_INDEX);
  const result = await index.query({ vector, topK: 5, includeMetadata: true });
  return result.matches.map((m) => m.metadata.text).join("\n\n---\n\n");
}

async function ask(query, history = []) {
  const context = await retrieve(query);
  const res = await groq.chat.completions.create({
    model: "llama-3.3-70b-versatile",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      ...history.slice(-10),
      { role: "user", content: `Context:\n${context}\n\n---\n\nQuestion: ${query}` },
    ],
  });
  return res.choices[0].message.content;
}

async function askStream(query, history, onChunk) {
  const context = await retrieve(query);
  const stream = await groq.chat.completions.create({
    model: "llama-3.3-70b-versatile",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      ...history.slice(-10),
      { role: "user", content: `Context:\n${context}\n\n---\n\nQuestion: ${query}` },
    ],
    stream: true,
  });
  let full = "";
  for await (const chunk of stream) {
    const text = chunk.choices[0]?.delta?.content || "";
    full += text;
    if (text) onChunk(text);
  }
  return full;
}

app.post("/vapi/chat/completions", async (req, res) => {
  const { messages } = req.body;
  const userMessage = messages.filter(m => m.role === "user").pop()?.content || "";
  const history = messages.filter(m => m.role !== "system");

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    await askStream(userMessage, history, (chunk) => {
      res.write(`data: ${JSON.stringify({
        choices: [{ delta: { content: chunk }, finish_reason: null }]
      })}\n\n`);
    });
    res.write(`data: ${JSON.stringify({
      choices: [{ delta: {}, finish_reason: "stop" }]
    })}\n\n`);
    res.write("data: [DONE]\n\n");
  } catch (err) {
    console.error("Vapi LLM error:", err);
  }
  res.end();
});

// ── Calendar (Cal.com — v2 API) ───────────────────────────────────────────────

const CAL_HEADERS = {
  "Content-Type": "application/json",
  "Authorization": `Bearer ${process.env.CALCOM_API_KEY}`,
  "cal-api-version": "2024-08-13",
};

async function getSlots() {
  const now = new Date();
  const end = new Date(now);
  end.setDate(end.getDate() + 7);
  const url = `https://api.cal.com/v2/slots/available?eventTypeId=${process.env.CALCOM_EVENT_TYPE_ID}&startTime=${now.toISOString()}&endTime=${end.toISOString()}`;
  const res = await fetch(url, { headers: CAL_HEADERS });
  const data = await res.json();
  console.log("Cal.com raw response:", JSON.stringify(data, null, 2));
  // v2 returns { status: "success", data: { slots: { "2024-01-01": [{time: "..."}], ... } } }
  const slotsByDay = data?.data?.slots || {};
  return Object.values(slotsByDay)
    .flat()
    .slice(0, 10)
    .map((s) => ({
      iso: s.time,
      human: new Date(s.time).toLocaleString("en-US", {
        weekday: "short",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        timeZoneName: "short",
      }),
    }));
}

async function bookSlot({ name, email, slotIso, notes = "" }) {
  const res = await fetch(
    `https://api.cal.com/v2/bookings`,
    {
      method: "POST",
      headers: CAL_HEADERS,
      body: JSON.stringify({
        eventTypeId: parseInt(process.env.CALCOM_EVENT_TYPE_ID),
        start: slotIso,
        attendee: {
          name,
          email,
          timeZone: "UTC",
          language: "en",
        },
        metadata: { notes },
      }),
    }
  );
  const data = await res.json();
  return {
    uid: data?.data?.uid,
    startTime: data?.data?.start,
    meetLink: data?.data?.meetingUrl,
  };
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get("/health", (_, res) => res.json({ ok: true }));

app.post("/chat", async (req, res) => {
  const { message, history = [] } = req.body;

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    let q = message;
    if (/available|schedule|book|slot|meeting|call|time/i.test(message)) {
      const slots = await getSlots();
      if (slots.length) {
        q += `\n\n[My available slots this week: ${slots.map((s) => s.human).join(", ")}]`;
      }
    }

    await askStream(q, history, (chunk) => {
      res.write(`data: ${JSON.stringify({ chunk })}\n\n`);
    });

    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
  } catch (err) {
    console.error("Chat error:", err);
    res.write(`data: ${JSON.stringify({ chunk: "Sorry, something went wrong. Please try again.", done: true })}\n\n`);
  }

  res.end();
});

app.get("/slots", async (_, res) => {
  try {
    const slots = await getSlots();
    res.json({ slots });
  } catch (err) {
    console.error("Slots error:", err);
    res.status(500).json({ error: "Could not fetch slots" });
  }
});

app.post("/book", async (req, res) => {
  const { name, email, slotIso, notes } = req.body;
  if (!name || !email || !slotIso)
    return res.status(400).json({ error: "name, email, slotIso required" });
  try {
    const booking = await bookSlot({ name, email, slotIso, notes });
    res.json({ ok: true, booking });
  } catch (err) {
    console.error("Booking error:", err);
    res.status(500).json({ error: "Booking failed" });
  }
});

// ── Vapi webhook ──────────────────────────────────────────────────────────────

app.post("/vapi", async (req, res) => {
  const { message } = req.body;

  if (message?.type === "function-call") {
    const { name, parameters } = message.functionCall;

    if (name === "getSlots") {
      try {
        const slots = await getSlots();
        return res.json({
          result: slots.length
            ? slots.map((s) => s.human).join(", ")
            : "No slots available this week.",
        });
      } catch {
        return res.json({ result: "Could not fetch availability right now." });
      }
    }

    if (name === "bookMeeting") {
      try {
        const booking = await bookSlot(parameters);
        return res.json({
          result: `Done! Meeting booked for ${booking.startTime}. A confirmation has been sent to ${parameters.email}.`,
        });
      } catch {
        return res.json({ result: "Booking failed. Please try again." });
      }
    }
  }

  if (message?.type === "transcript" && message.transcript) {
    try {
      const answer = await ask(message.transcript);
      return res.json({ result: answer });
    } catch {
      return res.json({ result: "I had trouble answering that. Could you rephrase?" });
    }
  }

  res.json({
    assistant: {
      name: `${YOUR_NAME} AI`,
      firstMessage: `Hi! I'm the AI representative for ${YOUR_NAME}. You can ask me about my background, projects, or book a call with me. How can I help?`,
      model: {
        provider: "groq",
        model: "llama-3.3-70b-versatile",
        systemPrompt: SYSTEM_PROMPT,
        functions: [
          {
            name: "getSlots",
            description: "Get available meeting slots for the next 7 days",
            parameters: { type: "object", properties: {} },
          },
          {
            name: "bookMeeting",
            description: "Book a meeting at a specific time slot",
            parameters: {
              type: "object",
              required: ["name", "email", "slotIso"],
              properties: {
                name: { type: "string", description: "Caller's full name" },
                email: { type: "string", description: "Caller's email address" },
                slotIso: { type: "string", description: "ISO datetime of the chosen slot from getSlots" },
                notes: { type: "string", description: "Optional notes about the meeting" },
              },
            },
          },
        ],
      },
      voice: { provider: "11labs", voiceId: "adam" },
      transcriber: { provider: "deepgram", model: "nova-2" },
      serverUrl: `${process.env.API_URL}/vapi`,
    },
  });
});

app.listen(process.env.PORT || 3001, () =>
  console.log(`Server running on http://localhost:${process.env.PORT || 3001}`)
);