// server.js — Express backend: chat (streaming SSE), booking, Vapi webhook
// All free: Gemini (LLM + embeddings), Pinecone (vector DB), Cal.com (calendar)

import "dotenv/config";
import express from "express";
import cors from "cors";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";

const app = express();
app.use(cors());
app.use(express.json());

const genai = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

// ── Change this ───────────────────────────────────────────────────────────────
const YOUR_NAME = "YOUR NAME HERE"; // e.g. "Arjun Sharma"
// ─────────────────────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are an AI persona representing ${YOUR_NAME}. Speak in first person as them.
Only answer based on the context provided. If something isn't in the context, say so honestly — never make up details.
Be specific about projects, technologies, and tradeoffs. Keep answers concise but compelling.`;

// ── RAG helpers ───────────────────────────────────────────────────────────────

async function embedQuery(text) {
  // Uses Gemini text-embedding-004 (free, 768-dim) — same model used in ingest.js
  const model = genai.getGenerativeModel({ model: "text-embedding-004" });
  const result = await model.embedContent(text);
  return result.embedding.values;
}

async function retrieve(query) {
  const vector = await embedQuery(query);
  const index = pinecone.index(process.env.PINECONE_INDEX);
  const result = await index.query({ vector, topK: 5, includeMetadata: true });
  return result.matches.map((m) => m.metadata.text).join("\n\n---\n\n");
}

async function ask(query, history = []) {
  const context = await retrieve(query);
  const model = genai.getGenerativeModel({
    model: "gemini-1.5-flash",
    systemInstruction: SYSTEM_PROMPT,
  });
  const chat = model.startChat({
    history: history.slice(-10).map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    })),
  });
  const result = await chat.sendMessage(
    `Context:\n${context}\n\n---\n\nQuestion: ${query}`
  );
  return result.response.text();
}

async function askStream(query, history, onChunk) {
  const context = await retrieve(query);
  const model = genai.getGenerativeModel({
    model: "gemini-1.5-flash",
    systemInstruction: SYSTEM_PROMPT,
  });
  const chat = model.startChat({
    history: history.slice(-10).map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    })),
  });
  const result = await chat.sendMessageStream(
    `Context:\n${context}\n\n---\n\nQuestion: ${query}`
  );
  let full = "";
  for await (const chunk of result.stream) {
    const text = chunk.text();
    full += text;
    onChunk(text);
  }
  return full;
}

// ── Calendar (Cal.com — free tier) ────────────────────────────────────────────

async function getSlots() {
  const now = new Date();
  const end = new Date(now);
  end.setDate(end.getDate() + 7);
  const url = `https://api.cal.com/v1/availability?apiKey=${process.env.CALCOM_API_KEY}&username=${process.env.CALCOM_USERNAME}&dateFrom=${now.toISOString()}&dateTo=${end.toISOString()}&eventTypeId=${process.env.CALCOM_EVENT_TYPE_ID}`;
  const res = await fetch(url);
  const data = await res.json();
  return Object.values(data.slots || {})
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
    `https://api.cal.com/v1/bookings?apiKey=${process.env.CALCOM_API_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        eventTypeId: parseInt(process.env.CALCOM_EVENT_TYPE_ID),
        start: slotIso,
        end: new Date(
          new Date(slotIso).getTime() + 30 * 60 * 1000
        ).toISOString(),
        responses: { name, email, notes },
        timeZone: "UTC",
        language: "en",
        metadata: {},
      }),
    }
  );
  const data = await res.json();
  return {
    uid: data.uid,
    startTime: data.startTime,
    meetLink: data.videoCallData?.url,
  };
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get("/health", (_, res) => res.json({ ok: true }));

// Chat — streaming SSE
app.post("/chat", async (req, res) => {
  const { message, history = [] } = req.body;

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    let q = message;
    // Inject live availability if the question is booking-related
    if (/available|schedule|book|slot|meeting|call|time/i.test(message)) {
      const slots = await getSlots();
      if (slots.length) {
        q += `\n\n[My available slots this week: ${slots
          .map((s) => s.human)
          .join(", ")}]`;
      }
    }

    await askStream(q, history, (chunk) => {
      res.write(`data: ${JSON.stringify({ chunk })}\n\n`);
    });

    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
  } catch (err) {
    console.error("Chat error:", err);
    res.write(
      `data: ${JSON.stringify({
        chunk: "Sorry, something went wrong. Please try again.",
        done: true,
      })}\n\n`
    );
  }

  res.end();
});

// Get available booking slots
app.get("/slots", async (_, res) => {
  try {
    const slots = await getSlots();
    res.json({ slots });
  } catch (err) {
    console.error("Slots error:", err);
    res.status(500).json({ error: "Could not fetch slots" });
  }
});

// Book a slot
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

// ── Vapi webhook (voice agent) ────────────────────────────────────────────────
// Vapi calls this URL during a phone call.
// On first call (no message type): returns the assistant config.
// On function-call: handles getSlots / bookMeeting.
// On transcript: does RAG and returns the answer.

app.post("/vapi", async (req, res) => {
  const { message } = req.body;

  // Function calls triggered by the voice agent mid-call
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

  // Transcript query — answer using RAG
  if (message?.type === "transcript" && message.transcript) {
    try {
      const answer = await ask(message.transcript);
      return res.json({ result: answer });
    } catch {
      return res.json({
        result: "I had trouble answering that. Could you rephrase?",
      });
    }
  }

  // First call — return assistant configuration to Vapi
  res.json({
    assistant: {
      name: `${YOUR_NAME} AI`,
      firstMessage: `Hi! I'm the AI representative for ${YOUR_NAME}. You can ask me about my background, projects, or book a call with me. How can I help?`,
      model: {
        provider: "google",
        model: "gemini-1.5-flash",
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
                email: {
                  type: "string",
                  description: "Caller's email address",
                },
                slotIso: {
                  type: "string",
                  description: "ISO datetime of the chosen slot from getSlots",
                },
                notes: {
                  type: "string",
                  description: "Optional notes about the meeting",
                },
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
  console.log(`✅ Server running on http://localhost:${process.env.PORT || 3001}`)
);
