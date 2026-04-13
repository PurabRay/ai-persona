# AI Persona — Purab Ray

A fully autonomous AI representative that can be called, chatted with, and used to book a real interview — no human intervention required.

## Live Links

| | |
|---|---|
| Chat | https://ai-persona-hdtk.onrender.com |
| Voice | +1 (253) 321-2312 |
| Repo | https://github.com/PurabRay/ai-persona |

> The voice number is a US number. Call via any VoIP service (Google Voice, Skype) or the Vapi dashboard web test at dashboard.vapi.ai.

---

## Architecture

```
                        User / Recruiter
                        /              \
                  Chat (HTTP)      Voice (Phone)
                       |                |
              Express Server (Node.js) — server.js
              /            |                    \
     Groq LLM        Pinecone (RAG)         Cal.com v2
  llama-3.3-70b    gemini-embedding-001      Booking API
                          |
                     ingest.js (run once)
                   resume.txt + GitHub READMEs
```

**Chat path:** Browser sends message to `/chat` → Pinecone RAG retrieval → Groq LLM streams SSE chunks back → rendered in real time.

**Voice path:** Vapi receives call → sends transcript to `/vapi/chat/completions` (OpenAI-compatible custom LLM endpoint) → RAG retrieval → Groq LLM streams response → ElevenLabs speaks it. `getSlots` and `bookMeeting` are registered as Vapi tools, triggered automatically when the caller asks about availability.

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Groq — llama-3.3-70b-versatile |
| Embeddings | Google Gemini gemini-embedding-001 (3072-dim) |
| Vector DB | Pinecone (serverless, cosine) |
| Voice | Vapi + ElevenLabs + Deepgram nova-2 |
| Calendar | Cal.com API v2 |
| Backend | Node.js + Express |
| Frontend | Vanilla HTML/CSS/JS (SSE streaming) |
| Hosting | Render |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/PurabRay/ai-persona
cd ai-persona
npm install
```

### 2. Environment variables

Create a `.env` file:

```env
GROQ_API_KEY=          # console.groq.com
GEMINI_API_KEY=        # aistudio.google.com
PINECONE_API_KEY=      # app.pinecone.io
PINECONE_INDEX=        # your index name
CALCOM_API_KEY=        # Cal.com > Settings > API Keys
CALCOM_EVENT_TYPE_ID=  # Cal.com > Event Types > URL id
CALCOM_USERNAME=       # your Cal.com username
API_URL=               # your public server URL
PORT=3001
```

### 3. Ingest resume and GitHub repos into Pinecone

```bash
npm run ingest
```

Runs once. Deletes and recreates the Pinecone index, embeds `resume.txt` and the READMEs of 10 priority GitHub repos, upserts them. Takes around 15 minutes on Gemini free tier (rate limited to 5 RPM).

### 4. Run locally

```bash
npm run dev
```

Open `http://localhost:3001` in your browser.

### 5. Deploy

Push to GitHub and connect to Render. Add all env vars in the Render dashboard. Build command: `npm install`, start command: `npm start`.

---

## Voice Agent Setup (Vapi)

1. Create an account at vapi.ai
2. Create a phone number
3. Create an assistant, set provider to Custom LLM, URL to `https://your-app.onrender.com/vapi/chat/completions`
4. Set org-level server URL to `https://your-app.onrender.com/vapi`
5. Assign the phone number to the assistant

The agent handles intro, Q&A, slot fetching, and booking automatically via two registered tools: `getSlots` and `bookMeeting`.

---

## Project Structure

```
ai-persona/
├── server.js      # Express backend: chat SSE, /slots, /book, /vapi webhook, /vapi/chat/completions
├── ingest.js      # One-time RAG ingestion: resume + GitHub READMEs into Pinecone
├── index.html     # Chat frontend with booking modal
├── resume.txt     # Source of truth for resume RAG
└── package.json
```
