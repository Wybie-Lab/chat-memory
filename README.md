# manila

Personal WhatsApp memory layer. Ingests messages from your WhatsApp, extracts durable facts about the people in your life via LLM, and stores them in a queryable memory (SQLite + vector embeddings). Query through a CLI or a small web UI with RAG-backed chat and a fact browser.

> **WhatsApp ban risk.** This uses [`whatsapp-web.js`](https://github.com/pedroslopez/whatsapp-web.js), which drives the Web client through Puppeteer and is not an official API. Personal accounts can be banned. Use at your own risk; do not run on an account you can't afford to lose.

## How it works

```
WhatsApp → raw_messages → conversation bursts → filter → extract → consolidate → facts + embeddings
                                                                                           ↓
                                                                             query (CLI / web RAG chat)
```

- **Burst-level extraction.** A burst = contiguous run of messages with <30 min gaps. Filter and extract run once per burst with full context, not per message.
- **Consolidation.** New facts are merged against existing memory for the same subject (`ADD` / `UPDATE` / `DELETE` / `DROP`).
- **Whitelist-only.** Only chats listed in `config/whitelist.json` are ingested. Everything else is dropped at the source.
- **LLM provider abstraction.** Default generative provider is OpenRouter through the Vercel AI SDK. Embeddings remain Cohere `embed-multilingual-v3.0` because the current vector index is 1024-dimensional.

## Requirements

- Node.js 20+
- An OpenRouter API key
- A Cohere API key (free tier is fine for personal use)

## Setup

```bash
npm install

cp .env.example .env
# set OPENROUTER_API_KEY, COHERE_API_KEY, and optionally OPENROUTER_MODEL

cp config/whitelist.example.json config/whitelist.json
# add the contacts you want to remember (wa_id like "393331234567@c.us")
```

## Usage

```bash
# Live ingest. First run prints a QR code — scan with WhatsApp → Linked Devices.
npm run dev

# Backfill the last 30 days of the whitelisted chats (run after the live client
# has authenticated at least once).
npm run backfill

# One-off import of a WhatsApp "Export Chat" .txt file.
npm run import-chat -- --file path/to/_chat.txt --wa-id 393331234567@c.us --me "Your Name"

# Process queued bursts (filter → extract → consolidate → embed). Idempotent.
npm run process

# Build/inspect the optional source-backed knowledge graph from active facts.
npm run rebuild-graph -- --limit 20 --dry-run
npm run rebuild-graph -- --force
# Set ENABLE_GRAPH=1 before `npm run process` to write graph rows during normal processing.

# Web UI: chat (RAG), fact browser, and a chat-export uploader.
npm run web
# → http://localhost:3000
```

## Storage

- `data/memory.db` — SQLite with `sqlite-vec`. Schema in `src/engine/storage/schema.sql`. Holds raw messages, bursts, facts, embeddings, graph rows, and a processing log.
- `data/session/` — `whatsapp-web.js` LocalAuth session (lets you skip the QR after the first scan).
- `config/whitelist.json` — your contacts (gitignored).

All of these are gitignored. Fact text is sent to OpenRouter for generation and Cohere for embeddings.

## License

MIT. See [LICENSE](LICENSE).
