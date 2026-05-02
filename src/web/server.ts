import 'dotenv/config';
import express, { type Request, type Response } from 'express';
import path from 'path';
import {
  openDb,
  listFacts,
  listCategories,
  searchFactsByVector,
} from '../memory/db';
import { createLLMProvider } from '../llm/claude';
import { loadWhitelist, syncWhitelistToDb } from '../config/whitelist';
import { importChat } from '../whatsapp/import-chat';

const dbPath = process.env.DB_PATH ?? './data/memory.db';
const whitelistPath = process.env.WHITELIST_PATH ?? './config/whitelist.json';
const port = Number(process.env.WEB_PORT ?? 3000);

const db = openDb(dbPath);
const provider = createLLMProvider();

// Whitelist is loaded on every request hitting /api/whitelist or /api/import-chat
// so edits to the JSON file take effect without restarting the server.
function getWhitelist() {
  const wl = loadWhitelist(whitelistPath);
  syncWhitelistToDb(db, wl);
  return wl;
}

const app = express();
// Chat exports can be a few MB for power chats; bump well past Express's
// default 100kb so the upload doesn't bounce.
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/facts', (req: Request, res: Response) => {
  try {
    const subject = typeof req.query.subject === 'string' ? req.query.subject : undefined;
    const category = typeof req.query.category === 'string' ? req.query.category : undefined;
    const q = typeof req.query.q === 'string' ? req.query.q : undefined;
    const limit = req.query.limit ? Math.min(Number(req.query.limit), 500) : 200;

    const facts = listFacts(db, { subject, category, contains: q }, limit);
    const categories = listCategories(db);
    res.json({ facts, categories });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.post('/api/chat', async (req: Request, res: Response) => {
  try {
    const { question } = req.body as { question?: unknown };
    if (typeof question !== 'string' || !question.trim()) {
      return res.status(400).json({ error: 'question is required' });
    }

    const { vector } = await provider.embed(question, 'query');
    const matches = searchFactsByVector(db, vector, 10);

    const factsForChat = matches.map((m) => ({
      id: m.id,
      content: m.content,
      confidence: m.confidence,
      category: m.category,
      subject: m.subject_wa_id,
    }));

    const { answer, usage } = await provider.chat({ question, facts: factsForChat });

    res.json({
      answer,
      citations: matches,
      usage,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/whitelist', (_req: Request, res: Response) => {
  try {
    const wl = getWhitelist();
    res.json({ entries: wl.entries() });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.post('/api/import-chat', (req: Request, res: Response) => {
  try {
    const { content, waId, me } = req.body as {
      content?: unknown;
      waId?: unknown;
      me?: unknown;
    };
    if (typeof content !== 'string' || !content.trim()) {
      return res.status(400).json({ error: 'content (the file text) is required' });
    }
    if (typeof waId !== 'string' || !waId.trim()) {
      return res.status(400).json({ error: 'waId is required' });
    }
    if (typeof me !== 'string' || !me.trim()) {
      return res.status(400).json({ error: 'me (your display name in the export) is required' });
    }

    const wl = getWhitelist();
    const wlEntry = wl.get(waId);
    if (!wlEntry) {
      return res.status(400).json({
        error: `${waId} is not in the whitelist (${whitelistPath}). Add it first, then retry.`,
      });
    }

    const stats = importChat(db, {
      content,
      waId,
      me,
      displayName: wlEntry.display_name ?? null,
    });

    if (stats.emitted === 0) {
      return res.status(422).json({
        error:
          "no messages emitted — check the 'me' display name matches the export's sender labels exactly, and the date format is DD/MM/YY",
        stats,
      });
    }

    res.json({ stats });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.listen(port, () => {
  console.log(`manila web: http://localhost:${port}`);
});
