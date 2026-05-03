import 'dotenv/config';
import express, { type Request, type Response } from 'express';
import path from 'path';
import {
  openDb,
  listFacts,
  listCategories,
  composeMemoryBlock,
  getBurstQueueStats,
  allActiveSubjects,
  getSubjectInfo,
  factsAboutSubjectWithSource,
  supersededFactsForSubject,
  listClusterSummariesForSubject,
} from '../engine';
import { createLLMProvider } from '../llm/claude';
import {
  loadWhitelist,
  syncWhitelistToDb,
  addContactToWhitelistFile,
} from '../sources/whatsapp/whitelist';
import { importChat } from '../sources/chat-export/import';
import { phoneToWaId } from '../sources/whatsapp/phone';
import { detectOtherSender } from '../sources/chat-export/parse';

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

    const composed = await composeMemoryBlock(db, provider, question);
    const { answer, usage } = await provider.chat({
      question,
      memoryBlock: composed.block,
    });

    res.json({
      answer,
      citations: composed.citations,
      preferences: composed.preferences,
      cluster_summaries: composed.cluster_summaries,
      episodes: composed.episodes,
      matched_subjects: composed.matched_subjects,
      memory_block: composed.block,
      budget: composed.budget,
      usage,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/subjects', (_req: Request, res: Response) => {
  try {
    res.json({ subjects: allActiveSubjects(db) });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/subjects/:wa_id', (req: Request<{ wa_id: string }>, res: Response) => {
  try {
    const waId = req.params.wa_id;
    const subject = getSubjectInfo(db, waId);
    if (!subject) {
      return res.status(404).json({ error: `no active facts for subject ${waId}` });
    }
    res.json({
      subject,
      cluster_summaries: listClusterSummariesForSubject(db, waId),
      facts: factsAboutSubjectWithSource(db, waId),
      superseded: supersededFactsForSubject(db, waId),
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/pipeline-status', (_req: Request, res: Response) => {
  try {
    const stats = getBurstQueueStats(db);
    res.json(stats);
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
    const { content, waId, phone, me, displayName } = req.body as {
      content?: unknown;
      waId?: unknown;
      phone?: unknown;
      me?: unknown;
      displayName?: unknown;
    };
    if (typeof content !== 'string' || !content.trim()) {
      return res.status(400).json({ error: 'content (the file text) is required' });
    }
    if (typeof me !== 'string' || !me.trim()) {
      return res.status(400).json({ error: 'me (your display name in the export) is required' });
    }

    let resolvedWaId: string;
    if (typeof waId === 'string' && waId.trim()) {
      resolvedWaId = waId.trim();
    } else if (typeof phone === 'string' && phone.trim()) {
      try {
        resolvedWaId = phoneToWaId(phone);
      } catch (err) {
        return res
          .status(400)
          .json({ error: err instanceof Error ? err.message : String(err) });
      }
    } else {
      return res
        .status(400)
        .json({ error: 'either phone or waId is required' });
    }

    const explicitDisplay =
      typeof displayName === 'string' && displayName.trim() ? displayName.trim() : null;
    const detectedDisplay = explicitDisplay ? null : detectOtherSender(content, me);
    const finalDisplay = explicitDisplay ?? detectedDisplay;

    const wl = getWhitelist();
    const existing = wl.get(resolvedWaId);
    let whitelistResult: { added: boolean; updated: boolean } = {
      added: false,
      updated: false,
    };
    if (!existing) {
      whitelistResult = addContactToWhitelistFile(whitelistPath, {
        wa_id: resolvedWaId,
        display_name: finalDisplay,
      });
      // Reload so the in-process whitelist + DB sync reflect the new entry.
      getWhitelist();
    } else if (!existing.display_name && finalDisplay) {
      whitelistResult = addContactToWhitelistFile(whitelistPath, {
        wa_id: resolvedWaId,
        display_name: finalDisplay,
      });
      getWhitelist();
    }

    const stats = importChat(db, {
      content,
      waId: resolvedWaId,
      me,
      displayName: finalDisplay ?? existing?.display_name ?? null,
    });

    if (stats.emitted === 0) {
      return res.status(422).json({
        error:
          "no messages emitted — check the 'me' display name matches the export's sender labels exactly, and the date format is DD/MM/YY",
        stats,
        wa_id: resolvedWaId,
        detected_display_name: detectedDisplay,
      });
    }

    res.json({
      stats,
      wa_id: resolvedWaId,
      display_name: finalDisplay ?? existing?.display_name ?? null,
      detected_display_name: detectedDisplay,
      whitelist: whitelistResult,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.listen(port, () => {
  console.log(`manila web: http://localhost:${port}`);
});
