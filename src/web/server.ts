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

const dbPath = process.env.DB_PATH ?? './data/memory.db';
const port = Number(process.env.WEB_PORT ?? 3000);

const db = openDb(dbPath);
const provider = createLLMProvider();

const app = express();
app.use(express.json());
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

app.listen(port, () => {
  console.log(`manila web: http://localhost:${port}`);
});
