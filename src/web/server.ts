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
  searchEntities,
  graphNeighborhood,
  graphCounts,
  graphTimeline,
  listGraphEntitiesWithStats,
  listKnowledgeEdges,
  graphForFact,
  planAgentRun,
  runCurator,
  drainPlannedAgentRuns,
  applyAgentRun,
  applyAgentAction,
  rejectAgentAction,
  listAgentRuns,
  getAgentRun,
  getAgentAction,
  listAgentActionsForRun,
  listMemoryThreads,
  getMemoryThread,
  listFactsInThread,
  listFactThreads,
  listConnectionsFromFact,
  listConnectionsToFact,
  type AgentRunStatus,
} from '../engine';
import { createLLMProvider } from '../llm';
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

app.get('/api/facts/:id/graph', (req: Request<{ id: string }>, res: Response) => {
  try {
    const factId = Number(req.params.id);
    if (!Number.isInteger(factId) || factId <= 0) {
      return res.status(400).json({ error: 'fact id must be a positive integer' });
    }
    res.json(graphForFact(db, factId));
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

app.get('/api/graph/search', (req: Request, res: Response) => {
  try {
    const q = typeof req.query.q === 'string' ? req.query.q.trim() : '';
    const limit = req.query.limit ? Math.min(Number(req.query.limit), 100) : 20;
    if (!q) return res.status(400).json({ error: 'q is required' });
    res.json({ entities: searchEntities(db, q, limit) });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/graph', (req: Request, res: Response) => {
  try {
    const entityLimit = req.query.entity_limit ? Math.min(Number(req.query.entity_limit), 1000) : 500;
    const edgeLimit = req.query.edge_limit ? Math.min(Number(req.query.edge_limit), 2000) : 1000;
    const at =
      typeof req.query.at === 'string' && req.query.at.trim()
        ? Number(req.query.at)
        : null;
    if (at !== null && (!Number.isFinite(at) || at <= 0)) {
      return res.status(400).json({ error: 'at must be a positive unix timestamp' });
    }
    res.json({
      counts: graphCounts(db, at),
      timeline: graphTimeline(db),
      entities: listGraphEntitiesWithStats(db, entityLimit, at),
      edges: listKnowledgeEdges(db, edgeLimit, at),
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/graph/timeline', (req: Request, res: Response) => {
  try {
    const limit = req.query.limit ? Math.min(Number(req.query.limit), 2000) : 500;
    res.json(graphTimeline(db, limit));
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/graph/entity/:id', (req: Request<{ id: string }>, res: Response) => {
  try {
    const entityId = Number(req.params.id);
    if (!Number.isInteger(entityId) || entityId <= 0) {
      return res.status(400).json({ error: 'entity id must be a positive integer' });
    }
    const limit = req.query.limit ? Math.min(Number(req.query.limit), 200) : 100;
    res.json({ edges: graphNeighborhood(db, entityId, { limit }) });
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

// ───────────── Curator (proposal-only v1) ─────────────

app.post('/api/curator/run', async (req: Request, res: Response) => {
  try {
    const body = req.body as {
      scope_type?: unknown;
      scope_ref?: unknown;
      budget_ops?: unknown;
      budget_llm_calls?: unknown;
    };
    const scopeType = body.scope_type;
    if (scopeType !== 'subject' && scopeType !== 'entity') {
      return res
        .status(400)
        .json({ error: "scope_type must be 'subject' or 'entity'" });
    }
    if (typeof body.scope_ref !== 'string' || !body.scope_ref.trim()) {
      return res.status(400).json({ error: 'scope_ref is required' });
    }
    const budgetOps =
      typeof body.budget_ops === 'number' && body.budget_ops > 0
        ? Math.floor(body.budget_ops)
        : undefined;
    const budgetLlmCalls =
      typeof body.budget_llm_calls === 'number' && body.budget_llm_calls > 0
        ? Math.floor(body.budget_llm_calls)
        : undefined;

    const runId = planAgentRun(db, {
      trigger: 'manual',
      scope_type: scopeType,
      scope_ref: body.scope_ref.trim(),
      budget_ops: budgetOps,
      budget_llm_calls: budgetLlmCalls,
    });
    const result = await runCurator(db, provider, runId);
    const actions = listAgentActionsForRun(db, runId);
    res.json({
      run: result.run,
      actions,
      proposed_action_count: result.proposed_action_count,
      llm_calls_used: result.llm_calls_used,
      reasoning: result.reasoning,
      error: result.error,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/curator/runs', (req: Request, res: Response) => {
  try {
    const status =
      typeof req.query.status === 'string'
        ? (req.query.status as AgentRunStatus)
        : undefined;
    const scopeType =
      typeof req.query.scope_type === 'string'
        ? (req.query.scope_type as 'subject' | 'entity')
        : undefined;
    const scopeRef =
      typeof req.query.scope_ref === 'string' ? req.query.scope_ref : undefined;
    const limit = req.query.limit ? Math.min(Number(req.query.limit), 200) : 50;
    res.json({
      runs: listAgentRuns(db, { status, scope_type: scopeType, scope_ref: scopeRef }, limit),
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/curator/runs/:id', (req: Request<{ id: string }>, res: Response) => {
  try {
    const runId = Number(req.params.id);
    if (!Number.isInteger(runId) || runId <= 0) {
      return res.status(400).json({ error: 'run id must be a positive integer' });
    }
    const run = getAgentRun(db, runId);
    if (!run) return res.status(404).json({ error: `run ${runId} not found` });
    res.json({ run, actions: listAgentActionsForRun(db, runId) });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// Apply ALL proposed actions in a run, in seq order. Stale actions are
// skipped; cluster summaries are refreshed after the fact mutations land.
app.post('/api/curator/runs/:id/apply', async (req: Request<{ id: string }>, res: Response) => {
  try {
    const runId = Number(req.params.id);
    if (!Number.isInteger(runId) || runId <= 0) {
      return res.status(400).json({ error: 'run id must be a positive integer' });
    }
    const approvedBy = approvedByFromBody(req.body);
    const result = await applyAgentRun(db, provider, runId, { approvedBy });
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// Apply ONE proposed action in isolation. Run-level status is left untouched
// so the caller can review/apply piecemeal before finalizing the run.
app.post(
  '/api/curator/actions/:id/apply',
  async (req: Request<{ id: string }>, res: Response) => {
    try {
      const actionId = Number(req.params.id);
      if (!Number.isInteger(actionId) || actionId <= 0) {
        return res.status(400).json({ error: 'action id must be a positive integer' });
      }
      const approvedBy = approvedByFromBody(req.body);
      const result = await applyAgentAction(db, provider, actionId, { approvedBy });
      res.json(result);
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  }
);

app.post('/api/curator/actions/:id/reject', (req: Request<{ id: string }>, res: Response) => {
  try {
    const actionId = Number(req.params.id);
    if (!Number.isInteger(actionId) || actionId <= 0) {
      return res.status(400).json({ error: 'action id must be a positive integer' });
    }
    const reason =
      typeof (req.body as { reason?: unknown })?.reason === 'string'
        ? ((req.body as { reason: string }).reason).trim()
        : '';
    if (!reason) return res.status(400).json({ error: 'reason is required' });
    const action = rejectAgentAction(db, actionId, reason);
    res.json({ action });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// Drain queued curator runs (status='planned'). Each run is executed via
// runCurator; per-run errors are caught so the drainer doesn't abort.
app.post('/api/curator/drain', async (req: Request, res: Response) => {
  try {
    const limit =
      typeof (req.body as { limit?: unknown })?.limit === 'number' &&
      (req.body as { limit: number }).limit > 0
        ? Math.min(Math.floor((req.body as { limit: number }).limit), 25)
        : undefined;
    const stats = await drainPlannedAgentRuns(db, provider, { limit });
    res.json(stats);
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/curator/actions/:id', (req: Request<{ id: string }>, res: Response) => {
  try {
    const actionId = Number(req.params.id);
    if (!Number.isInteger(actionId) || actionId <= 0) {
      return res.status(400).json({ error: 'action id must be a positive integer' });
    }
    const action = getAgentAction(db, actionId);
    if (!action) return res.status(404).json({ error: `action ${actionId} not found` });
    res.json({ action });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

function approvedByFromBody(body: unknown): string {
  const raw = (body as { approved_by?: unknown })?.approved_by;
  if (typeof raw === 'string' && raw.trim()) return raw.trim();
  return 'user:web';
}

// ───────────── Append-only memory: threads + connections ─────────────

app.get('/api/threads', (req: Request, res: Response) => {
  try {
    const owner =
      typeof req.query.owner_subject_wa_id === 'string'
        ? req.query.owner_subject_wa_id
        : undefined;
    const threads = listMemoryThreads(db, {
      owner_subject_wa_id: owner,
      active_only: true,
    });
    // Annotate each thread with a fact count for the list view (cheap query).
    const withCounts = threads.map((t) => {
      const facts = listFactsInThread(db, t.id, 1000);
      return { ...t, fact_count: facts.length };
    });
    res.json({ threads: withCounts });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/threads/:id', (req: Request<{ id: string }>, res: Response) => {
  try {
    const threadId = Number(req.params.id);
    if (!Number.isInteger(threadId) || threadId <= 0) {
      return res.status(400).json({ error: 'thread id must be a positive integer' });
    }
    const thread = getMemoryThread(db, threadId);
    if (!thread || thread.deleted_at !== null) {
      return res.status(404).json({ error: `thread ${threadId} not found` });
    }
    const facts = listFactsInThread(db, threadId, 500);
    res.json({ thread, facts });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/facts/:id/threads', (req: Request<{ id: string }>, res: Response) => {
  try {
    const factId = Number(req.params.id);
    if (!Number.isInteger(factId) || factId <= 0) {
      return res.status(400).json({ error: 'fact id must be a positive integer' });
    }
    res.json({ threads: listFactThreads(db, factId) });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.get('/api/facts/:id/connections', (req: Request<{ id: string }>, res: Response) => {
  try {
    const factId = Number(req.params.id);
    if (!Number.isInteger(factId) || factId <= 0) {
      return res.status(400).json({ error: 'fact id must be a positive integer' });
    }
    res.json({
      outgoing: listConnectionsFromFact(db, factId),
      incoming: listConnectionsToFact(db, factId),
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

app.listen(port, () => {
  console.log(`manila web: http://localhost:${port}`);
});
