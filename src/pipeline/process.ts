import {
  insertFact,
  insertEmbedding,
  markFiltered,
  logProcessing,
  listUnprocessedMessages,
  getRecentContext,
  type DB,
  type UnprocessedMessage,
} from '../memory/db';
import type { LLMProvider } from '../llm/provider';

const CONTEXT_WINDOW = 10;

export interface ProcessOptions {
  batchSize?: number;
  log?: (line: string) => void;
}

export interface ProcessStats {
  scanned: number;
  kept: number;
  dropped: number;
  facts: number;
  errors: number;
}

export async function processBatch(
  db: DB,
  provider: LLMProvider,
  opts: ProcessOptions = {}
): Promise<ProcessStats> {
  const limit = opts.batchSize ?? 10;
  const log = opts.log ?? (() => {});
  const messages = listUnprocessedMessages(db, limit);

  const stats: ProcessStats = {
    scanned: messages.length,
    kept: 0,
    dropped: 0,
    facts: 0,
    errors: 0,
  };

  for (const msg of messages) {
    try {
      const factsAdded = await processOne(db, provider, msg, log);
      if (factsAdded === null) {
        stats.dropped++;
      } else {
        stats.kept++;
        stats.facts += factsAdded;
      }
    } catch (err) {
      stats.errors++;
      log(`  msg ${msg.id} ERROR: ${err instanceof Error ? err.message : err}`);
    }
  }

  return stats;
}

async function processOne(
  db: DB,
  provider: LLMProvider,
  msg: UnprocessedMessage,
  log: (line: string) => void
): Promise<number | null> {
  const senderName = msg.direction === 'out' ? 'me' : (msg.chat_display_name ?? msg.sender_wa_id);
  const contactLabel = msg.chat_display_name ?? msg.chat_wa_id;

  const contextRows = getRecentContext(db, msg.contact_id, msg.id, CONTEXT_WINDOW);
  const recentContext = contextRows.map((r) =>
    r.direction === 'out' ? `me: ${r.body}` : `${contactLabel}: ${r.body}`
  );

  const filterResult = await provider.filter({
    body: msg.body,
    contactName: msg.chat_display_name,
    contactNotes: msg.chat_notes,
    isGroup: msg.is_group,
    senderName,
    recentContext,
  });

  logProcessing(db, {
    msg_id: msg.id,
    stage: 'filter',
    model: filterResult.usage.model,
    tokens_in: filterResult.usage.tokens_in,
    tokens_out: filterResult.usage.tokens_out,
  });

  if (!filterResult.keep) {
    markFiltered(db, msg.id, false);
    log(`  msg ${msg.id} drop: ${filterResult.reason}`);
    return null;
  }

  const extractResult = await provider.extract({
    body: msg.body,
    contactName: msg.chat_display_name,
    contactNotes: msg.chat_notes,
    ts: msg.ts,
    isGroup: msg.is_group,
    senderName,
    recentContext,
  });

  logProcessing(db, {
    msg_id: msg.id,
    stage: 'extract',
    model: extractResult.usage.model,
    tokens_in: extractResult.usage.tokens_in,
    tokens_out: extractResult.usage.tokens_out,
  });

  if (extractResult.facts.length === 0) {
    markFiltered(db, msg.id, true);
    log(`  msg ${msg.id} kept but extracted 0 facts`);
    return 0;
  }

  const embedded: Array<{
    fact: (typeof extractResult.facts)[number];
    vector: number[];
    embedUsage: { model: string; tokens_in: number; tokens_out: number };
  }> = [];
  for (const fact of extractResult.facts) {
    const embedResult = await provider.embed(fact.content);
    embedded.push({ fact, vector: embedResult.vector, embedUsage: embedResult.usage });
  }

  const writeTx = db.transaction(() => {
    for (const { fact, vector, embedUsage } of embedded) {
      const factId = insertFact(db, {
        source_msg_id: msg.id,
        subject: fact.subject,
        category: fact.category,
        content: fact.content,
        confidence: fact.confidence,
      });
      insertEmbedding(db, factId, vector);
      logProcessing(db, {
        msg_id: msg.id,
        stage: 'embed',
        model: embedUsage.model,
        tokens_in: embedUsage.tokens_in,
        tokens_out: embedUsage.tokens_out,
      });
    }
    markFiltered(db, msg.id, true);
  });
  writeTx();

  for (const { fact } of embedded) {
    log(`  msg ${msg.id} fact[${fact.category}]: ${fact.content}`);
  }

  return embedded.length;
}

export async function processUntilDrained(
  db: DB,
  provider: LLMProvider,
  opts: ProcessOptions = {}
): Promise<ProcessStats> {
  const total: ProcessStats = {
    scanned: 0,
    kept: 0,
    dropped: 0,
    facts: 0,
    errors: 0,
  };

  while (true) {
    const stats = await processBatch(db, provider, opts);
    total.scanned += stats.scanned;
    total.kept += stats.kept;
    total.dropped += stats.dropped;
    total.facts += stats.facts;
    total.errors += stats.errors;

    if (stats.scanned === 0) break;
  }

  return total;
}
