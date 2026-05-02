import 'dotenv/config';
import { createWAClient } from './whatsapp/client';
import {
  openDb,
  upsertContact,
  insertRawMessage,
  assignMessageToBurst,
  getLiveCutoff,
} from './memory/db';
import { loadWhitelist, syncWhitelistToDb } from './config/whitelist';
import { createLLMProvider } from './llm/claude';
import { processBatch } from './pipeline/process';

async function main() {
  const sessionPath = process.env.SESSION_PATH ?? './data/session';
  const dbPath = process.env.DB_PATH ?? './data/memory.db';
  const whitelistPath = process.env.WHITELIST_PATH ?? './config/whitelist.json';

  const db = openDb(dbPath);
  console.log(`db ready: ${dbPath}`);

  const whitelist = loadWhitelist(whitelistPath);
  syncWhitelistToDb(db, whitelist);
  console.log(`whitelist loaded: ${whitelist.size()} entries`);
  for (const e of whitelist.entries()) {
    console.log(`  • ${e.wa_id}${e.display_name ? ` (${e.display_name})` : ''}`);
  }

  const provider = createLLMProvider();

  const client = createWAClient({
    sessionPath,
    onMessage: async (msg) => {
      const chat = await msg.getChat();
      const chatWaId = chat.id._serialized;

      if (!whitelist.isAllowed(chatWaId)) {
        return;
      }

      const senderContact = await msg.getContact();
      const senderWaId = msg.fromMe ? (msg.to ?? chatWaId) : (msg.from ?? chatWaId);
      const direction: 'in' | 'out' = msg.fromMe ? 'out' : 'in';
      const ts = msg.timestamp ?? Math.floor(Date.now() / 1000);

      const contactId = upsertContact(db, {
        wa_id: chatWaId,
        display_name:
          whitelist.get(chatWaId)?.display_name ??
          chat.name ??
          senderContact.pushname ??
          senderContact.number ??
          null,
        is_group: chat.isGroup,
        ts,
      });

      const cutoff = getLiveCutoff(db, contactId);
      if (cutoff !== null && ts <= cutoff) {
        console.log(`[skip] ${chatWaId} msg @ ${ts} <= live_cutoff ${cutoff} (covered by import)`);
        return;
      }

      const inserted = insertRawMessage(db, {
        wa_msg_id: msg.id._serialized,
        contact_id: contactId,
        sender_wa_id: senderWaId,
        direction,
        body: msg.body ?? '',
        ts,
        media_type: msg.hasMedia ? msg.type : null,
        media_pointer: null,
      });

      if (inserted !== null && (msg.body ?? '').length > 0 && !msg.hasMedia) {
        assignMessageToBurst(db, inserted);
      }

      const tag = inserted === null ? 'DUP' : direction.toUpperCase();
      const senderLabel =
        senderContact.pushname ?? senderContact.name ?? senderContact.number ?? senderWaId;
      console.log(
        `[${tag}] ${chat.isGroup ? `(group: ${chat.name}) ` : ''}${senderLabel}: ${msg.body?.slice(0, 120) ?? ''}`
      );
    },
  });

  let running = true;
  let shuttingDown = false;
  const shutdown = async (signal: string) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`\n${signal} received — shutting down cleanly...`);
    running = false;
    try {
      await client.destroy();
    } catch (err) {
      console.error('destroy error:', err instanceof Error ? err.message : err);
    }
    process.exit(0);
  };
  process.on('SIGINT', () => void shutdown('SIGINT'));
  process.on('SIGTERM', () => void shutdown('SIGTERM'));

  const workerLoop = async () => {
    while (running) {
      try {
        const stats = await processBatch(db, provider, {
          batchSize: 5,
          log: (line) => console.log(`[pipeline] ${line}`),
        });
        if (stats.bursts_scanned > 0) {
          console.log(
            `[pipeline] bursts=${stats.bursts_scanned} kept=${stats.bursts_kept}/dropped=${stats.bursts_dropped} | facts: +${stats.facts_added} updated=${stats.facts_updated} deleted=${stats.facts_deleted} dup=${stats.facts_dropped} guarded=${stats.facts_guarded} | errors=${stats.errors}`
          );
        }
        await sleep(stats.bursts_scanned > 0 ? 500 : 5000);
      } catch (err) {
        console.error('[pipeline] loop error:', err);
        await sleep(10000);
      }
    }
  };

  console.log('manila starting...');
  await client.initialize();
  void workerLoop();
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
