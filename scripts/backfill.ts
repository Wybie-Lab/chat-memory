import 'dotenv/config';
import { createWAClient } from '../src/whatsapp/client';
import {
  openDb,
  upsertContact,
  insertRawMessage,
  rebuildAllBursts,
} from '../src/memory/db';
import { loadWhitelist, syncWhitelistToDb } from '../src/config/whitelist';

async function main() {
  const sessionPath = process.env.SESSION_PATH ?? './data/session';
  const dbPath = process.env.DB_PATH ?? './data/memory.db';
  const whitelistPath = process.env.WHITELIST_PATH ?? './config/whitelist.json';
  const days = Number(process.env.BACKFILL_DAYS ?? 30);
  const limit = Number(process.env.BACKFILL_LIMIT ?? 2000);

  const cutoff = Math.floor(Date.now() / 1000) - days * 86400;

  const db = openDb(dbPath);
  const whitelist = loadWhitelist(whitelistPath);
  syncWhitelistToDb(db, whitelist);

  if (whitelist.size() === 0) {
    console.log('whitelist is empty — nothing to backfill');
    process.exit(0);
  }

  console.log(`backfilling last ${days} days for ${whitelist.size()} chat(s)...`);

  const client = createWAClient({ sessionPath });

  let shuttingDown = false;
  const shutdown = async (signal: string) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`\n${signal} received — closing WhatsApp session cleanly...`);
    try {
      await client.destroy();
    } catch (err) {
      console.error('destroy error:', err instanceof Error ? err.message : err);
    }
    process.exit(130);
  };
  process.on('SIGINT', () => void shutdown('SIGINT'));
  process.on('SIGTERM', () => void shutdown('SIGTERM'));

  await new Promise<void>((resolve, reject) => {
    client.once('ready', () => resolve());
    client.once('auth_failure', (err) => reject(new Error(`auth failure: ${err}`)));
    client.initialize();
  });

  console.log('connected');

  let totalInserted = 0;
  let totalDuplicate = 0;
  let totalOlder = 0;

  for (const entry of whitelist.entries()) {
    const label = entry.display_name ? `${entry.wa_id} (${entry.display_name})` : entry.wa_id;
    try {
      const chat = await client.getChatById(entry.wa_id);
      const messages = await chat.fetchMessages({ limit });

      let inserted = 0;
      let duplicate = 0;
      let older = 0;

      for (const msg of messages) {
        if ((msg.timestamp ?? 0) < cutoff) {
          older++;
          continue;
        }
        const senderWaId = msg.fromMe ? (msg.to ?? entry.wa_id) : (msg.from ?? entry.wa_id);
        const direction: 'in' | 'out' = msg.fromMe ? 'out' : 'in';
        const ts = msg.timestamp ?? Math.floor(Date.now() / 1000);

        const contactId = upsertContact(db, {
          wa_id: entry.wa_id,
          display_name: entry.display_name ?? chat.name ?? null,
          is_group: chat.isGroup,
          ts,
        });

        const ok = insertRawMessage(db, {
          wa_msg_id: msg.id._serialized,
          contact_id: contactId,
          sender_wa_id: senderWaId,
          direction,
          body: msg.body ?? '',
          ts,
          media_type: msg.hasMedia ? msg.type : null,
          media_pointer: null,
        });

        if (ok === null) duplicate++;
        else inserted++;
      }

      console.log(
        `  ${label}: fetched=${messages.length} inserted=${inserted} duplicate=${duplicate} older=${older}`
      );
      totalInserted += inserted;
      totalDuplicate += duplicate;
      totalOlder += older;
    } catch (err) {
      console.error(`  ${label}: ERROR`, err instanceof Error ? err.message : err);
    }
  }

  console.log(
    `\ntotal: inserted=${totalInserted} duplicate=${totalDuplicate} older=${totalOlder}`
  );

  console.log('rebuilding conversation bursts from raw_messages...');
  const burstStats = rebuildAllBursts(db);
  console.log(
    `bursts: ${burstStats.bursts} burst(s) covering ${burstStats.messages} message(s)`
  );

  await client.destroy();
  process.exit(0);
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
