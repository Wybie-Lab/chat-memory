import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { openDb } from '../src/memory/db';
import { loadWhitelist, syncWhitelistToDb } from '../src/config/whitelist';
import { importChat } from '../src/whatsapp/import-chat';

interface Args {
  file: string;
  waId: string;
  me: string;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  const get = (flag: string): string | undefined => {
    const i = args.indexOf(flag);
    if (i < 0 || i === args.length - 1) return undefined;
    return args[i + 1];
  };
  const file = get('--file');
  const waId = get('--wa-id');
  const me = get('--me');
  if (!file || !waId || !me) {
    console.error('usage: tsx scripts/import-chat.ts --file <path> --wa-id <jid@c.us> --me "<your display name>"');
    process.exit(2);
  }
  return { file, waId, me };
}

async function main() {
  const args = parseArgs();
  const dbPath = process.env.DB_PATH ?? './data/memory.db';
  const whitelistPath = process.env.WHITELIST_PATH ?? './config/whitelist.json';

  const content = fs.readFileSync(path.resolve(args.file), 'utf-8');
  console.log(`reading ${args.file}: ${content.length.toLocaleString()} bytes`);

  const db = openDb(dbPath);
  const whitelist = loadWhitelist(whitelistPath);
  syncWhitelistToDb(db, whitelist);

  const wlEntry = whitelist.get(args.waId);
  if (!wlEntry) {
    console.error(`${args.waId} is not in the whitelist (${whitelistPath}). Add it first.`);
    process.exit(1);
  }

  const stats = importChat(db, {
    content,
    waId: args.waId,
    me: args.me,
    displayName: wlEntry.display_name ?? null,
  });

  console.log(
    `parsed: ${stats.parse.total} headers → text=${stats.parse.text} media=${stats.parse.media} skipped_system=${stats.parse.skipped_system} skipped_empty=${stats.parse.skipped_empty}`
  );
  if (stats.emitted === 0) {
    console.error('no messages emitted — check the --me display name matches the export and the date format is DD/MM/YY');
    process.exit(1);
  }
  console.log(`inserted=${stats.inserted} duplicate=${stats.duplicate}`);
  console.log(`bursts: ${stats.bursts} burst(s) covering ${stats.burst_messages} message(s)`);
  if (stats.live_cutoff_ts !== null) {
    console.log(
      `live_cutoff set to ${new Date(stats.live_cutoff_ts * 1000).toISOString()} for ${args.waId} — live ingest will skip msgs <= this ts`
    );
  }
  process.exit(0);
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
