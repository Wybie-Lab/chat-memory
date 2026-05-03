import 'dotenv/config';
import { openDb, rebuildAllBursts } from '../src/engine';

async function main() {
  const dbPath = process.env.DB_PATH ?? './data/memory.db';
  const db = openDb(dbPath);

  const facts = db.prepare('SELECT COUNT(*) AS n FROM facts').get() as { n: number };
  const embeddings = db
    .prepare('SELECT COUNT(*) AS n FROM fact_embeddings')
    .get() as { n: number };
  const bursts = db.prepare('SELECT COUNT(*) AS n FROM conversation_bursts').get() as {
    n: number;
  };
  const processed = db
    .prepare("SELECT COUNT(*) AS n FROM raw_messages WHERE filter_kept IS NOT NULL")
    .get() as { n: number };
  const logs = db.prepare('SELECT COUNT(*) AS n FROM processing_log').get() as { n: number };

  console.log(
    `before: facts=${facts.n} embeddings=${embeddings.n} bursts=${bursts.n} processed_msgs=${processed.n} logs=${logs.n}`
  );

  // Order matters: raw_messages.burst_id and facts.source_* are FK-tracked,
  // so wipe child rows / null out FK columns before deleting parent rows.
  const wipe = db.transaction(() => {
    db.exec('DELETE FROM facts;');
    db.exec('DELETE FROM fact_embeddings;');
    db.exec('DELETE FROM processing_log;');
    db.exec('UPDATE raw_messages SET filter_kept = NULL, processed_at = NULL, burst_id = NULL;');
    db.exec('DELETE FROM conversation_bursts;');
  });
  wipe();

  console.log('reset complete — raw_messages preserved, all derived state cleared');

  console.log('rebuilding bursts from raw_messages...');
  const rb = rebuildAllBursts(db);
  console.log(`bursts: ${rb.bursts} burst(s) covering ${rb.messages} message(s)`);
  console.log('ready for `npm run process`');
  process.exit(0);
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
