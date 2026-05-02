import 'dotenv/config';
import { openDb } from '../src/memory/db';

async function main() {
  const dbPath = process.env.DB_PATH ?? './data/memory.db';
  const db = openDb(dbPath);

  const facts = db.prepare('SELECT COUNT(*) AS n FROM facts').get() as { n: number };
  const embeddings = db
    .prepare('SELECT COUNT(*) AS n FROM fact_embeddings')
    .get() as { n: number };
  const processed = db
    .prepare("SELECT COUNT(*) AS n FROM raw_messages WHERE filter_kept IS NOT NULL")
    .get() as { n: number };
  const logs = db.prepare('SELECT COUNT(*) AS n FROM processing_log').get() as { n: number };

  console.log(`before: facts=${facts.n} embeddings=${embeddings.n} processed_msgs=${processed.n} logs=${logs.n}`);

  db.exec('DELETE FROM facts;');
  db.exec('DELETE FROM fact_embeddings;');
  db.exec('DELETE FROM processing_log;');
  db.exec('UPDATE raw_messages SET filter_kept = NULL, processed_at = NULL;');

  console.log('reset complete — raw_messages preserved, all derived state cleared');
  process.exit(0);
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
