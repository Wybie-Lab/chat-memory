import 'dotenv/config';
import {
  clearGraphTables,
  graphCounts,
  listActiveFactsForGraph,
  logProcessing,
  openDb,
  writeExtractedGraph,
  type GraphFactRow,
} from '../src/engine';
import { createLLMProvider } from '../src/llm';
import type { ExtractedGraph, FactCategory, GraphFactInput } from '../src/llm/provider';

interface Args {
  limit?: number;
  subject?: string;
  dryRun: boolean;
  force: boolean;
}

const FACT_CATEGORIES = new Set<string>([
  'preference',
  'event',
  'commitment',
  'fact',
  'relationship',
]);

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  const get = (flag: string): string | undefined => {
    const i = argv.indexOf(flag);
    if (i < 0 || i === argv.length - 1) return undefined;
    return argv[i + 1];
  };

  const limitRaw = get('--limit');
  const limit = limitRaw ? Number(limitRaw) : undefined;
  if (limitRaw && (!Number.isInteger(limit) || limit! <= 0)) {
    throw new Error('--limit must be a positive integer');
  }

  return {
    limit,
    subject: get('--subject'),
    dryRun: argv.includes('--dry-run'),
    force: argv.includes('--force'),
  };
}

async function main() {
  const args = parseArgs();
  const dbPath = process.env.DB_PATH ?? './data/memory.db';
  const db = openDb(dbPath);
  const provider = createLLMProvider();

  const facts = listActiveFactsForGraph(db, {
    subject: args.subject,
    limit: args.limit,
  });

  if (facts.length === 0) {
    console.log('no active facts found for graph rebuild');
    process.exit(0);
  }

  if (!args.dryRun && args.force) {
    const before = graphCounts(db);
    console.log(
      `clearing graph tables: entities=${before.entities} mentions=${before.mentions} edges=${before.edges}`
    );
    clearGraphTables(db);
  }

  if (!args.dryRun && !args.force) {
    console.log('writing graph rows without clearing first; pass --force to rebuild from scratch');
  }

  console.log(
    `${args.dryRun ? 'dry-running' : 'rebuilding'} graph for ${facts.length} fact(s)${
      args.subject ? ` subject=${args.subject}` : ''
    }`
  );

  let factsOk = 0;
  let factsErrored = 0;
  let entities = 0;
  let mentions = 0;
  let edges = 0;
  let skippedEdges = 0;

  for (const fact of facts) {
    if (!FACT_CATEGORIES.has(fact.category)) {
      console.log(`  fact ${fact.id} skipped: unknown category ${fact.category}`);
      continue;
    }

    const input = toGraphFactInput(fact);
    try {
      const result = await provider.extractGraphFromFact(input);
      factsOk++;

      if (args.dryRun) {
        printDryRun(fact, result.graph);
      } else {
        const written = writeExtractedGraph(
          db,
          { ...input, source_burst_id: fact.source_burst_id },
          result.graph
        );
        logProcessing(db, {
          burst_id: fact.source_burst_id,
          stage: 'graph_extract',
          model: result.usage.model,
          tokens_in: result.usage.tokens_in,
          tokens_out: result.usage.tokens_out,
        });
        entities += written.entities;
        mentions += written.mentions;
        edges += written.edges;
        skippedEdges += written.skipped_edges;
        console.log(
          `  fact ${fact.id}: entities=${written.entities} mentions=${written.mentions} edges=${written.edges} skipped_edges=${written.skipped_edges}`
        );
      }
    } catch (err) {
      factsErrored++;
      console.error(`  fact ${fact.id} ERROR: ${err instanceof Error ? err.message : err}`);
    }
  }

  if (args.dryRun) {
    console.log(`\ndry-run done: ok=${factsOk} errors=${factsErrored}`);
  } else {
    const after = graphCounts(db);
    console.log(
      `\ndone: facts_ok=${factsOk} errors=${factsErrored} written entities=${entities} mentions=${mentions} edges=${edges} skipped_edges=${skippedEdges}`
    );
    console.log(
      `graph totals: entities=${after.entities} mentions=${after.mentions} active_edges=${after.edges}`
    );
  }

  process.exit(factsErrored > 0 ? 1 : 0);
}

function toGraphFactInput(fact: GraphFactRow): GraphFactInput {
  return {
    fact_id: fact.id,
    subject: fact.subject_wa_id,
    category: fact.category as FactCategory,
    content: fact.content,
    confidence: fact.confidence,
    event_ts: fact.event_ts,
  };
}

function printDryRun(fact: GraphFactRow, graph: ExtractedGraph): void {
  console.log(`\n[fact:${fact.id}] (${fact.subject_wa_id}, ${fact.category}) ${fact.content}`);
  if (graph.entities.length === 0 && graph.edges.length === 0 && graph.mentions.length === 0) {
    console.log('  graph: empty');
    return;
  }
  for (const entity of graph.entities) {
    console.log(
      `  entity ${entity.local_id}: ${entity.type} "${entity.display_name}" conf=${entity.confidence.toFixed(2)}`
    );
  }
  for (const mention of graph.mentions) {
    console.log(
      `  mention ${mention.entity_local_id}: role=${mention.role} conf=${mention.confidence.toFixed(2)}${
        mention.mention_text ? ` text="${mention.mention_text}"` : ''
      }`
    );
  }
  for (const edge of graph.edges) {
    console.log(
      `  edge ${edge.source_local_id} -[${edge.predicate}]-> ${edge.target_local_id} conf=${edge.confidence.toFixed(2)}`
    );
  }
}

main().catch((err) => {
  console.error('fatal:', err instanceof Error ? err.message : err);
  process.exit(1);
});
