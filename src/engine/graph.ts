import {
  insertFactEntityMention,
  upsertEntity,
  upsertKnowledgeEdge,
  type DB,
  type GraphEntityInput,
} from './storage/db';
import type { ExtractedGraph, GraphFactInput } from '../llm/provider';

export interface GraphFactContext extends GraphFactInput {
  source_burst_id?: number | null;
}

export interface GraphWriteResult {
  entities: number;
  mentions: number;
  edges: number;
  skipped_edges: number;
}

export function writeExtractedGraph(
  db: DB,
  fact: GraphFactContext,
  graph: ExtractedGraph
): GraphWriteResult {
  const localToEntityId = new Map<string, number>();
  let entities = 0;
  let mentions = 0;
  let edges = 0;
  let skipped_edges = 0;

  const tx = db.transaction(() => {
    for (const entity of graph.entities) {
      const input = entityInputForExtractedEntity(entity, fact);
      const id = upsertEntity(db, input);
      localToEntityId.set(entity.local_id, id);
      entities++;
    }

    for (const mention of graph.mentions) {
      const entityId = localToEntityId.get(mention.entity_local_id);
      if (!entityId) continue;
      insertFactEntityMention(db, {
        fact_id: fact.fact_id,
        entity_id: entityId,
        role: mention.role,
        mention_text: mention.mention_text ?? null,
        confidence: mention.confidence,
      });
      mentions++;
    }

    for (const edge of graph.edges) {
      const sourceId = localToEntityId.get(edge.source_local_id);
      const targetId = localToEntityId.get(edge.target_local_id);
      if (!sourceId || !targetId || sourceId === targetId) {
        skipped_edges++;
        continue;
      }
      upsertKnowledgeEdge(db, {
        source_entity_id: sourceId,
        predicate: edge.predicate,
        target_entity_id: targetId,
        confidence: edge.confidence,
        source_fact_id: fact.fact_id,
        source_burst_id: fact.source_burst_id ?? null,
        event_ts: edge.event_ts ?? fact.event_ts ?? null,
        valid_from_ts: edge.valid_from_ts ?? null,
        valid_to_ts: edge.valid_to_ts ?? null,
        qualifiers: edge.qualifiers ?? {},
      });
      edges++;
    }
  });

  tx();
  return { entities, mentions, edges, skipped_edges };
}

function entityInputForExtractedEntity(
  entity: ExtractedGraph['entities'][number],
  fact: GraphFactContext
): GraphEntityInput {
  return {
    entity_type: entity.type,
    canonical_key: canonicalKeyForEntity(entity.type, entity.display_name, fact),
    display_name: entity.display_name.trim(),
    aliases: normalizedAliases(entity.aliases ?? [], entity.display_name),
    confidence: Math.min(entity.confidence, fact.confidence),
  };
}

export function canonicalKeyForEntity(
  type: string,
  displayName: string,
  fact?: Pick<GraphFactContext, 'subject'>
): string {
  const normalized = normalizeKey(displayName);
  if (type === 'person') {
    const subject = fact?.subject?.trim();
    if (subject && sameLooseName(subject, displayName)) {
      return canonicalPersonKey(subject);
    }
    return `person:${normalized}`;
  }
  if (type === 'place') return `place:${normalized}`;
  if (type === 'organization') return `org:${normalized}`;
  if (type === 'event') return `event:${normalized}`;
  if (type === 'preference_topic') return `topic:${normalized}`;
  if (type === 'date') return `date:${normalized}`;
  if (type === 'object') return `object:${normalized}`;
  return `concept:${normalized}`;
}

function canonicalPersonKey(subject: string): string {
  const normalized = normalizeKey(subject);
  if (normalized === 'me') return 'special:me';
  if (/@[cg]\.us$/.test(normalized)) return `wa:${normalized}`;
  return `person:${normalized}`;
}

function sameLooseName(a: string, b: string): boolean {
  return normalizeKey(a) === normalizeKey(b);
}

function normalizeKey(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^\p{L}\p{N}@._ -]+/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizedAliases(aliases: string[], displayName: string): string[] {
  const seen = new Set<string>([normalizeKey(displayName)]);
  const out: string[] = [];
  for (const alias of aliases) {
    const key = normalizeKey(alias);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(alias.trim());
  }
  return out;
}
