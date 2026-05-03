# Memory engine

A long-term memory layer for chat messages. Given a stream of messages from one or more sources, it extracts durable, atomic facts about each *person* in the conversation and exposes them as a queryable memory.

This document is the canonical reference for how the engine works. It's anchored in the code via `file:line` refs so it stays accurate as the code evolves — if a section drifts, fix the doc.

## Contents

1. [What it is, what it isn't](#1-what-it-is-what-it-isnt)
2. [Architecture overview](#2-architecture-overview)
3. [Data model](#3-data-model)
4. [Write path: messages → facts](#4-write-path-messages--facts)
5. [Read path: query → memory block](#5-read-path-query--memory-block)
6. [Public API](#6-public-api)
7. [Boundary rule](#7-boundary-rule)
8. [Operational notes](#8-operational-notes)

---

## 1. What it is, what it isn't

**Is.** A *per-subject* memory store. Every fact has a subject (`me`, `alex`, `mom`, …). Retrieval ranks facts by relevance to a query *and* by which subjects the query mentions. The engine is source-agnostic: anything that can produce a `(contact, message, ts, direction)` tuple can feed it.

**Isn't.** A single-user profile store. The proposal in `.context/attachments/pasted_text_2026-05-03_14-21-55.txt` describes a `user_id`-keyed memory; this engine is keyed by `subject_wa_id` so it can hold facts about *all* the people in your life. Don't conflate "user" with "subject."

**Isn't.** A general retrieval index. The output isn't search hits — it's a structured `<memory>` block (preferences, ranked facts with citations, prose subject summaries, recent episodes) shaped to be injected into an LLM's context window with a hard token budget.

---

## 2. Architecture overview

```
        sources/ (whatsapp live, chat-export importer, …)
              │
              │ engine.insertRawMessage + assignMessageToBurst
              ▼
        ┌──────────────────────────────┐
        │ raw_messages                 │
        │ conversation_bursts          │  ← burst = contiguous run of
        └──────────────┬───────────────┘     messages with <30 min gaps
                       │ processBatch
                       ▼
        ┌──────────────────────────────┐
        │ filter (Haiku 4.5)           │  keep or drop the whole burst
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ extract (Sonnet 4.6)         │  ExtractedFact[]
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ guard (regex, programmatic)  │  drop unresolved-subject markers
        └──────────────┬───────────────┘
                       │ group by normalized subject
                       ▼
        ┌──────────────────────────────┐
        │ consolidate (Sonnet 4.6)     │  ADD / UPDATE / DELETE / DROP
        │   per-subject merge logic    │  vs. the existing facts for
        └──────────────┬───────────────┘  that subject
                       ▼
        ┌──────────────────────────────┐
        │ embed (Cohere multilingual)  │  only kept facts
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ facts                        │
        │ fact_sources  (multi-source) │
        │ fact_embeddings (vec0)       │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ summarize (Sonnet 4.6)       │  per (subject, category)
        └──────────────┬───────────────┘  rolled-up prose summary
                       ▼
                 cluster_summaries
                       │
                       │ engine.composeMemoryBlock
                       ▼
                 ┌─────────────┐
                 │ <memory>    │  → injected into chat agent context
                 └─────────────┘
```

Every stage is documented in §4 (write) and §5 (read).

---

## 3. Data model

Schema lives in `src/engine/storage/schema.sql`. Migrations (additive only) live in `migrate()` in `src/engine/storage/db.ts`. SQLite + WAL mode + `sqlite-vec` for embeddings.

### `contacts`

One row per chat (1:1 or group). The `wa_id` is WhatsApp's JID format (`<digits>@c.us` or `<id>@g.us`).

```sql
contacts (
  id, wa_id UNIQUE, display_name, is_group, whitelisted, notes,
  first_seen_at, last_seen_at,
  live_cutoff_ts        -- imported messages stop here; live ingest
                        -- skips msgs <= this ts to avoid double-count
)
```

### `raw_messages`

Every message ever ingested. Source of truth — bursts, facts, and embeddings are all derivable from this table plus configuration.

```sql
raw_messages (
  id, wa_msg_id UNIQUE,        -- real id (live) or sha1 hash (import)
  contact_id → contacts,
  sender_wa_id, direction ('in'|'out'), body, ts,
  media_type, media_pointer,   -- non-null for media; body is empty
  burst_id → conversation_bursts,
  filter_kept, processed_at,   -- denormalized from the burst
  ingested_at
)
```

### `conversation_bursts`

A burst = contiguous run of messages for one contact with no gap larger than `BURST_GAP_SECONDS` (30 min, `db.ts:8`). Bursts are the unit of LLM filter/extract — single messages don't carry enough context.

```sql
conversation_bursts (
  id, contact_id, start_ts, end_ts, message_count,
  filter_kept,                 -- set by pipeline filter step
  processed_at                 -- set when pipeline finishes the burst
)
```

Built by `assignMessageToBurst` (live) or `rebuildAllBursts` (post-import).

### `facts`

Atomic, durable facts about *one* subject. Append-only at the row level — updates and deletions are recorded via `superseded_by_id` and `deleted_at`, not by mutating prior rows.

```sql
facts (
  id,
  source_msg_id → raw_messages,         -- legacy single-source columns;
  source_burst_id → conversation_bursts,-- still populated for back-compat
  subject_wa_id,                        -- can be 'me' OR a wa_id OR a name
  category,                             -- preference|event|commitment|fact|relationship
  content,                              -- self-contained sentence
  confidence,                           -- 0.0 - 1.0 from the extractor
  extracted_at,
  superseded_by_id → facts,             -- non-null = replaced by a newer fact
  deleted_at                            -- non-null = soft-deleted (no replacement)
)
```

Active facts are those where `superseded_by_id IS NULL AND deleted_at IS NULL`. Index `idx_facts_active` enforces fast queries on this.

### `fact_sources`

Many-to-many evidence trail: each fact can be backed by multiple bursts/messages. Used by retrieval scoring (more sources → higher importance).

```sql
fact_sources (
  fact_id → facts,
  source_burst_id → conversation_bursts,
  source_msg_id → raw_messages,
  attached_at
)
```

### `fact_embeddings` (virtual)

`sqlite-vec` `vec0` virtual table. 1024-dim Cohere `embed-multilingual-v3.0`. Embedding is dropped when a fact is superseded or deleted (`db.ts:550, 558`), so vector search never returns inactive facts.

```sql
fact_embeddings USING vec0 (
  fact_id PRIMARY KEY,
  embedding FLOAT[1024]
)
```

### `cluster_summaries`

Rolled-up prose summary per `(subject, category)`. Refreshed by the pipeline whenever facts in that group change. Used by retrieval as the `<subject_summaries>` section of the memory block — agents read prose better than triples.

```sql
cluster_summaries (
  id, subject_wa_id, category, summary,
  fact_count, fact_ids_json,
  last_refreshed_at,
  UNIQUE(subject_wa_id, category)
)
```

Only built once a cluster has ≥ `CLUSTER_SUMMARY_MIN_FACTS` (3, `pipeline.ts:51`).

### `processing_log`

Per-LLM-call accounting. One row per filter / extract / consolidate / embed / summarize call. Used by tools and budgets.

```sql
processing_log (
  msg_id, burst_id, stage, model,
  tokens_in, tokens_out, cost_usd, ts
)
```

---

## 4. Write path: messages → facts

Entry point: `processBatch(db, provider, opts)` in `src/engine/pipeline.ts`.

It pulls up to N unsettled bursts (where `end_ts <= now - BURST_GAP_SECONDS`), processes each in turn, and returns aggregate stats. `processUntilDrained` loops `processBatch` until empty.

For each burst, in order:

### 4.1 Filter (Haiku 4.5)

`provider.filterBurst(burstInput) → { keep: bool, reason: string }`.

Sees the full burst. Decides whether *anything* in it is worth remembering long-term. Bias is toward KEEP — false positives are easier to clean later than false negatives are to recover. Drop reasons are typically pure greetings, ephemeral logistics, or generic small talk.

If `keep === false`, the burst is marked processed and the pipeline skips to the next one. Cost: one Haiku call per burst (cheap, ~$0.001).

### 4.2 Extract (Sonnet 4.6)

`provider.extractBurst(burstInput) → { facts: ExtractedFact[] }`.

Sees the full burst with each line labeled by sender. Returns atomic facts, each with `subject` / `category` / `content` / `confidence`.

The prompt enforces strict subject attribution rules to prevent the most common failure mode: extracting facts about an *unnamed third person* (e.g., "she's flying to Tokyo" when "she" has no antecedent in this burst). The rule is to drop those facts entirely rather than fabricate a meta-fact like "the contact mentioned someone is flying to Tokyo." See `src/llm/claude.ts:60` for the full prompt.

### 4.3 Guard (programmatic)

`guardFacts(facts)` in `src/engine/guard.ts`.

Regex sentinel against the unnamed-third-person failure mode. Even with the strict prompt above, models sometimes produce `"unnamed person"`, `"a third person"`, or quoted foreign pronouns. We drop those rows hard. Programmatic, free, deterministic.

### 4.4 Consolidate (Sonnet 4.6)

For each candidate fact, decide vs. existing memory: `ADD` / `UPDATE` / `DELETE` / `DROP`.

Two paths based on subject memory size:

- **Small subjects** (≤ 30 active facts, `pipeline.ts:57`) — send all existing facts as the slate.
- **Large subjects** (> 30) — embed each candidate first, run a *subject-restricted* vector search, take top-12 per candidate, union them. This keeps the consolidate prompt bounded for power-chats.

The four ops:

- `ADD` — new info not covered. Insert as new fact.
- `UPDATE` — refines/replaces/contradicts ONE specific existing fact (named by `old_fact_id`). Insert new, mark old `superseded_by_id`.
- `DELETE` — existing fact is now wrong/obsolete; candidate provides no replacement. Soft-delete old.
- `DROP` — candidate is fully redundant; ignore it.

Validation in `validateOp` (`claude.ts:536`) rejects malformed responses (missing required fields, bad indices) — better to silently drop one op than fail the whole burst.

### 4.5 Embed (Cohere)

Only run for ops that produce a new fact (`ADD` and `UPDATE`). Embeddings are reused from the consolidation step's vector search when available, so we don't double-pay. Cohere `embed-multilingual-v3.0`, 1024-dim.

### 4.6 Persist (single transaction)

All ops for a burst commit atomically (`pipeline.ts:327-388`):

- `ADD`: insertFact + insertEmbedding + record-affected-cluster.
- `UPDATE`: insertFact (new) + insertEmbedding + markFactSuperseded(old → new) + record-affected-clusters (both old and new categories if they differ).
- `DELETE`: markFactDeleted (drops embedding too).
- `DROP`: noop.

Then `markBurstFiltered + markBurstProcessed` close the burst out.

### 4.7 Summarize affected clusters (Sonnet 4.6)

For every `(subject, category)` pair touched by an op, refresh its cluster summary. If the active count drops below 3, the summary is deleted. Otherwise we re-run the summarizer with all current active facts and `upsertClusterSummary`.

This is what makes retrieval cheap — readers see one paragraph per (subject, category) instead of N triples.

---

## 5. Read path: query → memory block

Entry point: `composeMemoryBlock(db, provider, question, opts)` in `src/engine/retrieval/memory-block.ts`.

Returns a `<memory>` XML block (string), citations array, and budget telemetry.

### 5.1 Query embedding

One Cohere `embed-multilingual-v3.0` call with `input_type: search_query`. ~10ms.

### 5.2 Retrieval context

`buildRetrievalContext(db, query)` in `score.ts`:

- Pull `allActiveSubjects(db)` — every distinct `subject_wa_id` with at least one active fact, plus its display_name.
- Run `matchSubjectsInQuery` — substring match the query against each subject's display_name and wa_id prefix. Single/two-letter tokens are skipped to avoid spurious hits ("is", "in"). Case-insensitive.

### 5.3 Hybrid retrieval

`hybridRetrieve` does:

1. `searchFactsByVector(db, queryEmbedding, k=30)` — pull top-30 by L2 distance from `vec0`.
2. `scoreCandidates` — blend semantic similarity with non-semantic signals:

```
score = w_semantic   * similarity(distance)         # 1.0 weight
      + w_recency    * exp(-ln2 * age_days / 365)   # 0.3 weight
      + w_confidence * fact.confidence              # 0.5 weight
      + w_entity     * (subject in matched ? 1 : 0) # 1.5 weight  ← biggest
      + w_importance * log10(1 + n_sources)         # 0.3 weight
```

Weights at `score.ts:18`. The entity-match weight (1.5) intentionally dominates: if your query mentions "Emma," facts about Emma rank above semantically-similar facts about anyone else.

3. Sort, slice top `rerankTopK` (default 20).

### 5.4 Memory block composition

Four sections, each with its own char budget (`memory-block.ts:24`):

- `<preferences>` — always-on. All active preference-category facts (300-token budget). Pulled by `activePreferences(db)`. Order: confidence DESC, extracted_at DESC.
- `<known_facts>` — top-20 hybrid-ranked facts (600-token budget). `[fact:ID]` citations.
- `<subject_summaries>` — cluster summaries for entity-matched subjects ∪ subjects from the top-5 ranked facts (400-token budget).
- `<recent_episodes>` — `event` and `commitment` facts within the last 60 days, filtered to relevant subjects (200-token budget).

Total budget: 1500 tokens (default). After per-section greedy fit, a global trim drops sections in priority order (recent_episodes → cluster_summaries → top_facts → preferences) until under cap.

### 5.5 Output shape

```xml
<memory>
  <preferences>
    - <content> [fact:ID]
  </preferences>
  <known_facts>
    - (subject) <content> [fact:ID]
  </known_facts>
  <subject_summaries>
    - subject (category): <prose summary>
  </subject_summaries>
  <recent_episodes>
    - YYYY-MM-DD (subject, category): <content> [fact:ID]
  </recent_episodes>
</memory>
```

This is what gets injected into the chat agent's context for the next turn.

---

## 6. Public API

The single import surface for *all* consumers (sources, web, cli, eval, scripts) is `src/engine/index.ts`. Everything in `engine/storage/`, `engine/pipeline.ts`, `engine/retrieval/*` is engine-internal.

Most-commonly-used exports:

| Function | Purpose |
|---|---|
| `openDb(path)` | Initialize sqlite + load `vec0` + run migrations |
| `upsertContact / insertRawMessage / assignMessageToBurst` | Source ingestion |
| `setLiveCutoff / getLiveCutoff` | Prevent live↔import double-count |
| `processBatch / processUntilDrained` | Drain bursts through the full pipeline |
| `composeMemoryBlock(db, provider, question)` | Read path — returns the `<memory>` block + telemetry |
| `getBurstQueueStats(db)` | `{ total, processed }` for progress UIs |
| `listFacts / listCategories / factsAboutSubject` | Browser/inspection |
| `searchFactsByVector / hybridRetrieve` | Lower-level retrieval (most callers want `composeMemoryBlock`) |

Full list with type re-exports lives in `src/engine/index.ts`.

---

## 7. Boundary rule

> The engine never imports from `src/sources/`, `src/web/`, `src/cli/`, or `scripts/`. Anything those consumers need from the engine must be exported from `src/engine/index.ts`. New code that crosses this boundary by reaching into `src/engine/storage/db.ts` directly is a code-review smell.

This is the property the reorg was for. Concrete enforcement:

```bash
# Should all return 0:
grep -r "engine/storage/db'"   src/ --include='*.ts' | grep -v '^src/engine/'
grep -r "engine/pipeline'"     src/ --include='*.ts' | grep -v '^src/engine/'
grep -r "engine/retrieval/"    src/ --include='*.ts' | grep -v '^src/engine/'
```

LLM types (`LLMProvider`, `ExtractedFact`, etc.) are an exception: they live in `src/llm/provider.ts` because they're the *contract* the engine depends on, not engine-owned types.

---

## 8. Operational notes

### WAL mode

`PRAGMA journal_mode = WAL` (`schema.sql:1`). The web UI can read facts while `processBatch` is mid-flight in another process — readers see committed bursts immediately, never block writers.

### Burst settling

`listUnprocessedBursts` (`db.ts:337`) only returns bursts whose `end_ts <= now - BURST_GAP_SECONDS`. This guarantees no late message can join a burst after the pipeline has touched it. The eval harness passes `includeUnsettled: true` to bypass this for synthetic transcripts.

### Live cutoff

Imported chat exports cover up to a known timestamp. The contact's `live_cutoff_ts` is set to that timestamp by the importer. Live ingest checks `getLiveCutoff(db, contactId)` and skips messages with `ts <= cutoff` so the same conversation isn't ingested twice. See `src/cli/live.ts:58`.

### Idempotence

- `wa_msg_id UNIQUE` makes raw-message inserts idempotent. Live: real WA message id. Import: `sha1(wa_id|ts|sender|body).slice(0,24)` — same export re-run → same hashes → all dupes.
- `markBurstProcessed` flips a flag; reprocessing a burst requires `scripts/reset-pipeline.ts` (which clears facts/embeddings/log and rebuilds bursts).
- Cluster summaries `ON CONFLICT DO UPDATE` so refresh is idempotent.

### Cost shape

Per burst, in expectation:

| Stage | Model | Frequency | Notes |
|---|---|---|---|
| filter | Haiku 4.5 | 1× per burst | cheap |
| extract | Sonnet 4.6 | 1× per kept burst | ~70% of bursts kept |
| consolidate | Sonnet 4.6 | 1× per (kept burst × subject) | typically 1–3 subjects per burst |
| embed | Cohere | 1× per ADD / UPDATE | reused across consolidate + persist |
| summarize | Sonnet 4.6 | 1× per affected cluster per burst | clusters with <3 facts skipped |
| chat (read) | Sonnet 4.6 | 1× per user question | + 1 Cohere embed for query |

Filter dominates volume; Sonnet stages dominate cost.

---

## See also

- `src/engine/storage/schema.sql` — full DDL.
- `src/engine/pipeline.ts` — the orchestrator.
- `src/llm/claude.ts` — all LLM prompts (filter, extract, consolidate, summarize, chat).
- `src/engine/retrieval/score.ts` — hybrid scoring weights and rationale.
- `src/engine/retrieval/memory-block.ts` — section budgets and assembly.
- `README.md` (repo root) — user-facing setup + usage.
