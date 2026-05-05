# Why I made my chat memory append-only

I've been building a long-term memory system for personal chat. The idea is simple: read every message, extract durable facts about the people you talk to, store them, query them later. *"What did Alex tell me about her job?"* — that kind of thing.

Most of my early design choices were obvious, copied from a hundred other LLM-memory systems. One choice surprised me. It's the one I want to write about.

I made memory **append-only**. Facts, once written, are never modified or deleted. Every change to memory is a *new* fact plus a typed edge to the old one. Both stay in the database forever.

This sounds like a small implementation detail. It isn't. It changes how the system reasons, what the agents inside it can do safely, and what kinds of questions you can ask.

Here's why I got there.

---

## Alex has a dog named Rex

Imagine the simplest version of the system. There's a `facts` table. Every row is one durable thing about one person. When new info arrives, you compare it to existing rows and decide what to do.

It's been six months of conversation. Memory says:

```
fact 22: Alex has a dog named Rex.
```

A new burst of messages arrives. Alex is upset. She mentions Rex passed away last week.

What does the system do?

The naive answer — and the answer in literally every "agent memory" library I looked at — is: **mark the old fact as wrong and write the new one.**

```sql
UPDATE facts SET superseded_by_id = 47 WHERE id = 22;
INSERT INTO facts (id=47, content='Rex passed away last week.');
```

Or even simpler:

```sql
UPDATE facts SET content = 'Alex had a dog named Rex; he passed away last week.' WHERE id = 22;
```

This works. Until it doesn't.

A week later you ask the system: *"Did Alex ever tell me about her dog?"* It correctly says yes — there's a fact about Rex. You ask: *"What was the dog's name?"* It says Rex.

You ask: *"Did Alex have a dog last March?"*

The system can't answer. It only knows what's true *now*. The fact "Alex has a dog Rex" was overwritten the moment Rex died. There's no way to recover the past state from what's stored.

That's the first thing that bothered me. **Memory built this way only knows the latest snapshot.** It's amnesia disguised as efficiency.

---

## The unstable shape of "supersede"

A lot of systems try to fix this with a tombstone column. You don't `UPDATE` the old row, you mark it superseded:

```sql
facts (
  id              INTEGER PRIMARY KEY,
  content         TEXT,
  superseded_by_id INTEGER REFERENCES facts(id),
  deleted_at      INTEGER
)
```

Reads filter `WHERE superseded_by_id IS NULL AND deleted_at IS NULL`. Better — the old row exists, you can technically reach it.

But you've still flattened seven different things into the same operation:

| Real-world thing that happened | What the system records |
|---|---|
| Alex moved cities | `update` |
| Alex's dog died | `update` |
| "Alex has a pet" → "Alex has a dog" (we learned more) | `update` |
| "Alex works at the bank" → "Alex works *part-time* at the bank" | `update` |
| "Alex lives in Rome" / "Alex lives in Milan" — both seem asserted | `update`? |
| The extractor hallucinated; Alex never had a dog | `update` |
| Two extractions of the same event from different bursts | `update` |

A relocation, a death, a clarification, a qualifier, an unresolved contradiction, a hallucinated retraction, a duplicate. All seven get the same edge. The model can't tell them apart later. Neither can you.

The third pain point is more subtle, and it took me longer to feel.

I was building an *agent* — an LLM that audits memory and proposes cleanups. It can find duplicates, surface contradictions, link related facts. The natural way to give it teeth is tools like `propose_update(old_id, new_content)` or `propose_delete(old_id)`.

The problem is that those tools are **destructive primitives**. A bad tool call permanently overwrites correct memory. An LLM agent that can `UPDATE` rows in your database can hallucinate one bad call into a chain of broken state. You'd have to wrap it in a heavy review process. The risk is asymmetric: a bad write costs you forever, a bad read costs you nothing.

I wanted an agent I'd actually let run autonomously. Destructive primitives didn't fit.

---

## The pivot

I started over. What if facts were immutable? You write them once. They never get modified. They never get marked deleted. They just exist.

Then how do you handle Rex dying? You add a *new* fact and connect it to the old one with a typed edge:

```
fact 22: Alex has a dog named Rex.    [stays active]
                  ↑
                  | predicate = state_change
                  |
fact 47: Rex passed away last week.   [stays active]
```

Both rows persist. The connection captures *how* they relate.

Schema-wise, the change is small:

```sql
CREATE TABLE fact_connections (
  id           INTEGER PRIMARY KEY,
  from_fact_id INTEGER NOT NULL REFERENCES facts(id),
  to_fact_id   INTEGER NOT NULL REFERENCES facts(id),
  predicate    TEXT NOT NULL CHECK (predicate IN (
    'update', 'state_change', 'expands', 'qualifies',
    'contradicts', 'retracts', 'same_as'
  )),
  reason       TEXT,
  confidence   REAL NOT NULL DEFAULT 1.0,
  created_at   INTEGER NOT NULL,
  UNIQUE(from_fact_id, to_fact_id, predicate)
);
```

The closed predicate enum — enforced as a SQL `CHECK` so the system can't invent garbage labels — is where the real expressive power lives:

| Predicate | Meaning | Example |
|---|---|---|
| `update`       | Same thing, new state                       | `lives in Berlin` → `lives in Lisbon` |
| `state_change` | Discrete event changed state                | `has a dog Rex` → `Rex passed away` |
| `expands`      | Same fact, more specific                    | `has a pet` → `has a dog Rex` |
| `qualifies`    | Adds a condition or nuance                  | `works at the bank` → `works part-time` |
| `contradicts`  | Mutual exclusion, no clear winner           | `lives in Rome` ↔ `lives in Milan` |
| `retracts`     | Old fact was wrong (extractor hallucinated) | `has a cat` → `actually a dog` |
| `same_as`      | Restating, deduped                          | two extractions of the same event |

These aren't just labels for humans. They have different read-time semantics:

- `update` and `state_change` form a chain. There's a function — I called it `latestInChain(factId)` — that walks those edges to the leaf and returns the most recent state. That's how you answer *"what's true now?"*
- `expands` adds detail without replacing. Both facts are simultaneously true.
- `qualifies` is similar but narrower in scope.
- `contradicts` deliberately doesn't pick a winner. It surfaces the conflict for the asker to resolve.
- `retracts` is the only edge that semantically removes information — but even then, the old fact stays in the table. Retrieval can choose to filter on it.
- `same_as` collapses duplicates without merging.

A single `superseded_by_id` flag couldn't carry any of this. Seven predicates can.

---

## What you get for free

Three things fell out of this design that I wasn't expecting.

**You can ask history questions.** *"Did Alex have a dog last March?"* Just look at fact 22 — it was active in March, never retracted, no contradicting edge before March. Yes. The naive system can't answer this. The supersede system technically can but you have to query the tombstone column. With append-only the question becomes a regular query.

**Provenance is automatic.** Every connection has a `reason` column and a `source_agent_action_id` back-pointer. You can ask the database *"why did fact 47 get connected to fact 22?"* and get a real answer. *"The new fact establishes that Rex passed away, providing a discrete state change to the existing fact about Alex's dog."* That comes from the connection's reason. With supersede, you'd have to log it separately or lose it.

**The agent gets safer.** This is the one I cared most about. Once you commit to immutability, the agent's tool catalog changes shape:

```
propose_connect(from_fact_id, to_fact_id, predicate, reason, confidence)
propose_assign_thread(fact_id, thread_id, reason)
propose_create_thread(name, attached_fact_ids, reason)
```

Notice what's missing. There's no `propose_update`, no `propose_delete`, no `propose_merge`. The agent literally cannot destroy memory. The worst case for a bad LLM call is a wrong edge — which is just another row you can ignore at read time, or delete from `fact_connections` later.

That's a different safety profile entirely. I let the curator run autonomously. I don't review every action. I trust the audit trail because *the audit trail is the actual data structure*.

---

## What it costs

I won't pretend this is free. There's a real trade-off, and it's on the read side.

With supersede, "current state of Alex's pet" is a single indexed query: `SELECT * FROM facts WHERE subject='alex' AND superseded_by_id IS NULL`. Done in microseconds.

With append-only, you have to walk edges. `latestInChain(22)` does a small graph traversal — usually 1–3 hops, but in principle unbounded. For the most common access pattern — "what do you know about X?" — the cost is a join, not a filter.

I dealt with this in two ways. First, I kept the existing legacy `superseded_by_id` columns alive for backward-compat reads on rows that predate the redesign; new rows simply never write them. Second, the read path *as of now* still uses the simpler "active flag" semantics. Connection-aware retrieval is a known TODO, not a shipped feature. The win so far is mostly on the write side: provenance, agent safety, history preservation. The read-side payoff is gradual.

That's an honest answer. I'm fine with it. The cost is a few milliseconds per read; the payoff is that I never lose Rex.

---

## This isn't actually new

The pattern is event sourcing. Datomic does it. Git does it (every commit is a new tree, never an overwrite). General ledger accounting does it. Every database that takes auditability seriously does some version of this.

What's new — at least to me — is applying it to **LLM memory specifically**. The standard frame in the agent-memory world is "scratchpad you mutate." It treats memory like a Python dict: read, update, write back. That worked when the unit of memory was a transient task context. It doesn't work when you want memory that lives for years and gets touched by autonomous agents.

The shift in mindset: **memory isn't state, it's a sequence of statements about the world.** Statements don't get retracted by being deleted. They get retracted by being explicitly retracted, on the record, with a reason. The fact that the system is full of LLMs making fallible judgments makes this more important, not less. Hallucinations should leave a trace. Corrections should be visible.

Once I framed it that way, supersede started looking obviously wrong. Like writing over a journal entry instead of crossing it out and dating the correction.

---

## The thing I'd build the same way again

Six commits and a few weeks in, this is the design choice I'm most certain about. Everything else — the LLM provider abstraction, the burst-grouping heuristic, the threading mechanism, the curator's tool catalog — has been revised at least once. The append-only invariant has only ever gotten more central.

If you're building a memory system right now, especially one that'll be touched by agents, I'd argue you should default to immutability. Make change additive. Type your edges. Don't destroy state to record that it changed.

Your future self — and your future agents — will be able to ask better questions of what they used to know.

---

*The system this post is about is open-source: [Wybie-Lab/chat-memory](https://github.com/Wybie-Lab/chat-memory). The full data model lives in [`src/engine/storage/schema.sql`](https://github.com/Wybie-Lab/chat-memory/blob/master/src/engine/storage/schema.sql); the technical reference is [`src/engine/README.md`](https://github.com/Wybie-Lab/chat-memory/blob/master/src/engine/README.md).*
