/**
 * Bounded-concurrency map. Runs `fn` over `items` with at most `limit`
 * concurrent in-flight calls and returns results in input order.
 *
 * Used to parallelize independent LLM/embed calls in the burst pipeline
 * without unbounded fan-out (which would hit provider 429s). DB writes
 * inside `fn` are still safe with better-sqlite3: its sync API blocks the
 * event loop, so writes serialize naturally between awaits.
 */
export async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>
): Promise<R[]> {
  if (items.length === 0) return [];
  const effective = Math.max(1, Math.min(limit, items.length));
  const results: R[] = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: effective }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return results;
}
