import type { ExtractedFact } from '../llm/provider';

/**
 * Programmatic guard against the "fabricated meta-fact" failure mode where the
 * extractor describes someone it can't actually identify. Sonnet (and similar
 * models) leak this through specific phrases — "a third person", "unnamed",
 * untranslated foreign pronouns in parens — when forced to substitute a
 * descriptor for a real name. Treating those phrases as drop-signals is more
 * reliable than asking the prompt to never produce them: the prompt has been
 * tightened and was still ignored.
 */
const UNRESOLVED_SUBJECT_MARKERS: Array<{ re: RegExp; label: string }> = [
  { re: /\bunnamed\b/i, label: '"unnamed"' },
  { re: /\b(a|the)?\s*third\s+person\b/i, label: '"third person"' },
  { re: /\b(a|the)?\s*third\s+party\b/i, label: '"third party"' },
  { re: /\bunidentified\b/i, label: '"unidentified"' },
  { re: /\bunknown\s+(person|individual|man|woman|guy|girl)\b/i, label: '"unknown person"' },
  { re: /\b(an?\s+)?other\s+person\b/i, label: '"other person"' },
  { re: /\bsomeone\s+(else|unspecified)\b/i, label: '"someone else/unspecified"' },
  // Quoted foreign pronoun in parens with optional English gloss — Sonnet's
  // tell that it preserved the original ambiguity instead of resolving it.
  { re: /\(\s*['"]?(lei|lui|loro|ella|él|ellos|ellas)['"]?/i, label: 'quoted foreign pronoun' },
];

export interface GuardedFact {
  fact: ExtractedFact;
  drop: boolean;
  reason?: string;
}

export function guardFacts(facts: ExtractedFact[]): GuardedFact[] {
  return facts.map(guardOne);
}

function guardOne(fact: ExtractedFact): GuardedFact {
  for (const { re, label } of UNRESOLVED_SUBJECT_MARKERS) {
    if (re.test(fact.content)) {
      return { fact, drop: true, reason: `unresolved-subject marker: ${label}` };
    }
  }
  return { fact, drop: false };
}
