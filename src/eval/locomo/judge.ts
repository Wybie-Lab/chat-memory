import { generateObject } from 'ai';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { z } from 'zod';

/**
 * LLM-as-judge for LOCOMO QA predictions.
 *
 * Token-F1 sandbags this benchmark hard: synonyms ("trans woman" vs
 * "Transgender woman"), date format ("2023-05-07" vs "7 May 2023"),
 * verbose-but-correct answers, and adversarial refusals where gold is
 * the literal string "undefined" all score 0 under F1 even when the
 * prediction is semantically right.
 *
 * The judge uses two prompts:
 *   - Adversarial (LOCOMO category 5): correct iff the prediction is a
 *     refusal ("unknown" / "I don't know" / "not mentioned"); wrong if
 *     it fabricates a specific answer.
 *   - All other categories: correct iff the prediction is semantically
 *     equivalent to the reference (lenient on synonyms / format / verbose
 *     prose; strict on wrong specifics or unjustified refusals).
 *
 * Calls go through OpenRouter directly (not the engine's LLMProvider)
 * because judging is an eval-only concern and shouldn't pollute the
 * provider contract.
 */

const JUDGE_MODEL = process.env.OPENROUTER_JUDGE_MODEL ?? process.env.OPENROUTER_MODEL ?? 'google/gemini-2.5-flash';

const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
  appName: process.env.OPENROUTER_APP_NAME ?? 'manila-memory',
  appUrl: process.env.OPENROUTER_APP_URL,
});

const JudgeSchema = z.object({
  correct: z.boolean(),
  reasoning: z.string(),
});

const SEMANTIC_PROMPT = `You evaluate whether a predicted answer is semantically correct, given a reference answer.

Be LENIENT about:
- Synonyms and paraphrases ("trans woman" ≡ "transgender woman", "counseling" ≡ "therapy")
- Date and number format ("2023-05-07" ≡ "7 May 2023" ≡ "May 7, 2023"; "2k" ≡ "2000")
- Capitalisation, punctuation, articles
- Verbose predictions that CONTAIN the reference answer plus extra context (still correct)
- Equivalent units, currencies, or phrasings ("ten dollars" ≡ "$10")
- Partial answers when the question allows them (e.g. listing one valid option when several exist)

Be STRICT about:
- Wrong specific facts (different name, different date, different place, contradicting numbers)
- "unknown" / refusal when the reference is a concrete answer
- Adding details that contradict the reference
- Hallucinated information not supported by the question/reference

Output JSON: { "correct": boolean, "reasoning": "one short sentence" }.`;

const ADVERSARIAL_PROMPT = `You evaluate whether a model correctly REFUSED to answer an unanswerable question.

The reference answer is "undefined", meaning: the information needed to answer this question is NOT present in the source conversation. The model should refuse.

A REFUSAL is correct: phrases like "unknown", "I don't know", "no information available", "not mentioned in the conversation", "the conversation doesn't say", or similar honest admissions.

A FABRICATED ANSWER is wrong: any specific claim presented as fact (a name, date, place, opinion, action), even if hedged with "probably" or "likely". The model is supposed to refuse, not guess.

Output JSON: { "correct": boolean, "reasoning": "one short sentence — say 'refused' or 'fabricated <X>'" }.`;

export interface JudgeInput {
  question: string;
  goldAnswer: string;
  predicted: string;
  category: number;
}

export interface JudgeResult {
  correct: boolean;
  reasoning: string;
  model: string;
}

export async function judgeAnswer(input: JudgeInput): Promise<JudgeResult> {
  const isAdversarial = input.category === 5;
  const systemPrompt = isAdversarial ? ADVERSARIAL_PROMPT : SEMANTIC_PROMPT;
  const userPrompt = isAdversarial
    ? `Question: ${input.question}\nPredicted answer: ${input.predicted}`
    : `Question: ${input.question}\nReference answer: ${input.goldAnswer}\nPredicted answer: ${input.predicted}`;

  // ai-sdk + openrouter types blow up TS inference when chained ("type
  // instantiation is excessively deep"). Suppress on the call-site; the
  // returned `object` is still validated against JudgeSchema at runtime.
  // @ts-expect-error -- ai-sdk + zod schema inference depth limit
  const result = await generateObject({
    model: openrouter(JUDGE_MODEL),
    system: systemPrompt,
    prompt: userPrompt,
    schema: JudgeSchema,
  });

  const obj = result.object as z.infer<typeof JudgeSchema>;
  return {
    correct: obj.correct,
    reasoning: obj.reasoning,
    model: JUDGE_MODEL,
  };
}
