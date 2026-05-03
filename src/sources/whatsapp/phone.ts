/**
 * Normalize a user-entered phone number into a WhatsApp wa_id of the form
 * `<digits>@c.us`. Accepts any input that contains a phone number — leading
 * "+", spaces, dashes, parentheses, dots are stripped. Throws if fewer than
 * 6 digits remain (probably typo, not a real number).
 *
 * Group jids (`<id>-<id>@g.us`) are not in scope for this helper; pass them
 * through directly via the `waId` field instead.
 */
export function phoneToWaId(phone: string): string {
  const digits = phone.replace(/\D+/g, '');
  if (digits.length < 6) {
    throw new Error(
      `phone number "${phone}" only has ${digits.length} digit(s) after stripping — too short`
    );
  }
  return `${digits}@c.us`;
}
