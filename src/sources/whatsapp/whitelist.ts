import fs from 'fs';
import path from 'path';
import type { DB } from '../../engine';

export interface WhitelistEntry {
  wa_id: string;
  display_name?: string | null;
  notes?: string | null;
}

interface WhitelistFile {
  contacts: WhitelistEntry[];
  groups: WhitelistEntry[];
}

export class Whitelist {
  private allowed: Map<string, WhitelistEntry>;

  constructor(file: WhitelistFile) {
    this.allowed = new Map();
    for (const e of [...file.contacts, ...file.groups]) {
      this.allowed.set(e.wa_id, e);
    }
  }

  isAllowed(wa_id: string): boolean {
    return this.allowed.has(wa_id);
  }

  get(wa_id: string): WhitelistEntry | undefined {
    return this.allowed.get(wa_id);
  }

  size(): number {
    return this.allowed.size;
  }

  entries(): WhitelistEntry[] {
    return Array.from(this.allowed.values());
  }
}

export function loadWhitelist(configPath: string): Whitelist {
  const abs = path.resolve(configPath);
  if (!fs.existsSync(abs)) {
    console.warn(`whitelist not found at ${abs} — no contacts will be processed`);
    return new Whitelist({ contacts: [], groups: [] });
  }
  const raw = fs.readFileSync(abs, 'utf-8');
  const parsed = JSON.parse(raw) as WhitelistFile;
  return new Whitelist(parsed);
}

/**
 * Append (or update) a single entry in the whitelist file. Idempotent: if
 * `wa_id` already exists, only `display_name` and `notes` are merged (existing
 * non-null values win to avoid clobbering hand edits). Creates the file with
 * the standard shape if it doesn't exist yet.
 */
export function addContactToWhitelistFile(
  configPath: string,
  entry: WhitelistEntry
): { added: boolean; updated: boolean } {
  const abs = path.resolve(configPath);
  let file: WhitelistFile;
  if (fs.existsSync(abs)) {
    file = JSON.parse(fs.readFileSync(abs, 'utf-8')) as WhitelistFile;
    if (!Array.isArray(file.contacts)) file.contacts = [];
    if (!Array.isArray(file.groups)) file.groups = [];
  } else {
    file = { contacts: [], groups: [] };
    fs.mkdirSync(path.dirname(abs), { recursive: true });
  }

  const isGroup = entry.wa_id.endsWith('@g.us');
  const bucket = isGroup ? file.groups : file.contacts;
  const existing = bucket.find((e) => e.wa_id === entry.wa_id);
  if (existing) {
    existing.display_name = existing.display_name ?? entry.display_name ?? null;
    existing.notes = existing.notes ?? entry.notes ?? null;
    fs.writeFileSync(abs, JSON.stringify(file, null, 2) + '\n');
    return { added: false, updated: true };
  }

  bucket.push({
    wa_id: entry.wa_id,
    display_name: entry.display_name ?? null,
    notes: entry.notes ?? null,
  });
  fs.writeFileSync(abs, JSON.stringify(file, null, 2) + '\n');
  return { added: true, updated: false };
}

export function syncWhitelistToDb(db: DB, wl: Whitelist): void {
  const now = Math.floor(Date.now() / 1000);
  const upsert = db.prepare(
    `INSERT INTO contacts (wa_id, display_name, is_group, whitelisted, notes, first_seen_at, last_seen_at)
     VALUES (?, ?, ?, 1, ?, ?, ?)
     ON CONFLICT(wa_id) DO UPDATE SET
       whitelisted = 1,
       display_name = COALESCE(excluded.display_name, contacts.display_name),
       notes = COALESCE(excluded.notes, contacts.notes)`
  );

  db.prepare('UPDATE contacts SET whitelisted = 0').run();

  const tx = db.transaction((entries: WhitelistEntry[]) => {
    for (const e of entries) {
      const isGroup = e.wa_id.endsWith('@g.us') ? 1 : 0;
      upsert.run(e.wa_id, e.display_name ?? null, isGroup, e.notes ?? null, now, now);
    }
  });
  tx(wl.entries());
}
