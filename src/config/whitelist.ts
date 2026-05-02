import fs from 'fs';
import path from 'path';
import type { DB } from '../memory/db';

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
