import { Client, LocalAuth, Message } from 'whatsapp-web.js';
import qrcode from 'qrcode-terminal';
import path from 'path';

export interface WAClientOptions {
  sessionPath: string;
  onMessage?: (msg: Message) => void | Promise<void>;
}

export function createWAClient(opts: WAClientOptions): Client {
  const client = new Client({
    authStrategy: new LocalAuth({
      clientId: 'manila',
      dataPath: path.resolve(opts.sessionPath),
    }),
    puppeteer: {
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    },
  });

  client.on('qr', (qr) => {
    console.log('\nScan this QR with WhatsApp → Linked Devices:\n');
    qrcode.generate(qr, { small: true });
  });

  client.on('authenticated', () => {
    console.log('authenticated');
  });

  client.on('auth_failure', (msg) => {
    console.error('auth failure:', msg);
  });

  client.on('ready', () => {
    console.log('ready — listening for messages');
  });

  client.on('disconnected', (reason) => {
    console.warn('disconnected:', reason);
  });

  if (opts.onMessage) {
    const handler = opts.onMessage;
    client.on('message', async (msg) => {
      try {
        await handler(msg);
      } catch (err) {
        console.error('onMessage error:', err);
      }
    });

    client.on('message_create', async (msg) => {
      if (msg.fromMe) {
        try {
          await handler(msg);
        } catch (err) {
          console.error('onMessage (outgoing) error:', err);
        }
      }
    });
  }

  return client;
}
