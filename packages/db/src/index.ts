// packages/db/src/index.ts
import postgres from 'postgres';
import * as schema from './schema';
import { drizzle } from 'drizzle-orm/postgres-js';

export const { tasks } = schema;
export async function connectDb(env: { HYPERDRIVE?: { connectionString: string }, DATABASE_URL?: string }) {
  const connectionString = env.HYPERDRIVE?.connectionString || env.DATABASE_URL;

  if (!connectionString) {
    throw new Error("No database connection string found in environment.");
  }

  const client = postgres(connectionString)
  return drizzle(client, { schema });
}