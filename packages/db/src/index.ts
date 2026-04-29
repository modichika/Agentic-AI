// packages/db/src/index.ts
import postgres from 'postgres';
import * as schema from './schema';
import { drizzle } from 'drizzle-orm/postgres-js';

export interface Env {
  DATABASE_URL?: string;
  HYPERDRIVE?: { connectionString: string };
}

export const { tasks } = schema;
export const { edges } = schema;

export async function connectDb(env: Env) {
  const connectionString = env.HYPERDRIVE?.connectionString || env.DATABASE_URL;

  if (!connectionString) {
    console.error("Environment keys found:", Object.keys(env));
    throw new Error("No database connection string found in environment.");
  }

  const client = postgres(connectionString)
  return drizzle(client, { schema });
}