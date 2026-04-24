// packages/db/src/index.ts
import postgres from 'postgres';
import * as schema from './schema';
import { drizzle } from 'drizzle-orm/postgres-js';

export const { tasks } = schema;
export async function connectDb(env: NodeJS.ProcessEnv) {
  const client = postgres(env.DATABASE_URL!,)
  return drizzle(client, { schema });
}