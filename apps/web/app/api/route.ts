import { connectDb, tasks } from "@repo/db";
import { TaskInsertSchema } from "../../../../packages/db/src/schema";
// Remove: import { env } from "process";

export async function POST(request: Request) {
  const db = await connectDb(process.env as any);
  const body: { name: string; description: string }  = await request.json();
  const validatedData = TaskInsertSchema.parse(body);
  // Insert the task into Neon
  const result = await db.insert(tasks).values({
    name: validatedData.name,
    description: validatedData.description,
  }).returning();

  return Response.json(result);
}


export async function GET() {
  try {
    // Access process.env directly
    const db = await connectDb(process.env as any); 
    const result = await db.select().from(tasks);

    return Response.json(result);
  } catch (error) {
    console.error(error);
    return Response.json({ error: "Database connection failed" }, { status: 500 });
  }
}
