import { connectDb, tasks } from "@repo/db";
// Remove: import { env } from "process";

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
