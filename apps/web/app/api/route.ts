import { connectDb, tasks } from "@repo/db";
import { TaskInsertSchema } from "../../../../packages/db/src/schema";
// Remove: import { env } from "process";

export async function POST(request: Request) {
  console.log("POST request received");
  try {
    const db = await connectDb(process.env as any);
    const body = await request.json();
    
    console.log("Validating data:", body);
    const validatedData = TaskInsertSchema.parse(body);

    const result = await db.insert(tasks).values({
      name: validatedData.name,
      description: validatedData.description,
    }).returning();

    console.log("Insert successful:", result);
    return Response.json(result);

  } catch (error: any) {
    console.error("Database Error:", error.message);
    return Response.json({ error: error.message }, { status: 500 });
  }
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
