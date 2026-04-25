import { type Env ,connectDb, tasks } from "@repo/db";
import { TaskInsertSchema } from "../../../../packages/db/src/schema";
import { getCloudflareContext } from "@opennextjs/cloudflare";


export async function POST(request: Request) {
  console.log("POST request received");
  try {
    const { env } = getCloudflareContext(); 
    const db = await connectDb(env as Env);
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
    const { env } = getCloudflareContext();
    const db = await connectDb(env as Env);
    const result = await db.select().from(tasks);

    return Response.json(result);
  } catch (error) {
    console.error(error);
    return Response.json({ error: "Database connection failed" }, { status: 500 });
  }
}
