import { type Env ,connectDb, edges, tasks } from "@repo/db";
import { TaskInsertSchema, EdgeInsertSchema, PostRequestSchema } from "../../../../packages/db/src/schema";
import { getCloudflareContext } from "@opennextjs/cloudflare";
import { eq } from "drizzle-orm";



export async function POST(request: Request) {
  console.log("POST request received");
  try {
    const { env } = getCloudflareContext(); 
    const db = await connectDb(env as Env);
    const body = await request.json();
    
    console.log("Validating data:", body);
    const validatedData = PostRequestSchema.parse(body);

    const [newTask] = await db.insert(tasks).values({
      name: validatedData.name,
      description: validatedData.description,
    }).returning();

    if(newTask?.id && validatedData.target_node_Id){
      await db.insert(edges).values({
        source_node_Id: newTask.id,
        target_node_Id: validatedData.target_node_Id,
        relationship_type: validatedData.relationship_type ?? "default",
      });
    }
    return Response.json(newTask);

} catch (error: any){
   return Response.json({ error: error.message }, { status: 400 })
}
}


export async function GET() {
  try {
    const { env } = getCloudflareContext();
    const db = await connectDb(env as Env);
    const graphData = await db.select({ taskId: tasks.id, taskName: tasks.name, connectedTo: edges.target_node_Id, type: edges.relationship_type }).from(tasks).leftJoin(edges, eq(tasks.id, edges.source_node_Id));

    return Response.json(graphData);
  } catch (error) {
    console.error(error);
    return Response.json({ error: "Database connection failed" }, { status: 500 });
  }
}
