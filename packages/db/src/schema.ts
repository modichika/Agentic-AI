import { pgTable, text, uuid }  from "drizzle-orm/pg-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";
import { z } from "zod";



export const tasks = pgTable("tasks", {
    id: uuid("id").primaryKey().defaultRandom(),
    name: text("name").notNull(),
    description: text("description").notNull(),
});

export const edges = pgTable("edges", {
    id: uuid("id").primaryKey().defaultRandom(),
    source_node_Id: uuid("sourceId").references(() => tasks.id).notNull(),
    target_node_Id: uuid("targetId").references(() => tasks.id).notNull(),
    relationship_type: text("relationshipType").notNull(),
});

export const PostRequestSchema = z.object({
    name: z.string(),
    description: z.string(),
    source_node_Id: z.string().uuid().optional(),
    target_node_Id: z.string().uuid().optional(),
    relationship_type: z.string().optional(),
});

export const TaskInsertSchema = createInsertSchema(tasks);
export const TaskSelectSchema = createSelectSchema(tasks);
export const EdgeInsertSchema = createInsertSchema(edges);
export const EdgeSelectSchema = createSelectSchema(edges);

export type Task = z.infer<typeof TaskInsertSchema>
export type Edge = z.infer<typeof EdgeInsertSchema>