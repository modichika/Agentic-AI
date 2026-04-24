import { pgTable, text, uuid }  from "drizzle-orm/pg-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";
import { z } from "zod";

export const tasks = pgTable("tasks", {
    id: uuid("id").primaryKey().defaultRandom(),
    name: text("name").notNull(),
    description: text("description").notNull(),
});

export const TaskInsertSchema = createInsertSchema(tasks);
export const TaskSelectSchema = createSelectSchema(tasks);

export type Task = z.infer<typeof TaskInsertSchema>
