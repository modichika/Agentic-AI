
import { CreateTask } from "@/components/button"

export default async function TableFooterExample() {
  return (
        <main>
       <section>
          <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-semibold tracking-tight">Welcome To Your Data</h1>
          <p className="text-muted-foreground">Create your data.</p>
          </div>
          </section>
          <div className="mt-4 flex items-center justify-between gap-2 md:mt-8">
                <CreateTask/>
            </div>
          
        </main>
        
  );
}
