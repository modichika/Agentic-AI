import Link from "next/link";
import { PencilIcon, PlusIcon, TrashIcon } from '@heroicons/react/24/outline';



export function CreateTask() {
  return (
    <div className="flex items-center justify-between ">
        <div className="flex items-center gap-2">
         
       
    <Link
      href="/add"
      className="flex h-10 items-center rounded-lg bg-grey-600 px-4 text-sm font-medium text-white transition-colors hover:bg-[#111111] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
    >
      <span className="hidden md:block">Create Task</span>{' '}
      <PlusIcon className="h-5 md:ml-4" />
    </Link>
     </div>
      </div>
  );
}

// export function UpdateTask({ id }: { id: string }) {
//   return (
//     <Link
//       href={`/${id}/edit`}
//       className="flex items-center justify-center rounded-full border p-1.5 hover:bg-white/10 transition-colors"
//     >
//       <PencilIcon className="w-4" />
//     </Link>
//   );
// }

// export function DeleteTask({ id }: { id: string }){
//   const deleteTaskWithId = deleteTask.bind(null, id);
//   return (
//     <form action={deleteTaskWithId}>
//      <button type="submit" className="flex items-center justify-center rounded-full border p-1.5 hover:bg-white/10 transition-colors">
//         <span className="sr-only">Delete</span>
//         <TrashIcon className="w-4" />
//       </button>
//     </form>
//   )
// }