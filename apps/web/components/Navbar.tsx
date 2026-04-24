import Link from "next/link";
import Image from "next/image";
import { Button } from "@ui/components/button";


export function Navbar(){
    return (
       <nav className="flex items-center justify-between px-6 py-4 border-b bg-background">
      <div className="font-bold text-xl">
        <Link href="/">Agentic-AI</Link>
      </div>
    </nav>
    );
}