"use-client";

import "./globals.css";
import type { Metadata } from "next";
import {Navbar} from "@/components/Navbar";
import { ThemeProvider } from "@/lib/theme_provider";


export const metadata: Metadata = {
  title: "Agentic-Kubeflow docs",
  description: "An Agent that has long-term memory for kubeflow docs in knowledge graph format",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
    <html lang="en" suppressHydrationWarning>
      <body>
         <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            disableTransitionOnChange
          >
            <Navbar/>
            {children}
          </ThemeProvider>
  
          
      </body>
    </html>
    </>
  );
}
