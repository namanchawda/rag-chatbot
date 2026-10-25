import type { Metadata } from "next";
import { AuthProvider } from "@/lib/auth";
import { THEME_STORAGE_KEY } from "@/lib/theme";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Chatbot",
  description: "Document-grounded AI chat, coming together piece by piece.",
};

const THEME_INIT_SCRIPT = `(function(){try{var theme=localStorage.getItem("${THEME_STORAGE_KEY}");if(theme==="light"||theme==="dark"){document.documentElement.setAttribute("data-theme",theme)}else{document.documentElement.removeAttribute("data-theme")}}catch(e){}})();`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="min-h-full font-sans">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
