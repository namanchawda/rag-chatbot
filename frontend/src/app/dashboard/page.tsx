"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { Splash } from "@/components/ui";

const COMING_SOON = [
  { label: "Documents", note: "Coming soon" },
  { label: "Chat", note: "Coming soon" },
  { label: "Database", note: "Coming soon" },
];

export default function DashboardPage() {
  const { status, user, signOut } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (status === "unauthenticated") {
      router.replace("/login");
    }
  }, [status, router]);

  function handleSignOut() {
    signOut();
    router.replace("/login");
  }

  if (status !== "authenticated") {
    return <Splash />;
  }

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="sticky top-0 z-10 border-b border-card-border bg-card backdrop-blur-2xl">
        <div className="flex h-14 items-center justify-between px-6">
          <div className="flex items-center gap-2.5">
            <span
              aria-hidden="true"
              className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent text-sm text-white"
            >
              ✦
            </span>
            <span className="text-[15px] font-semibold tracking-tight">
              RAG Chatbot
            </span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted">{user?.email}</span>
            <button
              type="button"
              onClick={handleSignOut}
              className="rounded-full border border-card-border px-4 py-1.5 text-sm font-medium text-foreground transition hover:bg-active active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <div className="flex flex-1">
        <aside className="hidden w-60 shrink-0 border-r border-card-border md:block">
          <nav aria-label="Main" className="flex flex-col gap-1 p-4">
            {COMING_SOON.map((item) => (
              <div
                key={item.label}
                aria-disabled="true"
                className="flex items-center justify-between rounded-xl px-3 py-2.5 opacity-50"
              >
                <span className="text-sm font-medium text-foreground">
                  {item.label}
                </span>
                <span className="rounded-full bg-active px-2 py-0.5 text-[11px] font-medium text-muted">
                  {item.note}
                </span>
              </div>
            ))}
          </nav>
        </aside>

        <main className="flex flex-1 items-center justify-center p-10">
          <div className="animate-appear max-w-md text-center">
            <h1 className="text-3xl font-semibold text-foreground">
              Welcome{user?.email ? `, ${user.email}` : ""} ✦
            </h1>
            <p className="mt-4 text-lg leading-8 text-muted">
              Your workspace is ready. Documents, chat, and database features
              are on their way — sign-in is all warmed up.
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}
