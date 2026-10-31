"use client";

import { useEffect, type ReactNode } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { Splash } from "@/components/ui";
import { ThemeToggle } from "@/components/theme-toggle";
import { AdminServiceIndicator } from "@/components/admin-services";

type NavItem = {
  label: string;
  href?: string;
  note?: string;
};

const NAV_ITEMS: NavItem[] = [
  { label: "Dashboard", href: "/dashboard" },
  { label: "Documents", href: "/documents" },
  { label: "Chat", href: "/chat" },
  { label: "Database", href: "/database" },
];

export default function AppLayout({ children }: { children: ReactNode }) {
  const { status, user, signOut } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

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
    <div className="app-shell flex flex-col overflow-hidden bg-background">
      <header className="z-10 shrink-0 border-b border-card-border bg-card backdrop-blur-2xl">
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
            {user?.is_admin === true && <AdminServiceIndicator />}
            <span className="text-sm text-muted">{user?.email}</span>
            <ThemeToggle />
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

      {/* Below the top bar: a fixed-height flex row. The main region
          scrolls internally; the shell itself never grows. */}
      <div className="flex min-h-0 flex-1">
        <aside className="app-scroll hidden w-60 shrink-0 overflow-y-auto border-r border-card-border md:block">
          <nav aria-label="Main" className="flex flex-col gap-1 p-4">
            {NAV_ITEMS.map((item) => {
              if (!item.href) {
                return (
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
                );
              }

              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.label}
                  href={item.href}
                  aria-current={isActive ? "page" : undefined}
                  className={`flex items-center rounded-xl px-3 py-2.5 text-sm font-medium transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                    isActive
                      ? "bg-active text-foreground"
                      : "text-muted hover:bg-active hover:text-foreground"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </aside>

        {/* Main content region: Dashboard/Documents/Database scroll in here
            while the top bar and sidebar stay put. */}
        <section className="app-scroll flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto overscroll-contain">
          {children}
        </section>
      </div>
    </div>
  );
}
