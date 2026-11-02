"use client";

import { useEffect, type ReactNode } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { Splash } from "@/components/ui";
import { ThemeToggle } from "@/components/theme-toggle";
import { AdminServiceIndicator } from "@/components/admin-services";
import { Logo } from "@/components/logo";

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

function MainNavigation({
  pathname,
  mobile = false,
}: {
  pathname: string;
  mobile?: boolean;
}) {
  return (
    <nav aria-label="Main" className="flex flex-col gap-1">
      {NAV_ITEMS.map((item) => {
        if (!item.href) {
          return (
            <div
              key={item.label}
              aria-disabled="true"
              className="flex min-h-11 items-center justify-between rounded-xl px-3 py-3 opacity-50"
            >
              <span className="text-base font-medium text-foreground">
                {item.label}
              </span>
              <span className="rounded-full bg-active px-2 py-0.5 text-xs font-medium text-muted">
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
            className={`flex min-h-11 items-center rounded-xl px-3 py-3 text-base font-medium transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
              isActive
                ? "bg-active text-foreground"
                : "text-muted hover:bg-active hover:text-foreground"
            } ${mobile ? "w-full" : ""}`}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

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
        <div className="relative flex h-16 items-center justify-between px-4 sm:px-6">
          <div className="flex items-center gap-2.5">
            <Logo size={32} />
            <span className="text-[15px] font-semibold tracking-tight">
              RAG Chatbot
            </span>
          </div>
          <div className="hidden items-center gap-2 md:flex lg:gap-3">
            {user?.is_admin === true && (
              <span className="hidden lg:block">
                <AdminServiceIndicator />
              </span>
            )}
            <span className="max-w-[180px] truncate text-sm text-muted">
              {user?.email}
            </span>
            <ThemeToggle />
            <button
              type="button"
              onClick={handleSignOut}
              className="min-h-11 rounded-full border border-card-border px-4 text-sm font-medium text-foreground transition hover:bg-active active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              Sign out
            </button>
          </div>
          <details className="relative md:hidden">
            <summary
              aria-label="Open menu"
              className="flex h-11 w-11 cursor-pointer list-none items-center justify-center rounded-full border border-card-border text-foreground transition hover:bg-active focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent [&::-webkit-details-marker]:hidden"
            >
              <svg
                aria-hidden="true"
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
              >
                <path d="M4 7h16M4 12h16M4 17h16" />
              </svg>
            </summary>
            <div className="absolute right-0 top-[calc(100%+0.5rem)] z-50 w-80 max-w-[calc(100vw-2rem)] rounded-2xl border border-card-border bg-background p-3 shadow-card">
              <MainNavigation pathname={pathname} mobile />
              <div className="mt-3 border-t border-card-border px-3 pt-3">
                <p className="break-all text-base text-muted">{user?.email}</p>
                <div className="mt-3 flex min-h-11 items-center justify-between gap-3">
                  <span className="text-base font-medium text-foreground">
                    Appearance
                  </span>
                  <ThemeToggle />
                </div>
                <button
                  type="button"
                  onClick={handleSignOut}
                  className="mt-3 min-h-11 w-full rounded-full border border-card-border px-4 text-base font-medium text-foreground transition hover:bg-active focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                >
                  Sign out
                </button>
              </div>
            </div>
          </details>
        </div>
      </header>

      {/* Below the top bar: a fixed-height flex row. The main region
          scrolls internally; the shell itself never grows. */}
      <div className="flex min-h-0 flex-1">
        <aside className="app-scroll hidden w-60 shrink-0 overflow-y-auto border-r border-card-border md:block">
          <div className="p-4">
            <MainNavigation pathname={pathname} />
          </div>
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
