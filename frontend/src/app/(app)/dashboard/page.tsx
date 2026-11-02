"use client";

import { useAuth } from "@/lib/auth";
import { AdminServicesCard } from "@/components/admin-services";
import Link from "next/link";

function ChatIcon() {
  return (
    <svg
      aria-hidden="true"
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H6l-3 2v-6a7.5 7.5 0 1 1 17-3.5Z" />
      <path d="M8 11h8M8 14h5" />
    </svg>
  );
}

function DocumentIcon() {
  return (
    <svg
      aria-hidden="true"
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M7 3h7l5 5v13H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" />
      <path d="M14 3v6h5M9 14h6M9 17h6" />
    </svg>
  );
}

function LibraryIcon() {
  return (
    <svg
      aria-hidden="true"
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 0 4 21.5v-16Z" />
      <path d="M4 5.5v16M8 7h8M8 11h8M8 15h5" />
    </svg>
  );
}

const WORKSPACE_LINKS = [
  {
    href: "/chat",
    title: "Chat",
    description: "Ask questions and get answers from your documents.",
    label: "Start a chat",
    icon: <ChatIcon />,
    primary: true,
  },
  {
    href: "/documents",
    title: "Documents",
    description: "Upload files and choose how they’re split.",
    label: "View documents",
    icon: <DocumentIcon />,
    primary: false,
  },
  {
    href: "/database",
    title: "Database",
    description: "Browse the text passages stored from your uploads.",
    label: "Open database",
    icon: <LibraryIcon />,
    primary: false,
  },
];

export default function DashboardPage() {
  const { user } = useAuth();

  return (
    <main className="flex flex-1 flex-col">
      <div className="animate-appear mx-auto w-full max-w-6xl px-5 py-8 sm:px-8 sm:py-12 lg:px-12 lg:py-14">
        <section
          aria-labelledby="dashboard-heading"
          className="relative overflow-hidden rounded-[28px] border border-card-border bg-card p-6 shadow-card sm:p-9 lg:p-11"
        >
          <div className="relative z-10 max-w-2xl">
            <p className="text-sm font-semibold uppercase tracking-[0.14em] text-accent">
              Welcome back
            </p>
            <h1
              id="dashboard-heading"
              className="mt-4 max-w-xl text-[34px] font-semibold leading-[1.08] tracking-[-0.04em] text-foreground sm:text-[40px]"
            >
              Retrieval-augmented generation
            </h1>
            <p className="mt-4 max-w-xl text-base leading-7 text-muted sm:text-lg sm:leading-8">
              Upload PDF or text files, then ask questions. Answers come only
              from your uploads and show the passages they’re based on.
            </p>
            {user?.email && (
              <p className="mt-3 text-sm font-medium text-foreground">
                Signed in as {user.email}
              </p>
            )}
            <div className="mt-6 flex flex-wrap gap-3">
              <Link
                href="/chat"
                className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full bg-accent px-6 text-base font-semibold text-white transition hover:bg-accent-hover active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                <ChatIcon />
                Start a chat
              </Link>
              <Link
                href="/documents"
                className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full border border-card-border bg-card px-6 text-base font-semibold text-foreground transition hover:bg-active active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                <DocumentIcon />
                Upload a document
              </Link>
            </div>
          </div>
          <div
            aria-hidden="true"
            className="pointer-events-none absolute -right-6 -top-10 hidden h-64 w-64 items-center justify-center xl:right-8 xl:top-1/2 xl:flex xl:-translate-y-1/2"
          >
            <div className="absolute h-36 w-40 rotate-[-12deg] rounded-2xl border border-card-border bg-background shadow-card" />
            <div className="absolute h-36 w-40 rotate-[7deg] rounded-2xl border border-card-border bg-background shadow-card" />
            <div className="relative flex h-36 w-40 flex-col justify-between rounded-2xl border border-card-border bg-background p-4 shadow-card">
              <DocumentIcon />
              <div className="space-y-2">
                <span className="block h-1.5 w-4/5 rounded-full bg-active" />
                <span className="block h-1.5 w-3/5 rounded-full bg-active" />
                <span className="block h-1.5 w-2/3 rounded-full bg-active" />
              </div>
              <span className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">
                Source material
              </span>
            </div>
          </div>
        </section>

        <section aria-label="Workspace shortcuts" className="mt-8">
          <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
            <h2 className="text-xl font-semibold text-foreground">
              Pick up where you need to
            </h2>
          </div>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {WORKSPACE_LINKS.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`group flex min-h-40 flex-col rounded-2xl border p-5 transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                  item.primary
                    ? "border-accent/25 bg-accent/[0.06] hover:bg-accent/[0.1]"
                    : "border-card-border bg-card hover:border-accent/30 hover:bg-active"
                }`}
              >
                <span
                  className={`flex h-11 w-11 items-center justify-center rounded-xl ${
                    item.primary
                      ? "bg-accent text-white"
                      : "bg-active text-accent"
                  }`}
                >
                  {item.icon}
                </span>
                <span className="mt-4 text-lg font-semibold text-foreground">
                  {item.title}
                </span>
                <span className="mt-1 text-base leading-6 text-muted">
                  {item.description}
                </span>
                <span className="mt-auto flex min-h-11 items-end pt-4 text-base font-semibold text-accent">
                  {item.label}
                  <span
                    aria-hidden="true"
                    className="ml-1 transition-transform group-hover:translate-x-1"
                  >
                    →
                  </span>
                </span>
              </Link>
            ))}
          </div>
        </section>

        {user?.is_admin === true && (
          <div className="mt-8">
            <AdminServicesCard />
          </div>
        )}
      </div>
    </main>
  );
}
