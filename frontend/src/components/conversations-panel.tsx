"use client";

import { useState } from "react";
import type { SessionItem } from "@/lib/api";

type SessionGroup = { label: string; items: SessionItem[] };

const DAY_MS = 24 * 60 * 60 * 1000;

// Backend already sorts by updated_at descending; this only buckets.
function groupSessionsByRecency(sessions: SessionItem[]): SessionGroup[] {
  const today: SessionItem[] = [];
  const yesterday: SessionItem[] = [];
  const week: SessionItem[] = [];
  const older: SessionItem[] = [];
  const now = new Date();
  const startOfToday = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate(),
  ).getTime();
  const startOfYesterday = startOfToday - DAY_MS;
  const startOfWeek = startOfToday - 6 * DAY_MS;
  for (const session of sessions) {
    const time = Date.parse(session.updated_at);
    if (Number.isNaN(time) || time < startOfWeek) older.push(session);
    else if (time >= startOfToday) today.push(session);
    else if (time >= startOfYesterday) yesterday.push(session);
    else week.push(session);
  }
  return (
    [
      ["Today", today],
      ["Yesterday", yesterday],
      ["Previous 7 days", week],
      ["Older", older],
    ] as const
  )
    .filter(([, items]) => items.length > 0)
    .map(([label, items]) => ({ label, items }));
}

function displayTitle(session: SessionItem): string {
  return session.title.trim() ? session.title : "New conversation";
}

const iconButtonClass =
  "flex h-7 w-7 items-center justify-center rounded-lg text-xs text-muted transition hover:bg-active hover:text-foreground active:scale-[0.95] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

function SessionRow({
  session,
  active,
  onSelect,
  onRename,
  onDelete,
}: {
  session: SessionItem;
  active: boolean;
  onSelect: (sessionId: string) => void;
  onRename: (sessionId: string, title: string) => Promise<void>;
  onDelete: (sessionId: string) => Promise<void>;
}) {
  const [mode, setMode] = useState<"view" | "rename" | "delete">("view");
  const [draft, setDraft] = useState("");
  const title = displayTitle(session);

  function startRename() {
    setDraft(session.title);
    setMode("rename");
  }

  function commitRename() {
    const trimmed = draft.trim();
    const valid = trimmed.length > 0 && trimmed.length <= 120;
    setMode("view");
    if (!valid || trimmed === session.title) return;
    void onRename(session.id, trimmed);
  }

  if (mode === "rename") {
    return (
      <li>
        <input
          autoFocus
          value={draft}
          maxLength={120}
          aria-label={`Rename conversation: ${title}`}
          onBlur={() => setMode("view")}
          onKeyDown={(event) => {
            event.stopPropagation();
            if (event.key === "Enter") {
              event.preventDefault();
              commitRename();
            } else if (event.key === "Escape") {
              event.preventDefault();
              setMode("view");
            }
          }}
          onChange={(event) => setDraft(event.target.value)}
          className="w-full rounded-xl border border-input-border bg-input px-3 py-2 text-sm text-foreground transition focus:border-accent focus:outline-none focus:ring-4 focus:ring-ring"
        />
      </li>
    );
  }

  return (
    <li
      className="group"
      onKeyDown={(event) => {
        if (event.key === "Escape" && mode === "delete") {
          event.stopPropagation();
          setMode("view");
        }
      }}
    >
      <div
        className={`flex items-center rounded-xl transition ${
          active ? "bg-active" : "hover:bg-active/60"
        }`}
      >
        <button
          type="button"
          onClick={() => onSelect(session.id)}
          aria-current={active ? "true" : undefined}
          className="min-w-0 flex-1 truncate rounded-xl px-3 py-2 text-left text-sm text-foreground transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          {title}
        </button>
        <div
          className={`flex shrink-0 items-center pr-1.5 transition-opacity ${
            mode === "delete"
              ? "opacity-100"
              : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"
          }`}
        >
          {mode === "view" ? (
            <>
              <button
                type="button"
                onClick={startRename}
                aria-label={`Rename conversation: ${title}`}
                className={iconButtonClass}
              >
                ✎
              </button>
              <button
                type="button"
                onClick={() => setMode("delete")}
                aria-label={`Delete conversation: ${title}`}
                className={iconButtonClass}
              >
                🗑
              </button>
            </>
          ) : (
            <span className="flex items-center gap-1.5 text-[11px] text-muted">
              <span className="pr-0.5">Delete?</span>
              <button
                type="button"
                onClick={() => {
                  setMode("view");
                  void onDelete(session.id);
                }}
                className="rounded-md border border-red-500/30 bg-red-500/10 px-1.5 py-0.5 font-medium text-red-600 transition hover:bg-red-500/20 active:scale-[0.95] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent dark:text-red-400"
              >
                Yes
              </button>
              <button
                type="button"
                onClick={() => setMode("view")}
                className="rounded-md border border-card-border px-1.5 py-0.5 font-medium text-foreground transition hover:bg-active active:scale-[0.95] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                No
              </button>
            </span>
          )}
        </div>
      </div>
    </li>
  );
}

export type ConversationsPanelProps = {
  sessions: SessionItem[] | null;
  error: string | null;
  activeSessionId: string | null;
  className?: string;
  onSelect: (sessionId: string) => void;
  onNewChat: () => void;
  onRename: (sessionId: string, title: string) => Promise<void>;
  onDelete: (sessionId: string) => Promise<void>;
  onRetry: () => void;
  onDismissError: () => void;
};

export function ConversationsPanel({
  sessions,
  error,
  activeSessionId,
  className = "",
  onSelect,
  onNewChat,
  onRename,
  onDelete,
  onRetry,
  onDismissError,
}: ConversationsPanelProps) {
  const groups = sessions ? groupSessionsByRecency(sessions) : [];
  const retryable = sessions === null && error !== null;

  return (
    <div className={`flex flex-col border-card-border bg-card/50 ${className}`}>
      <div className="shrink-0 border-b border-card-border p-2.5">
        <button
          type="button"
          onClick={onNewChat}
          className="w-full rounded-xl border border-card-border bg-input px-3 py-2 text-sm font-medium text-foreground transition hover:bg-active active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          ＋ New chat
        </button>
      </div>

      {error && (
        <div className="shrink-0 border-b border-card-border px-2.5 py-2">
          <div className="flex items-start gap-2">
            <p
              role="alert"
              className="flex-1 text-xs leading-relaxed text-red-600 dark:text-red-400"
            >
              {error}
            </p>
            {retryable && (
              <button
                type="button"
                onClick={onRetry}
                className="shrink-0 rounded-md border border-red-500/30 bg-red-500/10 px-2 py-1 text-[11px] font-medium text-red-600 transition hover:bg-red-500/20 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent dark:text-red-400"
              >
                Retry
              </button>
            )}
            <button
              type="button"
              onClick={onDismissError}
              aria-label="Dismiss error"
              className={`${iconButtonClass} shrink-0`}
            >
              ✕
            </button>
          </div>
        </div>
      )}

      <nav aria-label="Conversations" className="min-h-0 flex-1 overflow-y-auto p-2">
        {sessions === null && !error ? (
          <div aria-hidden="true" className="space-y-1.5">
            {Array.from({ length: 8 }).map((_, index) => (
              <div
                key={index}
                className="h-9 animate-pulse rounded-xl bg-active/70"
              />
            ))}
          </div>
        ) : sessions !== null && sessions.length === 0 ? (
          <p className="px-3 py-10 text-center text-sm leading-relaxed text-muted">
            No conversations yet.
            <br />
            Ask your first question and it will show up here.
          </p>
        ) : groups.length > 0 ? (
          groups.map((group) => (
            <div key={group.label} className="mb-5 last:mb-2">
              <h2 className="px-2 pb-1.5 pt-2 text-[11px] font-semibold uppercase tracking-wide text-muted">
                {group.label}
              </h2>
              <ul className="flex flex-col gap-0.5">
                {group.items.map((session) => (
                  <SessionRow
                    key={session.id}
                    session={session}
                    active={session.id === activeSessionId}
                    onSelect={onSelect}
                    onRename={onRename}
                    onDelete={onDelete}
                  />
                ))}
              </ul>
            </div>
          ))
        ) : null}
      </nav>
    </div>
  );
}
