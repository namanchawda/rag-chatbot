"use client";

import {
  Fragment,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ApiError,
  chunkStats,
  clearToken,
  getStoredToken,
  listChunks,
  listDocuments,
  type ChunkListResponse,
  type ChunkStats,
  type DocumentItem,
} from "@/lib/api";

const PAGE_SIZE = 20;
const PREVIEW_CHARS = 300;
const MAX_QUERY_CHARS = 200;
const SEARCH_DEBOUNCE_MS = 300;
const COPY_RESET_MS = 1600;

function escapeRegExp(term: string): string {
  return term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Split the text around case-insensitive matches of the search term so they
// can be wrapped in <mark> — never dangerouslySetInnerHTML.
function highlightText(text: string, query: string): ReactNode {
  const term = query.trim();
  if (!term) return text;
  let regex: RegExp;
  try {
    regex = new RegExp(`(${escapeRegExp(term)})`, "gi");
  } catch {
    return text;
  }
  // With a capture group, odd positions of split() are the matches.
  const parts = text.split(regex);
  return parts.map((part, index) =>
    index % 2 === 1 ? <mark key={index}>{part}</mark> : <Fragment key={index}>{part}</Fragment>,
  );
}

function formatNumber(value: number): string {
  return value.toLocaleString();
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="animate-appear rounded-[20px] border border-card-border bg-card px-5 py-4 shadow-card backdrop-blur-2xl">
      <p className="text-2xl font-semibold tracking-tight text-foreground">
        {formatNumber(value)}
      </p>
      <p className="mt-0.5 text-xs font-medium text-muted">{label}</p>
    </div>
  );
}

function ListSkeleton() {
  return (
    <div
      role="status"
      aria-label="Loading chunks"
      className="mt-6 space-y-3"
    >
      {[0, 1, 2, 3].map((index) => (
        <div
          key={index}
          className="animate-pulse rounded-[20px] border border-card-border bg-card p-5 shadow-card backdrop-blur-2xl"
        >
          <div className="h-4 w-2/5 rounded-full bg-active" />
          <div className="mt-3 h-4 w-full rounded-full bg-active" />
          <div className="mt-2 h-4 w-3/4 rounded-full bg-active" />
        </div>
      ))}
    </div>
  );
}

export default function DatabasePage() {
  const router = useRouter();

  const [stats, setStats] = useState<ChunkStats | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[] | null>(null);
  const [documentFilter, setDocumentFilter] = useState("");
  const [queryInput, setQueryInput] = useState("");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [data, setData] = useState<ChunkListResponse | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [statsError, setStatsError] = useState(false);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [copiedId, setCopiedId] = useState<string | null>(null);
  // Bumped by the Retry button so the fetch effect re-runs.
  const [reloadNonce, setReloadNonce] = useState(0);

  // Guards against stale responses: only the newest request may render.
  const requestIdRef = useRef(0);
  const copyTimerRef = useRef<number | null>(null);

  const readyDocuments = documents?.filter(
    (document) => document.status === "ready",
  );

  // Auth: everything here needs the token; without one, straight to login.
  useEffect(() => {
    if (!getStoredToken()) {
      clearToken();
      router.replace("/login");
    }
  }, [router]);

  // Load once (and again on retry): documents dropdown and stats.
  useEffect(() => {
    const token = getStoredToken();
    if (!token) return;
    let cancelled = false;
    listDocuments(token)
      .then((list) => {
        if (!cancelled) {
          setDocuments(list);
          setStatsError(false);
        }
      })
      .catch((error) => {
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
          router.replace("/login");
          return;
        }
        if (!cancelled) setStatsError(true);
      });
    chunkStats(token)
      .then((value) => {
        if (!cancelled) {
          setStats(value);
          setStatsError(false);
        }
      })
      .catch((error) => {
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
          router.replace("/login");
          return;
        }
        if (!cancelled) setStatsError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [router, reloadNonce]);

  // Search input is debounced; a change always re-reads page 1.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setQuery(queryInput);
      setPage(1);
    }, SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [queryInput]);

  const selectDocument = useCallback((documentId: string) => {
    setDocumentFilter(documentId);
    setPage(1);
  }, []);

  const clearFilters = useCallback(() => {
    setDocumentFilter("");
    setQueryInput("");
    setQuery("");
    setPage(1);
  }, []);

  // The actual fetch, re-run whenever page or filters change.
  useEffect(() => {
    const token = getStoredToken();
    if (!token) {
      clearToken();
      router.replace("/login");
      return;
    }

    const requestId = ++requestIdRef.current;
    const controller = new AbortController();

    void (async () => {
      try {
        const response = await listChunks(token, {
          documentId: documentFilter || null,
          q: query || null,
          limit: PAGE_SIZE,
          offset: (page - 1) * PAGE_SIZE,
          signal: controller.signal,
        });
        if (requestIdRef.current !== requestId) return;
        setData(response);
        setListError(null);
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        if (requestIdRef.current !== requestId) return;
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
          router.replace("/login");
          return;
        }
        setListError(
          error instanceof ApiError
            ? error.message
            : "Cannot reach the server. Check your connection and try again.",
        );
      }
    })();

    return () => {
      // A newer effect run (or unmount) aborts our in-flight request; stale
      // responses can never render because requestIdRef already moved past.
      controller.abort();
    };
  }, [documentFilter, query, page, reloadNonce, router]);

  useEffect(() => {
    return () => {
      if (copyTimerRef.current !== null) {
        window.clearTimeout(copyTimerRef.current);
      }
    };
  }, []);

  const toggleExpanded = useCallback((chunkId: string) => {
    setExpandedIds((current) => {
      const next = new Set(current);
      if (next.has(chunkId)) {
        next.delete(chunkId);
      } else {
        next.add(chunkId);
      }
      return next;
    });
  }, []);

  const copyChunk = useCallback(async (chunkId: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      try {
        const area = document.createElement("textarea");
        area.value = text;
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        document.body.removeChild(area);
      } catch {
        // Clipboard unavailable; the text stays selectable.
        return;
      }
    }
    setCopiedId(chunkId);
    if (copyTimerRef.current !== null) {
      window.clearTimeout(copyTimerRef.current);
    }
    copyTimerRef.current = window.setTimeout(() => {
      setCopiedId(null);
    }, COPY_RESET_MS);
  }, []);

  const noReadyDocuments =
    documents !== null &&
    documents.filter((document) => document.status === "ready").length === 0;
  const hasFilters = documentFilter !== "" || query.trim() !== "";
  const isLoadingFirstPage = data === null && listError === null;
  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  // "Showing X–Y of Z" for the aria-live region and the footer.
  const rangeStart = items.length > 0 ? (page - 1) * PAGE_SIZE + 1 : 0;
  const rangeEnd =
    items.length > 0 ? (page - 1) * PAGE_SIZE + items.length : 0;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <main className="flex-1 p-6 md:p-10">
      <div className="mx-auto max-w-5xl">
        <div className="animate-appear">
          <h1 className="text-3xl font-semibold text-foreground">Database</h1>
          <p className="mt-2 text-muted">
            Browse the exact chunks indexed from your documents. This view is
            read-only.
          </p>
        </div>

        <div className="mt-6 grid grid-cols-3 gap-3">
          <StatCard label="Documents" value={stats?.documents ?? 0} />
          <StatCard label="Chunks" value={stats?.chunks ?? 0} />
          <StatCard label="Characters" value={stats?.characters ?? 0} />
        </div>
        {statsError && (
          <p role="status" className="mt-2 text-xs text-muted">
            Could not load the totals — they may show 0. Retry the list below to
            refill them.
          </p>
        )}

        <div className="mt-6 flex flex-col gap-3 md:flex-row">
          <div className="md:w-64">
            <label
              htmlFor="database-document"
              className="mb-1.5 block text-xs font-medium text-muted"
            >
              Document
            </label>
            <select
              id="database-document"
              value={documentFilter}
              onChange={(event) => selectDocument(event.target.value)}
              disabled={documents === null}
              className="w-full rounded-xl border border-input-border bg-input px-3 py-2 text-sm text-foreground transition focus:border-accent focus:outline-none focus:ring-4 focus:ring-ring disabled:opacity-60"
            >
              <option value="">All documents</option>
              {readyDocuments?.map((document) => (
                <option key={document.id} value={document.id}>
                  {document.filename}
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1">
            <label
              htmlFor="database-search"
              className="mb-1.5 block text-xs font-medium text-muted"
            >
              Search chunk text
            </label>
            <div className="relative">
              <input
                id="database-search"
                type="search"
                value={queryInput}
                maxLength={MAX_QUERY_CHARS}
                onChange={(event) => setQueryInput(event.target.value)}
                placeholder="Find text inside chunks…"
                className="w-full rounded-xl border border-input-border bg-input px-3.5 py-2 pr-9 text-sm text-foreground placeholder:text-muted transition focus:border-accent focus:outline-none focus:ring-4 focus:ring-ring disabled:opacity-60"
              />
              {queryInput.length > 0 && (
                <button
                  type="button"
                  onClick={() => setQueryInput("")}
                  aria-label="Clear search"
                  className="absolute right-1.5 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-full text-muted transition hover:bg-active hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                >
                  ✕
                </button>
              )}
            </div>
          </div>
        </div>

        {noReadyDocuments ? (
          <div className="animate-appear mt-6 rounded-[24px] border border-card-border bg-card p-10 text-center shadow-card backdrop-blur-2xl">
            <p className="text-lg font-semibold text-foreground">
              Nothing indexed yet
            </p>
            <p className="mt-2 text-sm text-muted">
              Chunks appear here after a document finishes processing.{" "}
              <Link
                href="/documents"
                className="rounded-sm font-medium text-accent underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                Upload a PDF or TXT
              </Link>{" "}
              to get started.
            </p>
          </div>
        ) : isLoadingFirstPage ? (
          <ListSkeleton />
        ) : listError ? (
          <div
            role="alert"
            className="mt-6 flex flex-wrap items-center justify-between gap-4 rounded-[24px] border border-red-400/30 bg-red-500/10 p-5"
          >
            <p className="text-sm text-red-600 dark:text-red-400">
              {listError}
            </p>
            <button
              type="button"
              onClick={() => {
                // Bump the nonce so both the list fetch effect and the
                // documents/stats effect re-run.
                setListError(null);
                setReloadNonce((value) => value + 1);
              }}
              className="rounded-full border border-red-400/40 px-4 py-1.5 text-sm font-medium text-red-600 transition hover:bg-red-500/10 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent dark:text-red-400"
            >
              Retry
            </button>
          </div>
        ) : items.length === 0 ? (
          <div className="animate-appear mt-6 rounded-[24px] border border-card-border bg-card p-10 text-center shadow-card backdrop-blur-2xl">
            <p className="text-lg font-semibold text-foreground">
              {hasFilters ? "No chunks match" : "No chunks yet"}
            </p>
            <p className="mt-2 text-sm text-muted">
              {hasFilters
                ? "Try a different search, or clear the filters."
                : "Chunks appear here after your documents finish processing."}
            </p>
            {hasFilters && (
              <button
                type="button"
                onClick={clearFilters}
                className="mt-4 rounded-full border border-card-border px-4 py-1.5 text-sm font-medium text-muted transition hover:bg-active hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                Clear filters
              </button>
            )}
          </div>
        ) : (
          <>
            <p aria-live="polite" className="mt-6 px-1 text-sm text-muted">
              Showing {formatNumber(rangeStart)}–{formatNumber(rangeEnd)} of{" "}
              {formatNumber(total)} chunks
            </p>

            {/* Wide screens: real table with labeled headers. */}
            <div className="mt-3 hidden overflow-hidden rounded-[20px] border border-card-border bg-card shadow-card backdrop-blur-2xl md:block">
              <table className="w-full table-fixed border-collapse text-left text-sm">
                <caption className="sr-only">
                  Indexed chunks from your documents
                </caption>
                <thead>
                  <tr className="border-b border-card-border text-xs font-medium text-muted">
                    <th scope="col" className="w-48 shrink-0 px-4 py-3">
                      Source
                    </th>
                    <th scope="col" className="w-14 px-2 py-3 text-right">
                      #
                    </th>
                    <th scope="col" className="w-20 px-2 py-3 text-right">
                      Chars
                    </th>
                    <th scope="col" className="px-4 py-3">
                      Text
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((chunk) => {
                    const isExpanded = expandedIds.has(chunk.id);
                    const displayText =
                      chunk.text.length > PREVIEW_CHARS && !isExpanded
                        ? `${chunk.text.slice(0, PREVIEW_CHARS)}…`
                        : chunk.text;
                    return (
                      <tr
                        key={chunk.id}
                        className="border-b border-card-border/60 last:border-0 align-top"
                      >
                        <td className="px-4 py-3">
                          <p
                            className="truncate text-[13px] font-medium text-foreground"
                            title={chunk.source}
                          >
                            📄 {chunk.source}
                          </p>
                        </td>
                        <td className="px-2 py-3 text-right text-[13px] text-muted">
                          {chunk.chunk_index}
                        </td>
                        <td className="px-2 py-3 text-right text-[13px] text-muted">
                          {formatNumber(chunk.char_count)}
                        </td>
                        <td className="px-4 py-3">
                          <p className="break-words font-mono text-xs leading-relaxed text-muted">
                            {highlightText(displayText, query)}
                          </p>
                          <div className="mt-1.5 flex items-center gap-3">
                            {chunk.text.length > PREVIEW_CHARS && (
                              <button
                                type="button"
                                onClick={() => toggleExpanded(chunk.id)}
                                aria-expanded={isExpanded}
                                className="rounded-md text-xs font-medium text-accent transition hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                              >
                                {isExpanded ? "Show less" : "Show more"}
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={() => void copyChunk(chunk.id, chunk.text)}
                              className="rounded-md text-xs font-medium text-muted transition hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                            >
                              {copiedId === chunk.id ? "Copied" : "Copy"}
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Narrow screens: stacked cards instead of a squeezed table. */}
            <ul className="mt-3 space-y-3 md:hidden">
              {items.map((chunk, index) => {
                const isExpanded = expandedIds.has(chunk.id);
                const displayText =
                  chunk.text.length > PREVIEW_CHARS && !isExpanded
                    ? `${chunk.text.slice(0, PREVIEW_CHARS)}…`
                    : chunk.text;
                return (
                  <li
                    key={chunk.id}
                    className="animate-appear rounded-[20px] border border-card-border bg-card p-4 shadow-card backdrop-blur-2xl"
                    style={{ animationDelay: `${Math.min(index * 40, 200)}ms` }}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <p
                        className="truncate text-[13px] font-semibold text-foreground"
                        title={chunk.source}
                      >
                        📄 {chunk.source}
                      </p>
                      <p className="shrink-0 text-xs text-muted">
                        #{chunk.chunk_index} · {formatNumber(chunk.char_count)}{" "}
                        chars
                      </p>
                    </div>
                    <p className="mt-2 break-words font-mono text-xs leading-relaxed text-muted">
                      {highlightText(displayText, query)}
                    </p>
                    <div className="mt-2 flex items-center gap-3">
                      {chunk.text.length > PREVIEW_CHARS && (
                        <button
                          type="button"
                          onClick={() => toggleExpanded(chunk.id)}
                          aria-expanded={isExpanded}
                          className="rounded-md text-xs font-medium text-accent transition hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                        >
                          {isExpanded ? "Show less" : "Show more"}
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => void copyChunk(chunk.id, chunk.text)}
                        className="rounded-md text-xs font-medium text-muted transition hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                      >
                        {copiedId === chunk.id ? "Copied" : "Copy"}
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>

            <div className="mt-4 flex items-center justify-between gap-4">
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((current) => Math.max(1, current - 1))}
                className="rounded-full border border-card-border px-4 py-1.5 text-sm font-medium text-muted transition hover:bg-active hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-40"
              >
                ← Previous
              </button>
              <p className="text-xs text-muted">
                Page {formatNumber(page)} of {formatNumber(totalPages)}
              </p>
              <button
                type="button"
                disabled={
                  page >= totalPages || rangeEnd >= total
                }
                onClick={() => setPage((current) => current + 1)}
                className="rounded-full border border-card-border px-4 py-1.5 text-sm font-medium text-muted transition hover:bg-active hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-40"
              >
                Next →
              </button>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
