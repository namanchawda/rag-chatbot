"use client";

import {
  Suspense,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ApiError,
  clearToken,
  deleteSession,
  getPublicConfig,
  getStoredToken,
  listDocuments,
  listSessionMessages,
  listSessions,
  renameSession,
  type DocumentItem,
  type SessionItem,
  type StoredMessage,
} from "@/lib/api";
import { streamChat, type ChatSource } from "@/lib/chat";
import { ConversationsPanel } from "@/components/conversations-panel";

const MAX_QUESTION_CHARS = 2000;
const COUNTER_THRESHOLD = 1800;
const RERANK_STORAGE_KEY = "rag_chatbot_rerank";
const EXAMPLE_QUESTIONS = [
  "Summarize my documents in three bullets",
  "What are the key deadlines mentioned anywhere?",
  "Explain the main topic in simple terms",
];

type UserMessage = {
  id: string;
  role: "user";
  content: string;
};

type AssistantState = "streaming" | "done" | "error";

type AssistantMessage = {
  id: string;
  role: "assistant";
  question: string;
  content: string;
  sources: ChatSource[];
  state: AssistantState;
  errorMessage?: string;
  // Set for "retrying cannot help" failures (e.g. the daily message limit):
  // no Retry button is offered for that answer.
  noRetry?: boolean;
  // Rerank per-request state: what was requested for this answer and what
  // the backend actually did (reranked=false + rerankRequested=true → note).
  rerankRequested?: boolean;
  reranked?: boolean;
};

type Message = UserMessage | AssistantMessage;

let messageId = 0;

function nextId(): string {
  messageId += 1;
  return `m${messageId}`;
}

function loadStoredRerank(): boolean | null {
  try {
    const raw = window.localStorage.getItem(RERANK_STORAGE_KEY);
    if (raw === "true") return true;
    if (raw === "false") return false;
    return null;
  } catch {
    // localStorage may be unavailable; fall back to the server default.
    return null;
  }
}

function storeRerank(value: boolean): void {
  try {
    window.localStorage.setItem(RERANK_STORAGE_KEY, value ? "true" : "false");
  } catch {
    // Persisting is best-effort only.
  }
}

function mapHistoryMessage(message: StoredMessage): Message {
  if (message.role === "user") {
    return { id: message.id, role: "user", content: message.content };
  }
  return {
    id: message.id,
    role: "assistant",
    question: "",
    content: message.content,
    sources: message.sources ?? [],
    state: "done",
  };
}

function groupSourcesByFile(sources: ChatSource[]): [string, ChatSource[]][] {
  const groups = new Map<string, ChatSource[]>();
  for (const source of sources) {
    const list = groups.get(source.source) ?? [];
    list.push(source);
    groups.set(source.source, list);
  }
  return Array.from(groups.entries());
}

function TypingIndicator() {
  return (
    <span
      className="typing-dots inline-flex items-center gap-1.5 py-1.5"
      role="img"
      aria-label="Assistant is typing"
    >
      <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-muted/60" />
      <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-muted/60" />
      <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-muted/60" />
    </span>
  );
}

// react-markdown blocks raw HTML by default and is used here without
// dangerouslySetInnerHTML. remark-gfm adds tables/strikethrough/task lists;
// links open in a new tab so readers aren't navigated away from the chat.
// `table` is wrapped in a scrollable container so wide tables never overflow.
function AssistantMarkdown({ content }: { content: string }) {
  return (
    <div className="chat-markdown text-[15px] leading-relaxed text-foreground">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="chat-table-wrap">
              <table>{children}</table>
            </div>
          ),
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
        }}
      >
        {content}
      </Markdown>
    </div>
  );
}

function RerankSwitch({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label
      title="Re-scores retrieved passages for relevance, so sources fit the question better. Slightly slower."
      className="inline-flex cursor-pointer items-center gap-2 rounded-md text-xs font-medium text-muted transition hover:text-foreground"
    >
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        aria-label="Rerank results"
        className="peer sr-only"
      />
      <span
        aria-hidden="true"
        className="relative inline-flex h-5 w-9 shrink-0"
      >
        <span className="absolute inset-0 rounded-full bg-card-border transition peer-checked:bg-accent peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent" />
        <span className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform peer-checked:translate-x-4" />
      </span>
      Rerank results
    </label>
  );
}

function SourcesDisclosure({
  sources,
  fallback = false,
}: {
  sources: ChatSource[];
  fallback?: boolean;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-3 border-t border-card-border pt-2.5">
      {fallback && (
        <p className="mb-1.5 text-[11px] italic text-amber-600 dark:text-amber-400">
          Ranking unavailable, showing closest matches
        </p>
      )}
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="rounded-md text-xs font-medium text-muted transition hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        {open ? "▾" : "▸"} Sources ({sources.length})
      </button>
      {open && (
        <ul className="mt-2.5 space-y-3">
          {groupSourcesByFile(sources).map(([filename, items]) => (
            <li key={filename}>
              <p className="text-xs font-semibold text-foreground">📄 {filename}</p>
              <ul className="mt-1.5 space-y-1.5">
                {items.map((item) => (
                  <li
                    key={`${item.document_id}:${item.chunk_index}`}
                    className="rounded-lg bg-active px-3 py-2 text-xs leading-relaxed text-muted"
                  >
                    {/* Same raw numbering as the Database page's "#" column.
                        Old saved messages may lack chunk_index: no pill then. */}
                    {typeof item.chunk_index === "number" &&
                      Number.isFinite(item.chunk_index) && (
                        <span
                          title="Position of this passage in the document. You can see the full text in the Database tab."
                          className="mr-2 inline-block rounded-full border border-accent/20 bg-accent/10 px-2 py-0.5 align-middle text-[10px] font-semibold text-accent"
                        >
                          Chunk {item.chunk_index}
                        </span>
                      )}
                    {item.snippet}
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function HistorySkeleton() {
  return (
    <div role="status" className="space-y-4 pt-4">
      <span className="sr-only">Loading conversation…</span>
      <div className="flex justify-end">
        <div className="h-9 w-1/3 animate-pulse rounded-[20px] bg-active" />
      </div>
      <div className="flex justify-start">
        <div className="h-24 w-2/3 animate-pulse rounded-[20px] bg-active" />
      </div>
      <div className="flex justify-start">
        <div className="h-14 w-1/2 animate-pulse rounded-[20px] bg-active" />
      </div>
    </div>
  );
}

function ChatWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const urlSession = searchParams.get("session");

  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);
  const activeIdRef = useRef<string | null>(null);
  // The ?session= value we last acknowledged or wrote ourselves; a mismatch
  // means navigation we did not initiate (mount, back/forward, link).
  const urlWrittenRef = useRef<string | null>(null);

  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionItem[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [historyState, setHistoryState] = useState<
    "idle" | "loading" | "done" | "error"
  >("idle");
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [streamingQuestion, setStreamingQuestion] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[] | null>(null);
  const [isPinned, setIsPinned] = useState(true);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const drawerRef = useRef<HTMLDivElement>(null);
  const drawerToggleRef = useRef<HTMLButtonElement>(null);

  // Rerank: availability/defaults from /config, remembered choice from
  // localStorage. The switch stays hidden until /config says it is available.
  const [rerankAvailable, setRerankAvailable] = useState(false);
  const [rerankEnabled, setRerankEnabled] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const stored = loadStoredRerank();
    getPublicConfig()
      .then((config) => {
        if (cancelled) return;
        setRerankAvailable(config.rerank_available !== false);
        setRerankEnabled(stored ?? config.rerank_default !== false);
      })
      .catch(() => {
        // /config unavailable: keep rerank off (server default applies anyway).
        if (!cancelled) setRerankEnabled(stored ?? false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function handleRerankChange(value: boolean) {
    setRerankEnabled(value);
    storeRerank(value);
  }

  const readyCount =
    documents?.filter((document) => document.status === "ready").length ?? 0;

  const noteSessionActivity = useCallback((sessionId: string) => {
    const nowIso = new Date().toISOString();
    setSessions((current) => {
      if (!current) return current;
      const existing = current.find((session) => session.id === sessionId);
      const item: SessionItem = existing
        ? { ...existing, updated_at: nowIso }
        : { id: sessionId, title: "", created_at: nowIso, updated_at: nowIso };
      return [item, ...current.filter((session) => session.id !== sessionId)];
    });
  }, []);

  const loadSessions = useCallback(() => {
    const token = getStoredToken();
    if (!token) return;
    void (async () => {
      try {
        const list = await listSessions(token, 100);
        setSessions(list);
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
          router.replace("/login");
          return;
        }
        setListError(
          error instanceof ApiError
            ? error.message
            : "Could not load your conversations. Please try again.",
        );
      }
    })();
  }, [router]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const openSession = useCallback(
    (sessionId: string | null, writeUrl = false) => {
      // Abort any in-flight answer first; the partial answer is saved
      // server-side as if Stop was pressed.
      abortRef.current?.abort();
      abortRef.current = null;
      setDrawerOpen(false);
      setListError(null);
      setHistoryError(null);
      setStreamingQuestion(null);
      activeIdRef.current = sessionId;
      setActiveSessionId(sessionId);

      if (writeUrl) {
        urlWrittenRef.current = sessionId;
        router.replace(
          sessionId === null ? "/chat" : `/chat?session=${sessionId}`,
        );
      }

      if (sessionId === null) {
        setMessages([]);
        setHistoryState("idle");
        return;
      }

      setMessages([]);
      setHistoryState("loading");
      const token = getStoredToken();
      if (!token) {
        clearToken();
        router.replace("/login");
        return;
      }
      void (async () => {
        try {
          const stored = await listSessionMessages(token, sessionId);
          if (activeIdRef.current !== sessionId) return;
          setMessages(stored.map(mapHistoryMessage));
          setHistoryState("done");
        } catch (error) {
          if (activeIdRef.current !== sessionId) return;
          if (error instanceof ApiError && error.status === 401) {
            clearToken();
            router.replace("/login");
            return;
          }
          if (error instanceof ApiError && error.status === 404) {
            setHistoryError(
              "That conversation is no longer available — you are back to a fresh chat.",
            );
            activeIdRef.current = null;
            setActiveSessionId(null);
            setMessages([]);
            urlWrittenRef.current = null;
            router.replace("/chat");
            return;
          }
          setHistoryError(
            error instanceof ApiError
              ? error.message
              : "Could not load the conversation. Please try again.",
          );
          setHistoryState("error");
        }
      })();
    },
    [router],
  );

  // React to mount and to back/forward navigation; our own URL writes are
  // acknowledged via urlWrittenRef and skipped here.
  useEffect(() => {
    if (urlWrittenRef.current === urlSession) return;
    urlWrittenRef.current = urlSession;
    openSession(urlSession, false);
  }, [openSession, urlSession]);

  // Document caption: fetch the list once on mount.
  useEffect(() => {
    const token = getStoredToken();
    if (!token) {
      clearToken();
      router.replace("/login");
      return;
    }
    let cancelled = false;
    void listDocuments(token)
      .then((list) => {
        if (!cancelled) setDocuments(list);
      })
      .catch((error) => {
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
          router.replace("/login");
        }
        // Other failures simply leave the caption blank.
      });
    return () => {
      cancelled = true;
    };
  }, [router]);

  // Drawer focus handling: focus its first control when it opens, Escape
  // closes it, focus returns to the toggle button afterwards.
  useEffect(() => {
    if (!drawerOpen) return;
    const pendingFocus = drawerToggleRef.current;
    const panel = drawerRef.current;
    const firstControl = panel?.querySelector<HTMLElement>("button") ?? null;
    firstControl?.focus();
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        setDrawerOpen(false);
        pendingFocus?.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      pendingFocus?.focus();
    };
  }, [drawerOpen]);

  const isPinnedRef = useRef(true);
  const scrollToBottom = useCallback((smooth: boolean) => {
    const container = scrollRef.current;
    if (!container) return;
    container.scrollTo({
      top: container.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
  }, []);

  useLayoutEffect(() => {
    // Auto-scroll after every render while the user is still at the bottom.
    if (isPinnedRef.current) scrollToBottom(false);
  });

  function handleScroll() {
    const container = scrollRef.current;
    if (!container) return;
    const distance = container.scrollHeight - container.scrollTop - container.clientHeight;
    const nowPinned = distance < 80;
    isPinnedRef.current = nowPinned;
    setIsPinned(nowPinned);
  }

  function jumpToLatest() {
    isPinnedRef.current = true;
    setIsPinned(true);
    scrollToBottom(true);
  }

  const sendQuestion = useCallback(
    async (question: string) => {
      const token = getStoredToken();
      if (!token) {
        clearToken();
        router.replace("/login");
        return;
      }

      const requestId = ++requestIdRef.current;
      const startSessionId = activeIdRef.current;
      let streamSession = startSessionId;
      const controller = new AbortController();
      abortRef.current = controller;
      // Guards against late events writing into another conversation:
      // switching or a new chat aborts the stream AND bumps the request id.
      const bailed = () =>
        requestIdRef.current !== requestId || controller.signal.aborted;
      const isCurrent = () => !bailed() && activeIdRef.current === streamSession;

      const assistantId = nextId();
      setMessages((current) => [
        ...current,
        { id: nextId(), role: "user", content: question } satisfies UserMessage,
        {
          id: assistantId,
          role: "assistant",
          question,
          content: "",
          sources: [],
          state: "streaming",
          rerankRequested: rerankEnabled,
        } satisfies AssistantMessage,
      ]);
      setStreamingQuestion(question);
      setIsPinned(true);

      const patchAssistant = (patch: Partial<AssistantMessage>) => {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId && message.role === "assistant"
              ? { ...message, ...patch }
              : message,
          ),
        );
      };

      try {
        await streamChat(
          question,
          token,
          controller.signal,
          (event) => {
            if (bailed()) return;
            if (event.type === "session") {
              streamSession = event.sessionId;
              if (startSessionId === null) {
                // A fresh chat just became a real conversation: adopt the
                // backend-created session without interrupting the stream.
                activeIdRef.current = event.sessionId;
                setActiveSessionId(event.sessionId);
                urlWrittenRef.current = event.sessionId;
                router.replace(`/chat?session=${event.sessionId}`);
                const nowIso = new Date().toISOString();
                setSessions((current) => {
                  const existing = current?.find(
                    (session) => session.id === event.sessionId,
                  );
                  const item: SessionItem = existing
                    ? {
                        ...existing,
                        title: event.title || existing.title,
                        updated_at: nowIso,
                      }
                    : {
                        id: event.sessionId,
                        title: event.title,
                        created_at: nowIso,
                        updated_at: nowIso,
                      };
                  return [
                    item,
                    ...(current?.filter(
                      (session) => session.id !== event.sessionId,
                    ) ?? []),
                  ];
                });
              }
            } else if (event.type === "sources") {
              patchAssistant({
                sources: event.sources,
                reranked: event.reranked,
              });
            } else if (event.type === "token") {
              if (!isCurrent()) return;
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantId && message.role === "assistant"
                    ? { ...message, content: message.content + event.text }
                    : message,
                ),
              );
            } else if (event.type === "error") {
              patchAssistant({
                state: "error",
                errorMessage: event.message,
                noRetry: event.code === "service_unavailable",
              });
            } else {
              patchAssistant({ state: "done" });
            }
          },
          { sessionId: startSessionId, rerank: rerankEnabled },
        );
        // An abort via Stop returns normally: keep the partial answer.
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
          router.replace("/login");
          return;
        }
        if (
          error instanceof ApiError &&
          (error.status === 429 || error.status === 503)
        ) {
          // These requests cannot be helped by retrying from the chat UI.
          patchAssistant({
            state: "error",
            errorMessage: error.message,
            noRetry: true,
          });
        } else {
          patchAssistant({
            state: "error",
            errorMessage:
              error instanceof ApiError
                ? error.message
                : "Something went wrong. Please try again.",
          });
        }
      } finally {
        if (requestIdRef.current === requestId) {
          abortRef.current = null;
          setStreamingQuestion(null);
          // Never leave a message stuck in the streaming state (e.g. Stop).
          setMessages((current) =>
            current.map((message) =>
              message.role === "assistant" && message.state === "streaming"
                ? { ...message, state: "done" }
                : message,
            ),
          );
          // The conversation's answer finished (or stopped): newest activity.
          // streamSession was adopted from the session event, if any.
          if (streamSession !== null) noteSessionActivity(streamSession);
        }
      }
    },
    [noteSessionActivity, rerankEnabled, router],
  );

  function sendMessage() {
    const trimmed = input.trim();
    if (!trimmed || streamingQuestion !== null) return;
    void sendQuestion(trimmed);
    setInput("");
  }

  function stopStreaming() {
    abortRef.current?.abort();
  }

  function retryLast() {
    const lastAssistant = [...messages]
      .reverse()
      .find(
        (message) =>
          message.role === "assistant" &&
          message.state === "error" &&
          !message.noRetry,
      );
    if (lastAssistant?.role === "assistant" && lastAssistant.question) {
      void sendQuestion(lastAssistant.question);
    }
  }

  function handleChange(value: string) {
    setInput(value.slice(0, MAX_QUESTION_CHARS));
    // Auto-grow the composer.
    const element = textareaRef.current;
    if (element) {
      element.style.height = "auto";
      element.style.height = `${Math.min(element.scrollHeight, 176)}px`;
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    if (input.trim() && streamingQuestion === null) sendMessage();
  }

  const isSendEnabled =
    input.trim().length > 0 &&
    streamingQuestion === null &&
    historyState !== "loading";

  const lastAssistantError = [...messages]
    .reverse()
    .find(
      (message) =>
        message.role === "assistant" &&
        message.state === "error" &&
        !message.noRetry,
    );

  const isEmptyNewChat =
    activeSessionId === null &&
    messages.length === 0 &&
    historyState !== "loading";

  const handleRename = useCallback(
    async (sessionId: string, title: string) => {
      const token = getStoredToken();
      if (!token) {
        clearToken();
        router.replace("/login");
        return;
      }
      try {
        const updated = await renameSession(token, sessionId, title);
        setSessions((current) =>
          current
            ? current.map((session) =>
                session.id === sessionId ? updated : session,
              )
            : current,
        );
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
          router.replace("/login");
          return;
        }
        setListError(
          error instanceof ApiError
            ? error.message
            : "Could not rename the conversation. Please try again.",
        );
      }
    },
    [router],
  );

  const handleDelete = useCallback(
    async (sessionId: string) => {
      const token = getStoredToken();
      if (!token) {
        clearToken();
        router.replace("/login");
        return;
      }
      try {
        await deleteSession(token, sessionId);
        setSessions((current) =>
          current ? current.filter((session) => session.id !== sessionId) : current,
        );
        if (activeIdRef.current === sessionId) {
          openSession(null, true);
        }
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
          router.replace("/login");
          return;
        }
        setListError(
          error instanceof ApiError
            ? error.message
            : "Could not delete the conversation. Please try again.",
        );
      }
    },
    [openSession, router],
  );

  return (
    // Fills the shell's main region exactly; the message list is the only
    // scroller, so the composer stays visible no matter how long the chat.
    <main className="relative flex min-h-0 flex-1 overflow-hidden">
      <ConversationsPanel
        sessions={sessions}
        error={listError}
        activeSessionId={activeSessionId}
        className="hidden w-72 shrink-0 border-r min-[900px]:flex"
        onSelect={(sessionId) => openSession(sessionId, true)}
        onNewChat={() => {
          openSession(null, true);
          textareaRef.current?.focus();
        }}
        onRename={handleRename}
        onDelete={handleDelete}
        onRetry={loadSessions}
        onDismissError={() => setListError(null)}
      />

      {/* Slide-over drawer under ~900px */}
      {drawerOpen && (
        <div className="fixed inset-0 z-30 min-[900px]:hidden">
          <div
            aria-hidden="true"
            onClick={() => setDrawerOpen(false)}
            className="absolute inset-0 bg-black/40"
          />
          <div
            ref={drawerRef}
            role="dialog"
            aria-modal="true"
            aria-label="Conversations"
            className="animate-appear absolute inset-y-0 left-0 flex w-80 max-w-[85vw] flex-col border-r border-card-border bg-background shadow-card"
          >
            <ConversationsPanel
              sessions={sessions}
              error={listError}
              activeSessionId={activeSessionId}
              className="flex min-h-0 flex-1 flex-col"
              onSelect={(sessionId) => openSession(sessionId, true)}
              onNewChat={() => {
                openSession(null, true);
                textareaRef.current?.focus();
              }}
              onRename={handleRename}
              onDelete={handleDelete}
              onRetry={loadSessions}
              onDismissError={() => setListError(null)}
            />
          </div>
        </div>
      )}

      <div className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <div className="flex shrink-0 items-center gap-2 border-b border-card-border bg-card/60 px-4 py-2 min-[900px]:hidden">
          <button
            ref={drawerToggleRef}
            type="button"
            onClick={() => setDrawerOpen(true)}
            aria-label="Open conversations"
            className="rounded-lg px-2 py-1.5 text-sm font-medium text-muted transition hover:bg-active hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            ☰ Conversations
          </button>
        </div>

        {/* Message list: the one scrolling region. It is absolutely placed
            inside a relative wrapper so the view never scrolls the page, and
            "Jump to latest" can sit just above the composer. */}
        <div className="relative min-h-0 flex-1">
          <div
            ref={scrollRef}
            onScroll={handleScroll}
            className="app-scroll absolute inset-0 overflow-y-auto overscroll-contain px-6 pb-6 pt-8 md:px-10"
          >
          <div className="mx-auto max-w-3xl space-y-5">
            {isEmptyNewChat ? (
              <div className="animate-appear flex flex-col items-center justify-center pb-10 pt-14 text-center">
                <span
                  aria-hidden="true"
                  className="flex h-14 w-14 items-center justify-center rounded-[20px] bg-accent text-2xl text-white shadow-card"
                >
                  ✦
                </span>
                <h1 className="mt-5 text-2xl font-semibold tracking-tight text-foreground">
                  Hey — ask anything about your documents
                </h1>
                {documents === null ? (
                  <p className="mt-2 text-sm text-muted">Loading your library…</p>
                ) : readyCount > 0 ? (
                  <p className="mt-2 text-sm text-muted">
                    Chatting across {readyCount}{" "}
                    {readyCount === 1 ? "document" : "documents"} — answers stay
                    grounded in them.
                  </p>
                ) : (
                  <p className="mt-2 text-sm text-muted">
                    No ready documents yet.{" "}
                    <Link
                      href="/documents"
                      className="rounded-sm font-medium text-accent underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                    >
                      Upload a PDF or TXT
                    </Link>{" "}
                    in Documents and it will be ready in a moment.
                  </p>
                )}
                {readyCount > 0 && (
                  <div className="mt-8 flex flex-wrap justify-center gap-2.5">
                    {EXAMPLE_QUESTIONS.map((question) => (
                      <button
                        key={question}
                        type="button"
                        onClick={() => {
                          handleChange(question);
                          textareaRef.current?.focus();
                        }}
                        className="rounded-full border border-card-border bg-card px-4 py-2 text-sm text-muted shadow-card backdrop-blur-2xl transition hover:border-accent/50 hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : historyState === "loading" ? (
              <HistorySkeleton />
            ) : (
              messages.map((message) => {
                if (message.role === "user") {
                  return (
                    <div key={message.id} className="animate-appear flex justify-end">
                      <div className="max-w-[80%] whitespace-pre-wrap rounded-[20px] bg-accent px-4 py-2.5 text-[15px] leading-relaxed text-white shadow-card">
                        {message.content}
                      </div>
                    </div>
                  );
                }

                const isStreaming = message.state === "streaming";
                return (
                  <div
                    key={message.id}
                    className="animate-appear max-w-[85%] rounded-[20px] border border-card-border bg-card px-4 py-3 shadow-card backdrop-blur-2xl"
                  >
                    {message.content.length > 0 ? (
                      <div aria-live={isStreaming ? "polite" : undefined}>
                        <AssistantMarkdown content={message.content} />
                      </div>
                    ) : isStreaming ? (
                      <div aria-live="polite">
                        <TypingIndicator />
                      </div>
                    ) : (
                      <div aria-live="polite" className="text-sm text-muted">
                        —
                      </div>
                    )}

                    {message.errorMessage && (
                      <p
                        role="alert"
                        className="mt-2 rounded-xl border border-red-400/30 bg-red-500/10 px-3 py-2 text-sm text-red-600 dark:text-red-400"
                      >
                        {message.content.length > 0 && (
                          <span className="font-medium">
                            The answer stopped early.{" "}
                          </span>
                        )}
                        {message.errorMessage}
                      </p>
                    )}

                    {!isStreaming && message.sources.length > 0 && (
                      <SourcesDisclosure
                        sources={message.sources}
                        fallback={
                          message.rerankRequested === true &&
                          message.reranked === false
                        }
                      />
                    )}
                  </div>
                );
              })
            )}

            {historyError && (
              <p
                role="alert"
                className="mx-auto max-w-md rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-3 text-center text-sm text-red-600 dark:text-red-400"
              >
                {historyError}
              </p>
            )}
            {historyState === "error" && activeSessionId !== null && (
              <div className="flex justify-center">
                <button
                  type="button"
                  onClick={() => {
                    const failed = activeSessionId;
                    if (failed !== null) openSession(failed, true);
                  }}
                  className="rounded-full border border-card-border px-4 py-1.5 text-sm font-medium text-muted transition hover:bg-active hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                >
                  ↻ Try again
                </button>
              </div>
            )}

            {!streamingQuestion &&
              (lastAssistantError as AssistantMessage | undefined)?.question && (
                <button
                  type="button"
                  onClick={retryLast}
                  className="rounded-full border border-card-border px-4 py-1.5 text-sm font-medium text-muted transition hover:bg-active hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                >
                  ↻ Retry last question
                </button>
              )}
          </div>
          </div>

          {/* Anchored to the wrapper around the message list: sits directly
              above the composer whatever its height. */}
          {!isPinned && (
            <button
              type="button"
              onClick={jumpToLatest}
              className="absolute bottom-2 left-1/2 z-10 -translate-x-1/2 rounded-full border border-card-border bg-card px-4 py-1.5 text-xs font-medium text-muted shadow-card backdrop-blur-2xl transition hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              ↓ Jump to latest
            </button>
          )}
        </div>

        <div className="shrink-0 border-t border-card-border bg-card/80 px-6 py-4 backdrop-blur-2xl md:px-10">
          <div className="mx-auto max-w-3xl">
            {rerankAvailable && (
              <div className="mb-2 flex">
                <RerankSwitch
                  checked={rerankEnabled}
                  onChange={handleRerankChange}
                />
              </div>
            )}
            <div className="flex items-end gap-2">
              <textarea
                ref={textareaRef}
                rows={1}
                value={input}
                disabled={streamingQuestion !== null || historyState === "loading"}
                placeholder="Ask about your documents…"
                aria-label="Your question"
                onChange={(event) => handleChange(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                className="min-h-[44px] flex-1 resize-none overflow-hidden rounded-[20px] border border-input-border bg-input px-4 py-2.5 text-[15px] leading-relaxed text-foreground placeholder:text-muted transition focus:border-accent focus:outline-none focus:ring-4 focus:ring-ring disabled:opacity-60"
              />
              {streamingQuestion === null ? (
                <button
                  type="button"
                  onClick={sendMessage}
                  disabled={!isSendEnabled}
                  aria-label="Send question"
                  className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-accent text-lg text-white shadow-sm transition hover:bg-accent-hover active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-40"
                >
                  ↑
                </button>
              ) : (
                <button
                  type="button"
                  onClick={stopStreaming}
                  aria-label="Stop generating answer"
                  className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-input-border bg-input text-foreground shadow-sm transition hover:bg-active active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                >
                  <span
                    aria-hidden="true"
                    className="h-3.5 w-3.5 rounded-[5px] bg-current"
                  />
                </button>
              )}
            </div>
            {input.length >= COUNTER_THRESHOLD && (
              <p
                aria-live={input.length >= MAX_QUESTION_CHARS ? "polite" : "off"}
                className={`mt-2 text-right text-xs ${
                  input.length >= MAX_QUESTION_CHARS ? "text-red-500" : "text-muted"
                }`}
              >
                {input.length}/{MAX_QUESTION_CHARS}
              </p>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

export default function ChatPage() {
  return (
    <Suspense
      fallback={
        <main className="flex flex-1 items-center justify-center">
          <p className="text-sm text-muted">Loading…</p>
        </main>
      }
    >
      <ChatWorkspace />
    </Suspense>
  );
}
