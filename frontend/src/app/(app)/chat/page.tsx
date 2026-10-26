"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import Markdown from "react-markdown";
import {
  ApiError,
  clearToken,
  getStoredToken,
  listDocuments,
  type DocumentItem,
} from "@/lib/api";
import {
  streamChat,
  type ChatSource,
} from "@/lib/chat";

const MAX_QUESTION_CHARS = 2000;
const COUNTER_THRESHOLD = 1800;
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
};

type Message = UserMessage | AssistantMessage;

let messageId = 0;

function nextId(): string {
  messageId += 1;
  return `m${messageId}`;
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

function AssistantMarkdown({ content }: { content: string }) {
  return (
    <div className="chat-markdown text-[15px] leading-relaxed text-foreground">
      <Markdown>{content}</Markdown>
    </div>
  );
}

function SourcesDisclosure({ sources }: { sources: ChatSource[] }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-3 border-t border-card-border pt-2.5">
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

export default function ChatPage() {
  const router = useRouter();
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streamingQuestion, setStreamingQuestion] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[] | null>(null);
  const [isPinned, setIsPinned] = useState(true);

  const readyCount =
    documents?.filter((document) => document.status === "ready").length ?? 0;

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
        } satisfies AssistantMessage,
      ]);
      setStreamingQuestion(question);
      setIsPinned(true);

      const controller = new AbortController();
      abortRef.current = controller;

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
        await streamChat(question, token, controller.signal, (event) => {
          if (event.type === "sources") {
            patchAssistant({ sources: event.sources });
          } else if (event.type === "token") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId && message.role === "assistant"
                  ? { ...message, content: message.content + event.text }
                  : message,
              ),
            );
          } else if (event.type === "error") {
            patchAssistant({ state: "error", errorMessage: event.message });
          } else {
            patchAssistant({ state: "done" });
          }
        });
        // An abort via Stop returns normally: keep the partial answer.
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
          router.replace("/login");
          return;
        }
        patchAssistant({
          state: "error",
          errorMessage:
            error instanceof ApiError
              ? error.message
              : "Something went wrong. Please try again.",
        });
      } finally {
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
      }
    },
    [router],
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
        (message) => message.role === "assistant" && message.state === "error",
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
    input.trim().length > 0 && streamingQuestion === null;

  const lastAssistantError = [...messages]
    .reverse()
    .find(
      (message) => message.role === "assistant" && message.state === "error",
    );

  return (
    <main className="relative flex flex-1 flex-col overflow-hidden">
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-6 pb-6 pt-8 md:px-10"
      >
        <div className="mx-auto max-w-3xl space-y-5">
          {messages.length === 0 ? (
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
                    Upload one in Documents
                  </Link>{" "}
                  and it will be ready in a moment.
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
                    <SourcesDisclosure sources={message.sources} />
                  )}
                </div>
              );
            })
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

      {!isPinned && (
        <button
          type="button"
          onClick={jumpToLatest}
          className="absolute bottom-36 left-1/2 z-10 -translate-x-1/2 rounded-full border border-card-border bg-card px-4 py-1.5 text-xs font-medium text-muted shadow-card backdrop-blur-2xl transition hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          ↓ Jump to latest
        </button>
      )}

      <div className="shrink-0 border-t border-card-border bg-card/80 px-6 py-4 backdrop-blur-2xl md:px-10">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              disabled={streamingQuestion !== null}
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
    </main>
  );
}
