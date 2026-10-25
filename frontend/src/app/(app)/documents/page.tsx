"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  clearToken,
  deleteDocument,
  getStoredToken,
  listDocuments,
  uploadDocument,
  type DocumentItem,
} from "@/lib/api";

const ACCEPTED_EXTENSIONS = new Set(["pdf", "txt", "md", "html"]);
const FILE_INPUT_ACCEPT = ".pdf,.txt,.md,.html";
const MAX_FILE_BYTES = 25 * 1024 * 1024;
const MAX_FILE_LABEL = "PDF, TXT, Markdown, or HTML — up to 25 MB each";
const POLL_INTERVAL_MS = 2500;

function getExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot + 1).toLowerCase();
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function plural(count: number, singular: string): string {
  return `${count} ${count === 1 ? singular : `${singular}s`}`;
}

function StatusBadge({ document }: { document: DocumentItem }) {
  if (document.status === "processing") {
    return (
      <span
        className="inline-flex items-center gap-1.5 rounded-full bg-active px-2.5 py-1 text-xs font-medium text-muted"
        title="Processing"
      >
        <span
          aria-hidden="true"
          className="h-3 w-3 animate-spin rounded-full border-2 border-accent/30 border-t-accent"
        />
        Processing
      </span>
    );
  }

  if (document.status === "ready") {
    return (
      <span
        className="inline-flex items-center rounded-full bg-emerald-500/15 px-2.5 py-1 text-xs font-medium text-emerald-600 dark:text-emerald-400"
        title="Ready for chat"
      >
        Ready · {plural(document.chunk_count, "chunk")}
      </span>
    );
  }

  return (
    <span
      className="inline-flex items-center rounded-full bg-red-500/15 px-2.5 py-1 text-xs font-medium text-red-600 dark:text-red-400"
      title="Processing failed"
    >
      Failed
    </span>
  );
}

function SmallSpinner() {
  return (
    <span
      aria-hidden="true"
      className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current/30 border-t-current"
    />
  );
}

export default function DocumentsPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  const [documents, setDocuments] = useState<DocumentItem[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [rejectMessages, setRejectMessages] = useState<string[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [liveMessage, setLiveMessage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(
    null,
  );
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const loadDocuments = useCallback(async () => {
    const token = getStoredToken();
    if (!token) {
      clearToken();
      router.replace("/login");
      return;
    }
    try {
      const list = await listDocuments(token);
      setDocuments(list);
      setLoadError(null);
    } catch (error) {
      if (error instanceof ApiError) {
        if (error.status === 401) {
          clearToken();
          router.replace("/login");
          return;
        }
        setLoadError(error.message);
      } else {
        setLoadError(
          "Cannot reach the server. Check your connection and try again.",
        );
      }
    }
  }, [router]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadDocuments();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [loadDocuments]);

  const hasProcessing =
    documents?.some((document) => document.status === "processing") ?? false;

  useEffect(() => {
    if (!hasProcessing) return;
    const interval = window.setInterval(() => {
      void loadDocuments();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [hasProcessing, loadDocuments]);

  function openFilePicker() {
    if (isUploading) return;
    inputRef.current?.click();
  }

  const handleFiles = useCallback(
    async (fileList: FileList | File[]) => {
      const files = Array.from(fileList);
      if (files.length === 0 || isUploading) return;

      let currentFile = "";
      const rejected: string[] = [];
      const accepted = files.filter((file) => {
        currentFile = file.name;
        const extension = getExtension(file.name);
        if (!ACCEPTED_EXTENSIONS.has(extension)) {
          rejected.push(
            `"${file.name}" can't be uploaded. Supported types are .pdf, .txt, .md and .html.`,
          );
          return false;
        }
        if (file.size > MAX_FILE_BYTES) {
          rejected.push(
            `"${file.name}" is ${formatBytes(file.size)} — files can be up to 25 MB.`,
          );
          return false;
        }
        return true;
      });

      setRejectMessages(rejected);
      setUploadError(null);

      if (accepted.length === 0) return;

      const token = getStoredToken();
      if (!token) {
        clearToken();
        router.replace("/login");
        return;
      }

      setIsUploading(true);
      setLiveMessage(`Uploading ${plural(accepted.length, "file")}…`);

      try {
        for (const file of accepted) {
          currentFile = file.name;
          await uploadDocument(token, file);
        }
        setLiveMessage(`Uploaded ${plural(accepted.length, "file")}.`);
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
          router.replace("/login");
          return;
        }
        const detail =
          error instanceof ApiError
            ? error.message
            : "Something went wrong. Please try again.";
        setUploadError(`Couldn't upload "${currentFile}". ${detail}`);
      } finally {
        setIsUploading(false);
        await loadDocuments();
      }
    },
    [isUploading, loadDocuments, router],
  );

  async function confirmDelete(document: DocumentItem) {
    const token = getStoredToken();
    if (!token) {
      clearToken();
      router.replace("/login");
      return;
    }

    setDeletingId(document.id);
    setDeleteError(null);
    try {
      await deleteDocument(token, document.id);
      setDocuments((current) =>
        current ? current.filter((item) => item.id !== document.id) : current,
      );
      setConfirmingDeleteId(null);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        clearToken();
        router.replace("/login");
        return;
      }
      setDeleteError(
        error instanceof ApiError
          ? error.message
          : "Something went wrong. Please try again.",
      );
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <main className="flex-1 overflow-y-auto p-6 md:p-10">
      <div className="mx-auto max-w-3xl">
        <div className="animate-appear">
          <h1 className="text-3xl font-semibold text-foreground">Documents</h1>
          <p className="mt-2 text-muted">
            Upload files to build your knowledge base. Documents are chunked and
            indexed automatically.
          </p>
        </div>

        {loadError && (
          <div
            role="alert"
            className="mt-6 flex flex-wrap items-center justify-between gap-4 rounded-[24px] border border-red-400/30 bg-red-500/10 p-5"
          >
            <p className="text-sm text-red-600 dark:text-red-400">
              {loadError}
            </p>
            <button
              type="button"
              onClick={() => void loadDocuments()}
              className="rounded-full border border-red-400/40 px-4 py-1.5 text-sm font-medium text-red-600 transition hover:bg-red-500/10 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent dark:text-red-400"
            >
              Try again
            </button>
          </div>
        )}

        <div
          role="button"
          tabIndex={0}
          aria-label="Upload documents. Drag and drop or browse files."
          onClick={openFilePicker}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              openFilePicker();
            }
          }}
          onDragOver={(event) => {
            event.preventDefault();
            if (!isUploading) setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragOver(false);
            void handleFiles(event.dataTransfer.files);
          }}
          className={`animate-appear mt-8 flex w-full cursor-pointer flex-col items-center justify-center gap-2 rounded-[24px] border-2 border-dashed px-8 py-14 text-center transition ${
            dragOver
              ? "scale-[1.01] border-accent bg-accent/10"
              : "border-card-border bg-card/50 hover:border-accent/50 hover:bg-active"
          } ${isUploading ? "opacity-60" : ""}`}
        >
          <span
            aria-hidden="true"
            className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/10 text-xl text-accent"
          >
            ⬆
          </span>
          <span className="text-lg font-semibold text-foreground">
            Drag &amp; drop files here
          </span>
          <span className="text-sm text-muted">{MAX_FILE_LABEL}</span>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              openFilePicker();
            }}
            disabled={isUploading}
            className="mt-3 inline-flex h-10 items-center justify-center gap-2 rounded-full bg-accent px-5 text-sm font-medium text-white shadow-sm transition hover:bg-accent-hover active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isUploading && <SmallSpinner />}
            {isUploading ? "Uploading…" : "Browse files"}
          </button>
          {dragOver && (
            <span className="text-sm font-medium text-accent">
              Drop to upload
            </span>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          multiple
          accept={FILE_INPUT_ACCEPT}
          className="hidden"
          onChange={(event) => {
            void handleFiles(event.target.files ?? []);
            event.target.value = "";
          }}
        />

        <div aria-live="polite" className="mt-4 space-y-2">
          {rejectMessages.map((message) => (
            <p
              key={message}
              className="rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-2 text-sm text-red-600 dark:text-red-400"
            >
              {message}
            </p>
          ))}
          {uploadError && (
            <p
              role="alert"
              className="rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-2 text-sm text-red-600 dark:text-red-400"
            >
              {uploadError}
            </p>
          )}
          {liveMessage && (
            <p role="status" className="px-1 text-sm text-muted">
              {liveMessage}
            </p>
          )}
        </div>

        {documents === null && !loadError && (
          <ul className="mt-6 space-y-3" aria-hidden="true">
            {[0, 1, 2].map((index) => (
              <li
                key={index}
                className="animate-pulse rounded-[24px] border border-card-border bg-card p-5 shadow-card backdrop-blur-2xl"
              >
                <div className="h-4 w-2/5 rounded-full bg-active" />
                <div className="mt-3 flex gap-3">
                  <div className="h-4 w-32 rounded-full bg-active" />
                  <div className="h-4 w-24 rounded-full bg-active" />
                </div>
              </li>
            ))}
          </ul>
        )}

        {documents !== null && documents.length === 0 && !loadError && (
          <div className="animate-appear mt-6 rounded-[24px] border border-card-border bg-card p-10 text-center shadow-card backdrop-blur-2xl">
            <p className="text-lg font-semibold text-foreground">
              Upload your first document
            </p>
            <p className="mt-2 text-sm text-muted">
              Add a file above and it will appear here.
            </p>
          </div>
        )}

        {documents !== null && documents.length > 0 && (
          <>
            <ul className="mt-6 space-y-3">
              {documents.map((document, index) => {
                const isConfirming = confirmingDeleteId === document.id;
                const isDeleting = deletingId === document.id;
                return (
                  <li
                    key={document.id}
                    className="animate-appear rounded-[24px] border border-card-border bg-card p-5 shadow-card backdrop-blur-2xl"
                    style={{
                      animationDelay: `${Math.min(index * 50, 250)}ms`,
                    }}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <p
                          className="truncate text-[15px] font-semibold text-foreground"
                          title={document.filename}
                        >
                          {document.filename}
                        </p>
                        <p className="mt-0.5 text-sm text-muted">
                          Uploaded {formatDate(document.created_at)}
                        </p>
                      </div>
                      <div className="flex shrink-0 items-center gap-2.5">
                        <StatusBadge document={document} />
                        {isConfirming ? (
                          <div className="flex items-center gap-2">
                            {isDeleting ? (
                              <span
                                role="status"
                                className="inline-flex h-7 w-7 items-center justify-center text-red-500"
                              >
                                <SmallSpinner />
                              </span>
                            ) : (
                              <>
                                <button
                                  type="button"
                                  onClick={() => void confirmDelete(document)}
                                  className="rounded-full bg-red-500 px-3.5 py-1.5 text-xs font-semibold text-white transition hover:bg-red-600 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                                >
                                  Confirm
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setConfirmingDeleteId(null)}
                                  className="rounded-full border border-card-border px-3.5 py-1.5 text-xs font-medium text-muted transition hover:bg-active hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                                >
                                  Cancel
                                </button>
                              </>
                            )}
                          </div>
                        ) : (
                          <button
                            type="button"
                            onClick={() => {
                              setConfirmingDeleteId(document.id);
                              setDeleteError(null);
                            }}
                            aria-label={`Delete ${document.filename}`}
                            className="rounded-full border border-card-border px-3.5 py-1.5 text-xs font-medium text-muted transition hover:bg-active hover:text-red-500 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                          >
                            Delete
                          </button>
                        )}
                      </div>
                    </div>

                    {document.status === "failed" && document.error && (
                      <p className="mt-2 text-sm text-red-600 dark:text-red-400">
                        {document.error}
                      </p>
                    )}

                    {isConfirming && !isDeleting && (
                      <p className="mt-3 text-sm text-muted">
                        Delete &quot;{document.filename}&quot;? This can&apos;t
                        be undone.
                      </p>
                    )}
                    {deleteError && isConfirming && (
                      <p
                        role="alert"
                        className="mt-2 text-sm text-red-600 dark:text-red-400"
                      >
                        {deleteError}
                      </p>
                    )}
                  </li>
                );
              })}
            </ul>
            <p className="mt-4 px-1 text-sm text-muted">
              {plural(documents.length, "document")}
            </p>
          </>
        )}
      </div>
    </main>
  );
}
