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
import {
  dropZoneLabel,
  limitLabelFor,
  maxBytesFor,
  useUploadConfig,
} from "@/lib/upload-config";

const POLL_INTERVAL_MS = 2500;

function normalizeExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : `.${name.slice(dot + 1).toLowerCase()}`;
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

// The backend marks uploads swept off during downtime with this text; a
// friendly note explains the only cure (delete and re-upload), which
// Delete already supports for failed documents.
function isInterruptedUpload(error: string | null | undefined): boolean {
  return error !== null && error !== undefined && error.includes("interrupted");
}

function StatusBadge({ document }: { document: DocumentItem }) {
  if (document.status === "processing") {
    return (
      <span
        className="inline-flex min-h-8 items-center gap-1.5 rounded-full bg-active px-2.5 py-1 text-base font-medium text-muted"
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
        className="inline-flex min-h-8 items-center rounded-full bg-emerald-500/15 px-2.5 py-1 text-base font-medium text-emerald-600 dark:text-emerald-400"
        title="Ready for chat"
      >
        Ready · {plural(document.chunk_count, "chunk")}
      </span>
    );
  }

  return (
    <span
      className="inline-flex min-h-8 items-center rounded-full bg-red-500/15 px-2.5 py-1 text-base font-medium text-red-600 dark:text-red-400"
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
  const uploadConfig = useUploadConfig();

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
        const extension = normalizeExtension(file.name);
        if (!uploadConfig.allowedExtensions.includes(extension)) {
          rejected.push(
            `"${file.name}" can't be uploaded. Supported types are ${uploadConfig.allowedTypesLabel}.`,
          );
          return false;
        }
        const maxBytes = maxBytesFor(uploadConfig, extension);
        if (file.size > maxBytes) {
          rejected.push(
            `"${file.name}" is ${formatBytes(file.size)} — the upload limit for this file type is ${limitLabelFor(uploadConfig, extension)}.`,
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
        if (error instanceof ApiError && error.status === 409) {
          // Document quota reached: the server's own text, shown inline —
          // no retry loop, and later files would conflict the same way, so
          // the batch stops here.
          setUploadError(error.message);
          setLiveMessage(null);
        } else {
          const detail =
            error instanceof ApiError
              ? error.message
              : "Something went wrong. Please try again.";
          setUploadError(`Couldn't upload "${currentFile}". ${detail}`);
        }
      } finally {
        setIsUploading(false);
        await loadDocuments();
      }
    },
    [isUploading, loadDocuments, router, uploadConfig],
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
    <main className="flex-1 px-5 py-8 sm:px-8 sm:py-12 lg:px-12 lg:py-14">
      <div className="mx-auto w-full max-w-6xl">
        <div className="animate-appear">
          <p className="text-sm font-semibold uppercase tracking-[0.14em] text-accent">
            Your source library
          </p>
          <h1 className="mt-3 text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
            Documents
          </h1>
          <p className="mt-2 max-w-2xl text-base leading-7 text-muted">
            Upload files to build your knowledge base. Documents are chunked and
            indexed automatically.
          </p>
        </div>

        {loadError && (
          <div
            role="alert"
            className="mt-6 flex flex-wrap items-center justify-between gap-4 rounded-[24px] border border-red-400/30 bg-red-500/10 p-5"
          >
            <p className="text-base text-red-600 dark:text-red-400">
              {loadError}
            </p>
            <button
              type="button"
              onClick={() => void loadDocuments()}
              className="min-h-11 rounded-full border border-red-400/40 px-4 text-base font-medium text-red-600 transition hover:bg-red-500/10 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent dark:text-red-400"
            >
              Try again
            </button>
          </div>
        )}

        <div className="animate-appear mt-7 rounded-[28px] border border-card-border bg-card p-4 shadow-card sm:p-6">
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
            className={`flex min-h-64 w-full cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-5 py-9 text-center transition sm:min-h-72 sm:px-8 sm:py-12 ${
              dragOver
                ? "scale-[1.01] border-accent bg-accent/10"
                : "border-card-border bg-background/60 hover:border-accent/50 hover:bg-active"
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
            <span className="text-base leading-6 text-muted">
              {dropZoneLabel(uploadConfig)}
            </span>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                openFilePicker();
              }}
              disabled={isUploading}
              className="mt-3 inline-flex min-h-11 items-center justify-center gap-2 rounded-full bg-accent px-5 text-base font-medium text-white shadow-sm transition hover:bg-accent-hover active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isUploading && <SmallSpinner />}
              {isUploading ? "Uploading…" : "Browse files"}
            </button>
            {dragOver && (
              <span className="text-base font-medium text-accent">
                Drop to upload
              </span>
            )}
          </div>
        </div>

        <input
          ref={inputRef}
          type="file"
          multiple
          accept={uploadConfig.acceptValue}
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
              className="rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-2 text-base text-red-600 dark:text-red-400"
            >
              {message}
            </p>
          ))}
          {uploadError && (
            <p
              role="alert"
              className="rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-2 text-base text-red-600 dark:text-red-400"
            >
              {uploadError}
            </p>
          )}
          {liveMessage && (
            <p role="status" className="px-1 text-base text-muted">
              {liveMessage}
            </p>
          )}
        </div>

        {documents === null && !loadError && (
          <ul className="mt-6 space-y-3" aria-hidden="true">
            {[0, 1, 2].map((index) => (
              <li
                key={index}
                className="animate-pulse rounded-2xl border border-card-border bg-card p-5 shadow-card"
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
          <div className="animate-appear mt-6 rounded-2xl border border-card-border bg-card p-10 text-center shadow-card">
            <p className="text-lg font-semibold text-foreground">
              Upload your first document
            </p>
            <p className="mt-2 text-base text-muted">
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
                    className="animate-appear rounded-2xl border border-card-border bg-card p-4 shadow-card sm:p-5"
                    style={{
                      animationDelay: `${Math.min(index * 50, 250)}ms`,
                    }}
                  >
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
                      <div className="min-w-0">
                        <p
                          className="truncate text-base font-semibold text-foreground"
                          title={document.filename}
                        >
                          {document.filename}
                        </p>
                        <p className="mt-1 text-base text-muted">
                          Uploaded {formatDate(document.created_at)}
                        </p>
                      </div>
                      <div className="flex w-full shrink-0 flex-wrap items-center justify-between gap-2.5 sm:w-auto sm:justify-end">
                        <StatusBadge document={document} />
                        {isConfirming ? (
                          <div className="flex items-center gap-2">
                            {isDeleting ? (
                              <span
                                role="status"
                                className="inline-flex h-11 w-11 items-center justify-center text-red-500"
                              >
                                <SmallSpinner />
                              </span>
                            ) : (
                              <>
                                <button
                                  type="button"
                                  onClick={() => void confirmDelete(document)}
                                  className="min-h-11 rounded-full bg-red-500 px-4 text-sm font-semibold text-white transition hover:bg-red-600 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                                >
                                  Confirm
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setConfirmingDeleteId(null)}
                                  className="min-h-11 rounded-full border border-card-border px-4 text-sm font-medium text-muted transition hover:bg-active hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
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
                            className="min-h-11 rounded-full border border-card-border px-4 text-sm font-medium text-muted transition hover:bg-active hover:text-red-500 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                          >
                            Delete
                          </button>
                        )}
                      </div>
                    </div>

                    {document.status === "failed" &&
                      isInterruptedUpload(document.error) && (
                        <div
                          role="status"
                          className="mt-3 rounded-xl border border-amber-400/30 bg-amber-500/10 px-3 py-2.5"
                        >
                          <p className="text-base font-medium text-amber-600 dark:text-amber-400">
                            This upload was interrupted.
                          </p>
                          <p className="mt-0.5 text-base text-amber-600 dark:text-amber-400">
                            Delete it and upload again.
                          </p>
                        </div>
                      )}
                    {document.status === "failed" &&
                      !isInterruptedUpload(document.error) &&
                      document.error && (
                        <p className="mt-2 text-base text-red-600 dark:text-red-400">
                          {document.error}
                        </p>
                      )}

                    {isConfirming && !isDeleting && (
                      <p className="mt-3 text-base text-muted">
                        Delete &quot;{document.filename}&quot;? This can&apos;t
                        be undone.
                      </p>
                    )}
                    {deleteError && isConfirming && (
                      <p
                        role="alert"
                        className="mt-2 text-base text-red-600 dark:text-red-400"
                      >
                        {deleteError}
                      </p>
                    )}
                  </li>
                );
              })}
            </ul>
            <p className="mt-4 px-1 text-base text-muted">
              {plural(documents.length, "document")}
            </p>
          </>
        )}
      </div>
    </main>
  );
}
