"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  documentChunkingLabel,
  dropZoneLabel,
  hasStoredChunkingSettings,
  limitLabelFor,
  maxBytesFor,
  useUploadConfig,
  type ChunkingConfig,
  type ChunkingStrategyConfig,
} from "@/lib/upload-config";

const POLL_INTERVAL_MS = 2500;
const CHUNKING_STORAGE_KEY = "rag_chatbot_chunking";

type ChunkingSelection = {
  strategy: string;
  chunkSize: string;
  chunkOverlap: string;
};

type ResolvedChunkingSelection = {
  chunking_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
};

function defaultChunkingSelection(
  config: ChunkingConfig,
): ChunkingSelection {
  return {
    strategy: config.defaultStrategy,
    chunkSize: String(config.chunkSizeDefault),
    chunkOverlap: String(config.chunkOverlapDefault),
  };
}

function resolveChunkingSelection(
  config: ChunkingConfig,
  selection: ChunkingSelection,
): ResolvedChunkingSelection | null {
  const chunkSize = Number(selection.chunkSize);
  const chunkOverlap = Number(selection.chunkOverlap);
  if (
    !config.strategies.some((strategy) => strategy.key === selection.strategy) ||
    selection.chunkSize.trim().length === 0 ||
    !Number.isInteger(chunkSize) ||
    chunkSize < config.chunkSizeMin ||
    chunkSize > config.chunkSizeMax ||
    selection.chunkOverlap.trim().length === 0 ||
    !Number.isInteger(chunkOverlap) ||
    chunkOverlap < 0 ||
    chunkOverlap > config.chunkOverlapMax ||
    chunkOverlap * 2 >= chunkSize
  ) {
    return null;
  }
  return {
    chunking_strategy: selection.strategy,
    chunk_size: chunkSize,
    chunk_overlap: chunkOverlap,
  };
}

function readStoredChunkingSelection(): ChunkingSelection | null {
  try {
    const raw = window.localStorage.getItem(CHUNKING_STORAGE_KEY);
    if (!raw) return null;
    const stored: unknown = JSON.parse(raw);
    if (
      typeof stored !== "object" ||
      stored === null ||
      !("strategy" in stored) ||
      !("chunkSize" in stored) ||
      !("chunkOverlap" in stored) ||
      typeof stored.strategy !== "string" ||
      typeof stored.chunkSize !== "string" ||
      typeof stored.chunkOverlap !== "string"
    ) {
      return null;
    }
    const selection = {
      strategy: stored.strategy,
      chunkSize: stored.chunkSize,
      chunkOverlap: stored.chunkOverlap,
    };
    return selection;
  } catch {
    return null;
  }
}

function ChunkingStrategyCards({
  strategies,
  selected,
  onChange,
}: {
  strategies: ChunkingStrategyConfig[];
  selected: string;
  onChange: (value: string) => void;
}) {
  return (
    <fieldset>
      <legend className="text-base font-semibold text-foreground">
        Chunking strategy
      </legend>
      <p id="chunking-strategy-help" className="mt-1 text-sm leading-6 text-muted">
        Controls how your document is split into passages for search. You
        can&apos;t change it later without re-uploading.
      </p>
      <div
        role="radiogroup"
        aria-describedby="chunking-strategy-help"
        className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2"
      >
        {strategies.map((strategy) => {
          const isSelected = selected === strategy.key;
          return (
            <label key={strategy.key} className="cursor-pointer">
              <input
                type="radio"
                name="chunking-strategy"
                value={strategy.key}
                checked={isSelected}
                onChange={() => onChange(strategy.key)}
                className="peer sr-only"
              />
              <span
                className={`flex min-h-24 flex-col rounded-2xl border p-4 transition focus-within:ring-2 focus-within:ring-accent focus-within:ring-offset-2 ${
                  isSelected
                    ? "border-accent bg-accent/[0.08] shadow-sm"
                    : "border-card-border bg-background hover:border-accent/40 hover:bg-active"
                }`}
              >
                <span className="flex items-center justify-between gap-3">
                  <span className="text-base font-semibold text-foreground">
                    {strategy.label}
                  </span>
                  <span
                    aria-hidden="true"
                    className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border ${
                      isSelected
                        ? "border-accent bg-accent text-xs text-white"
                        : "border-input-border"
                    }`}
                  >
                    {isSelected ? "✓" : null}
                  </span>
                </span>
                <span className="mt-1 text-sm leading-5 text-muted">
                  {strategy.description}
                </span>
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

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

function ChunkingBadge({
  document,
  strategies,
}: {
  document: DocumentItem;
  strategies?: ChunkingStrategyConfig[];
}) {
  const hasSettings = hasStoredChunkingSettings(document);
  const label = documentChunkingLabel(document, strategies);
  return (
    <span
      className="inline-flex min-h-7 items-center rounded-full border border-card-border bg-active px-2.5 py-1 text-xs font-medium text-muted"
      title={
        hasSettings
          ? `${label} · size ${document.chunk_size} · overlap ${document.chunk_overlap}`
          : undefined
      }
    >
      {label}
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
  const [chunkingSelection, setChunkingSelection] =
    useState<ChunkingSelection | null>(readStoredChunkingSelection);
  const [chunkingEdited, setChunkingEdited] = useState(false);

  const chunkingConfig = uploadConfig.chunking;
  const storedSelectionIsValid =
    chunkingConfig !== null &&
    chunkingSelection !== null &&
    resolveChunkingSelection(chunkingConfig, chunkingSelection) !== null;
  const defaultSelection = useMemo(
    () => (chunkingConfig ? defaultChunkingSelection(chunkingConfig) : null),
    [chunkingConfig],
  );
  const effectiveChunkingSelection = chunkingConfig
    ? chunkingSelection &&
      (chunkingEdited || storedSelectionIsValid)
      ? chunkingSelection
      : defaultSelection
    : null;
  const resolvedChunkingSelection =
    chunkingConfig && effectiveChunkingSelection
      ? resolveChunkingSelection(chunkingConfig, effectiveChunkingSelection)
      : null;
  const chunkingValid =
    chunkingConfig === null || resolvedChunkingSelection !== null;

  useEffect(() => {
    if (
      !chunkingConfig ||
      !effectiveChunkingSelection ||
      !resolveChunkingSelection(chunkingConfig, effectiveChunkingSelection)
    ) {
      return;
    }
    try {
      window.localStorage.setItem(
        CHUNKING_STORAGE_KEY,
        JSON.stringify(effectiveChunkingSelection),
      );
    } catch {
      // Remembering the choice is best-effort only.
    }
  }, [
    chunkingConfig,
    effectiveChunkingSelection,
  ]);

  function updateChunkingSelection(update: Partial<ChunkingSelection>) {
    if (!chunkingConfig || !effectiveChunkingSelection) return;
    setChunkingEdited(true);
    setChunkingSelection((current) => ({
      ...(current && (chunkingEdited || storedSelectionIsValid)
        ? current
        : effectiveChunkingSelection),
      ...update,
    }));
  }

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
    if (isUploading || !chunkingValid) return;
    inputRef.current?.click();
  }

  const handleFiles = useCallback(
    async (fileList: FileList | File[]) => {
      const selectedChunking =
        chunkingConfig && effectiveChunkingSelection
          ? resolveChunkingSelection(
              chunkingConfig,
              effectiveChunkingSelection,
            )
          : null;
      const files = Array.from(fileList);
      if (
        files.length === 0 ||
        isUploading ||
        (chunkingConfig !== null && selectedChunking === null)
      ) {
        return;
      }

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
          await uploadDocument(token, file, selectedChunking ?? undefined);
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
        } else if (error instanceof ApiError && error.status === 422) {
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
    [
      chunkingConfig,
      effectiveChunkingSelection,
      isUploading,
      loadDocuments,
      router,
      uploadConfig,
    ],
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

  const chunkSizeValue = Number(effectiveChunkingSelection?.chunkSize);
  const chunkOverlapValue = Number(effectiveChunkingSelection?.chunkOverlap);
  const chunkSizeIsValid =
    chunkingConfig !== null &&
    (effectiveChunkingSelection?.chunkSize.trim().length ?? 0) > 0 &&
    Number.isInteger(chunkSizeValue) &&
    chunkSizeValue >= chunkingConfig.chunkSizeMin &&
    chunkSizeValue <= chunkingConfig.chunkSizeMax;
  const chunkingValidationMessage =
    chunkingConfig && effectiveChunkingSelection
      ? !chunkSizeIsValid
        ? `Chunk size must be an integer between ${chunkingConfig.chunkSizeMin} and ${chunkingConfig.chunkSizeMax}.`
        : effectiveChunkingSelection.chunkOverlap.trim().length === 0 ||
            !Number.isInteger(chunkOverlapValue) ||
            chunkOverlapValue < 0 ||
            chunkOverlapValue > chunkingConfig.chunkOverlapMax ||
            chunkOverlapValue * 2 >= chunkSizeValue
          ? `Overlap must be at least 0, no more than ${chunkingConfig.chunkOverlapMax}, and less than half of the chunk size.`
          : null
      : null;

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
          {chunkingConfig && effectiveChunkingSelection && (
            <section
              aria-label="Upload chunking settings"
              className="mb-5 rounded-2xl border border-card-border bg-background/70 p-4 sm:p-5"
            >
              <ChunkingStrategyCards
                strategies={chunkingConfig.strategies}
                selected={effectiveChunkingSelection.strategy}
                onChange={(strategy) => updateChunkingSelection({ strategy })}
              />
              <details className="mt-4 rounded-xl border border-card-border bg-card px-4">
                <summary className="flex min-h-11 cursor-pointer items-center text-base font-medium text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent">
                  Advanced
                </summary>
                <div className="grid gap-4 pb-4 sm:grid-cols-2">
                  <label
                    htmlFor="chunk-size"
                    className="text-sm font-medium text-foreground"
                  >
                    Chunk size (characters)
                    <input
                      id="chunk-size"
                      type="number"
                      inputMode="numeric"
                      min={chunkingConfig.chunkSizeMin}
                      max={chunkingConfig.chunkSizeMax}
                      step={1}
                      value={effectiveChunkingSelection.chunkSize}
                      aria-invalid={!chunkSizeIsValid}
                      aria-describedby={
                        !chunkSizeIsValid ? "chunking-validation" : undefined
                      }
                      onChange={(event) =>
                        updateChunkingSelection({
                          chunkSize: event.target.value,
                        })
                      }
                      className="mt-1.5 h-11 w-full rounded-xl border border-input-border bg-input px-3 text-base text-foreground outline-none focus:border-accent focus:ring-4 focus:ring-ring"
                    />
                    <span className="mt-1 block text-xs font-normal text-muted">
                      {chunkingConfig.chunkSizeMin}–{chunkingConfig.chunkSizeMax}
                    </span>
                  </label>
                  <label
                    htmlFor="chunk-overlap"
                    className="text-sm font-medium text-foreground"
                  >
                    Overlap (characters)
                    <input
                      id="chunk-overlap"
                      type="number"
                      inputMode="numeric"
                      min={0}
                      max={chunkingConfig.chunkOverlapMax}
                      step={1}
                      value={effectiveChunkingSelection.chunkOverlap}
                      aria-invalid={Boolean(chunkingValidationMessage)}
                      aria-describedby={
                        chunkingValidationMessage
                          ? "chunking-validation"
                          : undefined
                      }
                      onChange={(event) =>
                        updateChunkingSelection({
                          chunkOverlap: event.target.value,
                        })
                      }
                      className="mt-1.5 h-11 w-full rounded-xl border border-input-border bg-input px-3 text-base text-foreground outline-none focus:border-accent focus:ring-4 focus:ring-ring"
                    />
                    <span className="mt-1 block text-xs font-normal text-muted">
                      0–{chunkingConfig.chunkOverlapMax}
                    </span>
                  </label>
                </div>
              </details>
              <p
                aria-live="polite"
                className="mt-4 text-sm leading-6 text-muted"
              >
                Next uploads will use:{" "}
                <span className="font-semibold text-foreground">
                  {chunkingConfig.strategies.find(
                    (strategy) =>
                      strategy.key === effectiveChunkingSelection.strategy,
                  )?.label ?? "Recursive"}
                  {" · "}
                  size {effectiveChunkingSelection.chunkSize}
                  {" · "}
                  overlap {effectiveChunkingSelection.chunkOverlap}
                </span>
              </p>
              {chunkingValidationMessage && (
                <p
                  id="chunking-validation"
                  role="alert"
                  className="mt-2 text-sm text-red-600 dark:text-red-400"
                >
                  {chunkingValidationMessage}
                </p>
              )}
            </section>
          )}
          <div
            role="button"
            tabIndex={isUploading || !chunkingValid ? -1 : 0}
            aria-disabled={isUploading || !chunkingValid}
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
              if (!isUploading && chunkingValid) setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragOver(false);
              void handleFiles(event.dataTransfer.files);
            }}
            className={`flex min-h-64 w-full flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-5 py-9 text-center transition sm:min-h-72 sm:px-8 sm:py-12 ${
              dragOver
                ? "scale-[1.01] border-accent bg-accent/10"
                : "cursor-pointer border-card-border bg-background/60 hover:border-accent/50 hover:bg-active"
            } ${isUploading || !chunkingValid ? "cursor-not-allowed opacity-60" : ""}`}
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
              disabled={isUploading || !chunkingValid}
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
                        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
                          <p className="text-base text-muted">
                            Uploaded {formatDate(document.created_at)}
                          </p>
                          <ChunkingBadge
                            document={document}
                            strategies={chunkingConfig?.strategies}
                          />
                        </div>
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
