"use client";

import { useEffect, useState } from "react";
import { getPublicConfig } from "@/lib/api";

// Values used while /config is in flight or unreachable.
export const FALLBACK_EXTENSIONS = [".pdf", ".txt"];
export const FALLBACK_MAX_PDF_MB = 25;
export const FALLBACK_MAX_TEXT_MB = 2;

export type ChunkingStrategyConfig = {
  key: string;
  label: string;
  description: string;
};

export type ChunkingConfig = {
  strategies: ChunkingStrategyConfig[];
  defaultStrategy: string;
  chunkSizeDefault: number;
  chunkSizeMin: number;
  chunkSizeMax: number;
  chunkOverlapDefault: number;
  chunkOverlapMax: number;
};

const MIME_BY_EXTENSION: Record<string, string> = {
  ".pdf": "application/pdf",
  ".txt": "text/plain",
};

export type UploadConfig = {
  // Exactly what the server allows, e.g. [".pdf"] or [".pdf", ".txt"].
  allowedExtensions: string[];
  // Value for <input accept>, built from the allowed extensions.
  acceptValue: string;
  maxPdfBytes: number;
  maxTextBytes: number;
  // Human labels for copy and reject messages.
  pdfLimitLabel: string;
  textLimitLabel: string;
  allowedTypesLabel: string;
  chunking: ChunkingConfig | null;
};

function positiveMb(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : fallback;
}

function allowedTypesLabel(extensions: string[]): string {
  if (extensions.length <= 1) return extensions[0] ?? "no files";
  const all = [...extensions];
  const last = all.pop();
  return `${all.join(", ")} and ${last}`;
}

function buildAcceptValue(extensions: string[]): string {
  const parts: string[] = [];
  for (const extension of extensions) {
    parts.push(extension);
    const mime = MIME_BY_EXTENSION[extension];
    if (mime) parts.push(mime);
  }
  return parts.join(",");
}

function normalizeExtensions(allowed: unknown): string[] {
  if (!Array.isArray(allowed)) return [...FALLBACK_EXTENSIONS];
  const seen = new Set<string>();
  const extensions: string[] = [];
  for (const item of allowed) {
    if (typeof item !== "string") continue;
    const normalized = item.toLowerCase();
    if (!normalized.startsWith(".")) continue;
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    extensions.push(normalized);
  }
  return extensions.length > 0 ? extensions : [...FALLBACK_EXTENSIONS];
}

function validInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

function buildChunkingConfig(config: {
  chunking_strategies?: unknown;
  default_chunking_strategy?: unknown;
  chunk_size_default?: unknown;
  chunk_size_min?: unknown;
  chunk_size_max?: unknown;
  chunk_overlap_default?: unknown;
  chunk_overlap_max?: unknown;
}): ChunkingConfig | null {
  const strategies = config.chunking_strategies;
  const defaultStrategy = config.default_chunking_strategy;
  const chunkSizeDefault = config.chunk_size_default;
  const chunkSizeMin = config.chunk_size_min;
  const chunkSizeMax = config.chunk_size_max;
  const chunkOverlapDefault = config.chunk_overlap_default;
  const chunkOverlapMax = config.chunk_overlap_max;

  if (
    !Array.isArray(strategies) ||
    strategies.length === 0 ||
    !strategies.every(
      (strategy: unknown) =>
        typeof strategy === "object" &&
        strategy !== null &&
        "key" in strategy &&
        "label" in strategy &&
        "description" in strategy &&
        typeof strategy.key === "string" &&
        strategy.key.length > 0 &&
        typeof strategy.label === "string" &&
        typeof strategy.description === "string",
    ) ||
    typeof defaultStrategy !== "string" ||
    !strategies.some(
      (strategy) =>
        typeof strategy === "object" &&
        strategy !== null &&
        "key" in strategy &&
        strategy.key === defaultStrategy,
    ) ||
    !validInteger(chunkSizeDefault) ||
    !validInteger(chunkSizeMin) ||
    !validInteger(chunkSizeMax) ||
    !validInteger(chunkOverlapDefault) ||
    !validInteger(chunkOverlapMax) ||
    chunkSizeMin < 1 ||
    chunkSizeMin > chunkSizeDefault ||
    chunkSizeDefault > chunkSizeMax ||
    chunkOverlapMax < 0 ||
    chunkOverlapDefault < 0 ||
    chunkOverlapDefault > chunkOverlapMax ||
    chunkOverlapDefault * 2 >= chunkSizeDefault
  ) {
    return null;
  }

  return {
    strategies,
    defaultStrategy,
    chunkSizeDefault,
    chunkSizeMin,
    chunkSizeMax,
    chunkOverlapDefault,
    chunkOverlapMax,
  };
}

export function buildUploadConfig(server: unknown): UploadConfig {
  const config = (server ?? {}) as {
    allowed_extensions?: string[];
    max_upload_mb?: number;
    max_text_upload_mb?: number;
    chunking_strategies?: unknown;
    default_chunking_strategy?: unknown;
    chunk_size_default?: unknown;
    chunk_size_min?: unknown;
    chunk_size_max?: unknown;
    chunk_overlap_default?: unknown;
    chunk_overlap_max?: unknown;
  };
  const chunking = buildChunkingConfig(config);
  const extensions = normalizeExtensions(config.allowed_extensions);
  const maxPdfMb = positiveMb(config.max_upload_mb, FALLBACK_MAX_PDF_MB);
  // Older backends do not send a separate text limit; the fallback applies.
  const maxTextMb = positiveMb(config.max_text_upload_mb, FALLBACK_MAX_TEXT_MB);
  return {
    allowedExtensions: extensions,
    acceptValue: buildAcceptValue(extensions),
    maxPdfBytes: maxPdfMb * 1024 * 1024,
    maxTextBytes: maxTextMb * 1024 * 1024,
    pdfLimitLabel: `${maxPdfMb} MB`,
    textLimitLabel: `${maxTextMb} MB`,
    allowedTypesLabel: allowedTypesLabel(extensions),
    chunking,
  };
}

export const FALLBACK_UPLOAD_CONFIG: UploadConfig = buildUploadConfig(null);

export function useUploadConfig(): UploadConfig {
  const [config, setConfig] = useState<UploadConfig>(FALLBACK_UPLOAD_CONFIG);

  useEffect(() => {
    let cancelled = false;
    getPublicConfig()
      .then((server) => {
        if (!cancelled) setConfig(buildUploadConfig(server));
      })
      .catch(() => {
        // /config failed: keep the fallback values.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return config;
}

export function chunkingStrategyLabel(
  strategy: string | null | undefined,
  strategies?: ChunkingStrategyConfig[],
): string {
  if (!strategy) return "Recursive";
  const configured = strategies?.find((item) => item.key === strategy);
  if (configured) return configured.label;
  return strategy
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(" ");
}

export function hasStoredChunkingSettings(document: {
  chunking_strategy?: string | null;
  chunk_size?: number | null;
  chunk_overlap?: number | null;
}): document is {
  chunking_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
} {
  return (
    typeof document.chunking_strategy === "string" &&
    document.chunking_strategy.length > 0 &&
    typeof document.chunk_size === "number" &&
    Number.isFinite(document.chunk_size) &&
    typeof document.chunk_overlap === "number" &&
    Number.isFinite(document.chunk_overlap)
  );
}

export function documentChunkingLabel(
  document: {
    chunking_strategy?: string | null;
    chunk_size?: number | null;
    chunk_overlap?: number | null;
  } | null | undefined,
  strategies?: ChunkingStrategyConfig[],
): string {
  return document && hasStoredChunkingSettings(document)
    ? chunkingStrategyLabel(document.chunking_strategy, strategies)
    : "Recursive";
}

// Per-type byte limit: PDFs and text files each have their own cap.
export function maxBytesFor(
  config: UploadConfig,
  extension: string,
): number {
  return extension === ".txt" ? config.maxTextBytes : config.maxPdfBytes;
}

export function limitLabelFor(
  config: UploadConfig,
  extension: string,
): string {
  return extension === ".txt" ? config.textLimitLabel : config.pdfLimitLabel;
}

// Drop-zone copy, e.g. "PDF or TXT. PDFs up to 25 MB, text files up to 2 MB."
export function dropZoneLabel(config: UploadConfig): string {
  const names = config.allowedExtensions.map((extension) =>
    extension.slice(1).toUpperCase(),
  );
  let kinds: string;
  if (names.length === 0) kinds = "no files";
  else if (names.length === 1) kinds = names[0];
  else {
    const all = [...names];
    const last = all.pop();
    kinds = `${all.join(" or ")} or ${last}`;
  }

  const hasText = config.allowedExtensions.includes(".txt");
  const hasPdf = config.allowedExtensions.some(
    (extension) => extension !== ".txt",
  );
  let limits: string;
  if (hasPdf && hasText) {
    limits = `PDFs up to ${config.pdfLimitLabel}, text files up to ${config.textLimitLabel}`;
  } else if (hasText) {
    limits = `up to ${config.textLimitLabel}`;
  } else {
    limits = `up to ${config.pdfLimitLabel}`;
  }
  return `${kinds}. ${limits}.`;
}
