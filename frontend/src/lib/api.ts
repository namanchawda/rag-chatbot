import { trackWakeRequest } from "@/lib/wake";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const TOKEN_STORAGE_KEY = "rag_chatbot_token";

export class ApiError extends Error {
  readonly status: number;
  readonly retryAfter: number | undefined;

  constructor(status: number, message: string, retryAfter?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

export type User = {
  id: string;
  email: string;
  created_at: string;
  is_admin: boolean;
};

export type AuthResponse = {
  access_token: string;
  token_type: string;
  user: User;
};

export function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function storeToken(token: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
  } catch {
    // Storage unavailable; auth will not persist.
  }
}

export function clearToken(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    // Storage unavailable.
  }
}

export function extractErrorMessage(
  payload: unknown,
  fallback: string,
): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string" && detail.trim().length > 0) return detail;
    if (Array.isArray(detail)) {
      const parts = detail
        .map((item) => {
          const message = (item as { msg?: unknown }).msg;
          return typeof message === "string" ? message : "";
        })
        .filter(Boolean);
      if (parts.length > 0) return parts.join(", ");
    }
  }
  return fallback;
}

function retryAfterSeconds(response: Response): number | undefined {
  const raw = response.headers.get("Retry-After");
  if (raw === null) return undefined;
  const seconds = Number.parseInt(raw, 10);
  return Number.isNaN(seconds) || seconds <= 0 ? undefined : seconds;
}

export type DocumentStatus = "processing" | "ready" | "failed";

export type DocumentItem = {
  id: string;
  filename: string;
  status: DocumentStatus;
  chunk_count: number;
  error: string | null;
  created_at: string;
};

// A sleeping host may take up to a minute to answer, so there is NO
// client-side fetch timeout: normal requests get effectively unlimited
// headroom (far beyond 90s) and rely on the browser's own connection
// timeouts instead of aborting mid-wake.
const GET_RETRYABLE_STATUSES = new Set([502, 503, 504]);
const GET_RETRY_DELAYS_MS = [1000, 2000, 4000];

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function apiRequest<T>(
  path: string,
  options: {
    method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
    body?: unknown;
    token?: string | null;
    signal?: AbortSignal;
  } = {},
): Promise<T> {
  const { method = "GET", body, token, signal } = options;
  const isFormData = body instanceof FormData;
  // Never auto-retry POST/PATCH/DELETE (or anything but GET): they are not
  // guaranteed idempotent, and a wake-up delay is not worth double effects.
  const isGet = method === "GET";
  const releaseWake = trackWakeRequest();

  try {
    let attempt = 0;
    while (true) {
      let response: Response;
      try {
        response = await fetch(`${API_URL}${path}`, {
          method,
          headers: {
            ...(body !== undefined && !isFormData
              ? { "Content-Type": "application/json" }
              : {}),
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          ...(body !== undefined
            ? { body: isFormData ? body : JSON.stringify(body) }
            : {}),
          ...(signal ? { signal } : {}),
        });
      } catch (error) {
        // A caller abort is not a network error: bubble it up untouched so
        // no silent GET retry happens after the caller stopped caring.
        if (
          signal?.aborted ||
          (error instanceof Error && error.name === "AbortError")
        ) {
          throw error;
        }
        // Network error: idempotent GETs go again after a short backoff,
        // up to three more tries.
        if (isGet && attempt < GET_RETRY_DELAYS_MS.length) {
          await wait(GET_RETRY_DELAYS_MS[attempt]);
          attempt += 1;
          continue;
        }
        throw new ApiError(
          0,
          "Cannot reach the server. Check your connection and try again.",
        );
      }

      // Cold host or gateway hiccup on an idempotent GET: retry silently.
      if (
        isGet &&
        GET_RETRYABLE_STATUSES.has(response.status) &&
        attempt < GET_RETRY_DELAYS_MS.length
      ) {
        await wait(GET_RETRY_DELAYS_MS[attempt]);
        attempt += 1;
        continue;
      }

      let payload: unknown = null;
      const text = await response.text();
      if (text.length > 0) {
        try {
          payload = JSON.parse(text);
        } catch {
          payload = null;
        }
      }

      if (!response.ok) {
        const fallback =
          response.status === 401
            ? "Invalid email or password"
            : "Something went wrong. Please try again.";
        throw new ApiError(
          response.status,
          extractErrorMessage(payload, fallback),
          retryAfterSeconds(response),
        );
      }

      return payload as T;
    }
  } finally {
    releaseWake();
  }
}

export function login(email: string, password: string): Promise<AuthResponse> {
  return apiRequest<AuthResponse>("/auth/login", {
    method: "POST",
    body: { email, password },
  });
}

export function signup(email: string, password: string): Promise<AuthResponse> {
  return apiRequest<AuthResponse>("/auth/signup", {
    method: "POST",
    body: { email, password },
  });
}

export function getMe(token: string): Promise<User> {
  return apiRequest<User>("/auth/me", { token });
}

export function listDocuments(token: string): Promise<DocumentItem[]> {
  return apiRequest<DocumentItem[]>("/documents", { token });
}

export function uploadDocument(
  token: string,
  file: File,
): Promise<DocumentItem> {
  const formData = new FormData();
  formData.append("file", file);
  return apiRequest<DocumentItem>("/documents", {
    method: "POST",
    body: formData,
    token,
  });
}

export function deleteDocument(
  token: string,
  documentId: string,
): Promise<null> {
  return apiRequest<null>(`/documents/${documentId}`, {
    method: "DELETE",
    token,
  });
}

export type StoredChatSource = {
  document_id: string;
  source: string;
  chunk_index: number;
  snippet: string;
};
export type SessionItem = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type StoredMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: StoredChatSource[] | null;
  created_at: string;
};

export function listSessions(
  token: string,
  limit = 100,
): Promise<SessionItem[]> {
  return apiRequest<SessionItem[]>(`/sessions?limit=${limit}`, { token });
}

export function createSession(token: string): Promise<SessionItem> {
  return apiRequest<SessionItem>("/sessions", {
    method: "POST",
    body: {},
    token,
  });
}

export function listSessionMessages(
  token: string,
  sessionId: string,
): Promise<StoredMessage[]> {
  return apiRequest<StoredMessage[]>(`/sessions/${sessionId}/messages`, {
    token,
  });
}

export function renameSession(
  token: string,
  sessionId: string,
  title: string,
): Promise<SessionItem> {
  return apiRequest<SessionItem>(`/sessions/${sessionId}`, {
    method: "PATCH",
    body: { title },
    token,
  });
}

export function deleteSession(
  token: string,
  sessionId: string,
): Promise<null> {
  return apiRequest<null>(`/sessions/${sessionId}`, {
    method: "DELETE",
    token,
  });
}

export type PublicConfig = {
  signups_enabled: boolean;
  // All of the following may be absent on older backends; callers fall back.
  max_upload_mb?: number;
  max_text_upload_mb?: number;
  allowed_extensions?: string[];
  rerank_available?: boolean;
  rerank_default?: boolean;
};

// No auth: this endpoint exists precisely so the login page can ask it.
export function getPublicConfig(): Promise<PublicConfig> {
  return apiRequest<PublicConfig>("/config");
}

export type ServiceState =
  | "ok"
  | "invalid_key"
  | "exhausted"
  | "rate_limited"
  | "unavailable"
  | "error"
  | "unknown";

export type ServiceStatus = {
  state: ServiceState;
  http_status: number | null;
  message: string | null;
  last_success_at: string | null;
  last_error_at: string | null;
};

export type AdminStatus = {
  jina: ServiceStatus & {
    tokens_used_since_reset: number;
    tokens_used_today: number;
    requests_today: number;
    errors_today: number;
    tokens_budget: number;
    tokens_estimated_remaining: number;
    percent_remaining: number;
    low: boolean;
  };
  groq: ServiceStatus & {
    answers_last_24h: number;
  };
  checked_at: string;
};

export function getAdminStatus(
  token: string,
  signal?: AbortSignal,
): Promise<AdminStatus> {
  return apiRequest<AdminStatus>("/admin/status", { token, signal });
}

export function checkAdminStatus(token: string): Promise<AdminStatus> {
  return apiRequest<AdminStatus>("/admin/status/check", {
    method: "POST",
    token,
  });
}

export function resetAdminUsage(token: string): Promise<AdminStatus> {
  return apiRequest<AdminStatus>("/admin/status/reset-usage", {
    method: "POST",
    token,
  });
}

// --- Chunks browser (read-only; matches backend /chunks) -------------------

export type ChunkItem = {
  id: string;
  document_id: string;
  source: string;
  chunk_index: number;
  text: string;
  char_count: number;
  token_estimate: number;
};

export type ChunkListResponse = {
  items: ChunkItem[];
  total: number;
  limit: number;
  offset: number;
};

export type ChunkStats = {
  documents: number;
  chunks: number;
  characters: number;
};

export function chunkStats(token: string): Promise<ChunkStats> {
  return apiRequest<ChunkStats>("/chunks/stats", { token });
}

export function listChunks(
  token: string,
  options: {
    documentId?: string | null;
    q?: string | null;
    limit?: number;
    offset?: number;
    signal?: AbortSignal;
  } = {},
): Promise<ChunkListResponse> {
  const params = new URLSearchParams();
  if (options.documentId) params.set("document_id", options.documentId);
  if (options.q && options.q.length > 0) params.set("q", options.q);
  params.set("limit", String(options.limit ?? 20));
  params.set("offset", String(options.offset ?? 0));
  return apiRequest<ChunkListResponse>(`/chunks?${params.toString()}`, {
    token,
    signal: options.signal,
  });
}
