export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const TOKEN_STORAGE_KEY = "rag_chatbot_token";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export type User = {
  id: string;
  email: string;
  created_at: string;
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

function extractErrorMessage(payload: unknown, fallback: string): string {
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

export type DocumentStatus = "processing" | "ready" | "failed";

export type DocumentItem = {
  id: string;
  filename: string;
  status: DocumentStatus;
  chunk_count: number;
  error: string | null;
  created_at: string;
};

export async function apiRequest<T>(
  path: string,
  options: {
    method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
    body?: unknown;
    token?: string | null;
  } = {},
): Promise<T> {
  const { method = "GET", body, token } = options;
  const isFormData = body instanceof FormData;

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
    });
  } catch {
    throw new ApiError(0, "Cannot reach the server. Check your connection and try again.");
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
    throw new ApiError(response.status, extractErrorMessage(payload, fallback));
  }

  return payload as T;
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
