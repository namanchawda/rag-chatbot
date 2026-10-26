import { API_URL, ApiError, extractErrorMessage } from "@/lib/api";
import { trackWakeRequest } from "@/lib/wake";

export type ChatSource = {
  document_id: string;
  source: string; // filename
  chunk_index: number;
  snippet: string;
};

export type ChatEvent =
  | { type: "session"; sessionId: string; title: string }
  | { type: "sources"; sources: ChatSource[] }
  | { type: "token"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

type SseEvent = { event: string; data: string };

// Narrow no-break space and no-break space from model output become regular
// spaces so wrapping and copying behave like the rest of the app.
function normalizeTokenText(text: string): string {
  return text.replace(/[\u202f\u00a0]/g, " ");
}

// Minimal SSE parser. Events are separated by blank lines and may arrive
// split across arbitrary chunk boundaries, so a trailing partial event is
// returned as the remainder and prepended to the next chunk.
function parseSseBuffer(
  partial: string,
  chunk: string,
): { events: SseEvent[]; remainder: string } {
  const combined = partial + chunk;
  const events: SseEvent[] = [];
  const blocks = combined.split(/\r?\n\r?\n/);
  // The last element is either "" (clean finish) or an incomplete event.
  const remainder = blocks.pop() ?? "";

  for (const block of blocks) {
    let event = "message";
    const dataLines: string[] = [];
    for (const rawLine of block.split(/\r?\n/)) {
      if (rawLine.startsWith("event:")) {
        event = rawLine.slice(6).trim();
      } else if (rawLine.startsWith("data:")) {
        dataLines.push(rawLine.slice(5).trimStart());
      }
      // Comments and other SSE fields are ignored.
    }
    if (dataLines.length > 0) {
      events.push({ event, data: dataLines.join("\n") });
    }
  }
  return { events, remainder };
}

export async function streamChat(
  question: string,
  token: string,
  signal: AbortSignal,
  onEvent: (event: ChatEvent) => void,
  options: { sessionId?: string | null } = {},
): Promise<void> {
  // Tracked from send until the HTTP response arrives (or it is aborted),
  // exactly like every other request — one shared wake-up banner etc.
  const releaseWake = trackWakeRequest();
  let response: Response;
  try {
    response = await fetch(`${API_URL}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        Authorization: `Bearer ${token}`,
      },
      // Omit session_id for a fresh chat; the backend then creates a session
      // and reports it in the first "session" SSE event.
      body: JSON.stringify(
        options.sessionId
          ? { question, session_id: options.sessionId }
          : { question },
      ),
      signal,
    });
  } catch (error) {
    releaseWake();
    if ((error as Error).name === "AbortError") return;
    throw new ApiError(
      0,
      "Cannot reach the server. Check your connection and try again.",
    );
  }
  // Headers are in: the server answered, so the banner has no further job.
  releaseWake();

  if (response.status === 401) {
    // The page handles this by clearing the token and redirecting to login.
    throw new ApiError(401, "Your session has expired. Please sign in again.");
  }

  if (!response.ok || !response.body) {
    // Keep the server's own detail text (e.g. "Daily message limit reached
    // (50). Try again tomorrow."); the UI shows it verbatim as the backend
    // wrote it.
    let message = "Something went wrong. Please try again.";
    try {
      const payload = (await response.json()) as unknown;
      message = extractErrorMessage(payload, message);
    } catch {
      // Body was empty or not JSON: keep the fallback message.
    }
    throw new ApiError(
      response.status,
      message,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let partial = "";
  let sawError = false;
  let sawDone = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    const parsed = parseSseBuffer(partial, decoder.decode(value, { stream: true }));
    partial = parsed.remainder;

    for (const sse of parsed.events) {
      if (sse.event === "session") {
        // First event of a stream: the id of the conversation this answer
        // belongs to (new for fresh chats).
        try {
          const payload = JSON.parse(sse.data) as {
            session_id?: string;
            title?: string;
          };
          if (typeof payload.session_id === "string" && payload.session_id) {
            onEvent({
              type: "session",
              sessionId: payload.session_id,
              title: typeof payload.title === "string" ? payload.title : "",
            });
          }
        } catch {
          // Malformed session payload: keep going, the rest still works.
        }
      } else if (sse.event === "sources") {
        try {
          const payload = JSON.parse(sse.data) as { sources?: ChatSource[] };
          onEvent({ type: "sources", sources: payload.sources ?? [] });
        } catch {
          onEvent({ type: "sources", sources: [] });
        }
      } else if (sse.event === "token") {
        try {
          const payload = JSON.parse(sse.data) as { text?: string };
          if (typeof payload.text === "string" && payload.text.length > 0) {
            onEvent({
              type: "token",
              text: normalizeTokenText(payload.text),
            });
          }
        } catch {
          // Malformed token payload: skip it rather than corrupting the answer.
        }
      } else if (sse.event === "done") {
        sawDone = true;
      } else if (sse.event === "error") {
        sawError = true;
        let message = "Something went wrong. Please try again.";
        try {
          const payload = JSON.parse(sse.data) as { error?: string };
          if (typeof payload.error === "string" && payload.error.trim().length > 0) {
            message = payload.error;
          }
        } catch {
          // Keep the fallback message.
        }
        onEvent({ type: "error", message });
      }
      // Unknown event names are ignored for forward compatibility.
    }
  }

  if (sawError) return;
  if (!sawDone) {
    throw new ApiError(
      0,
      "The connection was interrupted before the answer completed.",
    );
  }
}
