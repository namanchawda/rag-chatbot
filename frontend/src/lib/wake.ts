// Shared wake-up tracking for the free-tier host that sleeps after idle.
//
// The API layer calls trackWakeRequest() around every request. If requests
// are still pending after SHOW_DELAY_MS, subscribers (the banner component)
// are told to show one shared banner — concurrent requests raise the same
// banner, never one per request. The banner clears as soon as responses
// arrive (or the request is aborted), i.e. when the pending count is empty.

const SHOW_DELAY_MS = 6000;

type Listener = (visible: boolean) => void;

const listeners = new Set<Listener>();
let pendingRequests = 0;
let visible = false;
let showTimer: ReturnType<typeof setTimeout> | null = null;

function notify(): void {
  for (const listener of listeners) {
    listener(visible);
  }
}

export function subscribeToWake(listener: Listener): () => void {
  listeners.add(listener);
  listener(visible);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * Report one in-flight request. The returned disposer must be called when
 * the response arrives, the call fails, or it is aborted.
 */
export function trackWakeRequest(): () => void {
  pendingRequests += 1;
  if (showTimer === null) {
    showTimer = setTimeout(() => {
      showTimer = null;
      if (pendingRequests > 0) {
        visible = true;
        notify();
      }
    }, SHOW_DELAY_MS);
  }

  let released = false;
  return () => {
    if (released) return;
    released = true;
    pendingRequests = Math.max(0, pendingRequests - 1);
    if (pendingRequests === 0) {
      if (showTimer !== null) {
        clearTimeout(showTimer);
        showTimer = null;
      }
      if (visible) {
        visible = false;
        notify();
      }
    }
  };
}
