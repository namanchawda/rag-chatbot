"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ApiError,
  checkAdminStatus,
  clearToken,
  getAdminStatus,
  getStoredToken,
  resetAdminUsage,
  type AdminStatus,
  type ServiceState,
} from "@/lib/api";
import { GlassCard } from "@/components/ui";

const TOKEN_FORMAT = new Intl.NumberFormat("en", {
  maximumFractionDigits: 0,
});
const COMPACT_TOKEN_FORMAT = new Intl.NumberFormat("en", {
  notation: "compact",
  maximumFractionDigits: 1,
});
const RELATIVE_TIME = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

function formatTokens(value: number): string {
  return TOKEN_FORMAT.format(value);
}

function serviceLabel(state: ServiceState): string {
  switch (state) {
    case "ok":
      return "OK";
    case "invalid_key":
      return "Invalid key";
    case "exhausted":
      return "Exhausted";
    case "rate_limited":
      return "Rate limited";
    case "unavailable":
      return "Unavailable";
    case "error":
      return "Error";
    default:
      return "Unknown";
  }
}

function relativeTime(value: string | null): string {
  if (!value) return "Never";
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return "Unknown";

  const seconds = Math.round((timestamp - Date.now()) / 1000);
  const absoluteSeconds = Math.abs(seconds);
  if (absoluteSeconds < 60) return RELATIVE_TIME.format(seconds, "second");
  const minutes = Math.round(seconds / 60);
  if (Math.abs(minutes) < 60) return RELATIVE_TIME.format(minutes, "minute");
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return RELATIVE_TIME.format(hours, "hour");
  return RELATIVE_TIME.format(Math.round(hours / 24), "day");
}

function statusDotClass(status: AdminStatus | null): string {
  if (!status) return "bg-muted";
  const states = [status.jina.state, status.groq.state];
  if (states.some((state) => state === "exhausted" || state === "invalid_key")) {
    return "bg-red-500";
  }
  if (
    status.jina.low ||
    states.some(
      (state) =>
        state === "rate_limited" ||
        state === "unavailable" ||
        state === "error",
    )
  ) {
    return "bg-amber-500";
  }
  if (states.every((state) => state === "ok")) return "bg-emerald-500";
  return "bg-muted";
}

function statusSummary(status: AdminStatus | null): string {
  if (!status) return "Service status unknown";
  const jinaLabel =
    status.jina.state === "ok" && status.jina.low
      ? "Jina low balance"
      : `Jina ${serviceLabel(status.jina.state)}`;
  const groqLabel = `Groq ${serviceLabel(status.groq.state)}`;
  const remaining = COMPACT_TOKEN_FORMAT.format(
    status.jina.tokens_estimated_remaining,
  );
  return `${jinaLabel} · about ${remaining} tokens left · ${groqLabel}`;
}

function badgeClass(state: ServiceState, low = false): string {
  if (state === "exhausted" || state === "invalid_key") {
    return "border-red-500/25 bg-red-500/10 text-red-600 dark:text-red-400";
  }
  if (
    low ||
    state === "rate_limited" ||
    state === "unavailable" ||
    state === "error"
  ) {
    return "border-amber-500/25 bg-amber-500/10 text-amber-700 dark:text-amber-400";
  }
  if (state === "ok") {
    return "border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400";
  }
  return "border-card-border bg-active text-muted";
}

function errorText(error: unknown): string {
  return error instanceof ApiError
    ? error.message
    : "Could not load service status. Please try again.";
}

function handleUnauthorized(error: unknown, router: ReturnType<typeof useRouter>): boolean {
  if (!(error instanceof ApiError) || error.status !== 401) return false;
  clearToken();
  router.replace("/login");
  return true;
}

export function AdminServiceIndicator() {
  const router = useRouter();
  const [status, setStatus] = useState<AdminStatus | null>(null);
  const [hidden, setHidden] = useState(false);
  const missingRef = useRef(false);

  useEffect(() => {
    let active = true;
    let requestId = 0;
    let controller: AbortController | null = null;

    const loadStatus = async () => {
      if (
        !active ||
        missingRef.current ||
        document.visibilityState !== "visible"
      ) {
        return;
      }
      const token = getStoredToken();
      if (!token) return;

      controller?.abort();
      const nextController = new AbortController();
      controller = nextController;
      const currentRequestId = ++requestId;
      try {
        const result = await getAdminStatus(token, nextController.signal);
        if (active && currentRequestId === requestId) setStatus(result);
      } catch (error) {
        if (nextController.signal.aborted || !active) return;
        if (error instanceof ApiError && error.status === 404) {
          missingRef.current = true;
          setHidden(true);
          return;
        }
        if (handleUnauthorized(error, router)) return;
        if (currentRequestId === requestId) setStatus(null);
      }
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        void loadStatus();
      } else {
        requestId += 1;
        controller?.abort();
      }
    };

    void loadStatus();
    const interval = window.setInterval(() => {
      if (document.visibilityState === "visible") void loadStatus();
    }, 5 * 60 * 1000);
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      active = false;
      requestId += 1;
      controller?.abort();
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [router]);

  if (hidden) return null;

  const summary = statusSummary(status);
  return (
    <Link
      href="/dashboard#services"
      title={summary}
      aria-label={`Services status: ${summary}. View service details.`}
      aria-live="polite"
      className="inline-flex items-center gap-2 rounded-full border border-card-border px-3 py-1.5 text-xs font-medium text-muted transition hover:bg-active hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
    >
      <span
        aria-hidden="true"
        className={`h-2 w-2 rounded-full ${statusDotClass(status)}`}
      />
      Services
    </Link>
  );
}

function ServiceBadge({
  state,
  low = false,
}: {
  state: ServiceState;
  low?: boolean;
}) {
  return (
    <span
      role="status"
      aria-live="polite"
      className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-medium ${badgeClass(state, low)}`}
    >
      {low && state === "ok" ? "Low balance" : serviceLabel(state)}
    </span>
  );
}

function JinaAdvice({ state }: { state: ServiceState }) {
  const advice: Partial<Record<ServiceState, string>> = {
    exhausted:
      "Replace JINA_API_KEY in the server environment, update JINA_TOKEN_BUDGET, then press Reset counter.",
    invalid_key: "Check JINA_API_KEY.",
    rate_limited: "Temporary. Wait a minute.",
    unavailable: "The provider may be down.",
  };
  const message = advice[state];
  return message ? <p className="mt-3 text-sm text-muted">{message}</p> : null;
}

function ServiceRowHeading({
  name,
  state,
  low,
}: {
  name: string;
  state: ServiceState;
  low?: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h3 className="text-base font-semibold text-foreground">{name}</h3>
      <ServiceBadge state={state} low={low} />
    </div>
  );
}

export function AdminServicesCard() {
  const router = useRouter();
  const [status, setStatus] = useState<AdminStatus | null>(null);
  const [hidden, setHidden] = useState(false);
  const [loading, setLoading] = useState(true);
  const [action, setAction] = useState<"check" | "reset" | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const [cooldownSeconds, setCooldownSeconds] = useState(0);
  const pollControllerRef = useRef<AbortController | null>(null);
  const pollSequenceRef = useRef(0);
  const missingRef = useRef(false);
  const actionInProgressRef = useRef(false);
  const cooldownUntilRef = useRef(0);

  const loadStatus = useCallback(async () => {
    if (
      missingRef.current ||
      document.visibilityState !== "visible" ||
      actionInProgressRef.current
    ) {
      return;
    }
    const token = getStoredToken();
    if (!token) return;

    pollControllerRef.current?.abort();
    const controller = new AbortController();
    pollControllerRef.current = controller;
    const sequence = ++pollSequenceRef.current;
    setLoading(true);
    try {
      const result = await getAdminStatus(token, controller.signal);
      if (sequence === pollSequenceRef.current) {
        setStatus(result);
        setErrorMessage(null);
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      if (error instanceof ApiError && error.status === 404) {
        missingRef.current = true;
        setHidden(true);
        return;
      }
      if (handleUnauthorized(error, router)) return;
      if (sequence === pollSequenceRef.current) {
        setErrorMessage(errorText(error));
      }
    } finally {
      if (sequence === pollSequenceRef.current) setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => void loadStatus(), 0);
    const interval = window.setInterval(() => {
      if (document.visibilityState === "visible") void loadStatus();
    }, 60 * 1000);
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        void loadStatus();
      } else {
        pollSequenceRef.current += 1;
        pollControllerRef.current?.abort();
        setLoading(false);
      }
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      pollSequenceRef.current += 1;
      pollControllerRef.current?.abort();
      window.clearTimeout(initialLoad);
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [loadStatus]);

  useEffect(() => {
    if (cooldownSeconds <= 0) return;
    const interval = window.setInterval(() => {
      const remaining = Math.ceil(
        (cooldownUntilRef.current - Date.now()) / 1000,
      );
      setCooldownSeconds(Math.max(0, remaining));
    }, 250);
    return () => window.clearInterval(interval);
  }, [cooldownSeconds]);

  const runAction = async (kind: "check" | "reset") => {
    const token = getStoredToken();
    if (!token) return;
    pollSequenceRef.current += 1;
    pollControllerRef.current?.abort();
    actionInProgressRef.current = true;
    setLoading(false);
    setAction(kind);
    setErrorMessage(null);
    setConfirmReset(false);
    try {
      const result =
        kind === "check"
          ? await checkAdminStatus(token)
          : await resetAdminUsage(token);
      setStatus(result);
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        missingRef.current = true;
        setHidden(true);
        return;
      }
      if (handleUnauthorized(error, router)) return;
      if (kind === "check" && error instanceof ApiError && error.status === 429) {
        const seconds = error.retryAfter ?? 1;
        cooldownUntilRef.current = Date.now() + seconds * 1000;
        setCooldownSeconds(seconds);
      }
      setErrorMessage(errorText(error));
    } finally {
      actionInProgressRef.current = false;
      setAction(null);
    }
  };

  if (hidden) return null;

  const jina = status?.jina;
  const groq = status?.groq;
  const percentage = Math.max(
    0,
    Math.min(100, jina?.percent_remaining ?? 0),
  );
  const secondaryButtonClass =
    "rounded-full border border-card-border px-4 py-2 text-sm font-medium text-muted transition hover:bg-active hover:text-foreground active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50";

  return (
    <GlassCard className="w-full max-w-2xl p-6 text-left">
      <section id="services" aria-labelledby="services-heading">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2
              id="services-heading"
              className="text-xl font-semibold text-foreground"
            >
              Services
            </h2>
            <p className="mt-1 text-sm text-muted">
              Provider health and estimated usage
            </p>
          </div>
          {loading && !status && (
            <span role="status" aria-live="polite" className="text-sm text-muted">
              Loading…
            </span>
          )}
        </div>

        <div className="mt-6 divide-y divide-card-border">
          <div className="pb-5">
            <ServiceRowHeading
              name="Jina"
              state={jina?.state ?? "unknown"}
              low={jina?.low}
            />
            <div className="mt-4">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="text-sm font-medium text-foreground">
                  Estimated remaining
                </span>
                <span className="text-sm tabular-nums text-muted">
                  {jina
                    ? `${formatTokens(jina.tokens_estimated_remaining)} / ${formatTokens(jina.tokens_budget)} tokens`
                    : "—"}
                  <span className="ml-1 text-xs">(estimate)</span>
                </span>
              </div>
              <div
                role="progressbar"
                aria-label="Estimated Jina tokens remaining"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={percentage}
                title="Computed from the usage this app has seen and can differ from Jina’s dashboard."
                className="mt-2 h-2 overflow-hidden rounded-full bg-active"
              >
                <div
                  className={`h-full rounded-full transition-[width] ${
                    jina?.low ? "bg-amber-500" : "bg-accent"
                  }`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
              <p className="mt-1.5 text-xs text-muted">
                Estimated from usage this app has seen; it can differ from
                Jina’s dashboard.
              </p>
            </div>
            <dl className="mt-4 grid gap-2 text-sm sm:grid-cols-3">
              <div>
                <dt className="text-muted">Tokens used today</dt>
                <dd className="mt-0.5 font-medium tabular-nums text-foreground">
                  {jina ? formatTokens(jina.tokens_used_today) : "—"}
                </dd>
              </div>
              <div>
                <dt className="text-muted">Last success</dt>
                <dd className="mt-0.5 font-medium text-foreground">
                  {relativeTime(jina?.last_success_at ?? null)}
                </dd>
              </div>
              <div>
                <dt className="text-muted">Last error</dt>
                <dd className="mt-0.5 font-medium text-foreground">
                  {relativeTime(jina?.last_error_at ?? null)}
                </dd>
              </div>
            </dl>
            {jina?.message && (
              <p className="mt-3 break-words rounded-xl bg-active px-3 py-2 text-sm text-muted">
                {jina.message}
              </p>
            )}
            {jina && !jina.message && (
              <p className="mt-3 text-sm text-muted">No recent status message.</p>
            )}
            {jina && <JinaAdvice state={jina.state} />}
          </div>

          <div className="pt-5">
            <ServiceRowHeading name="Groq" state={groq?.state ?? "unknown"} />
            <p className="mt-3 text-sm text-muted">
              <span className="font-medium text-foreground">
                {groq ? formatTokens(groq.answers_last_24h) : "—"}
              </span>{" "}
              answers in the last 24 hours
            </p>
            {groq?.message && (
              <p className="mt-3 break-words rounded-xl bg-active px-3 py-2 text-sm text-muted">
                {groq.message}
              </p>
            )}
            {groq && !groq.message && (
              <p className="mt-3 text-sm text-muted">No recent status message.</p>
            )}
          </div>
        </div>

        {errorMessage && (
          <p
            role="alert"
            aria-live="polite"
            className="mt-4 rounded-xl border border-red-400/30 bg-red-500/10 px-3 py-2 text-sm text-red-600 dark:text-red-400"
          >
            {errorMessage}
          </p>
        )}
        {cooldownSeconds > 0 && (
          <p role="status" aria-live="polite" className="mt-3 text-sm text-muted">
            Status check available in {cooldownSeconds}{" "}
            {cooldownSeconds === 1 ? "second" : "seconds"}.
          </p>
        )}
        {confirmReset && (
          <div
            role="group"
            aria-label="Confirm usage counter reset"
            className="mt-4 rounded-xl border border-amber-500/25 bg-amber-500/10 p-3"
          >
            <p className="text-sm text-foreground">
              Use this only after replacing the Jina key. This resets the
              app&apos;s usage estimate; it does not change the provider key.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => void runAction("reset")}
                disabled={action !== null}
                className={secondaryButtonClass}
              >
                Confirm reset
              </button>
              <button
                type="button"
                onClick={() => setConfirmReset(false)}
                className={secondaryButtonClass}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
        <div className="mt-5 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => void runAction("check")}
            disabled={action !== null || cooldownSeconds > 0}
            aria-label={
              cooldownSeconds > 0
                ? `Check now, available in ${cooldownSeconds} seconds`
                : "Check service status now"
            }
            className={secondaryButtonClass}
          >
            {action === "check"
              ? "Checking…"
              : cooldownSeconds > 0
                ? `Check now (${cooldownSeconds}s)`
                : "Check now"}
          </button>
          <button
            type="button"
            onClick={() => void loadStatus()}
            disabled={loading || action !== null}
            className={secondaryButtonClass}
          >
            {loading ? "Refreshing…" : "Refresh"}
          </button>
          {!confirmReset && (
            <button
              type="button"
              onClick={() => setConfirmReset(true)}
              disabled={action !== null}
              className={secondaryButtonClass}
            >
              Reset counter
            </button>
          )}
        </div>
      </section>
    </GlassCard>
  );
}
