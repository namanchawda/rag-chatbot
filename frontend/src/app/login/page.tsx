"use client";

import { useCallback, useEffect, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { ApiError, getPublicConfig } from "@/lib/api";
import { Field, PrimaryButton, Splash } from "@/components/ui";
import { ThemeToggle } from "@/components/theme-toggle";

type Mode = "signin" | "create";

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
// The backend caps auth attempts per IP for one minute; without the header
// we assume the same default the server uses.
const DEFAULT_RETRY_SECONDS = 60;

function friendlyError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "Invalid email or password";
    if (error.status === 409) return "Email already registered";
    return error.message;
  }
  return "Something went wrong. Please try again.";
}

export default function LoginPage() {
  const { status, signIn, signUp } = useAuth();
  const router = useRouter();

  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fieldErrors, setFieldErrors] = useState<{
    email?: string;
    password?: string;
  }>({});
  const [serverError, setServerError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  // null = signups enabled (or /config unavailable: default to showing the toggle).
  const [signupsClosed, setSignupsClosed] = useState<boolean>(false);
  // Seconds left on an auth rate limit before the submit button re-enables.
  const [countdown, setCountdown] = useState<number | null>(null);

  const openSignups = useCallback(() => {
    setSignupsClosed(false);
  }, []);

  useEffect(() => {
    let cancelled = false;
    getPublicConfig()
      .then((config) => {
        if (cancelled) return;
        const closed = !config.signups_enabled;
        setSignupsClosed(closed);
        if (closed) {
          // Never leave someone staring at a form nobody can submit.
          setMode((current) => (current === "create" ? "signin" : current));
        }
      })
      .catch(() => {
        // /config failing must not hide the toggle: opt back into showing it.
        openSignups();
      });
    return () => {
      cancelled = true;
    };
  }, [openSignups]);

  // One second per step; the last effect run stops the chain at zero.
  useEffect(() => {
    if (countdown === null || countdown <= 0) return;
    const timer = window.setTimeout(() => {
      setCountdown((value) => {
        const next = (value ?? 0) - 1;
        return next <= 0 ? null : next;
      });
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [countdown]);

  useEffect(() => {
    if (status === "authenticated") {
      router.replace("/dashboard");
    }
  }, [status, router]);

  function switchMode(next: Mode) {
    setMode(next);
    setFieldErrors({});
    setServerError(null);
  }

  function validate(): boolean {
    const errors: { email?: string; password?: string } = {};
    if (!EMAIL_PATTERN.test(email.trim())) {
      errors.email = "Please enter a valid email address.";
    }
    if (password.length < 8) {
      errors.password = "Password must be at least 8 characters.";
    }
    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setServerError(null);
    if (!validate() || submitting) return;

    setSubmitting(true);
    try {
      if (mode === "signin") {
        await signIn(email.trim(), password);
      } else {
        await signUp(email.trim(), password);
      }
      router.replace("/dashboard");
    } catch (error) {
      if (error instanceof ApiError && error.status === 429) {
        // Show the server's text and block submits until the limit lifts.
        setServerError(error.message);
        setCountdown(error.retryAfter ?? DEFAULT_RETRY_SECONDS);
      } else if (error instanceof ApiError && error.status === 403) {
        setServerError(error.message);
      } else {
        setServerError(friendlyError(error));
      }
    } finally {
      setSubmitting(false);
    }
  }

  if (status === "loading") {
    return <Splash />;
  }

  return (
    <main className="relative flex min-h-screen items-center bg-background px-5 py-20 sm:px-8 sm:py-24 lg:px-12 lg:py-16">
      <div className="absolute left-5 top-5 flex items-center gap-2.5 sm:left-8 sm:top-8 lg:left-12">
        <span
          aria-hidden="true"
          className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent text-lg text-white"
        >
          ✦
        </span>
        <span className="text-base font-semibold tracking-tight text-foreground">
          RAG Chatbot
        </span>
      </div>
      <ThemeToggle className="absolute right-4 top-4 sm:right-6 sm:top-6" />
      <div className="animate-appear mx-auto grid w-full max-w-6xl gap-6 sm:gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(380px,0.82fr)] lg:items-center">
        <section
          aria-labelledby="login-welcome"
          className="hidden rounded-[28px] border border-card-border bg-card p-7 shadow-card sm:block sm:p-9 lg:p-11"
        >
          <div className="max-w-xl">
            <p className="text-sm font-semibold uppercase tracking-[0.14em] text-accent">
              Your knowledge workspace
            </p>
            <h1
              id="login-welcome"
              className="mt-4 text-[34px] font-semibold leading-[1.08] tracking-[-0.04em] text-foreground lg:text-[42px]"
            >
              Your documents,
              <br className="hidden lg:block" /> in conversation.
            </h1>
            <p className="mt-4 max-w-lg text-base leading-7 text-muted lg:text-lg lg:leading-8">
              Bring your own sources into the conversation and find answers
              grounded in what you’ve collected.
            </p>
          </div>
          <div
            aria-hidden="true"
            className="relative mt-8 hidden h-40 items-center justify-center xl:flex"
          >
            <div className="absolute h-32 w-40 rotate-[-10deg] rounded-2xl border border-card-border bg-background shadow-card" />
            <div className="absolute h-32 w-40 rotate-[8deg] rounded-2xl border border-card-border bg-background shadow-card" />
            <div className="relative flex h-32 w-40 flex-col justify-between rounded-2xl border border-card-border bg-background p-4 shadow-card">
              <span className="text-accent">✦</span>
              <span className="space-y-2">
                <span className="block h-1.5 w-4/5 rounded-full bg-active" />
                <span className="block h-1.5 w-3/5 rounded-full bg-active" />
                <span className="block h-1.5 w-2/3 rounded-full bg-active" />
              </span>
              <span className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">
                Source material
              </span>
            </div>
          </div>
        </section>

        <section
          aria-labelledby="login-heading"
          className="mx-auto w-full max-w-md rounded-[28px] border border-card-border bg-card p-6 shadow-card sm:p-8 lg:p-9"
        >
          <div className="mb-7 text-center sm:mb-8">
            <span
              className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent text-2xl text-white shadow-sm sm:hidden"
              aria-hidden="true"
            >
              ✦
            </span>
            <h2
              id="login-heading"
              className="text-[26px] font-semibold tracking-[-0.03em] text-foreground"
            >
              {mode === "signin" ? "Welcome back" : "Create your account"}
            </h2>
            <p className="mt-2 text-base leading-6 text-muted">
              {mode === "signin"
                ? "Sign in to continue to your workspace."
                : "A few details and you're in."}
            </p>
          </div>

          {signupsClosed ? (
            <p
              role="status"
              aria-live="polite"
              className="mb-6 rounded-xl bg-input px-4 py-2.5 text-center text-sm text-muted"
            >
              Sign-ups are currently closed.
            </p>
          ) : (
            <div
              className="mb-6 grid grid-cols-2 gap-1 rounded-xl bg-input p-1"
              role="tablist"
              aria-label="Authentication mode"
            >
              {(
                [
                  ["signin", "Sign in"],
                  ["create", "Create account"],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  role="tab"
                  aria-selected={mode === value}
                  onClick={() => switchMode(value)}
                  className={`min-h-11 rounded-lg px-2 text-sm font-medium transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                    mode === value
                      ? "bg-card text-foreground shadow-sm"
                      : "text-muted hover:text-foreground"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="space-y-4">
            <Field
              id="email"
              label="Email"
              type="email"
              autoComplete="email"
              inputMode="email"
              placeholder="you@example.com"
              value={email}
              error={fieldErrors.email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <Field
              id="password"
              label="Password"
              type="password"
              autoComplete={
                mode === "signin" ? "current-password" : "new-password"
              }
              placeholder="••••••••"
              value={password}
              error={fieldErrors.password}
              onChange={(e) => setPassword(e.target.value)}
            />

            <div aria-live="polite" role="alert">
              {serverError && (
                <p className="rounded-xl bg-red-500/10 px-4 py-3 text-sm text-red-500">
                  {serverError}
                  {countdown !== null && countdown > 0 && (
                    <>
                      {" "}
                      Try again in {countdown}{" "}
                      {countdown === 1 ? "second" : "seconds"}.
                    </>
                  )}
                </p>
              )}
            </div>

            <PrimaryButton
              type="submit"
              loading={submitting}
              disabled={countdown !== null && countdown > 0}
              className="h-12 text-base font-semibold"
            >
              {countdown !== null && countdown > 0
                ? `Wait ${countdown}s`
                : mode === "signin"
                  ? "Sign in"
                  : "Create account"}
            </PrimaryButton>
          </form>

          {!signupsClosed && (
            <p className="mt-6 text-center text-base text-muted">
              {mode === "signin" ? "New here? " : "Already have an account? "}
              <button
                type="button"
                onClick={() =>
                  switchMode(mode === "signin" ? "create" : "signin")
                }
                className="min-h-11 rounded-sm px-1 font-semibold text-accent hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                {mode === "signin" ? "Create an account" : "Sign in"}
              </button>
            </p>
          )}
        </section>
      </div>
    </main>
  );
}
