"use client";

import { useCallback, useEffect, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { ApiError, getPublicConfig } from "@/lib/api";
import { Field, GlassCard, PrimaryButton, Splash } from "@/components/ui";
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
    <main className="relative flex min-h-screen items-center justify-center bg-background px-4">
      <ThemeToggle className="absolute right-4 top-4 md:right-6 md:top-6" />
      <div className="animate-appear w-full max-w-sm">
        <GlassCard className="p-8">
          <div className="mb-8 text-center">
            <p
              className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent text-2xl text-white shadow-sm"
              aria-hidden="true"
            >
              ✦
            </p>
            <h1 className="text-[26px] font-semibold text-foreground">
              {mode === "signin" ? "Welcome back" : "Create your account"}
            </h1>
            <p className="mt-1.5 text-[15px] text-muted">
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
                  className={`h-9 rounded-lg text-sm font-medium transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
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
            >
              {countdown !== null && countdown > 0
                ? `Wait ${countdown}s`
                : mode === "signin"
                  ? "Sign in"
                  : "Create account"}
            </PrimaryButton>
          </form>

          {!signupsClosed && (
            <p className="mt-6 text-center text-sm text-muted">
              {mode === "signin" ? "New here? " : "Already have an account? "}
              <button
                type="button"
                onClick={() =>
                  switchMode(mode === "signin" ? "create" : "signin")
                }
                className="font-medium text-accent hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                {mode === "signin" ? "Create an account" : "Sign in"}
              </button>
            </p>
          )}
        </GlassCard>
      </div>
    </main>
  );
}
