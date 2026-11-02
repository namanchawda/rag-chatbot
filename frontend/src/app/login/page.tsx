"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type ChangeEvent,
} from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import {
  ApiError,
  forgotPassword,
  getPublicConfig,
  resetPassword,
  resendVerification,
  verifyEmail,
} from "@/lib/api";
import { Field, PasswordField, PrimaryButton, Splash } from "@/components/ui";
import { ThemeToggle } from "@/components/theme-toggle";
import { Logo } from "@/components/logo";

type Mode = "signin" | "create";
type Step = "auth" | "verify" | "forgot-email" | "forgot-reset" | "forgot-done";

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const DEFAULT_RETRY_SECONDS = 60;
const RESEND_SECONDS = 60;
const EMAIL_LIMIT_MESSAGE =
  "Email delivery is temporarily unavailable. Please try again later.";

function friendlyError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "Invalid email or password";
    if (error.status === 409) return "Email already registered";
    if (error.status === 503) return EMAIL_LIMIT_MESSAGE;
    return error.message;
  }
  return "Something went wrong. Please try again.";
}

function secondsLeft(value: number | null): string | null {
  return value !== null && value > 0
    ? `Try again in ${value} ${value === 1 ? "second" : "seconds"}.`
    : null;
}

function UploadStepIcon() {
  return (
    <svg
      aria-hidden="true"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 15V4m0 0L8 8m4-4 4 4" />
      <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
    </svg>
  );
}

function ChatStepIcon() {
  return (
    <svg
      aria-hidden="true"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H6l-3 2v-6a7.5 7.5 0 1 1 17-3.5Z" />
      <path d="M8 11h8M8 14h5" />
    </svg>
  );
}

function CheckStepIcon() {
  return (
    <svg
      aria-hidden="true"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="9" />
      <path d="m8.5 12.2 2.4 2.4 4.6-5" />
    </svg>
  );
}

export default function LoginPage() {
  const { status, signIn, signUp, completeSignIn } = useAuth();
  const router = useRouter();
  const submittingRef = useRef(false);

  const [mode, setMode] = useState<Mode>("signin");
  const [step, setStep] = useState<Step>("auth");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [code, setCode] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [signupsClosed, setSignupsClosed] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [resendCountdown, setResendCountdown] = useState<number | null>(null);

  const openSignups = useCallback(() => setSignupsClosed(false), []);

  useEffect(() => {
    let cancelled = false;
    getPublicConfig()
      .then((config) => {
        if (cancelled) return;
        const closed = !config.signups_enabled;
        setSignupsClosed(closed);
        if (closed) setMode((current) => (current === "create" ? "signin" : current));
      })
      .catch(() => openSignups());
    return () => {
      cancelled = true;
    };
  }, [openSignups]);

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
    if (resendCountdown === null || resendCountdown <= 0) return;
    const timer = window.setTimeout(() => {
      setResendCountdown((value) => {
        const next = (value ?? 0) - 1;
        return next <= 0 ? null : next;
      });
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [resendCountdown]);

  useEffect(() => {
    if (status === "authenticated") router.replace("/dashboard");
  }, [status, router]);

  useEffect(() => {
    if (status === "loading") return;
    const focusTarget =
      step === "verify" || step === "forgot-reset"
        ? "auth-code"
        : step === "forgot-email"
          ? "forgot-email"
          : step === "forgot-done"
            ? "auth-heading"
            : "email";
    window.requestAnimationFrame(() => {
      document.getElementById(focusTarget)?.focus();
    });
  }, [step, status]);

  function clearFeedback() {
    setFormError(null);
    setStatusMessage(null);
    setCountdown(null);
  }

  function goToStep(next: Step) {
    setStep(next);
    setShowPassword(false);
    setShowNewPassword(false);
    setShowConfirmPassword(false);
  }

  function switchMode(next: Mode) {
    setMode(next);
    goToStep("auth");
    setPassword("");
    setConfirmation("");
    setCode("");
    clearFeedback();
  }

  function validEmail(value: string): boolean {
    if (EMAIL_PATTERN.test(value.trim())) return true;
    setFormError("Please enter a valid email address.");
    return false;
  }

  function handleRequestError(error: unknown) {
    if (error instanceof ApiError && error.status === 429) {
      setFormError(error.message);
      setCountdown(error.retryAfter ?? DEFAULT_RETRY_SECONDS);
      return;
    }
    setFormError(friendlyError(error));
  }

  async function submitAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (countdown !== null && countdown > 0) return;
    clearFeedback();
    if (submittingRef.current || !validEmail(email)) return;
    if (password.length < 8) {
      setFormError("Password must be at least 8 characters.");
      return;
    }

    submittingRef.current = true;
    setSubmitting(true);
    try {
      if (mode === "signin") {
        try {
          await signIn(email.trim(), password);
        } catch (error) {
          if (
            error instanceof ApiError &&
            error.status === 403 &&
            error.code === "email_not_verified"
          ) {
            goToStep("verify");
            setCode("");
            setStatusMessage("We’ve sent you a new code.");
            setResendCountdown(RESEND_SECONDS);
            return;
          }
          throw error;
        }
      } else {
        const response = await signUp(email.trim(), password);
        setEmail(response.email);
        setCode("");
        goToStep("verify");
        setStatusMessage(null);
        setResendCountdown(RESEND_SECONDS);
      }
    } catch (error) {
      handleRequestError(error);
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
      setShowPassword(false);
    }
  }

  async function submitVerification(value: string) {
    if (
      value.length !== 6 ||
      submittingRef.current ||
      (countdown !== null && countdown > 0)
    ) {
      return;
    }
    clearFeedback();
    submittingRef.current = true;
    setSubmitting(true);
    try {
      const response = await verifyEmail(email.trim(), value);
      completeSignIn(response);
      router.replace("/dashboard");
    } catch (error) {
      handleRequestError(error);
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  }

  function handleCodeChange(event: ChangeEvent<HTMLInputElement>) {
    const nextCode = event.target.value.replace(/\D/g, "").slice(0, 6);
    setCode(nextCode);
    setFormError(null);
    if (step === "verify" && nextCode.length === 6) {
      void submitVerification(nextCode);
    }
  }

  async function resendCode() {
    if (
      submittingRef.current ||
      (resendCountdown !== null && resendCountdown > 0) ||
      (countdown !== null && countdown > 0)
    ) {
      return;
    }
    clearFeedback();
    submittingRef.current = true;
    setSubmitting(true);
    try {
      if (step === "verify") {
        await resendVerification(email.trim());
      } else {
        await forgotPassword(email.trim());
      }
      setResendCountdown(RESEND_SECONDS);
      setCode("");
      setStatusMessage("New code sent.");
    } catch (error) {
      handleRequestError(error);
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  }

  async function sendResetCode(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (countdown !== null && countdown > 0) return;
    clearFeedback();
    if (submittingRef.current || !validEmail(email)) return;
    submittingRef.current = true;
    setSubmitting(true);
    try {
      await forgotPassword(email.trim());
      goToStep("forgot-reset");
      setCode("");
      setPassword("");
      setConfirmation("");
      setResendCountdown(RESEND_SECONDS);
      setStatusMessage(
        "If an account exists for that email, we’ve sent a code.",
      );
    } catch (error) {
      handleRequestError(error);
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  }

  async function submitPasswordReset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (countdown !== null && countdown > 0) return;
    clearFeedback();
    if (submittingRef.current) return;
    if (code.length !== 6) {
      setFormError("Enter the 6-digit code.");
      return;
    }
    if (password.length < 8) {
      setFormError("Password must be at least 8 characters.");
      return;
    }
    if (password !== confirmation) {
      setFormError("Passwords don’t match.");
      return;
    }

    submittingRef.current = true;
    setSubmitting(true);
    try {
      await resetPassword(email.trim(), code, password);
      setPassword("");
      setConfirmation("");
      setCode("");
      goToStep("forgot-done");
      setStatusMessage("Password updated");    } catch (error) {
      handleRequestError(error);
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
      setShowNewPassword(false);
      setShowConfirmPassword(false);
    }
  }

  function useDifferentEmail() {
    setCode("");
    setPassword("");
    setConfirmation("");
    goToStep("forgot-email");
    setResendCountdown(null);
    clearFeedback();
  }

  if (status === "loading") return <Splash />;

  const verifyMode = step === "verify";
  const forgotEmailMode = step === "forgot-email";
  const forgotResetMode = step === "forgot-reset";
  const forgotDoneMode = step === "forgot-done";
  const showAuthForm = step === "auth";
  const canResend = resendCountdown === null || resendCountdown <= 0;
  const passwordMatches = confirmation.length > 0 && password === confirmation;
  const passwordMismatch =
    confirmation.length > 0 && password !== confirmation;

  return (
    <main className="relative flex min-h-screen items-center bg-background px-5 pb-14 pt-24 sm:px-8 sm:pb-20 sm:pt-28 lg:px-12 lg:pb-12 lg:pt-28">
      <div className="absolute left-5 top-5 flex items-center gap-2.5 sm:left-8 sm:top-8 lg:left-12">
        <Logo size={36} />
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
              Retrieval-augmented generation
            </p>
            <h1
              id="login-welcome"
              className="mt-4 text-[34px] font-semibold leading-[1.08] tracking-[-0.04em] text-foreground lg:text-[40px]"
            >
              Answers grounded in your documents, not guessed.
            </h1>
            <p className="mt-4 max-w-lg text-base leading-7 text-muted lg:text-lg lg:leading-8">
              A RAG pipeline you can control. Upload PDF or text files, choose how they're split into passages, and every answer is generated only from the passages retrieved for your question, with the sources shown.
            </p>
            <ol className="mt-8 max-w-lg space-y-4">
              <li className="flex items-start gap-3.5">
                <span
                  aria-hidden="true"
                  className="mt-0.5 flex h-9 w-9 flex-none items-center justify-center rounded-full bg-active text-accent"
                >
                  <UploadStepIcon />
                </span>
                <span>
                  <span className="block text-[15px] font-semibold text-foreground">
                    Chunk
                  </span>
                  <span className="mt-0.5 block text-sm leading-6 text-muted">
                    Choose a strategy (recursive, fixed size, sentence or paragraph) and set the size and overlap.
                  </span>
                </span>
              </li>
              <li className="flex items-start gap-3.5">
                <span
                  aria-hidden="true"
                  className="mt-0.5 flex h-9 w-9 flex-none items-center justify-center rounded-full bg-active text-accent"
                >
                  <ChatStepIcon />
                </span>
                <span>
                  <span className="block text-[15px] font-semibold text-foreground">
                    Retrieve
                  </span>
                  <span className="mt-0.5 block text-sm leading-6 text-muted">
                    Your question is embedded and matched against your passages in pgvector, with optional reranking to drop weak matches.
                  </span>
                </span>
              </li>
              <li className="flex items-start gap-3.5">
                <span
                  aria-hidden="true"
                  className="mt-0.5 flex h-9 w-9 flex-none items-center justify-center rounded-full bg-active text-accent"
                >
                  <CheckStepIcon />
                </span>
                <span>
                  <span className="block text-[15px] font-semibold text-foreground">
                    Generate
                  </span>
                  <span className="mt-0.5 block text-sm leading-6 text-muted">
                    The LLM answers only from the retrieved passages, and shows each one so you can verify it.
                  </span>
                </span>
              </li>
            </ol>
          </div>
        </section>

        <section
          aria-labelledby="auth-heading"
          className="mx-auto w-full max-w-md rounded-[28px] border border-card-border bg-card p-6 shadow-card backdrop-blur-2xl sm:p-8 lg:p-9"
        >
          <div className="mb-7 text-center sm:mb-8">
            <Logo size={48} className="mx-auto mb-4 block shadow-sm sm:hidden" />
            <h2
              id="auth-heading"
              tabIndex={-1}
              className="text-[26px] font-semibold tracking-[-0.03em] text-foreground focus:outline-none"
            >
              {verifyMode
                ? "Verify your email"
                : forgotEmailMode
                  ? "Forgot password?"
                  : forgotResetMode
                    ? "Choose a new password"
                    : forgotDoneMode
                      ? "Password updated"
                      : mode === "signin"
                        ? "Welcome back"
                        : "Create your account"}
            </h2>
            <p className="mt-2 text-base leading-6 text-muted">
              {verifyMode
                ? `We sent a 6-digit code to ${email}.`
                : forgotEmailMode
                  ? "Enter the email address for your account."
                  : forgotResetMode
                    ? `Enter the code sent to ${email} and choose a new password.`
                    : forgotDoneMode
                      ? "Your new password is ready. Sign in to continue."
                      : mode === "signin"
                        ? "Sign in to continue to your workspace."
                        : "A few details and you're in."}
            </p>
          </div>

          {showAuthForm && !signupsClosed && (
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

          {signupsClosed && showAuthForm && mode === "signin" && (
            <p
              role="status"
              aria-live="polite"
              className="mb-6 rounded-xl bg-input px-4 py-2.5 text-center text-sm text-muted"
            >
              Sign-ups are currently closed.
            </p>
          )}

          {showAuthForm && (
            <form onSubmit={submitAuth} noValidate className="space-y-4">
              <Field
                id="email"
                label="Email"
                type="email"
                autoComplete="email"
                inputMode="email"
                placeholder="you@example.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
              <PasswordField
                id="password"
                label="Password"
                autoComplete={
                  mode === "signin" ? "current-password" : "new-password"
                }
                placeholder="••••••••"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                show={showPassword}
                onToggleShow={setShowPassword}
              />
              {mode === "signin" && (
                <div className="-mt-2 text-right">
                  <button
                    type="button"
                    onClick={() => {
                      clearFeedback();
                      goToStep("forgot-email");
                    }}
                    className="min-h-11 rounded-sm px-1 text-sm font-medium text-accent underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                  >
                    Forgot password?
                  </button>
                </div>
              )}
              <div aria-live="assertive" role="alert">
                {formError && (
                  <p className="rounded-xl bg-red-500/10 px-4 py-3 text-sm text-red-500">
                    {formError}
                    {secondsLeft(countdown) && (
                      <> {secondsLeft(countdown)}</>
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
          )}

          {verifyMode && (
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void submitVerification(code);
              }}
              noValidate
              className="space-y-4"
            >
              <Field
                id="auth-code"
                label="6-digit code"
                type="text"
                inputMode="numeric"
                autoComplete="one-time-code"
                maxLength={6}
                pattern="[0-9]{6}"
                placeholder="000000"
                value={code}
                onChange={handleCodeChange}
                aria-describedby="verify-code-hint"
              />
              <p id="verify-code-hint" className="text-sm text-muted">
                Can’t see it? Check your spam folder.
              </p>
              {statusMessage && (
                <p role="status" aria-live="polite" className="text-sm text-accent">
                  {statusMessage}
                </p>
              )}
              <div aria-live="assertive" role="alert">
                {formError && (
                  <p className="rounded-xl bg-red-500/10 px-4 py-3 text-sm text-red-500">
                    {formError}
                    {secondsLeft(countdown) && <> {secondsLeft(countdown)}</>}
                  </p>
                )}
              </div>
              <PrimaryButton
                type="submit"
                loading={submitting}
                disabled={
                  code.length !== 6 ||
                  (countdown !== null && countdown > 0)
                }
                className="h-12 text-base font-semibold"
              >
                {countdown !== null && countdown > 0
                  ? `Wait ${countdown}s`
                  : "Verify email"}
              </PrimaryButton>
              <button
                type="button"
                onClick={() => void resendCode()}
                disabled={
                  submitting ||
                  !canResend ||
                  (countdown !== null && countdown > 0)
                }
                className="min-h-11 w-full rounded-xl px-3 text-base font-medium text-accent hover:bg-active focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-60"
              >
                {resendCountdown !== null && resendCountdown > 0
                  ? `Resend code in ${resendCountdown}s`
                  : "Resend code"}
              </button>
              <button
                type="button"
                onClick={() => {
                    switchMode(signupsClosed ? "signin" : "create");
                  setEmail("");
                }}
                className="min-h-11 w-full rounded-sm px-2 text-sm font-medium text-muted underline-offset-2 hover:text-accent hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                Wrong email? Go back
              </button>
            </form>
          )}

          {forgotEmailMode && (
            <form onSubmit={sendResetCode} noValidate className="space-y-4">
              <Field
                id="forgot-email"
                label="Email"
                type="email"
                autoComplete="email"
                inputMode="email"
                placeholder="you@example.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
              {statusMessage && (
                <p role="status" aria-live="polite" className="text-sm text-muted">
                  {statusMessage}
                </p>
              )}
              <div aria-live="assertive" role="alert">
                {formError && (
                  <p className="rounded-xl bg-red-500/10 px-4 py-3 text-sm text-red-500">
                    {formError}
                    {secondsLeft(countdown) && <> {secondsLeft(countdown)}</>}
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
                  : "Send code"}
              </PrimaryButton>
              <button
                type="button"
                onClick={() => {
                  goToStep("auth");
                  setMode("signin");
                  clearFeedback();
                }}
                className="min-h-11 w-full rounded-sm px-2 text-sm font-medium text-muted underline-offset-2 hover:text-accent hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                Back to sign in
              </button>
            </form>
          )}

          {forgotResetMode && (
            <form
              onSubmit={submitPasswordReset}
              noValidate
              className="space-y-4"
            >
              <Field
                id="auth-code"
                label="6-digit code"
                type="text"
                inputMode="numeric"
                autoComplete="one-time-code"
                maxLength={6}
                pattern="[0-9]{6}"
                placeholder="000000"
                value={code}
                onChange={handleCodeChange}
              />
              <div>
                <PasswordField
                  id="new-password"
                  label="New password"
                  autoComplete="new-password"
                  placeholder="At least 8 characters"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  show={showNewPassword}
                  onToggleShow={setShowNewPassword}
                />
                <p
                  aria-live="polite"
                  className={`mt-1.5 text-sm ${
                    password.length >= 8 ? "text-accent" : "text-muted"
                  }`}
                >
                  {password.length >= 8
                    ? "Password is at least 8 characters."
                    : "Use at least 8 characters."}
                </p>
              </div>
              <PasswordField
                id="confirm-password"
                label="Confirm new password"
                autoComplete="new-password"
                placeholder="Enter your new password again"
                value={confirmation}
                onChange={(event) => setConfirmation(event.target.value)}
                show={showConfirmPassword}
                onToggleShow={setShowConfirmPassword}
              />
              <p
                aria-live="polite"
                className={`text-sm ${
                  passwordMatches
                    ? "text-accent"
                    : passwordMismatch
                      ? "text-red-500"
                      : "text-muted"
                }`}
              >
                {passwordMatches
                  ? "Passwords match."
                  : passwordMismatch
                    ? "Passwords don’t match."
                    : "Enter your password again to confirm."}
              </p>
              {statusMessage && (
                <p role="status" aria-live="polite" className="text-sm text-muted">
                  {statusMessage}
                </p>
              )}
              <div aria-live="assertive" role="alert">
                {formError && (
                  <p className="rounded-xl bg-red-500/10 px-4 py-3 text-sm text-red-500">
                    {formError}
                    {secondsLeft(countdown) && <> {secondsLeft(countdown)}</>}
                  </p>
                )}
              </div>
              <PrimaryButton
                type="submit"
                loading={submitting}
                disabled={
                  submitting ||
                  password.length < 8 ||
                  !passwordMatches ||
                  code.length !== 6 ||
                  (countdown !== null && countdown > 0)
                }
                className="h-12 text-base font-semibold"
              >
                {countdown !== null && countdown > 0
                  ? `Wait ${countdown}s`
                  : "Update password"}
              </PrimaryButton>
              <button
                type="button"
                onClick={() => void resendCode()}
                disabled={
                  submitting ||
                  !canResend ||
                  (countdown !== null && countdown > 0)
                }
                className="min-h-11 w-full rounded-xl px-3 text-base font-medium text-accent hover:bg-active focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-60"
              >
                {resendCountdown !== null && resendCountdown > 0
                  ? `Resend code in ${resendCountdown}s`
                  : "Resend code"}
              </button>
              <button
                type="button"
                onClick={useDifferentEmail}
                className="min-h-11 w-full rounded-sm px-2 text-sm font-medium text-muted underline-offset-2 hover:text-accent hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                Use a different email
              </button>
            </form>
          )}

          {forgotDoneMode && (
            <div className="space-y-4 text-center">
              <p role="status" aria-live="polite" className="text-base text-accent">
                Password updated
              </p>
              <PrimaryButton
                type="button"
                onClick={() => {
                  setMode("signin");
                  goToStep("auth");
                  setPassword("");
                  setConfirmation("");
                  setCode("");
                  setStatusMessage(null);
                }}
                className="h-12 text-base font-semibold"
              >
                Return to sign in
              </PrimaryButton>
            </div>
          )}

          {showAuthForm && !signupsClosed && (
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
