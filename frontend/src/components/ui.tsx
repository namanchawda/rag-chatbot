"use client";

import { useRef, type ButtonHTMLAttributes, type InputHTMLAttributes } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  loading?: boolean;
};

export function PrimaryButton({
  loading = false,
  disabled,
  children,
  className,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={`flex h-11 w-full items-center justify-center gap-2 rounded-full bg-accent px-5 text-[15px] font-medium text-white shadow-sm transition hover:bg-accent-hover active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-60 ${className ?? ""}`}
    >
      {loading && (
        <span
          aria-hidden="true"
          className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"
        />
      )}
      {loading ? "Please wait…" : children}
    </button>
  );
}

type InputProps = InputHTMLAttributes<HTMLInputElement> & {
  label: string;
  error?: string;
};

export function Field({ label, error, id, className, ...rest }: InputProps) {
  const describedBy = [
    rest["aria-describedby"],
    error ? `${id}-error` : undefined,
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <div className={className}>
      <label
        htmlFor={id}
        className="mb-1.5 block text-base font-medium text-foreground"
      >
        {label}
      </label>
      <input
        id={id}
        {...rest}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy || undefined}
        className={`h-12 w-full rounded-xl border border-input-border bg-input px-4 text-base text-foreground placeholder:text-muted focus:border-accent focus:outline-none focus:ring-4 focus:ring-ring transition-shadow ${
          error ? "border-red-400" : ""
        }`}
      />
      {error && (
        <p id={`${id}-error`} className="mt-1.5 text-sm text-red-500">
          {error}
        </p>
      )}
    </div>
  );
}

export function EyeIcon() {
  return (
    <svg
      aria-hidden="true"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

export function EyeOffIcon() {
  return (
    <svg
      aria-hidden="true"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 3l18 18" />
      <path d="M10.6 5.8c.5-.2.9-.3 1.4-.3 6 0 9.5 6.5 9.5 6.5a17.6 17.6 0 0 1-3 3.8" />
      <path d="M6.6 6.8a17.3 17.3 0 0 0-4.1 5.2S6 18.5 12 18.5c1.6 0 3-.4 4.3-1" />
      <path d="M9.9 10.2a3 3 0 0 0 4 4.2" />
    </svg>
  );
}

type PasswordFieldProps = Omit<InputProps, "type"> & {
  show: boolean;
  onToggleShow: (next: boolean) => void;
};

export function PasswordField({
  show,
  onToggleShow,
  label,
  error,
  id,
  className,
  ...rest
}: PasswordFieldProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  function toggleVisibility() {
    const input = inputRef.current;
    const hasSelection =
      input !== null && typeof input.selectionStart === "number";
    const start = hasSelection ? input.selectionStart : null;
    const end = hasSelection ? input.selectionEnd : null;
    const direction = hasSelection
      ? (input.selectionDirection ?? "none")
      : "none";
    onToggleShow(!show);
    if (input && start !== null && end !== null) {
      window.requestAnimationFrame(() => {
        input.setSelectionRange(start, end, direction);
      });
    }
  }

  const describedBy = [
    rest["aria-describedby"],
    error ? `${id}-error` : undefined,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={className}>
      <label
        htmlFor={id}
        className="mb-1.5 block text-base font-medium text-foreground"
      >
        {label}
      </label>
      <div className="relative">
        <input
          ref={inputRef}
          id={id}
          type={show ? "text" : "password"}
          {...rest}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy || undefined}
          className={`h-12 w-full rounded-xl border border-input-border bg-input pl-4 pr-12 text-base text-foreground placeholder:text-muted focus:border-accent focus:outline-none focus:ring-4 focus:ring-ring transition-shadow ${
            error ? "border-red-400" : ""
          }`}
        />
        <button
          type="button"
          aria-label={show ? "Hide password" : "Show password"}
          aria-pressed={show}
          onClick={toggleVisibility}
          className="absolute right-2 top-1/2 flex h-9 w-9 -translate-y-1/2 items-center justify-center rounded-lg text-muted transition hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          {show ? <EyeOffIcon /> : <EyeIcon />}
        </button>
      </div>
      {error && (
        <p id={`${id}-error`} className="mt-1.5 text-sm text-red-500">
          {error}
        </p>
      )}
    </div>
  );
}

export function GlassCard({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-[24px] border border-card-border bg-card shadow-card backdrop-blur-2xl ${className ?? ""}`}
    >
      {children}
    </div>
  );
}

export function Splash() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div
        role="status"
        aria-live="polite"
        className="animate-appear text-sm text-muted"
      >
        Loading…
      </div>
    </div>
  );
}
