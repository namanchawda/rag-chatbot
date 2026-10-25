"use client";

import { useCallback, useEffect, useState } from "react";
import {
  applyThemeChoice,
  getStoredTheme,
  storeTheme,
  type ThemeChoice,
} from "@/lib/theme";

function SunIcon() {
  return (
    <svg
      aria-hidden="true"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg
      aria-hidden="true"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z" />
    </svg>
  );
}

function MonitorIcon() {
  return (
    <svg
      aria-hidden="true"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <path d="M8 21h8m-4-4v4" />
    </svg>
  );
}

export function ThemeToggle({ className }: { className?: string }) {
  const [choice, setChoice] = useState<ThemeChoice>("system");

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const stored = getStoredTheme();
      if (stored) setChoice(stored);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const select = useCallback((next: ThemeChoice) => {
    setChoice(next);
    storeTheme(next);
    applyThemeChoice(next);
  }, []);

  return (
    <div
      role="group"
      aria-label="Theme"
      className={`inline-flex items-center gap-1 rounded-full bg-input p-1 ${className ?? ""}`}
    >
      {(
        [
          ["light", "Light theme", <SunIcon key="sun" />],
          ["dark", "Dark theme", <MoonIcon key="moon" />],
          ["system", "System theme", <MonitorIcon key="monitor" />],
        ] as const
      ).map(([value, label, icon]) => (
        <button
          key={value}
          type="button"
          title={label}
          aria-label={label}
          aria-pressed={choice === value}
          onClick={() => select(value)}
          className={`flex h-7 w-7 items-center justify-center rounded-full transition active:scale-[0.94] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
            choice === value
              ? "bg-card text-foreground shadow-sm"
              : "text-muted hover:text-foreground"
          }`}
        >
          {icon}
        </button>
      ))}
    </div>
  );
}
