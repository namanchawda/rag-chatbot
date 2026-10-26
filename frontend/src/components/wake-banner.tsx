"use client";

import { useEffect, useState } from "react";
import { subscribeToWake } from "@/lib/wake";

export function WakeBanner() {
  const [visible, setVisible] = useState(false);
  useEffect(() => subscribeToWake(setVisible), []);
  if (!visible) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed left-1/2 top-4 z-50 flex max-w-[92vw] -translate-x-1/2 items-center gap-2.5 rounded-full border border-card-border bg-card px-4 py-2 text-sm text-foreground shadow-card backdrop-blur-2xl"
    >
      <span
        aria-hidden="true"
        className="motion-reduce:animate-none h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-accent/30 border-t-accent"
      />
      Waking up the server. This can take up to a minute on the free plan.
    </div>
  );
}
