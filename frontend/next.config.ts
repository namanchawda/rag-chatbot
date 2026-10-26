import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // All routes are "use client" pages with browser-side auth and data
  // fetching, so `instant = false` cannot be exported from them (invalid in
  // Client Components). Cache Components / Partial Prefetching therefore
  // provide no shell to validate and are disabled to skip instant-navigation
  // validation for /dashboard, /chat, and /documents.
  cacheComponents: false,
  partialPrefetching: false,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
