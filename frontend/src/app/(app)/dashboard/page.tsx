"use client";

import { useAuth } from "@/lib/auth";

export default function DashboardPage() {
  const { user } = useAuth();

  return (
    <main className="flex flex-1 items-center justify-center p-10">
      <div className="animate-appear max-w-md text-center">
        <h1 className="text-3xl font-semibold text-foreground">
          Welcome{user?.email ? `, ${user.email}` : ""} ✦
        </h1>
        <p className="mt-4 text-lg leading-8 text-muted">
          Your workspace is ready. Chat and database features are still on their
          way — documents are available in the sidebar.
        </p>
      </div>
    </main>
  );
}
