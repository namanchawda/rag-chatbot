"use client";

import { useAuth } from "@/lib/auth";
import { AdminServicesCard } from "@/components/admin-services";

export default function DashboardPage() {
  const { user } = useAuth();

  return (
    <main
      className={
        user?.is_admin === true
          ? "flex flex-1 flex-col items-center justify-center gap-8 p-10"
          : "flex flex-1 items-center justify-center p-10"
      }
    >
      <div className="animate-appear max-w-md text-center">
        <h1 className="text-3xl font-semibold text-foreground">
          Welcome{user?.email ? `, ${user.email}` : ""} ✦
        </h1>
        <p className="mt-4 text-lg leading-8 text-muted">
          Your workspace is ready. Chat answers come only from your own
          documents — head to Documents to upload a PDF or TXT, or browse your
          indexed content under Database.
        </p>
      </div>
      {user?.is_admin === true && <AdminServicesCard />}
    </main>
  );
}
