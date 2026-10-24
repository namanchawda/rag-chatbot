"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  clearToken,
  getMe,
  getStoredToken,
  login as apiLogin,
  signup as apiSignup,
  storeToken,
  type User,
} from "@/lib/api";

type AuthStatus = "loading" | "authenticated" | "unauthenticated";

type AuthContextValue = {
  status: AuthStatus;
  user: User | null;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      let token: string | null = null;
      try {
        token = getStoredToken();
      } catch {
        token = null;
      }
      if (!token) {
        setStatus("unauthenticated");
        return;
      }
      try {
        const me = await getMe(token);
        if (cancelled) return;
        setUser(me);
        setStatus("authenticated");
      } catch (error) {
        if (cancelled) return;
        if ((error as { status?: number }).status === 401) {
          clearToken();
          setStatus("unauthenticated");
        } else {
          // Network or server failure: treat as authenticated with cached
          // token rather than logging the user out spuriously.
          setStatus("authenticated");
        }
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const response = await apiLogin(email, password);
    storeToken(response.access_token);
    setUser(response.user);
    setStatus("authenticated");
  }, []);

  const signUp = useCallback(async (email: string, password: string) => {
    const response = await apiSignup(email, password);
    storeToken(response.access_token);
    setUser(response.user);
    setStatus("authenticated");
  }, []);

  const signOut = useCallback(() => {
    clearToken();
    setUser(null);
    setStatus("unauthenticated");
  }, []);

  const value = useMemo(
    () => ({ status, user, signIn, signUp, signOut }),
    [status, user, signIn, signUp, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
