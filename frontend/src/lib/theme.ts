export const THEME_STORAGE_KEY = "rag_chatbot_theme";

export type ThemeChoice = "light" | "dark" | "system";

export function getStoredTheme(): ThemeChoice | null {
  if (typeof window === "undefined") return null;
  try {
    const value = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (value === "light" || value === "dark" || value === "system") {
      return value;
    }
    return null;
  } catch {
    return null;
  }
}

export function storeTheme(choice: ThemeChoice): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, choice);
  } catch {
    return;
  }
}

export function applyThemeChoice(choice: ThemeChoice): void {
  if (typeof document === "undefined") return;
  if (choice === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", choice);
  }
}
