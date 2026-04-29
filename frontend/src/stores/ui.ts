import { create } from "zustand";

/**
 * Ephemeral UI state. Anything persisted lives in the server (sessions,
 * versions, chat messages) and is read via TanStack Query. Anything in
 * this store dies on reload by design.
 */
interface UIState {
  // Phase 1 search box draft
  orgSearchQuery: string;
  setOrgSearchQuery: (s: string) => void;

  // Stub user identity for V0 (auth is a header-trust shim until Entra ID lands).
  // Persist to localStorage so reloads keep the same email.
  userEmail: string;
  setUserEmail: (s: string) => void;
}

const PERSIST_KEY = "drw.user_email";

export const useUI = create<UIState>((set) => ({
  orgSearchQuery: "",
  setOrgSearchQuery: (s) => set({ orgSearchQuery: s }),

  userEmail:
    (typeof localStorage !== "undefined" &&
      localStorage.getItem(PERSIST_KEY)) ||
    "",
  setUserEmail: (s) => {
    if (typeof localStorage !== "undefined") localStorage.setItem(PERSIST_KEY, s);
    set({ userEmail: s });
  },
}));
