import { NavLink, Outlet } from "react-router-dom";
import { useEffect, useState } from "react";
import { getStats } from "../api/library";

const tabs = [
  { to: "/ask", label: "Ask", icon: "💬" },
  { to: "/search", label: "Search", icon: "🔎" },
  { to: "/library", label: "Library", icon: "📚" },
  { to: "/ingest", label: "Ingest", icon: "📥" },
];

export default function Layout() {
  const [stats, setStats] = useState<{
    embedding_model: string;
    generation_model: string;
    store: Record<string, unknown>;
  } | null>(null);

  useEffect(() => {
    getStats()
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  return (
    <div className="min-h-full flex flex-col">
      <header className="border-b border-slate-200 bg-white sticky top-0 z-10">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-md bg-brand-500 text-white font-bold">
              m
            </span>
            <span className="font-semibold tracking-tight">medlit</span>
            <span className="hidden sm:inline text-xs text-slate-500 ml-1">
              biomedical literature RAG
            </span>
          </div>
          <div className="hidden md:flex text-xs text-slate-500 gap-3">
            {stats && (
              <>
                <span>
                  embed: <code className="text-slate-700">{stats.embedding_model}</code>
                </span>
                <span>
                  gen: <code className="text-slate-700">{stats.generation_model}</code>
                </span>
                <span>
                  vectors: <code className="text-slate-700">{String(stats.store?.vectors_count ?? "?")}</code>
                </span>
              </>
            )}
          </div>
        </div>
        <nav className="mx-auto max-w-6xl px-4 flex overflow-x-auto">
          {tabs.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              className={({ isActive }) =>
                `px-3 py-2 -mb-px border-b-2 text-sm font-medium whitespace-nowrap ${
                  isActive
                    ? "border-brand-500 text-brand-600"
                    : "border-transparent text-slate-600 hover:text-slate-900"
                }`
              }
            >
              <span className="mr-1">{t.icon}</span>
              {t.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
        <Outlet />
      </main>

      <footer className="text-center text-xs text-slate-400 py-4">
        <a
          href="https://github.com/Rodrigo-Coloma/sagebase"
          target="_blank"
          rel="noreferrer"
          className="hover:text-slate-600"
        >
          medlit on GitHub
        </a>
      </footer>
    </div>
  );
}
