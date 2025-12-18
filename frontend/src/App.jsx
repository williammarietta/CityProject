import React, { useEffect, useMemo, useRef, useState } from "react";

const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8000";

export default function App() {
  const [activeTab, setActiveTab] = useState("search");

  // ---------------- Search Assistant state ----------------
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);
  const [searchResultHtml, setSearchResultHtml] = useState("");
  const [usedLLM, setUsedLLM] = useState(false);
  const [searchError, setSearchError] = useState("");

  // For suggestions timing
  const suggestAbortRef = useRef(null);

  // ---------------- Visual Helper state ----------------
  const [vhFile, setVhFile] = useState(null);
  const [vhImagePreview, setVhImagePreview] = useState("");
  const [vhGlass, setVhGlass] = useState(false);
  const [vhGrease, setVhGrease] = useState(false);
  const [vhCorrugated, setVhCorrugated] = useState(false);
  const [vhPlastic12, setVhPlastic12] = useState(false);
  const [vhResultHtml, setVhResultHtml] = useState("");
  const [vhIsLoading, setVhIsLoading] = useState(false);
  const [vhError, setVhError] = useState("");

  // ---------------- Search Assistant helpers ----------------

  async function fetchSuggestions(text) {
    const q = (text || "").trim();
    if (!q) {
      setSuggestions([]);
      return;
    }

    try {
      setIsLoadingSuggestions(true);

      // Abort previous in-flight request
      if (suggestAbortRef.current) {
        suggestAbortRef.current.abort();
      }
      const controller = new AbortController();
      suggestAbortRef.current = controller;

      const resp = await fetch(
        `${BACKEND_URL}/api/suggestions?q=${encodeURIComponent(q)}`,
        { signal: controller.signal }
      );
      if (!resp.ok) throw new Error("Suggestion request failed");
      const data = await resp.json();
      setSuggestions(data?.suggestions || []);
    } catch (e) {
      // If aborted, ignore
      if (e?.name !== "AbortError") {
        setSuggestions([]);
      }
    } finally {
      setIsLoadingSuggestions(false);
    }
  }

  useEffect(() => {
    const q = (query || "").trim();
    if (!q) {
      setSuggestions([]);
      return;
    }

    const t = setTimeout(() => {
      fetchSuggestions(q);
    }, 250);

    return () => clearTimeout(t);
  }, [query]);

  async function runSearch(qOverride = null) {
    const q = (qOverride ?? query ?? "").trim();
    if (!q) return;

    setSearchError("");
    setSearchResultHtml("");
    setUsedLLM(false);

    try {
      const resp = await fetch(`${BACKEND_URL}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "Search failed");
      }

      const data = await resp.json();
      setSearchResultHtml(data?.html || "");
      setUsedLLM(Boolean(data?.used_llm || data?.usedLLM));
    } catch (e) {
      setSearchError(String(e?.message || e));
    }
  }

  function onPickSuggestion(name) {
    setQuery(name);
    setSuggestions([]);
    runSearch(name);
  }

  // ---------------- Visual Helper helpers ----------------

  function onPickFile(file) {
    setVhFile(file);
    setVhResultHtml("");
    setVhError("");

    if (!file) {
      setVhImagePreview("");
      return;
    }

    const url = URL.createObjectURL(file);
    setVhImagePreview(url);
  }

  async function runVisualHelper() {
    if (!vhFile) {
      setVhError("Please choose an image first.");
      return;
    }

    setVhIsLoading(true);
    setVhError("");
    setVhResultHtml("");

    try {
      const form = new FormData();
      form.append("file", vhFile);
      form.append("glass", String(vhGlass));
      form.append("grease", String(vhGrease));
      form.append("corrugated", String(vhCorrugated));
      form.append("plastic12", String(vhPlastic12));

      const resp = await fetch(`${BACKEND_URL}/api/visual-helper`, {
        method: "POST",
        body: form,
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "Visual Helper failed");
      }

      const data = await resp.json();
      setVhResultHtml(data?.html || "");
    } catch (e) {
      setVhError(String(e?.message || e));
    } finally {
      setVhIsLoading(false);
    }
  }

  const tabClass = (t) =>
    `tab ${activeTab === t ? "tab-active" : ""}`.trim();

  const canSearch = useMemo(() => (query || "").trim().length > 0, [query]);

  return (
    <div className="page">
      <header className="header">
        <div className="title">Chesapeake Sorting Assistant</div>
        <div className="subtitle">
          Search-based guidance + optional photo helper (beta)
        </div>
      </header>

      <div className="tabs">
        <button className={tabClass("search")} onClick={() => setActiveTab("search")}>
          Search Assistant
        </button>
        <button className={tabClass("visual")} onClick={() => setActiveTab("visual")}>
          Visual Helper (Beta)
        </button>
      </div>

      {activeTab === "search" && (
        <section className="card">
          <div className="card-title">Search for an item</div>

          <div className="search-row">
            <input
              className="search-input"
              value={query}
              placeholder='Try: "pizza box", "battery", "plastic bottle"...'
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") runSearch();
              }}
            />
            <button className="btn" disabled={!canSearch} onClick={() => runSearch()}>
              Search
            </button>
          </div>

          {isLoadingSuggestions && (
            <p className="muted-text">Loading suggestions…</p>
          )}

          {!isLoadingSuggestions && suggestions.length > 0 && (
            <div className="suggestions">
              <div className="muted-text">Suggestions (click one):</div>
              <ul className="suggestion-list">
                {suggestions.map((s) => (
                  <li key={s}>
                    <button className="suggestion-item" onClick={() => onPickSuggestion(s)}>
                      {s}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {!isLoadingSuggestions && suggestions.length === 0 && query && (
            <p className="muted-text">
              No suggestions yet. You can still finish typing your item and press Search,
              even if it doesn&apos;t appear in this list.
            </p>
          )}

          {!query && !isLoadingSuggestions && (
            <p className="muted-text">
              Start typing to see suggestions, or type any item and press Search.
            </p>
          )}

          {searchError && <div className="error">{searchError}</div>}

          {searchResultHtml && (
            <div className="result-area">
              {usedLLM && (
                <div className="pill">
                  Used LLM fallback (OpenAI) for this result
                </div>
              )}
              <div
                className="result-html"
                dangerouslySetInnerHTML={{ __html: searchResultHtml }}
              />
            </div>
          )}
        </section>
      )}

      {activeTab === "visual" && (
        <section className="card">
          <div className="card-title">Visual Helper (Beta)</div>
          <div className="muted-text">
            Upload a photo and optionally check any obvious cues. These checkboxes
            override the ML model.
          </div>

          <div className="visual-row">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => onPickFile(e.target.files?.[0] || null)}
            />
          </div>

          {vhImagePreview && (
            <div className="preview">
              <img src={vhImagePreview} alt="preview" />
            </div>
          )}

          <div className="checkbox-grid">
            <div className="checkbox-item">
              <input
                id="vh-glass"
                type="checkbox"
                checked={vhGlass}
                onChange={(e) => setVhGlass(e.target.checked)}
              />
              <label htmlFor="vh-glass">Glass</label>
            </div>

            <div className="checkbox-item">
              <input
                id="vh-grease"
                type="checkbox"
                checked={vhGrease}
                onChange={(e) => setVhGrease(e.target.checked)}
              />
              <label htmlFor="vh-grease">Visible grease / food</label>
            </div>

            <div className="checkbox-item">
              <input
                id="vh-corrugated"
                type="checkbox"
                checked={vhCorrugated}
                onChange={(e) => setVhCorrugated(e.target.checked)}
              />
              <label htmlFor="vh-corrugated">Corrugated cardboard (shipping box)</label>
            </div>

            <div className="checkbox-item">
              <input
                id="vh-plastic12"
                type="checkbox"
                checked={vhPlastic12}
                onChange={(e) => setVhPlastic12(e.target.checked)}
              />
              <label htmlFor="vh-plastic12">
                Plastic #1 or #2 (check the triangle!)
              </label>
            </div>
          </div>

          <div className="visual-actions">
            <button className="btn" onClick={runVisualHelper} disabled={vhIsLoading}>
              {vhIsLoading ? "Analyzing…" : "Analyze photo"}
            </button>
          </div>

          {vhError && <div className="error">{vhError}</div>}

          {vhResultHtml && (
            <div className="result-area">
              <div
                className="result-html"
                dangerouslySetInnerHTML={{ __html: vhResultHtml }}
              />
            </div>
          )}
        </section>
      )}

      <footer className="footer muted-text">
        Built for Chesapeake, VA drop-off recycling guidance.
      </footer>
    </div>
  );
}
