// frontend/src/App.jsx
import { useEffect, useState } from "react";
import "./App.css";

const API_BASE = "https://cityproject.onrender.com";

function App() {
  // Which tab is active: "search" or "visual"
  const [activeTab, setActiveTab] = useState("search");

  // ----- Search Assistant state -----
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);
  const [resultHtml, setResultHtml] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState("");

  // ----- Visual Helper state -----
  const [imageFile, setImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState("");
  const [vhGlass, setVhGlass] = useState(false);
  const [vhGrease, setVhGrease] = useState(false);
  const [vhCorrugated, setVhCorrugated] = useState(false);
  const [vhPlastic12, setVhPlastic12] = useState(false);
  const [vhResultHtml, setVhResultHtml] = useState("");
  const [vhError, setVhError] = useState("");
  const [vhIsLoading, setVhIsLoading] = useState(false);

  // Fetch suggestions whenever query changes
  useEffect(() => {
    let ignore = false;

    async function getSuggestions() {
      const q = query.trim();
      if (!q) {
        setSuggestions([]);
        return;
      }

      setIsLoadingSuggestions(true);
      try {
        const response = await fetch(
          `${API_BASE}/api/suggestions?q=${encodeURIComponent(q)}`
        );
        if (!response.ok) throw new Error("Suggestion request failed");
        const data = await response.json();
        if (!ignore) setSuggestions(data.suggestions || []);
      } catch (e) {
        if (!ignore) setSuggestions([]);
      } finally {
        if (!ignore) setIsLoadingSuggestions(false);
      }
    }

    // small debounce
    const t = setTimeout(getSuggestions, 200);
    return () => {
      ignore = true;
      clearTimeout(t);
    };
  }, [query]);

  async function runSearch() {
    const q = query.trim();
    if (!q) return;

    setError("");
    setResultHtml("");
    setIsSearching(true);

    try {
      const response = await fetch(`${API_BASE}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      if (!response.ok) throw new Error("Network response was not ok");
      const data = await response.json();
      setResultHtml(data.html || "");
    } catch (err) {
      console.error(err);
      setError("Search failed. Please try again.");
    } finally {
      setIsSearching(false);
    }
  }

  async function runVisualHelper() {
    if (!imageFile) {
      setVhError("Please upload an image first.");
      return;
    }

    setVhError("");
    setVhResultHtml("");
    setVhIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", imageFile);
      formData.append("glass", String(vhGlass));
      formData.append("grease", String(vhGrease));
      formData.append("corrugated", String(vhCorrugated));
      formData.append("plastic12", String(vhPlastic12));

      const response = await fetch(`${API_BASE}/api/visual-helper`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Visual Helper request failed");
      const data = await response.json();
      setVhResultHtml(data.html || "");
    } catch (err) {
      console.error(err);
      setVhError("Visual Helper failed. Please try again.");
    } finally {
      setVhIsLoading(false);
    }
  }

  function onSuggestionClick(itemName) {
    setQuery(itemName);
    setSuggestions([]);
    setTimeout(runSearch, 0);
  }

  function onPickImage(file) {
    setImageFile(file || null);
    setVhResultHtml("");
    setVhError("");

    if (!file) {
      setImagePreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(file);
    setImagePreviewUrl(url);
  }

  return (
    <div className="page">
      <header className="header">
        <div className="title">Chesapeake Sorting Assistant</div>
        <div className="subtitle">
          Search-based guidance + optional photo helper (beta)
        </div>
      </header>

      <div className="tabs">
        <button
          className={`tab ${activeTab === "search" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("search")}
        >
          Search Assistant
        </button>
        <button
          className={`tab ${activeTab === "visual" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("visual")}
        >
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
            <button className="btn" onClick={runSearch} disabled={isSearching}>
              {isSearching ? "Searching..." : "Search"}
            </button>
          </div>

          {isLoadingSuggestions && <p className="muted-text">Loading…</p>}

          {!isLoadingSuggestions && suggestions.length > 0 && (
            <div className="suggestions">
              <ul className="suggestion-list">
                {suggestions.map((s) => (
                  <li key={s}>
                    <button
                      className="suggestion-item"
                      onClick={() => onSuggestionClick(s)}
                    >
                      {s}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {!isLoadingSuggestions && suggestions.length === 0 && query && (
            <p className="muted-text">
              No suggestions yet. You can still finish typing your item and press
              Search — we'll check it even if it isn't in the list (and may use
              OpenAI as a fallback if needed).
            </p>
          )}

          {!query && !isLoadingSuggestions && (
            <p className="muted-text">
              Start typing to see suggestions, or type any item and press Search.
            </p>
          )}

          {error && <div className="error">{error}</div>}

          {resultHtml && (
            <div
              className="result-html"
              dangerouslySetInnerHTML={{ __html: resultHtml }}
            />
          )}
        </section>
      )}

      {activeTab === "visual" && (
        <section className="card">
          <div className="card-title">Visual Helper (Beta)</div>

          <div className="visual-row">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => onPickImage(e.target.files?.[0] || null)}
            />
          </div>

          {imagePreviewUrl && (
            <div className="preview">
              <img src={imagePreviewUrl} alt="preview" />
            </div>
          )}

          <div className="checkbox-grid">
            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={vhGlass}
                onChange={(e) => setVhGlass(e.target.checked)}
              />
              Glass
            </label>

            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={vhGrease}
                onChange={(e) => setVhGrease(e.target.checked)}
              />
              Visible grease / food
            </label>

            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={vhCorrugated}
                onChange={(e) => setVhCorrugated(e.target.checked)}
              />
              Corrugated cardboard (shipping box)
            </label>

            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={vhPlastic12}
                onChange={(e) => setVhPlastic12(e.target.checked)}
              />
              Plastic #1 or #2 (check the triangle!)
            </label>
          </div>

          <div className="visual-actions">
            <button className="btn" onClick={runVisualHelper} disabled={vhIsLoading}>
              {vhIsLoading ? "Analyzing..." : "Analyze photo"}
            </button>
          </div>

          {vhError && <div className="error">{vhError}</div>}

          {vhResultHtml && (
            <div
              className="result-html"
              dangerouslySetInnerHTML={{ __html: vhResultHtml }}
            />
          )}
        </section>
      )}

      <footer className="footer muted-text">
        Built for Chesapeake, VA drop-off recycling guidance.
      </footer>
    </div>
  );
}

export default App;
