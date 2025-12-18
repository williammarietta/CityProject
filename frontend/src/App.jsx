// frontend/src/App.jsx
import { useState } from "react";
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
  const [vhImageFile, setVhImageFile] = useState(null);
  const [vhImagePreview, setVhImagePreview] = useState("");
  const [vhGlass, setVhGlass] = useState(false);
  const [vhGrease, setVhGrease] = useState(false);
  const [vhCorrugated, setVhCorrugated] = useState(false);
  const [vhPlastic12, setVhPlastic12] = useState(false);
  const [vhResultHtml, setVhResultHtml] = useState("");
  const [vhIsLoading, setVhIsLoading] = useState(false);

  // ---------------- Search Assistant handlers ----------------

  const handleQueryChange = async (event) => {
    const value = event.target.value;
    setQuery(value);
    setError("");
    setResultHtml("");

    const trimmed = value.trim();

    if (trimmed.length === 0) {
      setSuggestions([]);
      return;
    }

    setIsLoadingSuggestions(true);
    try {
      const response = await fetch(
        `${API_BASE}/api/suggestions?q=` + encodeURIComponent(trimmed)
      );
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      const data = await response.json();
      setSuggestions((data.choices ?? data.suggestions ?? []));
    } catch (err) {
      console.error(err);
      setError("Could not load suggestions. Make sure backend_api.py is running.");
    } finally {
      setIsLoadingSuggestions(false);
    }
  };

  const handleSuggestionClick = (itemName) => {
    setQuery(itemName);
    setSuggestions([]);
    setError("");
    handleSearch(itemName);
  };

  const handleSearch = async (overrideQuery) => {
    const q = (overrideQuery ?? query).trim();
    if (q.length === 0) {
      setError("Please type the name of an item first.");
      setResultHtml("");
      return;
    }

    setIsSearching(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      const data = await response.json();
      setResultHtml(data.html || "");
    } catch (err) {
      console.error(err);
      setError("Something went wrong talking to the backend. Is backend_api.py running?");
      setResultHtml("");
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSearch();
    }
  };

  // ---------------- Visual Helper handlers ----------------

  const handleVisualFileChange = (event) => {
    const file = event.target.files && event.target.files[0];
    setVhImageFile(file || null);
    setVhResultHtml("");
    setError("");

    if (file) {
      const url = URL.createObjectURL(file);
      setVhImagePreview(url);
    } else {
      setVhImagePreview("");
    }
  };

  const handleVisualAnalyze = async () => {
    setError("");
    setVhIsLoading(true);
    setVhResultHtml("");

    try {
      const formData = new FormData();
      if (vhImageFile) {
        formData.append("image", vhImageFile);
        formData.append("file",vhImageFile);
      }
      formData.append("glass", String(vhGlass));
      formData.append("grease", String(vhGrease));
      formData.append("corrugated", String(vhCorrugated));
      formData.append("plastic12", String(vhPlastic12));

      const response = await fetch(`${API_BASE}/api/visual-helper`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      setVhResultHtml(data.html || "");
    } catch (err) {
      console.error(err);
      setError("Something went wrong talking to the Visual Helper. Is backend_api.py running?");
      setVhResultHtml("");
    } finally {
      setVhIsLoading(false);
    }
  };

  // ---------------- Tab switching ----------------

  const switchToSearch = () => {
    setActiveTab("search");
    setError("");
  };

  const switchToVisual = () => {
    setActiveTab("visual");
    setError("");
  };

  const trimmedQuery = query.trim();
  const showNoSuggestionsHelp =
    trimmedQuery.length > 0 && !isLoadingSuggestions && suggestions.length === 0;

  return (
    <div className="App">
      <header className="app-header">
        <h1>Chesapeake Sorting Assistant</h1>
        <p className="app-subtitle">For City of Chesapeake, VA recycling drop-off locations</p>
      </header>

      <main className="app-main">
        {error && <p className="error-text">{error}</p>}

        <div className="tabs">
          <button
            type="button"
            onClick={switchToSearch}
            className={`tab-button ${activeTab === "search" ? "tab-button-active" : ""}`}
          >
            Search Assistant (Recommended)
          </button>
          <button
            type="button"
            onClick={switchToVisual}
            className={`tab-button ${activeTab === "visual" ? "tab-button-active" : ""}`}
          >
            Visual Helper (Beta)
          </button>
        </div>

        {activeTab === "search" && (
          <section className="tab-content">
            <h2>Search Assistant</h2>
            <p className="tab-description">
              Start typing the name of an item (for example: <b>"microwave"</b>, <b>"battery"</b>, or{" "}
              <b>"bottle"</b>). We&apos;ll tell you whether it belongs in Mixed Recyclables,
              Corrugated Cardboard, Household Hazardous Waste, or Trash / Not Accepted.
            </p>

            <div className="search-box">
              <label className="search-label">Search for an item</label>
              <div className="search-row">
                <input
                  className="search-input"
                  type="text"
                  value={query}
                  onChange={handleQueryChange}
                  onKeyDown={handleKeyDown}
                  placeholder='Try: "pizza box", "battery", "plastic bottle"'
                />
                <button
                  type="button"
                  className="search-button"
                  onClick={() => handleSearch()}
                  disabled={isSearching}
                >
                  {isSearching ? "Searching..." : "Search"}
                </button>
              </div>

              <div className="suggestions-area">
                {isLoadingSuggestions ? (
                  <p className="loading-text">Loading suggestions...</p>
                ) : suggestions.length > 0 ? (
                  <ul className="suggestions-list">
                    {suggestions.map((item) => (
                      <li key={item}>
                        <button
                          type="button"
                          className="suggestion-item"
                          onClick={() => handleSuggestionClick(item)}
                        >
                          {item}
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : showNoSuggestionsHelp ? (
                  <p className="muted-text">
                    No suggestions yet — keep typing. If your item isn’t listed, you can still press{" "}
                    <b>Search</b> and we’ll try to help anyway (including an AI fallback when needed).
                  </p>
                ) : (
                  <p className="muted-text">Suggestions will appear here as you type.</p>
                )}
              </div>

              <div className="result-card-wrapper">
                {resultHtml ? (
                  <div
                    className="result-card"
                    dangerouslySetInnerHTML={{ __html: resultHtml }}
                  />
                ) : (
                  <div className="result-card-placeholder">
                    <p className="muted-text">
                      Result panel will show City of Chesapeake guidance for your item.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {activeTab === "visual" && (
          <section className="tab-content secondary-tab">
            <h2>Visual Helper (Beta)</h2>
            <p className="tab-description">
              Upload a photo of an item. Use the checkboxes for hard rules (glass, grease/food, corrugated).
            </p>

            <div className="visual-upload">
              <input
                className="visual-file-input"
                type="file"
                accept="image/*"
                onChange={handleVisualFileChange}
              />
              <p className="muted-text">
                Tip: For best results, take the photo in good lighting with the item centered.
              </p>
            </div>

            <div className="visual-checkboxes">
              <div className="checkbox-item">
                <input
                  id="vh-glass"
                  type="checkbox"
                  checked={vhGlass}
                  onChange={(e) => setVhGlass(e.target.checked)}
                />
                <label htmlFor="vh-glass">Glass item</label>
              </div>

              <div className="checkbox-item">
                <input
                  id="vh-grease"
                  type="checkbox"
                  checked={vhGrease}
                  onChange={(e) => setVhGrease(e.target.checked)}
                />
                <label htmlFor="vh-grease">Visible grease / food residue</label>
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
                <label htmlFor="vh-plastic12">Plastic #1 or #2 (check the triangle!)</label>
              </div>
            </div>

            <div className="visual-actions">
              <button
                type="button"
                className="search-button"
                onClick={handleVisualAnalyze}
                disabled={vhIsLoading}
              >
                {vhIsLoading ? "Analyzing..." : "Analyze Photo"}
              </button>
            </div>

            <div className="visual-preview">
              {vhImagePreview && (
                <div>
                  <p className="muted-text">Preview (local only):</p>
                  <img
                    src={vhImagePreview}
                    alt="Uploaded item preview"
                    className="preview-image"
                  />
                </div>
              )}
            </div>

            <div className="result-card-wrapper">
              {vhResultHtml ? (
                <div
                  className="result-card"
                  dangerouslySetInnerHTML={{ __html: vhResultHtml }}
                />
              ) : (
                <div className="result-card-placeholder">
                  <p className="muted-text">
                    Visual Helper result will appear here after you upload a photo and click Analyze.
                  </p>
                </div>
              )}
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>
          This guidance applies only to City of Chesapeake, VA recycling drop-off locations and may
          differ from rules in other cities.
        </p>
      </footer>
    </div>
  );
}

export default App;
