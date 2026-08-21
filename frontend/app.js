/* ─────────────────────────────────────────────────────────────────────────────
   VisualFind — React Application (CDN React + Babel)
   Image-based Product Search using ResNet50 + FAISS
   ─────────────────────────────────────────────────────────────────────────── */

const { useState, useRef, useCallback, useEffect, useMemo } = React;

// ── API base URL detection ────────────────────────────────────────────────────
// Local dev  : frontend served from file:// or localhost:8080, backend on :8000
//              → use absolute URL http://localhost:8000
// Docker     : frontend served by Nginx on port 80/8080, backend proxied via /api/
//              → use relative /api prefix (Nginx rewrites /api/* → backend:8000/*)
const _IS_DOCKER = window.location.port !== "8080" &&
                   window.location.protocol !== "file:";
const API_BASE   = _IS_DOCKER ? "/api" : "http://localhost:8000";


// ── Helpers ───────────────────────────────────────────────────────────────────

function getSimilarityClass(score) {
  if (score >= 0.75) return "similarity-high";
  if (score >= 0.55) return "similarity-medium";
  return "similarity-low";
}

function getRankClass(index) {
  if (index === 0) return "rank-1";
  if (index === 1) return "rank-2";
  if (index === 2) return "rank-3";
  return "";
}

function formatMs(ms) {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

// ── Shopping Links ────────────────────────────────────────────────────────────

function buildSearchQuery(product) {
  const parts = [
    product.brand && product.brand !== "Unknown Brand" ? product.brand : "",
    product.subcategory && product.subcategory !== "Unknown" ? product.subcategory : "",
    product.category && product.category !== "Unknown" ? product.category : "",
  ].filter(Boolean);
  return encodeURIComponent(parts.join(" ").trim());
}

const SHOPPING_SITES = [
  {
    id: "amazon",
    label: "Amazon",
    icon: "🛒",
    color: "#ff9900",
    getUrl: (q) => `https://www.amazon.in/s?k=${q}`,
  },
  {
    id: "flipkart",
    label: "Flipkart",
    icon: "🏪",
    color: "#2874f0",
    getUrl: (q) => `https://www.flipkart.com/search?q=${q}`,
  },
  {
    id: "myntra",
    label: "Myntra",
    icon: "👗",
    color: "#ff3f6c",
    getUrl: (q) => `https://www.myntra.com/${encodeURIComponent(q.replace(/%20/g, "-"))}`,
  },
  {
    id: "ajio",
    label: "AJIO",
    icon: "✨",
    color: "#e85f00",
    getUrl: (q) => `https://www.ajio.com/search/?text=${q}`,
  },
];

function ShoppingLinks({ product, compact = false }) {
  const query = buildSearchQuery(product);
  if (!query) return null;

  if (compact) {
    return (
      <div className="shopping-links-compact" onClick={(e) => e.stopPropagation()}>
        <span className="shopping-label-compact">Shop:</span>
        {SHOPPING_SITES.map((site) => (
          <a
            key={site.id}
            href={site.getUrl(query)}
            target="_blank"
            rel="noopener noreferrer"
            className="shopping-btn-compact"
            style={{ "--shop-color": site.color }}
            title={`Search on ${site.label}`}
            id={`shop-${site.id}-${product.cid}`}
          >
            {site.icon} {site.label}
          </a>
        ))}
      </div>
    );
  }

  return (
    <div className="shopping-links-full">
      <div className="shopping-title">🛍️ Shop This Product</div>
      <div className="shopping-grid">
        {SHOPPING_SITES.map((site) => (
          <a
            key={site.id}
            href={site.getUrl(query)}
            target="_blank"
            rel="noopener noreferrer"
            className="shopping-btn-full"
            style={{ "--shop-color": site.color }}
            id={`modal-shop-${site.id}-${product.cid}`}
          >
            <span className="shopping-btn-icon">{site.icon}</span>
            <span className="shopping-btn-label">{site.label}</span>
            <span className="shopping-btn-arrow">↗</span>
          </a>
        ))}
      </div>
    </div>
  );
}

// ── ApiStatusBadge ────────────────────────────────────────────────────────────

function ApiStatusBadge({ status }) {
  const labels = { online: "API Online", offline: "API Offline", checking: "Connecting…" };
  return (
    <div className={`api-status api-status-${status}`} title={`Backend status: ${status}`}>
      <span className="api-status-dot" />
      <span className="api-status-label">{labels[status]}</span>
    </div>
  );
}

// ── Nav ───────────────────────────────────────────────────────────────────────

function Nav({ apiStatus, onHistoryToggle, hasHistory }) {
  return (
    <nav className="nav">
      <div className="nav-inner">
        <a href="#" className="nav-logo">
          <div className="nav-logo-icon">👁</div>
          <span className="nav-logo-text">VisualFind</span>
        </a>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <span className="nav-badge">ResNet50 + FAISS</span>
          {hasHistory && (
            <button
              id="history-toggle-btn"
              className="nav-history-btn"
              onClick={onHistoryToggle}
              title="View search history"
            >
              🕐 History
            </button>
          )}
          <ApiStatusBadge status={apiStatus} />
        </div>
      </div>
    </nav>
  );
}

// ── Hero ──────────────────────────────────────────────────────────────────────

function Hero({ catalogSize }) {
  return (
    <div className="hero">
      <div className="hero-tag">
        <span className="hero-tag-dot"></span>
        AI-Powered Visual Search
      </div>
      <h1 className="hero-title">
        <span className="hero-title-line1">Find Products</span>
        <span className="hero-title-line2">By Image, Not Words</span>
      </h1>
      <p className="hero-subtitle">
        Upload any product photo and our deep learning engine will instantly
        surface the most visually similar items from the catalog.
      </p>
      <div className="hero-stats">
        <div className="hero-stat">
          <div className="hero-stat-num">
            {catalogSize > 0 ? catalogSize.toLocaleString() : "—"}
          </div>
          <div className="hero-stat-label">Products Indexed</div>
        </div>
        <div className="hero-stat">
          <div className="hero-stat-num">2048</div>
          <div className="hero-stat-label">Embedding Dims</div>
        </div>
        <div className="hero-stat">
          <div className="hero-stat-num">ResNet50</div>
          <div className="hero-stat-label">Fine-Tuned CNN</div>
        </div>
        <div className="hero-stat">
          <div className="hero-stat-num">FAISS</div>
          <div className="hero-stat-label">ANN Index</div>
        </div>
      </div>
    </div>
  );
}

// ── HowItWorks ────────────────────────────────────────────────────────────────

function HowItWorks() {
  const steps = [
    {
      icon: "📸",
      title: "Upload a Photo",
      desc: "Drag & drop or browse any product image. JPEG, PNG, or WebP supported.",
      color: "#4f8ef7",
    },
    {
      icon: "🧠",
      title: "Fine-Tuned ResNet50",
      desc: "A ResNet50 CNN fine-tuned with triplet loss on footwear brands extracts a 2048-dim embedding tuned for shoe similarity — shape, texture, color, style.",
      color: "#9b6fef",
    },
    {
      icon: "⚡",
      title: "FAISS Similarity Search",
      desc: "Cosine similarity over L2-normalized vectors. Top-K nearest neighbors retrieved in milliseconds.",
      color: "#f0c040",
    },
  ];

  return (
    <div className="how-it-works">
      <div className="how-it-works-header">
        <span className="section-label">How It Works</span>
        <p className="section-sublabel">Three steps from photo to results</p>
      </div>
      <div className="how-steps">
        {steps.map((step, i) => (
          <div key={i} className="how-step-wrapper">
            <div className="how-step" style={{ "--step-color": step.color }}>
              <div className="how-step-icon">{step.icon}</div>
              <div className="how-step-num">0{i + 1}</div>
              <div className="how-step-title">{step.title}</div>
              <div className="how-step-desc">{step.desc}</div>
            </div>
            {i < steps.length - 1 && <div className="how-connector">→</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── DropZone (with image preview) ─────────────────────────────────────────────

function DropZone({ onFile, hasImage, previewUrl }) {
  const inputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) onFile(file);
  }, [onFile]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setIsDragOver(false), []);
  const handleClick = () => inputRef.current?.click();
  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onFile(file);
  };

  return (
    <div
      id="drop-zone"
      className={`drop-zone ${isDragOver ? "drag-over" : ""} ${hasImage ? "has-image" : ""}`}
      onClick={handleClick}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      role="button"
      tabIndex={0}
      aria-label="Upload product image"
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") handleClick(); }}
    >
      <input
        id="file-input"
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleChange}
      />

      {previewUrl ? (
        <div className="drop-preview-wrapper">
          <img src={previewUrl} alt="Selected product" className="drop-preview-image" />
          <div className="drop-preview-overlay">
            <div className="drop-preview-hint">
              <span>📁</span>
              <span>Click or drop to replace</span>
            </div>
          </div>
        </div>
      ) : (
        <>
          <div className="drop-icon">📸</div>
          <div className="drop-title">Drop your product image here</div>
          <div className="drop-subtitle">or click to browse from your device</div>
          <div className="drop-formats">
            {["JPEG", "PNG", "WebP"].map((f) => (
              <span key={f} className="format-chip">{f}</span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ── QueryPanel ────────────────────────────────────────────────────────────────

function QueryPanel({ queryImageUrl, isLoading, topK, onTopKChange, onSearch, onClear, hasImage }) {
  return (
    <div className="query-panel">
      <div className="query-panel-label">Query Image</div>
      <div className="query-image-container">
        {queryImageUrl ? (
          <img src={queryImageUrl} alt="Query" className="query-image-preview" />
        ) : (
          <div className="query-placeholder">
            <div className="query-placeholder-icon">🔍</div>
            <div>Your image will appear here</div>
          </div>
        )}
      </div>

      <div className="settings-row">
        <div className="top-k-control">
          <label className="top-k-label" htmlFor="top-k-select">Results:</label>
          <select
            id="top-k-select"
            className="top-k-select"
            value={topK}
            onChange={(e) => onTopKChange(Number(e.target.value))}
          >
            {[6, 9, 12, 16, 20].map((n) => (
              <option key={n} value={n}>{n} items</option>
            ))}
          </select>
        </div>

        {hasImage && !isLoading && (
          <button
            id="clear-btn"
            className="search-btn search-btn-secondary"
            onClick={onClear}
          >
            Clear
          </button>
        )}
      </div>

      <button
        id="search-btn"
        className="search-btn search-btn-primary"
        onClick={onSearch}
        disabled={!hasImage || isLoading}
        style={{ width: "100%" }}
      >
        {isLoading ? "Searching…" : "🔍 Find Similar Products"}
      </button>
    </div>
  );
}

// ── StatsBar ──────────────────────────────────────────────────────────────────

function StatsBar({ results, queryTime }) {
  if (!results || results.length === 0) return null;
  const sims = results.map((r) => r.similarity);
  const highest = Math.max(...sims);
  const avg = sims.reduce((a, b) => a + b, 0) / sims.length;

  return (
    <div className="stats-bar">
      <div className="stat-pill">
        <span className="stat-pill-icon">⚡</span>
        <span className="stat-pill-label">Query time</span>
        <span className="stat-pill-value">{formatMs(queryTime)}</span>
      </div>
      <div className="stat-pill">
        <span className="stat-pill-icon">📦</span>
        <span className="stat-pill-label">Results</span>
        <span className="stat-pill-value">{results.length}</span>
      </div>
      <div className="stat-pill stat-pill-gold">
        <span className="stat-pill-icon">🏆</span>
        <span className="stat-pill-label">Best match</span>
        <span className="stat-pill-value">{Math.round(highest * 100)}%</span>
      </div>
      <div className="stat-pill">
        <span className="stat-pill-icon">📊</span>
        <span className="stat-pill-label">Avg similarity</span>
        <span className="stat-pill-value">{Math.round(avg * 100)}%</span>
      </div>
    </div>
  );
}

// ── FilterBar ─────────────────────────────────────────────────────────────────

function FilterBar({ results, filters, onFilterChange }) {
  if (!results || results.length === 0) return null;

  const categories = useMemo(() => (
    ["all", ...new Set(results.map((r) => r.category).filter(Boolean))]
  ), [results]);

  const genders = useMemo(() => (
    ["all", ...new Set(
      results.flatMap((r) => (r.gender || "").split(";").map((g) => g.trim())).filter(Boolean)
    )]
  ), [results]);

  const simLevels = [
    ["all", "All"],
    ["high", "High ≥ 75%"],
    ["medium", "Medium ≥ 55%"],
  ];

  return (
    <div className="filter-bar">
      <div className="filter-group">
        <span className="filter-group-label">Category</span>
        <div className="filter-chips">
          {categories.map((cat) => (
            <button
              key={cat}
              className={`filter-chip ${filters.category === cat ? "active" : ""}`}
              onClick={() => onFilterChange("category", cat)}
            >
              {cat === "all" ? "All" : cat}
            </button>
          ))}
        </div>
      </div>

      <div className="filter-group">
        <span className="filter-group-label">Gender</span>
        <div className="filter-chips">
          {genders.map((g) => (
            <button
              key={g}
              className={`filter-chip ${filters.gender === g ? "active" : ""}`}
              onClick={() => onFilterChange("gender", g)}
            >
              {g === "all" ? "All" : g}
            </button>
          ))}
        </div>
      </div>

      <div className="filter-group">
        <span className="filter-group-label">Similarity</span>
        <div className="filter-chips">
          {simLevels.map(([val, label]) => (
            <button
              key={val}
              className={`filter-chip ${filters.similarity === val ? "active" : ""}`}
              onClick={() => onFilterChange("similarity", val)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── ProductDetailModal ────────────────────────────────────────────────────────

function ProductDetailModal({ product, onClose }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", handler);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  if (!product) return null;

  const simPct = Math.round(product.similarity * 100);
  const simClass = getSimilarityClass(product.similarity);

  const metaFields = [
    ["Category", product.category],
    ["Subcategory", product.subcategory],
    ["Brand", product.brand],
    ["Gender", product.gender],
    ["Material", product.material],
    ["Heel Height", product.heel_height],
    ["Closure", product.closure],
    ["Toe Style", product.toe_style],
    ["Product ID", product.cid],
  ].filter(([, v]) => v);

  return (
    <div
      className="modal-overlay"
      onClick={onClose}
      id="product-modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Product details"
    >
      <div
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
        id="product-modal"
      >
        <button
          className="modal-close"
          onClick={onClose}
          id="modal-close-btn"
          aria-label="Close modal"
        >
          ✕
        </button>

        <div className="modal-body">
          {/* Left: Image */}
          <div className="modal-image-col">
            {product.image_b64 ? (
              <img
                src={product.image_b64}
                alt={`${product.brand} ${product.subcategory}`}
                className="modal-image"
              />
            ) : (
              <div className="modal-image-placeholder">👟</div>
            )}
            <div className={`modal-sim-badge ${simClass}`}>{simPct}% Match</div>
          </div>

          {/* Right: Info */}
          <div className="modal-info-col">
            <div className="modal-brand">{product.brand || "Unknown Brand"}</div>
            <div className="modal-product-name">
              {product.category} — {product.subcategory}
            </div>

            <div className="modal-sim-section">
              <div className="modal-sim-label">Cosine Similarity</div>
              <div className="modal-sim-track">
                <div className="modal-sim-fill" style={{ width: `${simPct}%` }} />
              </div>
              <div className="modal-sim-value">{simPct}%</div>
            </div>

            <div className="modal-meta-grid">
              {metaFields.map(([label, value]) => (
                <div key={label} className="modal-meta-item">
                  <div className="modal-meta-label">{label}</div>
                  <div className="modal-meta-value">{value}</div>
                </div>
              ))}
            </div>

            <ShoppingLinks product={product} compact={false} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── SearchHistory ─────────────────────────────────────────────────────────────

function SearchHistory({ history, onReplay, onClearHistory, onClose }) {
  if (!history || history.length === 0) return null;

  return (
    <div className="history-panel" id="history-panel">
      <div className="history-header">
        <span className="history-title">🕐 Search History</span>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <button
            className="search-btn search-btn-secondary"
            onClick={onClearHistory}
            style={{ padding: "5px 12px", fontSize: "0.78rem" }}
            id="clear-history-btn"
          >
            Clear All
          </button>
          <button
            className="history-close-btn"
            onClick={onClose}
            id="history-close-btn"
            aria-label="Close history"
          >
            ✕
          </button>
        </div>
      </div>

      <div className="history-list">
        {history.slice(0, 10).map((entry, i) => (
          <div
            key={i}
            className="history-item"
            onClick={() => onReplay(entry)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === "Enter") onReplay(entry); }}
          >
            <img src={entry.previewUrl} alt="History" className="history-thumb" />
            <div className="history-meta">
              <div className="history-results">{entry.resultCount} results found</div>
              <div className="history-time">
                {new Date(entry.timestamp).toLocaleDateString()} ·{" "}
                {new Date(entry.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </div>
            </div>
            <div className="history-replay">Replay →</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── ProductCard ───────────────────────────────────────────────────────────────

function ProductCard({ product, rank, onClick }) {
  const simPct = Math.round(product.similarity * 100);
  const simClass = getSimilarityClass(product.similarity);
  const rankClass = getRankClass(rank);

  const tags = [
    product.gender && { label: product.gender, type: "gender" },
    product.material && product.material.split(";")[0],
    product.closure && product.closure.split(";")[0],
  ].filter(Boolean);

  return (
    <div
      className="product-card"
      style={{ animationDelay: `${rank * 50}ms` }}
      onClick={() => onClick(product)}
      role="button"
      tabIndex={0}
      aria-label={`${product.brand} ${product.subcategory}, ${simPct}% match`}
      onKeyDown={(e) => { if (e.key === "Enter") onClick(product); }}
    >
      <div className="card-image-wrapper">
        {product.image_b64 ? (
          <img
            src={product.image_b64}
            alt={`${product.brand} ${product.subcategory}`}
            className="card-image"
            loading="lazy"
          />
        ) : (
          <div className="card-image-placeholder">👟</div>
        )}

        <div className={`rank-badge ${rankClass}`}>{rank + 1}</div>
        <div className={`similarity-badge ${simClass}`}>{simPct}% match</div>
      </div>

      <div className="card-body">
        <div className="card-brand">{product.brand || "Unknown Brand"}</div>
        <div className="card-category">{product.category}</div>
        <div className="card-subcategory">{product.subcategory}</div>

        <div className="card-tags">
          {tags.slice(0, 3).map((tag, i) => (
            typeof tag === "object" ? (
              <span key={i} className={`card-tag card-tag-${tag.type}`}>{tag.label}</span>
            ) : (
              <span key={i} className="card-tag">{tag}</span>
            )
          ))}
        </div>

        <div className="similarity-bar-wrapper">
          <div className="similarity-bar">
            <div
              className="similarity-bar-fill"
              style={{ width: `${simPct}%` }}
            />
          </div>
          <span className="similarity-pct">{simPct}%</span>
        </div>

        <ShoppingLinks product={product} compact={true} />
      </div>
    </div>
  );
}

// ── ResultsGrid ───────────────────────────────────────────────────────────────

function ResultsGrid({ results, queryTime, onCardClick, filters, onFilterChange }) {
  const filteredResults = useMemo(() => {
    if (!results) return [];
    return results.filter((r) => {
      if (filters.category !== "all" && r.category !== filters.category) return false;
      if (filters.gender !== "all") {
        const genders = (r.gender || "").split(";").map((g) => g.trim());
        if (!genders.includes(filters.gender)) return false;
      }
      if (filters.similarity === "high" && r.similarity < 0.75) return false;
      if (filters.similarity === "medium" && r.similarity < 0.55) return false;
      return true;
    });
  }, [results, filters]);

  if (!results || results.length === 0) return null;

  return (
    <div className="results-section" id="results-section">
      <div className="results-header">
        <h2 className="results-title">Similar Products Found</h2>
        <div className="results-meta">
          <span className="results-count">
            {filteredResults.length !== results.length
              ? `${filteredResults.length} of ${results.length} results`
              : `${results.length} results`}
          </span>
          <span className="results-time">⚡ {formatMs(queryTime)}</span>
        </div>
      </div>

      <FilterBar
        results={results}
        filters={filters}
        onFilterChange={onFilterChange}
      />

      <div className="results-grid" id="results-grid">
        {filteredResults.map((product, i) => (
          <ProductCard
            key={`${product.cid}-${i}`}
            product={product}
            rank={results.indexOf(product)}
            onClick={onCardClick}
          />
        ))}
      </div>

      {filteredResults.length === 0 && (
        <div className="state-box">
          <div className="state-icon">🔎</div>
          <div className="state-title">No matches for current filters</div>
          <div className="state-text">
            Try adjusting the category, gender, or similarity threshold.
          </div>
        </div>
      )}
    </div>
  );
}

// ── LoadingState ──────────────────────────────────────────────────────────────

function LoadingState() {
  return (
    <div className="loading-overlay">
      <div className="spinner" />
      <div className="loading-text">Analyzing your image…</div>
      <div className="loading-sub">
        Extracting ResNet50 embeddings · Searching FAISS index
      </div>
    </div>
  );
}

// ── ErrorState ────────────────────────────────────────────────────────────────

function ErrorState({ message, onRetry }) {
  return (
    <div className="state-box state-error">
      <div className="state-icon">⚠️</div>
      <div className="state-title">Search Failed</div>
      <div
        className="state-text"
        style={{ whiteSpace: "pre-wrap", textAlign: "left", maxWidth: "480px" }}
      >
        {message}
      </div>
      {onRetry && (
        <button
          id="retry-btn"
          className="search-btn search-btn-secondary"
          onClick={onRetry}
          style={{ marginTop: "1rem" }}
        >
          Try Again
        </button>
      )}
    </div>
  );
}

// ── EmptyResults ──────────────────────────────────────────────────────────────

function EmptyResults() {
  return (
    <div className="state-box">
      <div className="state-icon">🔍</div>
      <div className="state-title">No results found</div>
      <div className="state-text">
        The catalog may not contain visually similar items for this image.
        Try a different product photo.
      </div>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

function App() {
  // ── Core state ──────────────────────────────────────────────────────────────
  const [queryFile, setQueryFile]         = useState(null);
  const [queryImageUrl, setQueryImageUrl] = useState(null);
  const [previewUrl, setPreviewUrl]       = useState(null);   // FileReader data-URL for DropZone
  const [topK, setTopK]                   = useState(12);
  const [isLoading, setIsLoading]         = useState(false);
  const [results, setResults]             = useState(null);
  const [queryTime, setQueryTime]         = useState(0);
  const [error, setError]                 = useState(null);
  const [catalogSize, setCatalogSize]     = useState(0);

  // ── UI state ─────────────────────────────────────────────────────────────────
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [filters, setFilters]                 = useState({ category: "all", gender: "all", similarity: "all" });
  const [apiStatus, setApiStatus]             = useState("checking");
  const [showHistory, setShowHistory]         = useState(false);

  // ── Search history (localStorage) ───────────────────────────────────────────
  const [searchHistory, setSearchHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem("vf_history") || "[]"); }
    catch { return []; }
  });

  // ── API health polling ───────────────────────────────────────────────────────
  useEffect(() => {
    const check = () => {
      fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) })
        .then((r) => setApiStatus(r.ok ? "online" : "offline"))
        .catch(() => setApiStatus("offline"));
    };
    check();
    const timer = setInterval(check, 10000);
    return () => clearInterval(timer);
  }, []);

  // ── Catalog stats ────────────────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/stats`)
      .then((r) => r.json())
      .then((d) => setCatalogSize(d.catalog_size || 0))
      .catch(() => {});
  }, []);

  // ── File selection ───────────────────────────────────────────────────────────
  const handleFile = useCallback((file) => {
    setQueryFile(file);
    setQueryImageUrl(URL.createObjectURL(file));
    setResults(null);
    setError(null);
    setFilters({ category: "all", gender: "all", similarity: "all" });

    // FileReader for inline drop zone preview
    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const handleClear = useCallback(() => {
    setQueryFile(null);
    setQueryImageUrl(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
  }, []);

  // ── Search ───────────────────────────────────────────────────────────────────
  const handleSearch = useCallback(async () => {
    if (!queryFile) return;
    setIsLoading(true);
    setError(null);
    setResults(null);
    setFilters({ category: "all", gender: "all", similarity: "all" });

    const formData = new FormData();
    formData.append("file", queryFile);
    formData.append("top_k", String(topK));

    try {
      const res = await fetch(`${API_BASE}/search`, { method: "POST", body: formData });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResults(data.results);
      setQueryTime(data.query_time_ms);

      // Persist to history (keep last 10)
      if (previewUrl) {
        const entry = {
          previewUrl,
          resultCount: data.total_results,
          timestamp: Date.now(),
          results: data.results,
          queryTime: data.query_time_ms,
        };
        setSearchHistory((prev) => {
          const next = [entry, ...prev].slice(0, 10);
          try { localStorage.setItem("vf_history", JSON.stringify(next)); } catch {}
          return next;
        });
      }

      setTimeout(() => {
        document.getElementById("results-section")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 100);
    } catch (err) {
      if (err.name === "TypeError" && err.message.includes("fetch")) {
        setError(
          "Cannot reach the API server. Two things to check:\n\n" +
          "1. Make sure the FastAPI backend is running:\n" +
          "   → Double-click start_server.bat\n\n" +
          "2. If you opened index.html directly (file://), use the frontend server instead:\n" +
          "   → Double-click start_frontend.bat\n" +
          "   → Then visit: http://localhost:8080"
        );
      } else {
        setError(err.message || "Search failed. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  }, [queryFile, topK, previewUrl]);

  // ── Handlers ─────────────────────────────────────────────────────────────────
  const handleFilterChange = useCallback((key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleReplay = useCallback((entry) => {
    setResults(entry.results);
    setQueryTime(entry.queryTime);
    setPreviewUrl(entry.previewUrl);
    setQueryImageUrl(entry.previewUrl);
    setShowHistory(false);
    setFilters({ category: "all", gender: "all", similarity: "all" });
    setTimeout(() => {
      document.getElementById("results-section")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 150);
  }, []);

  const handleClearHistory = useCallback(() => {
    setSearchHistory([]);
    try { localStorage.removeItem("vf_history"); } catch {}
  }, []);

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      <Nav
        apiStatus={apiStatus}
        onHistoryToggle={() => setShowHistory((v) => !v)}
        hasHistory={searchHistory.length > 0}
      />

      <main className="main">
        <Hero catalogSize={catalogSize} />

        <HowItWorks />

        {showHistory && (
          <SearchHistory
            history={searchHistory}
            onReplay={handleReplay}
            onClearHistory={handleClearHistory}
            onClose={() => setShowHistory(false)}
          />
        )}

        <div className="upload-section">
          <DropZone
            onFile={handleFile}
            hasImage={!!queryFile}
            fileName={queryFile?.name}
            previewUrl={previewUrl}
          />
          <QueryPanel
            queryImageUrl={queryImageUrl}
            isLoading={isLoading}
            topK={topK}
            onTopKChange={setTopK}
            onSearch={handleSearch}
            onClear={handleClear}
            hasImage={!!queryFile}
          />
        </div>

        <div className="section-divider" />

        {isLoading && <LoadingState />}

        {error && !isLoading && (
          <ErrorState message={error} onRetry={handleSearch} />
        )}

        {results !== null && !isLoading && !error && (
          results.length === 0 ? (
            <EmptyResults />
          ) : (
            <>
              <StatsBar results={results} queryTime={queryTime} />
              <ResultsGrid
                results={results}
                queryTime={queryTime}
                catalogSize={catalogSize}
                onCardClick={setSelectedProduct}
                filters={filters}
                onFilterChange={handleFilterChange}
              />
            </>
          )
        )}
      </main>

      <footer className="footer">
        <strong>VisualFind</strong> — Image-based Product Search ·
        ResNet50 + FAISS · FastAPI · React ·{" "}
        <span style={{ color: "var(--accent-gold)" }}>UT-Zappos50K Dataset</span>
      </footer>

      {selectedProduct && (
        <ProductDetailModal
          product={selectedProduct}
          onClose={() => setSelectedProduct(null)}
        />
      )}
    </div>
  );
}

// ── Mount ─────────────────────────────────────────────────────────────────────
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
