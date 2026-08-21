# Image-Based Product Search System — Implementation Plan

## Overview

Build a production-quality, end-to-end **visual similarity search engine** for footwear using the **UT-Zappos50K** dataset. Users upload a shoe image and receive the top-K most visually similar products ranked by cosine similarity, powered by a ResNet50 CNN backbone + FAISS ANN indexing, served via FastAPI, and presented in a premium React UI.

---

## Dataset Overview

The workspace already contains the **UT-Zappos50K** dataset:
- **`ut-zap50k-images/`** — ~50,000 shoe product images organized in `Category > SubCategory > Brand > image.jpg` hierarchy (currently only `Boots/Ankle` subdirectory visible, but metadata covers all categories)
- **`ut-zap50k-data/meta-data.csv`** — 50,026 rows with columns: `CID, Category, SubCategory, HeelHeight, Insole, Closure, Gender, Material, ToeStyle`
- Categories include: Boots, Shoes, Sandals, Slippers (with subcategories like Ankle, Mid-Calf, Oxfords, Loafers, Heels, Flats, Sneakers, etc.)

---

## Architecture

```
User Upload (React)
      ↓
FastAPI /search endpoint
      ↓
Image Preprocessing (Pillow/OpenCV → 224×224, normalize)
      ↓
ResNet50 Feature Extractor (PyTorch, pretrained, remove classifier head)
      → 2048-dim embedding
      ↓
FAISS IndexFlatIP (cosine similarity via L2-normalized vectors)
      ↓
Top-K nearest neighbors (image paths + metadata)
      ↓
JSON response → React renders ranked results grid
```

**Offline catalog indexing pipeline:**
```
Catalog images → Batch embedding extraction → .npy embeddings file + FAISS index file
```

---

## Project Folder Structure

```
Image-based Product Search/
├── backend/
│   ├── main.py               # FastAPI app
│   ├── embedder.py           # ResNet50 feature extractor
│   ├── indexer.py            # FAISS index builder & loader
│   ├── catalog.py            # Dataset path & metadata loader
│   ├── build_index.py        # One-time offline indexing script
│   ├── requirements.txt
│   └── data/
│       ├── embeddings.npy    # Pre-computed catalog embeddings
│       ├── index.faiss       # FAISS index
│       └── catalog.json      # CID → image path + metadata mapping
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── SearchBar.jsx       # Drag & drop image upload
│   │   │   ├── ResultsGrid.jsx     # Visual results display
│   │   │   ├── ProductCard.jsx     # Individual product tile
│   │   │   └── LoadingSpinner.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   └── package.json
├── ut-zap50k-data/           # existing
├── ut-zap50k-images/         # existing
└── README.md
```

---

## Open Questions

> [!IMPORTANT]
> **Dataset Scope — How many images to index?**
> The full UT-Zappos50K has ~50,000 images. Indexing all of them will take 15–45 minutes on first run and ~1–3GB disk space for embeddings. You have two options:
> - **Full catalog (50k images)** — best portfolio value, slow first build
> - **Subset (~5,000 images)** — fast demo, ~2 min build time
>
> **Recommendation:** Start with a 5k-image subset for fast iteration, then optionally switch to full catalog.

> [!IMPORTANT]
> **GPU availability?**
> Embedding extraction is much faster on GPU. Do you have a CUDA-capable GPU available? (Affects `requirements.txt` torch version and build time)

> [!NOTE]
> **Python environment** — Do you have Python already installed? Any preferred virtual environment tool? (venv, conda, etc.)

---

## Proposed Changes

### Component 1: Backend (Python / FastAPI)

#### [NEW] `backend/requirements.txt`
- `fastapi`, `uvicorn[standard]`
- `torch`, `torchvision` (CPU or CUDA)
- `faiss-cpu` (or `faiss-gpu`)
- `pillow`, `numpy`, `pandas`, `python-multipart`

#### [NEW] `backend/catalog.py`
Parses `meta-data.csv` and the `ut-zap50k-images/` directory tree to produce a unified catalog mapping: `{ cid: { image_path, category, subcategory, brand, gender, material } }`

#### [NEW] `backend/embedder.py`
- Loads `ResNet50` (pretrained on ImageNet) via `torchvision.models`
- Removes the final FC classification layer (keeps pooling output → **2048-dim vector**)
- Normalizes embeddings to unit L2 norm (enables cosine similarity via dot product in FAISS)
- Supports single image inference and batch inference

#### [NEW] `backend/indexer.py`
- Wraps FAISS `IndexFlatIP` (inner product = cosine on normalized vectors)
- `build(embeddings, metadata)` → saves `index.faiss` + `catalog.json`
- `load()` → loads persisted index
- `search(query_embedding, k=12)` → returns top-k results with similarity scores

#### [NEW] `backend/build_index.py`
One-time offline script:
1. Walk the `ut-zap50k-images/` tree to find all `.jpg` files
2. Match file paths to metadata via CID from the filename
3. Batch-process images through `embedder.py` (batch size 32)
4. Save `embeddings.npy`, build and save `index.faiss`, write `catalog.json`

#### [NEW] `backend/main.py`
FastAPI application with endpoints:
- `GET /health` — health check
- `POST /search` — accepts multipart image upload, returns top-K similar products
- `GET /images/{image_path}` — serves catalog images as static files
- `GET /stats` — returns catalog size, index stats

CORS configured for `localhost:5173` (Vite dev server).

---

### Component 2: Frontend (React + Vite)

#### [NEW] `frontend/` (Vite + React project)

**Design system:** Dark/luxury aesthetic with glassmorphism panels, deep navy-to-purple gradient background, gold accent colors. Typography: `Inter` from Google Fonts.

#### [NEW] `frontend/src/index.css`
Full design token system:
- CSS custom properties for colors, spacing, shadows
- Glassmorphism card styles
- Smooth transitions and hover micro-animations
- Responsive grid layout

#### [NEW] `frontend/src/App.jsx`
Main layout:
- Hero header with animated gradient title
- Image upload zone (drag & drop or click to upload)
- Query image preview panel
- Results section with similarity score bars

#### [NEW] `frontend/src/components/SearchBar.jsx`
- Drag-and-drop file drop zone with animated border
- File validation (accepts `.jpg`, `.jpeg`, `.png`, `.webp`)
- Preview of uploaded image
- "Find Similar Products" CTA button with loading state

#### [NEW] `frontend/src/components/ResultsGrid.jsx`
- Masonry-style or uniform grid layout for results
- Animated entrance (staggered fade-in)
- Filter chips for category/gender (bonus)

#### [NEW] `frontend/src/components/ProductCard.jsx`
- Product image (fetched from FastAPI `/images/` endpoint)
- Similarity score badge (e.g. "94% match")
- Category, subcategory, brand labels
- Subtle hover scale + glow effect

---

### Component 3: README & Documentation

#### [NEW] `README.md`
Full GitHub-ready README with:
- Project description and architecture diagram (ASCII)
- Dataset details
- Installation & setup guide
- How to build the index
- How to run backend + frontend
- API documentation
- Sample results screenshots
- Limitations & future improvements

---

## Implementation Phases

### Phase 1: Backend Core
1. Set up Python virtual environment and install dependencies
2. Implement `catalog.py` — parse dataset structure
3. Implement `embedder.py` — ResNet50 feature extraction
4. Implement `indexer.py` — FAISS index management
5. Implement `build_index.py` — offline catalog indexing
6. **Run index build** (this is the critical step)
7. Implement `main.py` — FastAPI search API
8. Test API with curl/httpie

### Phase 2: Frontend
1. Scaffold Vite + React project
2. Build design system in `index.css`
3. Build `SearchBar` with drag-and-drop
4. Build `ProductCard` and `ResultsGrid`
5. Wire up API calls
6. Polish animations and responsive layout

### Phase 3: Integration & Polish
1. End-to-end test (upload → results)
2. Add loading states and error handling
3. Write `README.md`
4. Capture screenshots/demo

---

## Verification Plan

### Automated Checks
```bash
# Backend health
curl http://localhost:8000/health

# Test search with a sample image
curl -X POST http://localhost:8000/search \
  -F "file=@sample.jpg" \
  -F "top_k=12"
```

### Manual Verification
1. Upload a boot image → verify top results are all boots
2. Upload a sandal image → verify top results are sandals
3. Check similarity scores are in (0, 1] range
4. Verify product cards show correct metadata (category, brand, gender)
5. Verify drag-and-drop works on the React frontend
6. Test responsive layout on different window sizes
