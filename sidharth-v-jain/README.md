# Emoji Semantic Search

Search emojis by meaning instead of keyword, using `all-MiniLM-L6-v2` embeddings and ChromaDB.

- `backend/` — FastAPI app. Builds an in-memory ChromaDB collection from the official emoji names (via the `emoji` package) on startup, exposes `GET /search?q=...&top_k=10`.
- `frontend/` — Streamlit app with a search box that calls the backend and displays results in a grid.

## Run locally

```bash
# backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# frontend (separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

