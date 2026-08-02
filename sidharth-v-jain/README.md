# Emoji Semantic Search

## What it does

Type a description, and get back the 10 emojis whose meaning is closest to it. For example, search "food" and you'll also surface things like 🍕 or 🍨 even if the word "food" never appears in their official name.

I chose this topic out of a real frustration of mine - picking an emoji usually means scrolling through a picker or guessing the exact keyword the emoji is tagged with. Since emojis already have official descriptive names (e.g. "party popper", "face with tears of joy"), embedding those names and comparing them semantically to a query makes finding emojis way easier.

Frontend: https://semantic-emoji-picker.streamlit.app/
Backend: https://month-02-sprint-a-projects-production.up.railway.app/docs

## Tech Stack

- **FastAPI** — backend, exposes `GET /search`
- **ChromaDB** — stores and queries the emoji embeddings
- **sentence-transformers (`all-MiniLM-L6-v2`)** — turns emoji names and queries into embeddings
- **`emoji` (Python package)** — source of official emoji names
- **Streamlit** — frontend search box and results grid

## How to run locally

```bash
# backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# runs on http://localhost:8000

# frontend (separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
# runs on http://localhost:8501
```

## Environment variables

| Variable | Used by | Required | Description |
|---|---|---|---|
| `BACKEND_URL` | frontend | No (defaults to `http://localhost:8000`) | Base URL of the FastAPI backend. |
| `PORT` | backend | No (set automatically by Railway) | Port the FastAPI server binds to. |
