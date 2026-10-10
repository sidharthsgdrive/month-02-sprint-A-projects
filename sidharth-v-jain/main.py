import logging
import os
import time
from contextlib import asynccontextmanager

import chromadb
import emoji
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("emoji-search")

logger.info("Loading sentence-transformer model")
model_start = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Model loaded in %.1f seconds", time.time() - model_start)

logger.info("Creating ChromaDB collection")
client = chromadb.Client()
collection = client.create_collection(
    name="emojis",
    metadata={"hnsw:space": "cosine"},
)


def build_index():
    start = time.time()
    logger.info("Starting emoji index build")

    names = []
    chars = []
    for char, data in emoji.EMOJI_DATA.items():
        if data.get("status") != emoji.STATUS["fully_qualified"]:
            continue
        names.append(data["en"].strip(":").replace("_", " "))
        chars.append(char)

    logger.info("Collected %d emojis in %.1f seconds", len(names), time.time() - start)

    if not names:
        logger.warning("No emojis found; skipping index build")
        return

    logger.info("Generating embeddings with batch_size=32")
    embeddings = model.encode(
        names,
        show_progress_bar=False,
        batch_size=32,
    ).tolist()
    logger.info("Embeddings generated in %.1f seconds", time.time() - start)

    logger.info("Adding records to ChromaDB")
    collection.add(
        ids=[str(i) for i in range(len(names))],
        documents=names,
        embeddings=embeddings,
        metadatas=[{"emoji": char} for char in chars],
    )
    logger.info(
        "Index build complete: %d records in %.1f seconds",
        collection.count(),
        time.time() - start,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: beginning index build")
    try:
        build_index()
        logger.info("Startup: index build finished")
        yield
    except Exception:
        logger.exception("Application lifespan failed")
        raise
    finally:
        logger.info("Application shutting down")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "emojis_indexed": collection.count()}


@app.get("/search")
def search(q: str, top_k: int = 10):
    logger.info("Search request: q=%r top_k=%d", q, top_k)

    try:
        query_embedding = model.encode(q).tolist()
        count = collection.count()

        if count == 0:
            logger.warning("Search requested but emoji index is empty")
            return []

        safe_top_k = min(max(top_k, 1), count)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_top_k,
        )

        response = [
            {
                "emoji": metadata["emoji"],
                "name": document,
                "score": round(1 - distance, 3),
            }
            for metadata, document, distance in zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0],
            )
        ]

        logger.info("Search completed: returned %d results", len(response))
        return response

    except Exception:
        logger.exception("Search request failed")
        raise
