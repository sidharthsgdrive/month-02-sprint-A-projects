import logging
import os
import time
from contextlib import asynccontextmanager

import chromadb
import emoji
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

# Send application logs to Railway's container logs.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("emoji-search")

logger.info("Importing application and loading embedding model")
model_load_start = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Embedding model loaded in %.1f seconds", time.time() - model_load_start)

logger.info("Creating ChromaDB client and emoji collection")
client = chromadb.Client()
collection = client.create_collection(
    name="emojis",
    metadata={"hnsw:space": "cosine"},
)


def build_index():
    start = time.time()
    logger.info("Starting emoji index build")

    names, chars = [], []
    for char, data in emoji.EMOJI_DATA.items():
        if data.get("status") != emoji.STATUS["fully_qualified"]:
            continue
        names.append(data["en"].strip(":").replace("_", " "))
        chars.append(char)

    logger.info(
        "Collected %d fully-qualified emojis in %.1f seconds",
        len(names),
        time.time() - start,
    )

    if not names:
        logger.warning("No emojis found; skipping index creation")
        return

    logger.info("Generating embeddings (batch_size=32)")
    embeddings = model.encode(
        names,
        show_progress_bar=False,
        batch_size=32,
    ).tolist()
    logger.info(
        "Generated embeddings for %d emojis in %.1f seconds",
        len(names),
        time.time() - start,
    )

    logger.info("Adding emoji documents and embeddings to ChromaDB")
    collection.add(
        ids=[str(i) for i in range(len(names))],
        documents=names,
        embeddings=embeddings,
        metadatas=[{"emoji": char} for char in chars],
    )
    logger.info(
        "Emoji index build completed: %d records in %.1f seconds",
        collection.count(),
        time.time() - start,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI lifespan startup: beginning index build")
    try:
        build_index()
        logger.info("FastAPI lifespan startup: index build complete")
        yield
    except Exception:
        logger.exception("Application startup or request lifespan failed")
        raise
    finally:
        logger.info("FastAPI lifespan shutdown")


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
    logger.info("Search request received: q=%r top_k=%d", q, top_k)
    try:
        query_embedding = model.encode(q).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(top_k, 1), max(collection.count(), 1)),
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

