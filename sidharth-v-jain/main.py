from contextlib import asynccontextmanager

import chromadb
import emoji
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection(name="emojis", metadata={"hnsw:space": "cosine"})


def build_index():
    names, chars = [], []
    for char, data in emoji.EMOJI_DATA.items():
        if data.get("status") != emoji.STATUS["fully_qualified"]:
            continue
        names.append(data["en"].strip(":").replace("_", " "))
        chars.append(char)

    embeddings = model.encode(names, show_progress_bar=False).tolist()
    collection.add(
        ids=[str(i) for i in range(len(names))],
        documents=names,
        embeddings=embeddings,
        metadatas=[{"emoji": char} for char in chars],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()
    yield


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
    query_embedding = model.encode(q).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    return [
        {"emoji": metadata["emoji"], "name": document, "score": round(1 - distance, 3)}
        for metadata, document, distance in zip(
            results["metadatas"][0], results["documents"][0], results["distances"][0]
        )
    ]
