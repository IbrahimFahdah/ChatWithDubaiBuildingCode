"""
Run once to build the FAISS index from dubai_chunks.json.
Usage: python build_index.py
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import requests

CHUNKS_PATH  = Path(__file__).parent.parent / "dubai_chunks.json"
INDEX_PATH   = Path(__file__).parent / "index.faiss"
META_PATH    = Path(__file__).parent / "index_meta.pkl"
OLLAMA_URL   = "http://localhost:11434/api/embeddings"
EMBED_MODEL  = "nomic-embed-text"


def embed_texts(texts: list[str]) -> np.ndarray:
    vecs = []
    total = len(texts)
    for i, text in enumerate(texts):
        r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": f"search_document: {text}"})
        r.raise_for_status()
        vecs.append(r.json()["embedding"])
        if (i + 1) % 10 == 0 or (i + 1) == total:
            pct = (i + 1) * 100 // total
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"  [{bar}] {i+1}/{total} ({pct}%)", flush=True)
    return np.array(vecs, dtype="float32")


def main():
    print(f"Loading chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  {len(chunks)} chunks loaded.")

    print(f"Embedding with '{EMBED_MODEL}' via Ollama (this takes ~5 min)...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # L2-normalise for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-9, None)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"FAISS index saved: {INDEX_PATH}")

    meta = [
        {
            "id":         c["id"],
            "section":    c.get("section"),
            "page_start": c.get("page_start"),
            "page_end":   c.get("page_end"),
            "text":       c["text"],
        }
        for c in chunks
    ]
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"Metadata saved: {META_PATH}")
    print("Done. You can now start api.py.")


if __name__ == "__main__":
    main()
