"""
FastAPI RAG server for Dubai Building Code.
Usage: uvicorn api:app --reload --port 8000
"""

import pickle
import re
import textwrap
from pathlib import Path

import faiss
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

INDEX_PATH    = Path(__file__).parent / "index.faiss"
META_PATH     = Path(__file__).parent / "index_meta.pkl"
OLLAMA_URL    = "http://localhost:11434/api/chat"
EMBED_URL     = "http://localhost:11434/api/embed"
EMBED_MODEL   = "nomic-embed-text"
OLLAMA_MODEL  = "gemma4:e4b"
TOP_K         = 5
MIN_SCORE     = 0.45   # nomic-embed-text with search_query/search_document prefixes

app = FastAPI(title="Dubai Building Code RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_resources():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("Index not found. Run build_index.py first.")

    app.state.index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        app.state.meta = pickle.load(f)
    app.state.id_to_index = {c["id"]: i for i, c in enumerate(app.state.meta)}

    tokenized = [re.findall(r"\w+", c["text"].lower()) for c in app.state.meta]
    app.state.bm25 = BM25Okapi(tokenized)
    print(f"Loaded {app.state.index.ntotal} vectors + BM25 index. Ready.")


def embed_query(text: str) -> np.ndarray:
    try:
        r = requests.post(
            EMBED_URL,
            json={"model": EMBED_MODEL, "input": f"search_query: {text}"},
            timeout=60,
        )
        r.raise_for_status()
        vec = np.array(r.json()["embeddings"][0], dtype="float32")
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama embeddings service is not running. Start it with: ollama serve",
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {e}")

    norm = np.linalg.norm(vec)
    if norm <= 1e-9:
        raise HTTPException(status_code=500, detail="Embedding model returned a zero vector.")
    vec = vec / norm
    return vec.reshape(1, -1)


def retrieve(question: str, k: int = TOP_K):
    k = max(1, min(int(k), len(app.state.meta)))

    # Vector search casts a wider net so BM25 can rerank the candidate set.
    vec = embed_query(question)
    search_k = min(k * 6, len(app.state.meta))
    v_scores, v_indices = app.state.index.search(vec, search_k)
    vector_scores: dict[int, float] = {
        int(idx): float(score)
        for score, idx in zip(v_scores[0], v_indices[0])
        if int(idx) >= 0
    }

    # BM25 on the raw question; BM25Okapi already handles IDF internally.
    tokens    = re.findall(r"\w+", question.lower())
    bm25_raw  = app.state.bm25.get_scores(tokens)
    bm25_norm = bm25_raw / (float(bm25_raw.max()) or 1.0)

    # Union of vector candidates and BM25 leaders, then filter by a quality gate.
    bm25_top = set(int(i) for i in np.argsort(bm25_norm)[::-1][:k * 6])
    candidates = set(vector_scores) | bm25_top

    fused: dict[int, float] = {}
    for i in candidates:
        v = vector_scores.get(i, 0.0)
        b = float(bm25_norm[i])
        if v < MIN_SCORE and b < 0.4:
            continue
        fused[i] = 0.65 * v + 0.35 * b

    top_idx = sorted(fused, key=fused.get, reverse=True)[:k]

    # Include the next chunk when it lands on the same or adjacent page.
    seen, extra = set(top_idx), []
    for i in top_idx:
        nxt = i + 1
        if nxt < len(app.state.meta) and nxt not in seen:
            if app.state.meta[nxt]["page_start"] <= app.state.meta[i]["page_end"] + 1:
                extra.append(nxt)
                seen.add(nxt)

    all_idx = sorted(set(top_idx) | set(extra), key=lambda i: (app.state.meta[i]["page_start"], i))
    return [{**app.state.meta[i], "score": round(fused.get(i, 0), 4)} for i in all_idx]


def strip_section_prefix(text: str, section: str | None) -> str:
    if not section:
        return text.strip()
    prefix = f"[{section}] "
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    return text.strip()


def merge_overlapping_text(left: str, right: str, max_overlap_words: int = 80) -> str:
    left_words = left.split()
    right_words = right.split()
    max_overlap = min(len(left_words), len(right_words), max_overlap_words)

    for size in range(max_overlap, 0, -1):
        if left_words[-size:] == right_words[:size]:
            return " ".join(left_words + right_words[size:])

    return f"{left} {right}".strip()


def merge_chunks_for_prompt(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []

    merged: list[dict] = []
    current = None

    for chunk in chunks:
        chunk_text = strip_section_prefix(chunk["text"], chunk.get("section"))
        if current and chunk.get("section") == current.get("section"):
            current["text"] = merge_overlapping_text(current["text"], chunk_text)
            current["page_start"] = min(current["page_start"], chunk.get("page_start") or current["page_start"])
            current["page_end"] = max(current["page_end"], chunk.get("page_end") or current["page_end"])
            current["source_count"] += 1
            continue

        if current:
            merged.append(current)

        current = {
            "section": chunk.get("section"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "text": chunk_text,
            "source_count": 1,
        }

    if current:
        merged.append(current)

    return merged


def expand_chunks_for_prompt(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []

    expanded_idx: set[int] = set()

    for chunk in chunks:
        idx = app.state.id_to_index.get(chunk["id"])
        if idx is None:
            continue

        section = app.state.meta[idx].get("section")
        expanded_idx.add(idx)
        if not section:
            continue

        prev_idx = idx - 1
        while prev_idx >= 0 and app.state.meta[prev_idx].get("section") == section:
            expanded_idx.add(prev_idx)
            prev_idx -= 1

        next_idx = idx + 1
        while next_idx < len(app.state.meta) and app.state.meta[next_idx].get("section") == section:
            expanded_idx.add(next_idx)
            next_idx += 1

    ordered_idx = sorted(expanded_idx, key=lambda i: (app.state.meta[i]["page_start"], i))
    return [{**app.state.meta[i], "score": 0.0} for i in ordered_idx]


def build_prompt(question: str, chunks: list) -> str:
    if not chunks:
        return (
            f"Question: {question}\n"
            "Answer: The Dubai Building Code sections retrieved are not relevant enough "
            "to answer this question accurately. Please rephrase or ask about a specific section."
        )

    prompt_chunks = expand_chunks_for_prompt(chunks)
    merged_chunks = merge_chunks_for_prompt(prompt_chunks)
    context_parts = []
    for i, c in enumerate(merged_chunks, 1):
        label = f"Section {c['section']}" if c["section"] else f"Pages {c['page_start']}-{c['page_end']}"
        context_parts.append(f"[{i}] {label}\n{c['text'].strip()}")

    context = "\n\n".join(context_parts)
    return textwrap.dedent(f"""
        You are an expert on the Dubai Building Code.
        The excerpts below are from the PDF and may be fragmented due to multi-column or table formatting - read every line carefully.
        Instructions:
        - Start with a short direct answer when possible.
        - Then present the supporting requirements as short bullet points.
        - If the excerpts cover different sections or rule sets, keep them separate.
        - Extract and present ALL relevant requirements found across ALL excerpts, including every row of any table.
        - If you see a table with rows like "Studio: 1 bay", "1 Bedroom: 1 bay" etc., list every row.
        - Use ONLY information explicitly stated in the excerpts.
        - Do NOT invent numbers, thresholds, or requirements not present in the text.
        - If the excerpts do not answer the question, say "This specific requirement is not covered in the retrieved sections."
        - Cite the section number or page at the end of every bullet or statement.
        - Prefer plain language, but preserve technical wording when it matters for compliance.

        --- Dubai Building Code Excerpts ---
        {context}
        ------------------------------------

        Question: {question}
        Answer:
    """).strip()


def call_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":    OLLAMA_MODEL,
                "stream":   False,
                "think":    False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {
                    "temperature":    0.1,
                    "num_predict":    512,
                    "repeat_penalty": 1.3,
                },
            },
            timeout=120,
        )
        if not response.ok:
            detail = response.text.strip()
            raise HTTPException(
                status_code=502,
                detail=f"Ollama chat failed ({response.status_code}): {detail or response.reason}",
            )
        return response.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama is not running. Start it with: ollama serve")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=TOP_K, ge=1, le=50)


class SourceChunk(BaseModel):
    id: int
    section: str | None
    page_start: int | None
    page_end: int | None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve(req.question, k=req.top_k)

    if not chunks:
        return AskResponse(
            answer="No sufficiently relevant sections found in the Dubai Building Code for this question. Try rephrasing.",
            sources=[],
        )

    prompt = build_prompt(req.question, chunks)
    answer = call_ollama(prompt)

    return AskResponse(
        answer=answer,
        sources=[SourceChunk(**{k: v for k, v in c.items() if k != "text"}) for c in chunks],
    )


@app.get("/")
def ui():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "vectors": app.state.index.ntotal, "model": OLLAMA_MODEL}
