"""
FastAPI RAG server — Dubai Building Code (cloud deployment).
LLM  : Groq API   (set GROQ_API_KEY env var)
Embed: Nomic API  (set NOMIC_API_KEY env var)
Usage: uvicorn api:app --host 0.0.0.0 --port $PORT

Retrieval pipeline:
  prose_pool  FAISS + BM25 hybrid → top PROSE_K
  fact_pool   FAISS + BM25 + section overlap + intent boost → top FACT_K
  late_fuse   deduplicate + per-section diversity cap        → LLM context
"""

import os
import pickle
import re
import textwrap
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────────────────────

INDEX_PATH = Path(__file__).parent / "index.faiss"
META_PATH  = Path(__file__).parent / "index_meta.pkl"

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

NOMIC_API_KEY   = os.environ["NOMIC_API_KEY"]
NOMIC_EMBED_URL = "https://api-atlas.nomic.ai/v1/embedding/text"

CANDIDATE_K     = 80   # FAISS candidates per pool
PROSE_K         = 3
FACT_K          = 3
MAX_PER_SECTION = 2    # diversity cap

_REQUIREMENT_TERMS = {
    "requirement", "requirements", "required", "require",
    "rule", "rules", "criteria", "standard", "standards",
    "minimum", "minimums",
}


# ── App startup ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("Index files not found. Copy index.faiss and index_meta.pkl into this folder.")

    app.state.index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        app.state.meta = pickle.load(f)
    app.state.id_to_index = {c["id"]: i for i, c in enumerate(app.state.meta)}

    prose_docs = [(i, c) for i, c in enumerate(app.state.meta) if c.get("chunk_type") == "text"]
    fact_docs  = [(i, c) for i, c in enumerate(app.state.meta) if c.get("chunk_type") == "fact"]

    app.state.prose_indices = [i for i, _ in prose_docs]
    app.state.fact_indices  = [i for i, _ in fact_docs]
    app.state.prose_bm25 = _build_bm25(prose_docs)
    app.state.fact_bm25  = _build_bm25(fact_docs)

    print(
        f"Loaded {app.state.index.ntotal} vectors "
        f"({len(app.state.prose_indices)} prose, {len(app.state.fact_indices)} fact). Ready."
    )
    yield


app = FastAPI(title="Dubai Building Code RAG API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_bm25(docs: list[tuple[int, dict]]) -> BM25Okapi | None:
    if not docs:
        return None
    return BM25Okapi([re.findall(r"\w+", c["text"].lower()) for _, c in docs])


def _vector_scores(vec: np.ndarray, k: int) -> dict[int, float]:
    scores, indices = app.state.index.search(vec, min(k, app.state.index.ntotal))
    return {int(idx): float(s) for s, idx in zip(scores[0], indices[0]) if idx >= 0}


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_query(text: str) -> np.ndarray:
    try:
        r = requests.post(
            NOMIC_EMBED_URL,
            headers={"Authorization": f"Bearer {NOMIC_API_KEY}"},
            json={
                "model":     "nomic-embed-text-v1",
                "texts":     [f"search_query: {text}"],
                "task_type": "search_query",
            },
            timeout=30,
        )
        r.raise_for_status()
        vec = np.array(r.json()["embeddings"][0], dtype="float32")
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, f"Embedding request failed: {e}")

    norm = np.linalg.norm(vec)
    if norm <= 1e-9:
        raise HTTPException(500, "Embedding returned a zero vector.")
    return (vec / norm).reshape(1, -1)


# ── Query expansion ───────────────────────────────────────────────────────────

def _expand_tokens(question: str) -> list[str]:
    tokens = re.findall(r"\w+", question.lower())
    expanded = list(tokens)
    for t in tokens:
        if t.endswith("ies") and len(t) > 4:
            expanded.append(f"{t[:-3]}y")
        elif t.endswith("s") and len(t) > 3:
            expanded.append(t[:-1])
    if set(tokens) & _REQUIREMENT_TERMS:
        expanded.extend([
            "shall", "required", "minimum", "not", "less", "than",
            "area", "dimension", "dimensions", "size", "sizes",
            "clearance", "height", "width",
        ])
    return expanded


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _fact_boost(q_tokens: set[str], chunk: dict) -> float:
    """Intent-based score adjustment for fact chunks."""
    if not (q_tokens & _REQUIREMENT_TERMS):
        return 0.0
    haystack = f"{(chunk.get('section') or '')} {chunk['text']}".lower()
    positive = [
        "shall", "minimum", "not less than", "required", "requirement",
        "requirements", "conform", "provided", "clearance", "dimension",
        "area", "height", "width",
    ]
    boost = min(0.35, 0.07 * sum(1 for s in positive if s in haystack))
    if "minimum" in (chunk.get("section") or "").lower():
        boost += 0.12
    # Penalise occupancy-load tables when the query is about physical dimensions
    negative = {"population", "occupancy", "rate", "estimation"}
    if not (q_tokens & negative) and any(t in haystack for t in negative):
        boost -= 0.30
    return boost


def retrieve_prose(question: str, vec: np.ndarray) -> list[dict]:
    if not app.state.prose_indices or not app.state.prose_bm25:
        return []

    v_scores = _vector_scores(vec, CANDIDATE_K * 2)
    tokens   = _expand_tokens(question)
    bm25_raw = app.state.prose_bm25.get_scores(tokens)
    bm25_max = float(bm25_raw.max()) or 1.0

    fused: dict[int, float] = {}
    for pos, meta_idx in enumerate(app.state.prose_indices):
        v = v_scores.get(meta_idx, 0.0)
        b = float(bm25_raw[pos]) / bm25_max
        if v < 0.40 and b < 0.30:
            continue
        fused[meta_idx] = 0.65 * v + 0.35 * b

    top = sorted(fused, key=fused.get, reverse=True)[:PROSE_K]
    return [{"score": round(fused[i], 4), **app.state.meta[i]} for i in top]


def retrieve_facts(question: str, vec: np.ndarray) -> list[dict]:
    if not app.state.fact_indices or not app.state.fact_bm25:
        return []

    v_scores = _vector_scores(vec, CANDIDATE_K * 2)
    tokens   = _expand_tokens(question)
    bm25_raw = app.state.fact_bm25.get_scores(tokens)
    bm25_max = float(bm25_raw.max()) or 1.0
    q_tokens = set(tokens)

    fused: dict[int, float] = {}
    for pos, meta_idx in enumerate(app.state.fact_indices):
        v = v_scores.get(meta_idx, 0.0)
        b = float(bm25_raw[pos]) / bm25_max
        if v < 0.40 and b < 0.20:
            continue
        section       = (app.state.meta[meta_idx].get("section") or "").lower()
        section_bonus = min(0.15, sum(1 for t in q_tokens if len(t) > 3 and t in section) * 0.05)
        fused[meta_idx] = (
            0.60 * v + 0.25 * b + section_bonus
            + _fact_boost(q_tokens, app.state.meta[meta_idx])
        )

    top = sorted(fused, key=fused.get, reverse=True)[:FACT_K]
    return [{"score": round(fused[i], 4), **app.state.meta[i]} for i in top]


def late_fuse(prose: list[dict], facts: list[dict]) -> list[dict]:
    seen: set[int] = set()
    section_count: dict[str, int] = {}
    final: list[dict] = []
    for chunk in facts + prose:   # facts first — they carry the precise values
        cid     = chunk.get("id")
        section = chunk.get("section") or ""
        if cid in seen or section_count.get(section, 0) >= MAX_PER_SECTION:
            continue
        seen.add(cid)
        section_count[section] = section_count.get(section, 0) + 1
        final.append(chunk)
    return final


def retrieve(question: str) -> list[dict]:
    vec = embed_query(question)
    return late_fuse(retrieve_prose(question, vec), retrieve_facts(question, vec))


# ── Prompt construction ───────────────────────────────────────────────────────

def _strip_section_prefix(text: str, section: str | None) -> str:
    prefix = f"[{section}] "
    return text[len(prefix):].strip() if section and text.startswith(prefix) else text.strip()


def _merge_overlapping(left: str, right: str, max_overlap: int = 80) -> str:
    lw, rw = left.split(), right.split()
    n = min(len(lw), len(rw), max_overlap)
    for size in range(n, 0, -1):
        if lw[-size:] == rw[:size]:
            return " ".join(lw + rw[size:])
    return f"{left} {right}".strip()


def _expand_to_section(chunks: list[dict]) -> list[dict]:
    """Expand each retrieved chunk to all siblings in the same section."""
    expanded: set[int] = set()
    for chunk in chunks:
        idx = app.state.id_to_index.get(chunk["id"])
        if idx is None:
            continue
        section = app.state.meta[idx].get("section")
        expanded.add(idx)
        if not section:
            continue
        i = idx - 1
        while i >= 0 and app.state.meta[i].get("section") == section:
            expanded.add(i); i -= 1
        i = idx + 1
        while i < len(app.state.meta) and app.state.meta[i].get("section") == section:
            expanded.add(i); i += 1
    ordered = sorted(expanded, key=lambda i: (app.state.meta[i]["page_start"], i))
    return [{**app.state.meta[i], "score": 0.0} for i in ordered]


def _merge_for_prompt(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []
    merged, current = [], None
    for chunk in chunks:
        text = _strip_section_prefix(chunk["text"], chunk.get("section"))
        if current and chunk.get("section") == current["section"]:
            current["text"]       = _merge_overlapping(current["text"], text)
            current["page_start"] = min(current["page_start"], chunk.get("page_start") or current["page_start"])
            current["page_end"]   = max(current["page_end"],   chunk.get("page_end")   or current["page_end"])
            continue
        if current:
            merged.append(current)
        current = {
            "section":    chunk.get("section"),
            "page_start": chunk.get("page_start"),
            "page_end":   chunk.get("page_end"),
            "text":       text,
        }
    if current:
        merged.append(current)
    return merged


def build_prompt(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return (
            f"Question: {question}\n"
            "Answer: The retrieved sections are not relevant enough to answer this question accurately."
        )
    merged = _merge_for_prompt(_expand_to_section(chunks))
    context_parts = []
    for i, c in enumerate(merged, 1):
        label = f"Section {c['section']}" if c["section"] else f"Pages {c['page_start']}-{c['page_end']}"
        context_parts.append(f"[{i}] {label}\n{c['text'].strip()}")
    context = "\n\n".join(context_parts)
    return textwrap.dedent(f"""
        You are an expert on the Dubai Building Code.
        The excerpts below are from the PDF and may be fragmented due to multi-column or table formatting — read every line carefully.

        Instructions:
        - Start with a short direct answer when possible.
        - Present supporting requirements as short bullet points.
        - If excerpts cover different sections, keep them separate.
        - Extract ALL relevant requirements across ALL excerpts, including every table row.
        - Use ONLY information explicitly stated in the excerpts.
        - Do NOT invent numbers, thresholds, or requirements not present in the text.
        - If the excerpts do not answer the question, say "This specific requirement is not covered in the retrieved sections."
        - Cite the section number or page at the end of every bullet or statement.
        - Preserve technical wording when it matters for compliance.

        --- Dubai Building Code Excerpts ---
        {context}
        ------------------------------------

        Question: {question}
        Answer:
    """).strip()


# ── LLM ──────────────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    try:
        r = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       GROQ_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens":  512,
            },
            timeout=60,
        )
        if not r.ok:
            raise HTTPException(502, f"Groq API error ({r.status_code}): {r.text.strip()}")
        return r.json()["choices"][0]["message"]["content"].strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Schemas & routes ──────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    id: int
    section: str | None
    page_start: int | None
    page_end: int | None
    score: float
    chunk_type: str | None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    chunks = retrieve(req.question)
    if not chunks:
        return AskResponse(
            answer="No sufficiently relevant sections found for this question. Try rephrasing.",
            sources=[],
        )
    answer = call_llm(build_prompt(req.question, chunks))
    return AskResponse(
        answer=answer,
        sources=[
            SourceChunk(**{k: v for k, v in c.items() if k in SourceChunk.model_fields})
            for c in chunks
        ],
    )


@app.get("/")
def ui():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "vectors": app.state.index.ntotal, "model": GROQ_MODEL}
