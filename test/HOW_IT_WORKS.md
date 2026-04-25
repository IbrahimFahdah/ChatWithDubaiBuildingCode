# How It Works — Query Pipeline

This describes the full flow from the moment you submit a question to receiving an answer.

```
You type: "What is the minimum ceiling height?"
                    │
                    ▼
            1. EMBED THE QUERY
            nomic-embed-text converts your question
            into a 768-number vector
            (search_query: "What is the minimum...")
                    │
                    ▼
            2. VECTOR SEARCH (FAISS)
            Your query vector is compared against
            all chunk vectors in index.faiss
            using cosine similarity → top 30 candidates
                    │
                    ▼
            3. BM25 KEYWORD SEARCH
            Simultaneously, your question words are
            matched against chunk text using BM25
            (like a smart keyword search) → top 30 candidates
                    │
                    ▼
            4. FUSION SCORING
            Both results are merged:
            score = 0.65 × vector_score + 0.35 × bm25_score
            Low-scoring chunks filtered out (< threshold)
            → top 5 chunks selected
                    │
                    ▼
            5. CHUNK EXPANSION
            For each top chunk, the full surrounding
            section is pulled in (all chunks with the
            same section label, e.g. "B.4.2.2 Building height")
                    │
                    ▼
            6. MERGE OVERLAPPING TEXT
            Adjacent chunks from the same section
            are stitched together, deduplicating
            any overlapping words
                    │
                    ▼
            7. BUILD PROMPT
            The merged text blocks are formatted
            into a structured prompt with instructions
            telling the LLM to cite sections and
            not invent numbers
                    │
                    ▼
            8. LLM INFERENCE (Ollama / gemma4:e4b)
            The prompt is sent to your local model
            which generates the answer
                    │
                    ▼
            9. RESPONSE
            Answer text + source list
            (section, page range, score) → back to you
```

## Key Point

The LLM never "remembers" the building code. It only sees the excerpts retrieved
for that specific question, which means it cannot hallucinate rules that are not
present in those excerpts.

## Files Involved

| File | Role |
|------|------|
| `index.faiss` | Stores all chunk vectors (built once by `build_index.py`) |
| `index_meta.pkl` | Stores chunk text + metadata (section, page) alongside each vector |
| `api.py` | Runs steps 1–9 on every `/ask` request |

## Important: Embedding Model Must Match

`index.faiss` was built using **nomic-embed-text**. Queries must be embedded with
the same model, otherwise the vectors are in different spaces and results will be
wrong. The model is pulled via `ollama pull nomic-embed-text`.
