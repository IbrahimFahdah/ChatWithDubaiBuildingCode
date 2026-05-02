# How It Works — Query Pipeline

This describes the full flow from the moment you submit a question to receiving an answer.

```
You type: "What is the minimum ceiling height?"
                    │
                    ▼
            1. EMBED THE QUERY
            nomic-embed-text converts your question
            into a 768-number vector
            (prefix: "search_query: ...")
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    2a. PROSE POOL        2b. FACT POOL
    Text paragraphs       Table rows rendered
    and descriptions      as natural language
          │                   │
    FAISS vector search   FAISS vector search
    + BM25 keyword        + BM25 keyword
    fused 65/35           fused 60/25
          │                   │
    Filter weak           Filter weak
    candidates            candidates
          │                   │
    FlashRank             Section title overlap
    cross-encoder         bonus + normative
    rerank → top 3        language boost → top 3
    (test only)           scores normalised 0–1
          │                   │
          └─────────┬─────────┘
                    ▼
            3. LATE FUSION
            Facts listed first (they carry exact values)
            Deduplicate by chunk ID
            Cap at 2 chunks per section (diversity)
            → up to 6 chunks total
                    │
                    ▼
            4. SECTION EXPANSION
            Each retrieved chunk is expanded to include
            all sibling chunks in the same section,
            ordered by page number
                    │
                    ▼
            5. MERGE OVERLAPPING TEXT
            Adjacent chunks from the same section
            are stitched together, deduplicating
            any overlapping words
                    │
                    ▼
            6. BUILD PROMPT
            Merged text blocks formatted with strict
            instructions: cite sections, bullet points,
            no invented numbers
                    │
                    ▼
            7. LLM INFERENCE
            Ollama → gemma4:e4b (test)
            Groq  → llama-3.3-70b-versatile (deployment)
            temperature 0.1, 512 tokens
                    │
                    ▼
            8. RESPONSE
            Answer text + sources
            (section, page range, score, chunk_type)
```

## Two Chunk Types

| Type | Source | Example |
|------|--------|---------|
| `text` | Prose paragraphs from the PDF | "Habitable rooms shall have a minimum ceiling height of..." |
| `fact` | One table row rendered as a sentence | "For residential living space: minimum area 10.5 m² and minimum dimension 3 m. [B.5.2, Table B.3]" |

Facts are retrieved through a separate pool with no cross-encoder reranking — the vector + BM25 + section overlap scoring is better suited to structured data than a prose-trained cross-encoder.

## Key Point

The LLM never "remembers" the building code. It only sees the excerpts retrieved
for that specific question, which means it cannot hallucinate rules that are not
present in those excerpts.

## Files Involved

| File | Role |
|------|------|
| `index.faiss` | 4,440 chunk vectors built via Colab (nomic-embed-text-v1) |
| `index_meta.pkl` | Chunk text + metadata (section, page, chunk_type, table, subject) |
| `api.py` | Runs the full pipeline on every `/ask` request |

## Important: Embedding Model Must Match

`index.faiss` was built using **nomic-embed-text-v1**. Queries must be embedded
with the same model — the local Ollama `nomic-embed-text` and the Nomic API
`nomic-embed-text-v1` produce compatible vectors.
