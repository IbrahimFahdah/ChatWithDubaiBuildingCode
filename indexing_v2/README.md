# indexing_v2 — Why This Exists

## What was wrong with the original pipeline (`indexing/`)

The original pipeline used **PyPDF2** to extract text from the PDF and a single-pass
loop to assign each word to the last-detected section heading. This caused a silent but
critical retrieval failure.

### The specific bug

The Dubai Building Code PDF's layout places table data rows **above** the section
heading that governs them. On pages 56–57, the PDF reading order is:

```
[Table B.2 continuation — correctly B.5.1]
Residential living space (bedroom, living room)  10.5  3    ← Table B.3 rows
Residential studio                               21    3
...
Table B.3 Minimum room sizes                               ← caption
B.5.2 Minimum room sizes                                   ← heading detected here
The net area shall be not less than...
```

Because the `B.5.2 Minimum room sizes` heading was only detected **after** the table
rows had already been processed, every row of Table B.3 was stamped with the label
`B.5.1 Occupant loads` instead of `B.5.2 Minimum room sizes`.

### Impact on retrieval

When a user asked *"what are the requirements for bedrooms?"*, the system returned
window glazing rules, lux levels, and bathroom ratios — but never:

- Minimum net area: **10.5 m²**
- Minimum dimensions: **3 m** (length and width)
- Minimum clear height: **2.7 m**

These are the most fundamental bedroom requirements in the code. They were in the index
but under the wrong section label, so the hybrid vector + BM25 retriever ranked them
below daylighting chunks and they fell outside the `TOP_K = 5` cutoff.

### Why not just fix `extract_chunks.py`?

A retro-caption heuristic was added to `extract_chunks.py` (see git history) to
retroactively relabel table rows when a table caption appeared close to an upcoming
section heading. It helped but was a patch, not a fix:

- PyPDF2 has no bounding box information, so table regions cannot be cleanly excluded
  from the text stream
- Table rows still appear garbled (flattened, no structure) in the text chunks
- A separate `extract_table_facts.py` would still be needed for structured table data
- Two scripts using two different libraries (PyPDF2 + pdfplumber) with no shared
  layout model is fragile

---

## Why pymupdf4llm

**pymupdf4llm** is a thin wrapper around PyMuPDF specifically designed for LLM
ingestion. Tested against pages 56–59 of this PDF, it produces:

```markdown
## B.5.2 Minimum room sizes          ← heading BEFORE table (correct order)

The net area and clear dimension...

| Occupancy/use                         | Minimum area (m2) | Min dimension (m) |
|---------------------------------------|-------------------|-------------------|
| Residential living space              |                   |                   |
| (bedroom, living room)                | 10.5              | 3                 |
| Residential studio                    | 21                | 3                 |
...

Table B.3 Minimum room sizes
```

Key improvements over PyPDF2:

| Problem | PyPDF2 | pymupdf4llm |
|---|---|---|
| Table rows before section heading | Mislabeled under wrong section | Heading always precedes content in markdown output |
| Table structure | Flattened to a single text stream | Clean `\| col \| col \|` rows |
| Section detection | Regex against garbled text | `##` markers from font size analysis |
| Retro-caption fix needed | Yes | No |
| Separate table extractor needed | Yes (pdfplumber) | No — same pass |

---

## What the new pipeline does

One script (`build_chunks.py`) replaces both `extract_chunks.py` and the proposed
`extract_table_facts.py`:

```
PDF
 └── pymupdf4llm (one pass)
      ├── Markdown with ## section headings and | table rows
      ├── Text chunks  — prose between tables, labeled by section
      └── Fact chunks  — one sentence per table row, labeled by section + table name

 → merge → dubai_chunks.json → embed → FAISS + BM25 index
```

### Fact chunk example

```
B.5.2 Minimum room sizes (Table B.3): Residential living space
(bedroom, living room) — minimum net area 10.5 m², minimum
dimension (length and width) 3 m.
```

This chunk contains every token a bedroom query hits: *bedroom*, *minimum*, *area*,
*dimension*, *10.5*, *3 m*, *B.5.2*, *Table B.3*. It will rank above daylighting
chunks for any bedroom sizing query.

---

## Files

| File | Purpose |
|---|---|
| `build_chunks.py` | Single script: PDF → text chunks + fact chunks → `dubai_chunks.json` |
| `dubai_chunks.json` | Output — input to the Colab embedding notebook |

The Colab embedding notebook (`indexing/dubai_build_index_colab.ipynb`) is unchanged
and still used to produce `index.faiss` and `index_meta.pkl`.
