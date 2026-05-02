"""
build_chunks.py  —  PDF -> pymupdf4llm Markdown -> text chunks + fact chunks -> dubai_chunks.json

Replaces indexing/extract_chunks.py. Key differences:
  - Uses pymupdf4llm so section headings appear BEFORE their table content (correct order)
  - Tables are parsed as structured rows — no garbled flat text
  - Produces two chunk types per section:
      text  — sliding-window prose chunks (same strategy as before)
      fact  — one retrieval-optimized sentence per table row
  - No retro-caption heuristic needed; reading order is already correct in the markdown

Output: indexing_v2/dubai_chunks.json  (fed into the existing Colab embedding notebook)
"""

import json
import re
import sys
from pathlib import Path

import pymupdf4llm

PDF_PATH      = Path(__file__).parent.parent / "indexing" / "Dubai Building Code_English_2021 Edition.pdf"
OUT_PATH      = Path(__file__).parent / "dubai_chunks.json"
TARGET_WORDS  = 200
OVERLAP_WORDS = 40

if OVERLAP_WORDS >= TARGET_WORDS:
    raise ValueError(f"OVERLAP_WORDS ({OVERLAP_WORDS}) must be < TARGET_WORDS ({TARGET_WORDS})")

# ── Patterns ──────────────────────────────────────────────────────────────────

# Section codes: B.5.2, K.5.3.1, etc.  Title must start with a capital letter.
_SECTION_RE = re.compile(
    r"^(?P<code>[A-K]\.\d{1,2}(?:\.\d{1,2}){0,4})"
    r"\s+(?P<title>[A-Z][a-zA-Z].{0,80}?)$"
)

_TABLE_ROW_RE = re.compile(r"^\|.+\|$")           # any | … | line
_TABLE_SEP_RE = re.compile(r"^\|[-:\s|]+\|$")     # |---|---| separator
_TABLE_CAP_RE = re.compile(r"^Table\s+[A-K]\.\d+\b", re.IGNORECASE)

# Lines that carry no content value
_NOISE_RE = re.compile(
    r"^Dubai Building Code\s*$"
    r"|^Part\s+[A-K]:\s+\w"         # "Part B: Architecture"
    r"|^[A-K]\s+\d+\s*$"            # "B 37"  (section page labels)
    r"|^[*\s]*==>.+<==\s*\**$"      # "**==> picture … <==**"
    r"|^-{5,}",                      # "-----" picture block dividers
    re.IGNORECASE,
)
_PIC_START_RE = re.compile(r"Start of picture text", re.IGNORECASE)
_PIC_END_RE   = re.compile(r"End of picture text",   re.IGNORECASE)

# ── Table type detection ──────────────────────────────────────────────────────
_REQ_TABLE_RE = re.compile(
    r"\bminimum\b|\bmaximum\b|\brequirement[s]?\b|\bspecification[s]?\b"
    r"|\bclearance[s]?\b|\bwidth[s]?\b|\bheight[s]?\b|\barea\b"
    r"|\bdimension[s]?\b|\bparking\b|\bratio[s]?\b|\bsetback\b",
    re.IGNORECASE,
)
_CLASS_TABLE_RE = re.compile(
    r"\bclassif\w+\b|\bcategor\w+\b|\bgrade[s]?\b",
    re.IGNORECASE,
)
# Cell value is self-contained normative/reference text — use as-is
_SELF_CONTAINED_RE = re.compile(
    r"^shall\b|^must\b|^refer to\b|^see \b|^in accordance\b|^per \b",
    re.IGNORECASE,
)

# Mojibake: UTF-8 sequences misread as cp1252, ordered longest-match first
_MOJIBAKE = [
    ("â€”", "–"),   # â€" -> en dash
    ("â€“", "—"),   # â€" -> em dash
    ("â€™", "’"),   # â€™ -> right single quote
    ("â€˜", "‘"),   # â€˜ -> left single quote
    ("â€œ", "“"),   # â€œ -> left double quote
    ("â€",        "”"),   # â€  -> right double quote
    ("Â ",        " "),        # Â\xa0 -> non-breaking space
]


# ── Text helpers ──────────────────────────────────────────────────────────────

def fix_mojibake(text: str) -> str:
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    return text


def detect_section(heading_line: str) -> str | None:
    """
    Return 'CODE first-5-title-words' if the ## heading is a real section code
    (e.g. B.5.2, K.5.3.1).  Returns None for headings like '## Key'.
    """
    text = heading_line.lstrip("#").strip()
    text = re.sub(r"\*+", "", text).strip()       # strip **bold**
    m = _SECTION_RE.match(text)
    if not m:
        return None
    title_words = m.group("title").split()[:5]
    return f"{m.group('code')} {' '.join(title_words)}"


def clean_cell(raw: str) -> str:
    """Normalise a markdown table cell."""
    raw = fix_mojibake(raw)
    raw = raw.replace("<br>", " ")
    raw = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", raw)   # **bold**
    raw = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw)    # [text](url)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def is_empty_row(cells: list[str]) -> bool:
    return all(c == "" for c in cells)


# ── Table parser ──────────────────────────────────────────────────────────────

def parse_table(lines: list[str], start: int) -> tuple[list[str], list[list[str]], int]:
    """
    Parse a markdown table beginning at lines[start].
    Returns (headers, data_rows, next_line_idx).

    Handles two split-row patterns common in PDF-converted tables:
      1. Inherited first cell — a row whose first column is empty inherits
         the first-column value from the row above (merged cells in the PDF).
         e.g.  | Residential | Accommodation | 5.0 |
               |             | Labour accom  | 3.7 |
      2. Continuation row — a row with fewer columns than the header is
         appended to the previous row's first cell.
         e.g.  | Residential living space |
               | (bedroom, living room)   | 10.5 | 3 |
    """
    headers: list[str] = []
    data_rows: list[list[str]] = []
    header_done     = False
    last_first_cell = ""
    i = start

    while i < len(lines):
        line = lines[i].strip()
        if not _TABLE_ROW_RE.match(line):
            break

        cells = [clean_cell(c) for c in line[1:-1].split("|")]

        if _TABLE_SEP_RE.match(line):
            header_done = True
            i += 1
            continue

        if is_empty_row(cells):
            i += 1
            continue

        if not header_done:
            headers = cells
        else:
            # Pattern 2: continuation row (fewer columns than header)
            if headers and 0 < len(cells) < len(headers) and data_rows:
                prev = data_rows[-1]
                merged_first = f"{prev[0]} {cells[0]}".strip() if cells[0] else prev[0]
                data_rows[-1] = [merged_first] + prev[1:]
                i += 1
                continue

            # Pattern 1: inherited first cell (empty first column = merged cell above)
            if cells[0] == "" and last_first_cell:
                cells[0] = last_first_cell

            if cells[0]:
                last_first_cell = cells[0]

            data_rows.append(cells)

        i += 1

    return headers, data_rows, i


def find_table_caption(lines: list[str], table_start: int, table_end: int) -> str:
    """
    Search for a 'Table X.Y …' caption in the 5 lines before the table
    and the 5 lines after it.  Before takes priority (some PDFs place
    captions above the table).
    """
    window = 5
    # Check before
    for j in range(max(0, table_start - window), table_start):
        cap = lines[j].strip()
        if _TABLE_CAP_RE.match(cap):
            return cap
    # Check after
    for j in range(table_end, min(table_end + window, len(lines))):
        cap = lines[j].strip()
        if _TABLE_CAP_RE.match(cap):
            return cap
    return ""


# ── Fact + text chunk builders ────────────────────────────────────────────────

def _table_type(section: str | None, table_name: str, headers: list[str]) -> str:
    """Classify a table as requirement / classification / parameter."""
    context = f"{section or ''} {table_name} {' '.join(headers)}"
    if _REQ_TABLE_RE.search(context):
        return "requirement"
    if _CLASS_TABLE_RE.search(context):
        return "classification"
    return "parameter"


def _join_attrs(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def make_fact(
    section: str | None,
    table_name: str,
    headers: list[str],
    row: list[str],
) -> dict | None:
    """
    Render one table row as a natural-language sentence and return a dict with:
      text        — the sentence used for embedding and cross-encoder scoring
      subject     — first-column value (entity being described)
      table_type  — requirement | classification | parameter

    Template per table type:
      requirement    For {subject}: {attr} {value} [and {attr} {value}].
      classification {subject} is classified as {value} [{attr}].
      parameter      For {subject}: {attr} = {value}.

    When a cell value is self-contained normative text ("Shall meet ...",
    "Refer to ...") it is used verbatim without the header prefix.
    """
    if not row:
        return None

    subject = row[0].strip()
    if not subject:
        return None

    # Attribute pairs: skip column 0 (that's the subject)
    attr_pairs: list[tuple[str, str]] = []
    for h, v in zip(headers[1:] if len(headers) > 1 else [], row[1:]):
        v = v.strip()
        if not v or v in ("-", "—", "–"):
            continue
        attr_pairs.append((h.strip(), v))

    if not attr_pairs and len(row) == 1:
        return None

    ttype    = _table_type(section, table_name, headers)
    citation = (
        f"[{section}, {table_name}]" if section and table_name
        else f"[{section}]" if section
        else f"[{table_name}]" if table_name
        else ""
    )

    # Build attribute phrases
    phrases: list[str] = []
    for h, v in attr_pairs:
        if _SELF_CONTAINED_RE.match(v):
            # Value is already a complete predicate — use as-is, drop header
            phrases.append(v)
        elif ttype == "requirement":
            phrases.append(f"{h.lower()} {v}")
        elif ttype == "classification":
            phrases.append(f"{h}: {v}" if h else v)
        else:
            phrases.append(f"{h} = {v}" if h else v)

    # Compose sentence
    if not phrases:
        text = f"{subject}. {citation}".strip()
    elif ttype == "requirement":
        text = f"For {subject}: {_join_attrs(phrases)}. {citation}".strip()
    elif ttype == "classification":
        text = f"{subject} is classified as {_join_attrs(phrases)}. {citation}".strip()
    else:
        text = f"For {subject}: {_join_attrs(phrases)}. {citation}".strip()

    return {"text": text, "subject": subject, "table_type": ttype}


def make_text_chunks(
    section: str | None,
    word_buffer: list[tuple[str, int]],
    start_id: int,
) -> list[dict]:
    """Sliding-window chunker over (word, page_num) pairs."""
    if not word_buffer:
        return []
    chunks = []
    cid    = start_id
    prefix = f"[{section}] " if section else ""
    i = 0
    while i < len(word_buffer):
        end    = min(i + TARGET_WORDS, len(word_buffer))
        window = word_buffer[i:end]
        chunks.append({
            "id":         cid,
            "text":       prefix + " ".join(w for w, _ in window),
            "page_start": window[0][1],
            "page_end":   window[-1][1],
            "section":    section,
            "chunk_type": "text",
        })
        cid += 1
        if end == len(word_buffer):
            break
        i += TARGET_WORDS - OVERLAP_WORDS
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading {PDF_PATH} ...")
    if not PDF_PATH.exists():
        print(f"ERROR: {PDF_PATH} not found.")
        sys.exit(1)

    pages = pymupdf4llm.to_markdown(str(PDF_PATH), page_chunks=True)
    print(f"  {len(pages)} pages")

    all_chunks: list[dict]          = []
    chunk_id                        = 1
    current_section: str | None     = None
    text_buffer: list[tuple[str, int]] = []

    def flush_text():
        nonlocal chunk_id, text_buffer
        new = make_text_chunks(current_section, text_buffer, chunk_id)
        all_chunks.extend(new)
        chunk_id += len(new)
        text_buffer = []

    for page in pages:
        page_num   = page["metadata"]["page_number"]
        lines      = page["text"].splitlines()
        in_picture = False
        i = 0

        while i < len(lines):
            line     = lines[i]
            stripped = line.strip()

            # ── Picture blocks ─────────────────────────────────────────────
            if _PIC_START_RE.search(stripped):
                in_picture = True
                i += 1
                continue
            if _PIC_END_RE.search(stripped):
                in_picture = False
                i += 1
                continue
            if in_picture:
                i += 1
                continue

            # ── Noise / blank ──────────────────────────────────────────────
            if not stripped or _NOISE_RE.match(stripped):
                i += 1
                continue

            # ── Section heading (## …) ─────────────────────────────────────
            if stripped.startswith("#"):
                sec = detect_section(stripped)
                if sec:
                    flush_text()
                    current_section = sec
                i += 1
                continue

            # ── Table ──────────────────────────────────────────────────────
            if _TABLE_ROW_RE.match(stripped):
                flush_text()
                table_start          = i
                headers, rows, next_i = parse_table(lines, i)
                table_name           = find_table_caption(lines, table_start, next_i)

                for row in rows:
                    result = make_fact(current_section, table_name, headers, row)
                    if not result:
                        continue
                    all_chunks.append({
                        "id":         chunk_id,
                        "text":       result["text"],
                        "page_start": page_num,
                        "page_end":   page_num,
                        "section":    current_section,
                        "chunk_type": "fact",
                        "table":      table_name,
                        "subject":    result["subject"],
                        "table_type": result["table_type"],
                    })
                    chunk_id += 1

                i = next_i
                continue

            # ── Prose ──────────────────────────────────────────────────────
            clean = fix_mojibake(stripped)
            for word in clean.split():
                text_buffer.append((word, page_num))
            i += 1

    flush_text()

    if not all_chunks:
        print("ERROR: no chunks produced. Verify the PDF path.")
        sys.exit(1)

    text_n = sum(1 for c in all_chunks if c.get("chunk_type") == "text")
    fact_n = sum(1 for c in all_chunks if c.get("chunk_type") == "fact")
    print(f"  {len(all_chunks):,} chunks  ({text_n:,} text  +  {fact_n:,} fact)")

    OUT_PATH.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
