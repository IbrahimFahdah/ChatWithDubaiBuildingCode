"""
Extract Dubai Building Code PDF -> dubai_chunks.json for rag_api/build_index.py

Chunking strategy: split on section boundaries first, then slide a word window
within each section. This guarantees no chunk ever straddles two sections, so
every chunk's prefix accurately reflects its content.
"""

import json
import re
import sys
from pathlib import Path
from PyPDF2 import PdfReader

PDF_PATH      = Path("Dubai Building Code_English_2021 Edition.pdf")
OUT_PATH      = Path("dubai_chunks.json")
TARGET_WORDS  = 200
OVERLAP_WORDS = 40

if OVERLAP_WORDS >= TARGET_WORDS:
    raise ValueError(f"OVERLAP_WORDS ({OVERLAP_WORDS}) must be less than TARGET_WORDS ({TARGET_WORDS})")

_SECTION_CODE_RE = re.compile(
    r"^\s*(?:\d+\s+)?"         # allow page/line-number prefixes before the heading
    r"(?P<code>"
    r"[A-K]\.\d{1,2}(?:\.\d{1,2}){0,4}"   # up to 5 levels: B.7.2.6.1
    r"|(?:Chapter|Section|Article|Appendix)\s+\d+"
    r")"
    r"\s+(?P<title>[A-Z][a-zA-Z].{0,60}?)\s*$",
)

# Splits cases like "heightB.4.2.2 Building height" → "height\nB.4.2.2 Building height"
_CONCAT_SECTION_RE = re.compile(r"([a-zA-Z])([A-K]\.\d{1,2}(?:\.\d{1,2}){0,4}\s)")


def clean_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = text.replace("\u00ad", "")
    return text.strip()


def detect_section(text: str) -> str | None:
    """Return '<code> <first-5-words-of-title>' or None."""
    # Normalise PDF artefact: codes concatenated to preceding text e.g. "heightB.4.2.2 Title"
    text = _CONCAT_SECTION_RE.sub(r"\1\n\2", text.strip())
    for line in text.splitlines():
        m = _SECTION_CODE_RE.match(line.strip())
        if m:
            title_words = m.group("title").split()[:5]
            return f"{m.group('code')} {' '.join(title_words)}"
    return None


def chunk_section(section_label: str | None,
                  sec_words: list[tuple[str, int]],
                  start_id: int) -> list[dict]:
    """Slide a word window over one section's words."""
    chunks = []
    chunk_id = start_id
    i = 0
    prefix = f"[{section_label}] " if section_label else ""

    while i < len(sec_words):
        end = min(i + TARGET_WORDS, len(sec_words))
        window = sec_words[i:end]
        text = prefix + " ".join(w for w, _ in window)
        chunks.append({
            "id":         chunk_id,
            "text":       text,
            "page_start": window[0][1],
            "page_end":   window[-1][1],
            "section":    section_label,
        })
        chunk_id += 1
        if end == len(sec_words):
            break
        i += TARGET_WORDS - OVERLAP_WORDS

    return chunks


def main():
    print(f"Reading {PDF_PATH} ...")
    reader = PdfReader(PDF_PATH)
    print(f"  {len(reader.pages)} pages")

    page_texts: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            page_texts.append((i, text))

    # Build word list with per-line section tracking
    words: list[tuple[str, int]] = []
    word_sections: list[str | None] = []
    current_section: str | None = None

    for page_num, text in page_texts:
        for line in text.splitlines():
            sec = detect_section(line)
            if sec:
                current_section = sec
            for word in line.split():
                words.append((word, page_num))
                word_sections.append(current_section)

    print(f"  Total words: {len(words):,}")

    # Group words into contiguous runs sharing the same section label
    sections: list[tuple[str | None, list[tuple[str, int]]]] = []
    if words:
        cur_sec = word_sections[0]
        cur_run: list[tuple[str, int]] = [words[0]]
        for j in range(1, len(words)):
            if word_sections[j] != cur_sec:
                sections.append((cur_sec, cur_run))
                cur_sec = word_sections[j]
                cur_run = [words[j]]
            else:
                cur_run.append(words[j])
        sections.append((cur_sec, cur_run))

    # Chunk each section independently so no chunk crosses a boundary
    all_chunks: list[dict] = []
    next_id = 1
    for sec_label, sec_words in sections:
        new_chunks = chunk_section(sec_label, sec_words, next_id)
        all_chunks.extend(new_chunks)
        next_id += len(new_chunks)

    if not all_chunks:
        print("ERROR: no text extracted from PDF. Verify the file is not scanned-only.")
        sys.exit(1)

    print(f"  {len(all_chunks)} chunks (target={TARGET_WORDS} words, overlap={OVERLAP_WORDS})")
    with_section = sum(1 for c in all_chunks if c["section"])
    avg_words    = sum(
        len(c["text"].split()) - (len(c["section"].split()) + 2 if c["section"] else 0)
        for c in all_chunks
    ) // len(all_chunks)
    print(f"  Sections detected: {with_section} ({with_section*100//len(all_chunks)}%)")
    print(f"  Avg words/chunk: {avg_words}")

    OUT_PATH.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
