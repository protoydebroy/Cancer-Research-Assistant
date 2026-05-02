"""
CANCER CHATBOT — STEP 1: PDF INGESTION PIPELINE
================================================
Reads 19 cancer PDFs from the Indian Cancer Society,
parses text (with pdfplumber fallback), chunks into
~500-token segments with 10% overlap, and writes
ready-to-embed chunks to a JSONL file.

Usage:
  1. Drop your 19 PDFs into  ./cancer_pdfs/
  2. python step1_ingest.py
  3. Output: ./data/chunks.jsonl  (one JSON object per line)
"""

import os
import json
import re
import hashlib
from pathlib import Path
from datetime import datetime

import fitz        # PyMuPDF
import pdfplumber  # fallback for tables / rotated text

# ── Token estimation (offline, no tiktoken BPE download needed) ──────────────
# OpenAI cl100k_base averages ~4 chars per token for English medical text.
# Accurate to ~±10%; exact counts come from the embedding API.
CHARS_PER_TOKEN = 4

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR          = Path("cancer_pdfs")
OUTPUT_DIR       = Path("data")
CHUNKS_FILE      = OUTPUT_DIR / "chunks.jsonl"
CHUNK_CHARS      = 500 * CHARS_PER_TOKEN   # 2,000 chars ≈ 500 tokens
OVERLAP_CHARS    = 50  * CHARS_PER_TOKEN   # 200 chars  ≈ 50 tokens
MIN_CHUNK_CHARS  = 50  * CHARS_PER_TOKEN   # discard tiny trailing pieces
# ─────────────────────────────────────────────────────────────────────────────

CANCER_KEYWORDS = {
    "breast":     ["breast"],
    "cervical":   ["cervical", "cervix"],
    "oral":       ["oral", "mouth"],
    "bladder":    ["bladder"],
    "eye":        ["eye", "ocular", "retinoblastoma"],
    "pancreas":   ["pancreas", "pancreatic"],
    "colorectal": ["colorectal", "colon", "rectal", "rectum"],
    "esophageal": ["esophageal", "oesophageal", "esophagus"],
    "kidney":     ["kidney", "renal"],
    "laryngeal":  ["laryngeal", "larynx"],
    "liver":      ["liver", "hepatocellular"],
    "lung":       ["lung"],
    "ovarian":    ["ovarian", "ovary"],
    "prostate":   ["prostate"],
    "skin":       ["skin", "melanoma"],
    "stomach":    ["stomach", "gastric"],
    "testicular": ["testicular", "testis"],
    "thyroid":    ["thyroid"],
    "uterine":    ["uterine", "uterus", "endometrial"],
}


def detect_cancer_type(filename: str, first_page_text: str) -> str:
    combined = (filename + " " + first_page_text[:500]).lower()
    for cancer, kws in CANCER_KEYWORDS.items():
        if any(kw in combined for kw in kws):
            return cancer
    return "unknown"


def extract_pymupdf(pdf_path: Path) -> list:
    pages = []
    doc = fitz.open(str(pdf_path))
    for num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        pages.append({"page_num": num, "text": text})
    doc.close()
    return pages


def extract_pdfplumber(pdf_path: Path) -> list:
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages.append({"page_num": num, "text": text})
    return pages


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\.{4,}', '', text)
    return text.strip()


def chunk_text(text: str) -> list:
    """Sliding-window char chunking that approximates token boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        # Snap end to nearest space to avoid mid-word cuts
        if end < len(text):
            snap = text.rfind(' ', start, end)
            if snap > start:
                end = snap
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def build_chunk_id(source: str, page: int, idx: int) -> str:
    raw = f"{source}:{page}:{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def process_pdf(pdf_path: Path) -> list:
    print(f"  Processing: {pdf_path.name}")

    pages = extract_pymupdf(pdf_path)
    total_chars = sum(len(p["text"]) for p in pages)

    if total_chars < 200:
        print(f"    [!] Sparse text, trying pdfplumber...")
        pages = extract_pdfplumber(pdf_path)
        total_chars = sum(len(p["text"]) for p in pages)

    if total_chars < 200:
        print(f"    [x] Likely scanned — OCR needed. Skipping.")
        return []

    first_text = pages[0]["text"] if pages else ""
    cancer_type = detect_cancer_type(pdf_path.stem, first_text)
    print(f"    Cancer type : {cancer_type}")

    all_chunks = []
    global_idx = 0

    for page in pages:
        raw = clean_text(page["text"])
        if not raw:
            continue

        for local_idx, chunk_str in enumerate(chunk_text(raw)):
            all_chunks.append({
                "chunk_id":    build_chunk_id(pdf_path.name, page["page_num"], local_idx),
                "source":      pdf_path.name,
                "cancer_type": cancer_type,
                "page_num":    page["page_num"],
                "chunk_index": global_idx,
                "token_count": estimate_tokens(chunk_str),
                "char_count":  len(chunk_str),
                "text":        chunk_str,
                "ingested_at": datetime.utcnow().isoformat() + "Z",
            })
            global_idx += 1

    print(f"    [ok] {len(pages)} pages -> {global_idx} chunks")
    return all_chunks


def run_ingestion():
    print("=" * 60)
    print("  STEP 1 — PDF INGESTION PIPELINE")
    print("=" * 60)

    if not PDF_DIR.exists():
        PDF_DIR.mkdir(parents=True)
        print(f"\n  Created '{PDF_DIR}/' — add your 19 PDFs there and rerun.\n")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n  No PDFs found in '{PDF_DIR}/' — add files and rerun.\n")
        return

    print(f"\n  Found {len(pdf_files)} PDF(s)\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for pdf_path in pdf_files:
        all_chunks.extend(process_pdf(pdf_path))

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  PDFs processed  : {len(pdf_files)}")
    print(f"  Total chunks    : {len(all_chunks)}")

    if all_chunks:
        avg_tok = sum(c["token_count"] for c in all_chunks) / len(all_chunks)
        print(f"  Avg token count : {avg_tok:.0f} (estimated)")

        cancer_counts = {}
        for c in all_chunks:
            cancer_counts[c["cancer_type"]] = cancer_counts.get(c["cancer_type"], 0) + 1

        print(f"\n  Chunks by cancer type:")
        for ct, cnt in sorted(cancer_counts.items(), key=lambda x: -x[1]):
            bar = "#" * (cnt // 2)
            print(f"    {ct:<18} {cnt:>4} chunks  {bar}")

    print(f"\n  Output -> {CHUNKS_FILE}")
    print("=" * 60)

    if all_chunks:
        s = all_chunks[0]
        print("\n  SAMPLE CHUNK:")
        print(f"  chunk_id    : {s['chunk_id']}")
        print(f"  source      : {s['source']}")
        print(f"  cancer_type : {s['cancer_type']}")
        print(f"  page_num    : {s['page_num']}")
        print(f"  token_count : {s['token_count']} (est.)")
        print(f"  text[:300]  :\n")
        print("  " + s['text'][:300].replace("\n", "\n  "))
    print()


if __name__ == "__main__":
    run_ingestion()
