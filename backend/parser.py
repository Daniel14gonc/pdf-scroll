"""Extract a stream of reading elements (paragraphs + images) from a PDF.

Two passes:
  1. Walk every page once, collect raw lines (text + bbox + font height) and
     image blocks. Save image bytes to disk as we go.
  2. With global stats in hand (font-size distribution, repeated-line
     frequency, TOC pages), filter into paragraphs and detect chapter
     headings by font-size tier (mirroring the paginas-reader.html approach).
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF


# ---------- output types ----------

@dataclass
class Paragraph:
    kind: Literal["text"]
    text: str
    page: int
    y: float


@dataclass
class Image:
    kind: Literal["image"]
    url_path: str
    width: int
    height: int
    page: int
    y: float


Element = Paragraph | Image


@dataclass
class Toc:
    title: str
    page: int
    level: int


# ---------- internal scratch types ----------

@dataclass
class _Line:
    text: str
    page: int               # 1-based
    y: float                # top y
    h: float                # dominant font size on the line
    x_start: float
    x_end: float


@dataclass
class _ImageHit:
    page: int
    y: float
    url_path: str
    width: int
    height: int


_HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")
_MULTI_SPACE = re.compile(r"[ \t]+")
_PAGE_NUM_LINE = re.compile(r"^\s*[ivxlcdm\d]{1,5}\s*$", re.IGNORECASE)
_LETTERS = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]")
_TOC_DOTS = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúÑñ].*?(\.\s*){2,}\s*\d{1,4}\s*$")
_TOC_TRAIL_NUM = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúÑñ].{2,}\s+\d{1,4}\s*$")
_TOC_HEADER = re.compile(
    r"^(contents|table of contents|índice|indice|contenido|sumario)$",
    re.IGNORECASE,
)

_MIN_IMAGE_PIXELS = 100 * 100


def _clean_text(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.replace("\n", " ")
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _looks_like_chrome(text: str) -> bool:
    if not text:
        return True
    if _PAGE_NUM_LINE.match(text):
        return True
    if len(text) < 3:
        return True
    return False


# ---------- pass 1: raw extraction ----------

def _walk_pages(doc: fitz.Document, image_out_dir: Path) -> tuple[
    list[list[_Line]],     # lines per page
    list[_ImageHit],       # all image hits
    list[float],            # all font sizes (for stats)
]:
    image_out_dir.mkdir(parents=True, exist_ok=True)
    pages_lines: list[list[_Line]] = []
    images: list[_ImageHit] = []
    all_sizes: list[float] = []
    image_index = 0

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        d = page.get_text("dict")
        page_lines: list[_Line] = []

        for block in d.get("blocks", []):
            btype = block.get("type")
            if btype == 0:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    parts: list[str] = []
                    sizes: list[float] = []
                    x_start = float("inf")
                    x_end = float("-inf")
                    y_top = float("inf")
                    for s in spans:
                        txt = s.get("text", "")
                        if not txt:
                            continue
                        parts.append(txt)
                        sz = float(s.get("size", 0) or 0)
                        if sz > 0:
                            sizes.append(sz)
                            all_sizes.append(sz)
                        bbox = s.get("bbox", (0, 0, 0, 0))
                        x_start = min(x_start, bbox[0])
                        x_end = max(x_end, bbox[2])
                        y_top = min(y_top, bbox[1])
                    raw = "".join(parts).strip()
                    if not raw:
                        continue
                    h = max(sizes) if sizes else 0.0
                    page_lines.append(_Line(
                        text=raw, page=page_num,
                        y=y_top if y_top != float("inf") else 0.0,
                        h=h, x_start=x_start if x_start != float("inf") else 0.0,
                        x_end=x_end if x_end != float("-inf") else 0.0,
                    ))

            elif btype == 1:
                bbox = block.get("bbox", (0, 0, 0, 0))
                width = int(bbox[2] - bbox[0])
                height = int(bbox[3] - bbox[1])
                if width * height < _MIN_IMAGE_PIXELS:
                    continue
                img_bytes = block.get("image")
                ext = block.get("ext", "png")
                if not img_bytes:
                    continue
                fname = f"{image_index}.{ext}"
                (image_out_dir / fname).write_bytes(img_bytes)
                images.append(_ImageHit(
                    page=page_num, y=bbox[1],
                    url_path=f"images/{fname}",
                    width=width, height=height,
                ))
                image_index += 1

        pages_lines.append(page_lines)

    return pages_lines, images, all_sizes


# ---------- global analyses on raw lines ----------

def _running_lines(pages_lines: list[list[_Line]]) -> set[str]:
    """Lines that look like running headers/footers (repeated short text)."""
    counter: Counter[str] = Counter()
    for page in pages_lines:
        if not page:
            continue
        # Only consider top-most and bottom-most lines on the page.
        for L in (page[0], page[-1]):
            key = re.sub(r"\s+", " ", L.text).strip().lower()
            if 2 < len(key) < 80:
                counter[key] += 1
    n_pages = len(pages_lines)
    threshold = max(5, n_pages // 4)
    return {k for k, c in counter.items() if c >= threshold}


def _toc_pages(pages_lines: list[list[_Line]]) -> set[int]:
    """Pages that look like the book's own table of contents."""
    toc: set[int] = set()
    for idx, page in enumerate(pages_lines):
        if len(page) < 4:
            continue
        hits = 0
        for L in page:
            t = L.text.strip()
            if _TOC_DOTS.search(t) or _TOC_TRAIL_NUM.search(t):
                hits += 1
        title_hit = any(_TOC_HEADER.match(L.text.strip()) for L in page[:3])
        if (hits >= 4 and hits / len(page) > 0.4) or (title_hit and hits >= 3):
            toc.add(idx)        # 0-based page index
    return toc


def _percentile(arr: list[float], p: float) -> float:
    if not arr:
        return 0.0
    a = sorted(arr)
    return a[min(len(a) - 1, int(p * len(a)))]


def _detect_chapters_by_font(
    pages_lines: list[list[_Line]],
    all_sizes: list[float],
    toc_pages: set[int],
    running: set[str],
) -> list[Toc]:
    """Heading-by-font-size detector ported from paginas-reader.html.

    Candidates are lines in the largest font tier of the book that look like
    titles (short, has letters, not a running header, not on TOC pages).
    """
    if not all_sizes:
        return []
    med = statistics.median(all_sizes)
    big = _percentile(all_sizes, 0.97) or med * 2

    # Words that are common in front-matter/TOC labels and should never be
    # treated as chapter titles even if rendered in a large font.
    stopword_titles = {
        "contents", "table of contents", "índice", "indice", "contenido",
        "sumario", "copyright", "dedication", "acknowledgements",
        "acknowledgments", "preface", "foreword", "introduction",
        "title", "author", "isbn",
    }

    # Merge consecutive lines on the same page that share roughly the same
    # large font size — these are multi-line chapter titles.
    merged_per_page: list[list[_Line]] = []
    for page in pages_lines:
        merged: list[_Line] = []
        for L in page:
            if merged and abs(merged[-1].h - L.h) < 0.5 and (L.y - merged[-1].y) < L.h * 2.5 and L.h > med * 1.4:
                merged[-1] = _Line(
                    text=(merged[-1].text + " " + L.text).strip(),
                    page=merged[-1].page,
                    y=merged[-1].y,
                    h=max(merged[-1].h, L.h),
                    x_start=min(merged[-1].x_start, L.x_start),
                    x_end=max(merged[-1].x_end, L.x_end),
                )
            else:
                merged.append(L)
        merged_per_page.append(merged)

    candidates: list[_Line] = []
    text_freq: Counter[str] = Counter()
    for page in pages_lines:
        for L in page:
            text_freq[re.sub(r"\s+", " ", L.text).strip()] += 1

    for idx, page in enumerate(merged_per_page):
        for L in page:
            text = re.sub(r"\s+", " ", L.text).strip()
            if idx in toc_pages:
                continue
            if not (2 <= len(text) <= 90):
                continue
            if not _LETTERS.search(text):
                continue
            key = text.lower()
            if key in running or key in stopword_titles:
                continue
            candidates.append(_Line(
                text=text, page=L.page, y=L.y, h=L.h,
                x_start=L.x_start, x_end=L.x_end,
            ))

    def ok(c: _Line) -> bool:
        return text_freq.get(c.text, 0) <= 3

    threshold = max(med * 1.7, big * 0.88)
    heads = [c for c in candidates if ok(c) and c.h >= threshold]
    if len(heads) < 2:
        threshold = med * 1.5
        heads = [c for c in candidates if ok(c) and c.h >= threshold]

    # Same title repeated → keep the largest instance (real opener, not the index).
    by_title: dict[str, _Line] = {}
    for h in heads:
        k = h.text.lower()
        if k not in by_title or h.h > by_title[k].h:
            by_title[k] = h

    # One chapter per page → keep the longest title on that page.
    by_page: dict[int, _Line] = {}
    for h in by_title.values():
        prev = by_page.get(h.page)
        if not prev or len(h.text) > len(prev.text):
            by_page[h.page] = h

    chosen = sorted(by_page.values(), key=lambda l: (l.page, l.y))
    return [Toc(title=l.text, page=l.page, level=1) for l in chosen]


# ---------- pass 2: build paragraphs ----------

def _body_x_bounds(lines: list[_Line]) -> tuple[float, float] | None:
    """Estimate the main text column's left/right edges from multi-word lines.

    Marginalia (glossary terms, side notes) tends to be 1-3 words and sits
    outside this column; we use the bounds to filter them out.
    """
    multi = [L for L in lines if len(L.text.split()) >= 5]
    if len(multi) < 3:
        return None
    lefts = sorted(L.x_start for L in multi)
    rights = sorted(L.x_end for L in multi)
    # Inner quartile-ish range — robust to a few outliers.
    left = lefts[len(lefts) // 4]
    right = rights[-(len(rights) // 4) - 1]
    return left, right


def _is_marginalia(L: _Line, bounds: tuple[float, float] | None) -> bool:
    if bounds is None:
        return False
    body_left, body_right = bounds
    margin = 30.0
    short = len(L.text.split()) <= 3
    too_far_left = L.x_start < body_left - margin
    too_far_right = L.x_end > body_right + margin and L.x_start > body_left + margin
    return short and (too_far_left or too_far_right)


def _join_lines(lines: list[str]) -> str:
    """Join lines preserving '\n' between them so the hyphenation regex
    in _clean_text can repair line-break hyphenation ('histori-\ncal' →
    'historical'). _clean_text then collapses remaining '\n' to spaces.
    """
    return "\n".join(lines)


def _build_paragraphs(
    pages_lines: list[list[_Line]],
    running: set[str],
    toc_pages: set[int],
) -> list[Paragraph]:
    out: list[Paragraph] = []
    for idx, page in enumerate(pages_lines):
        if idx in toc_pages:
            continue
        bounds = _body_x_bounds(page)
        kept: list[_Line] = []
        for L in page:
            key = re.sub(r"\s+", " ", L.text).strip().lower()
            if key in running:
                continue
            if _looks_like_chrome(L.text.strip()):
                continue
            if _is_marginalia(L, bounds):
                continue
            kept.append(L)
        if not kept:
            continue
        gaps: list[float] = []
        for k in range(1, len(kept)):
            g = kept[k].y - kept[k - 1].y
            if g > 0:
                gaps.append(g)
        med_gap = statistics.median(gaps) if gaps else 0.0
        cur_text: list[str] = []
        cur_y: float | None = None
        for k, L in enumerate(kept):
            new_para = False
            if k > 0:
                gap = L.y - kept[k - 1].y
                if med_gap > 0 and gap > med_gap * 1.6:
                    new_para = True
            if new_para and cur_text:
                text = _clean_text(_join_lines(cur_text))
                if text:
                    out.append(Paragraph(kind="text", text=text, page=kept[k - 1].page, y=cur_y or 0))
                cur_text = []
                cur_y = None
            if cur_y is None:
                cur_y = L.y
            cur_text.append(L.text)
        if cur_text:
            text = _clean_text(_join_lines(cur_text))
            if text:
                out.append(Paragraph(kind="text", text=text, page=kept[-1].page, y=cur_y or 0))
    return out


# ---------- public entrypoint ----------

def extract_elements(pdf_bytes: bytes, image_out_dir: Path) -> tuple[list[Element], list[Toc], int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = doc.page_count

    pages_lines, image_hits, all_sizes = _walk_pages(doc, image_out_dir)
    running = _running_lines(pages_lines)
    toc_pages = _toc_pages(pages_lines)

    paragraphs = _build_paragraphs(pages_lines, running, toc_pages)

    # Interleave images with paragraphs in reading order, per page.
    elements: list[Element] = []
    by_page_para: dict[int, list[Paragraph]] = {}
    for p in paragraphs:
        by_page_para.setdefault(p.page, []).append(p)
    by_page_img: dict[int, list[_ImageHit]] = {}
    for img in image_hits:
        by_page_img.setdefault(img.page, []).append(img)

    for pg in range(1, page_count + 1):
        merged: list[Element] = []
        merged.extend(by_page_para.get(pg, []))
        for h in by_page_img.get(pg, []):
            merged.append(Image(
                kind="image", url_path=h.url_path,
                width=h.width, height=h.height, page=h.page, y=h.y,
            ))
        merged.sort(key=lambda e: e.y)
        elements.extend(merged)

    # Chapters: prefer the PDF's own outline, fall back to font-size detection.
    toc_raw = doc.get_toc(simple=True)
    embedded = [Toc(title=t[1].strip(), page=t[2], level=t[0]) for t in toc_raw if t[1].strip()]
    if embedded:
        chapters = embedded
    else:
        chapters = _detect_chapters_by_font(pages_lines, all_sizes, toc_pages, running)

    doc.close()
    return elements, chapters, page_count
