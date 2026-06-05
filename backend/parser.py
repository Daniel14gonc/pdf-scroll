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

# Vector-figure detection thresholds.
_VEC_MIN_STROKES = 4              # cluster needs at least this many drawings
_VEC_MIN_AREA_FRAC = 0.018        # cluster bbox >= this fraction of page area
_VEC_CLUSTER_GAP = 18.0           # vertical gap (units) that breaks clusters
_VEC_RENDER_SCALE = 2.0           # rasterization scale (2 = "@2x")
_VEC_PADDING = 6.0                # padding around bbox before rendering


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

def _detect_vector_regions(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    """Cluster vector drawings on a page; return bboxes of figure-sized clusters.

    Many diagrams (and most math equations rendered as paths) come through
    PyMuPDF as drawings, not images. We cluster nearby drawings by vertical
    proximity and keep clusters that look substantial.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    if not drawings:
        return []

    page_area = page.rect.width * page.rect.height
    rects: list[tuple[float, float, float, float]] = []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        w = r.x1 - r.x0
        h = r.y1 - r.y0
        if w * h < 16:  # skip rules, underlines, single ticks
            continue
        rects.append((r.x0, r.y0, r.x1, r.y1))

    if not rects:
        return []

    # Cluster by vertical proximity: items within _VEC_CLUSTER_GAP go together.
    rects.sort(key=lambda r: r[1])
    clusters: list[list[tuple[float, float, float, float]]] = [[rects[0]]]
    for r in rects[1:]:
        last = clusters[-1]
        prev_bottom = max(p[3] for p in last)
        if r[1] - prev_bottom < _VEC_CLUSTER_GAP:
            last.append(r)
        else:
            clusters.append([r])

    out: list[tuple[float, float, float, float]] = []
    for cluster in clusters:
        if len(cluster) < _VEC_MIN_STROKES:
            continue
        x0 = min(r[0] for r in cluster)
        y0 = min(r[1] for r in cluster)
        x1 = max(r[2] for r in cluster)
        y1 = max(r[3] for r in cluster)
        area = (x1 - x0) * (y1 - y0)
        if area < page_area * _VEC_MIN_AREA_FRAC:
            continue
        out.append((x0, y0, x1, y1))
    return out


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
        page_image_bboxes: list[tuple[float, float, float, float]] = []

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
                page_image_bboxes.append(bbox)

        # --- Vector figures: rasterize clustered drawings as their own images.
        for bbox in _detect_vector_regions(page):
            # Skip regions that overlap heavily with an already-extracted raster
            # image — avoid double-rendering.
            x0, y0, x1, y1 = bbox
            duplicate = any(
                not (x1 < ib[0] or x0 > ib[2] or y1 < ib[1] or y0 > ib[3])
                and (min(x1, ib[2]) - max(x0, ib[0])) * (min(y1, ib[3]) - max(y0, ib[1]))
                > 0.5 * (x1 - x0) * (y1 - y0)
                for ib in page_image_bboxes
            )
            if duplicate:
                continue
            pad = _VEC_PADDING
            clip = fitz.Rect(
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(page.rect.width, x1 + pad),
                min(page.rect.height, y1 + pad),
            )
            try:
                pix = page.get_pixmap(
                    clip=clip,
                    matrix=fitz.Matrix(_VEC_RENDER_SCALE, _VEC_RENDER_SCALE),
                    alpha=False,
                )
            except Exception:
                continue
            fname = f"{image_index}.png"
            (image_out_dir / fname).write_bytes(pix.tobytes("png"))
            images.append(_ImageHit(
                page=page_num, y=clip.y0,
                url_path=f"images/{fname}",
                width=int(clip.width), height=int(clip.height),
            ))
            image_index += 1
            page_image_bboxes.append((clip.x0, clip.y0, clip.x1, clip.y1))

        # --- Drop text lines whose center falls inside any image bbox on this
        # page. PyMuPDF often extracts garbled text from inside figures
        # (especially math); now that we have the figure rasterized, we don't
        # want that broken text mixed back in.
        if page_image_bboxes:
            def inside_any(L: _Line) -> bool:
                cx = (L.x_start + L.x_end) / 2
                cy = L.y
                for ib in page_image_bboxes:
                    if ib[0] <= cx <= ib[2] and ib[1] <= cy <= ib[3]:
                        return True
                return False
            page_lines = [L for L in page_lines if not inside_any(L)]

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
    med_size: float,
) -> list[Paragraph]:
    """Build paragraphs, forcing breaks around heading-sized lines.

    A line whose font size is notably larger than the body's median is treated
    as a heading: we cut the current paragraph before it and start a fresh one
    after it. Without this, headings like "Chapter 13" get glued to the body
    text that follows them on the same page.
    """
    HEADING_RATIO = 1.3
    out: list[Paragraph] = []

    def is_heading(L: _Line) -> bool:
        return med_size > 0 and L.h > med_size * HEADING_RATIO

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

        def flush(page_num: int):
            nonlocal cur_text, cur_y
            if cur_text:
                text = _clean_text(_join_lines(cur_text))
                if text:
                    out.append(Paragraph(kind="text", text=text, page=page_num, y=cur_y or 0))
                cur_text = []
                cur_y = None

        for k, L in enumerate(kept):
            heading_now = is_heading(L)
            heading_prev = k > 0 and is_heading(kept[k - 1])
            new_para = False
            if k > 0:
                gap = L.y - kept[k - 1].y
                if med_gap > 0 and gap > med_gap * 1.6:
                    new_para = True
                if heading_now or heading_prev:
                    new_para = True
            if new_para:
                flush(kept[k - 1].page)
            if cur_y is None:
                cur_y = L.y
            cur_text.append(L.text)
        flush(kept[-1].page)
    return out


# ---------- public entrypoint ----------

def extract_elements(pdf_bytes: bytes, image_out_dir: Path) -> tuple[list[Element], list[Toc], int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = doc.page_count

    pages_lines, image_hits, all_sizes = _walk_pages(doc, image_out_dir)
    running = _running_lines(pages_lines)
    toc_pages = _toc_pages(pages_lines)
    med_size = statistics.median(all_sizes) if all_sizes else 0.0

    paragraphs = _build_paragraphs(pages_lines, running, toc_pages, med_size)

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
