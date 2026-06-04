"""Build cards from a stream of text+image elements.

The whole point: a card never mixes sentences from different paragraphs.
Within a paragraph we greedily pack sentences up to TARGET_WORDS, with a
small overflow allowance so we don't strand a tiny tail sentence.

Images become their own cards inserted in reading order.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable

from parser import Element, Image, Paragraph, Toc

TARGET_WORDS = 60
MAX_WORDS = 85
MIN_TAIL_WORDS = 15


@dataclass
class Card:
    id: int
    type: str            # "text" | "image"
    page: int
    # text-only fields
    paragraph_id: int = -1
    is_paragraph_start: bool = False
    is_paragraph_end: bool = False
    text: str = ""
    word_count: int = 0
    # image-only fields
    image_url: str = ""
    image_width: int = 0
    image_height: int = 0


_ABBREV = {
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "rev", "hon",
    "vs", "etc", "eg", "ie", "no", "vol", "pp", "p", "ch", "fig",
    "cf", "ed", "eds", "trans", "approx",
}

_SENT_SPLIT = re.compile(
    r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])"
)


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    raw = _SENT_SPLIT.split(text)
    merged: list[str] = []
    for part in raw:
        part = part.strip()
        if not part:
            continue
        if merged:
            last_word = re.search(r"(\w+)\.\s*$", merged[-1])
            if last_word and last_word.group(1).lower() in _ABBREV:
                merged[-1] = merged[-1] + " " + part
                continue
        merged.append(part)
    return merged


def _chunk_paragraph(sentences: list[str]) -> list[list[str]]:
    if not sentences:
        return []
    groups: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for i, sent in enumerate(sentences):
        sent_words = len(sent.split())
        remaining_after = sum(len(s.split()) for s in sentences[i + 1:])
        if current and current_words + sent_words > MAX_WORDS:
            groups.append(current)
            current = []
            current_words = 0
        current.append(sent)
        current_words += sent_words
        is_last = i == len(sentences) - 1
        if is_last:
            continue
        over_target = current_words >= TARGET_WORDS
        tail_would_be_orphan = remaining_after < MIN_TAIL_WORDS and current_words + remaining_after <= MAX_WORDS
        if over_target and not tail_would_be_orphan:
            groups.append(current)
            current = []
            current_words = 0
    if current:
        groups.append(current)
    return groups


def build_cards(elements: Iterable[Element]) -> list[Card]:
    cards: list[Card] = []
    card_id = 0
    paragraph_id = 0
    for el in elements:
        if isinstance(el, Image):
            cards.append(Card(
                id=card_id,
                type="image",
                page=el.page,
                image_url=el.url_path,
                image_width=el.width,
                image_height=el.height,
            ))
            card_id += 1
            continue

        # Text paragraph.
        sentences = _split_sentences(el.text)
        if not sentences:
            continue
        groups = _chunk_paragraph(sentences)
        for g_idx, group in enumerate(groups):
            text = " ".join(group)
            cards.append(Card(
                id=card_id,
                type="text",
                page=el.page,
                paragraph_id=paragraph_id,
                is_paragraph_start=(g_idx == 0),
                is_paragraph_end=(g_idx == len(groups) - 1),
                text=text,
                word_count=len(text.split()),
            ))
            card_id += 1
        paragraph_id += 1
    return cards


def map_chapters(cards: list[Card], toc: list[Toc], page_count: int) -> list[dict]:
    """Map each TOC entry to the first card on (or after) its page."""
    if not cards or not toc:
        return []
    chapters: list[dict] = []
    for entry in toc:
        match = next((c for c in cards if c.page >= entry.page), None)
        if match is None:
            continue
        chapters.append({
            "title": entry.title,
            "level": entry.level,
            "page": entry.page,
            "card_index": match.id,
        })
    return chapters


def cards_to_dicts(cards: list[Card]) -> list[dict]:
    return [asdict(c) for c in cards]
