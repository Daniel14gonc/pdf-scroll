"""Disk-backed book library.

Layout under ``data/books/``:

    index.json                 # [{id, filename, added_at, card_count, ...}]
    <book_id>/
        book.json              # full {cards, chapters, ...}
        images/<n>.<ext>       # extracted images, referenced by cards
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "books"
INDEX_PATH = DATA_DIR / "index.json"


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def book_id_for(pdf_bytes: bytes) -> str:
    return hashlib.sha1(pdf_bytes).hexdigest()[:16]


def book_dir(book_id: str) -> Path:
    return DATA_DIR / book_id


def image_dir(book_id: str) -> Path:
    return book_dir(book_id) / "images"


def load_index() -> list[dict]:
    _ensure_dirs()
    if not INDEX_PATH.exists():
        return []
    try:
        return json.loads(INDEX_PATH.read_text())
    except json.JSONDecodeError:
        return []


def save_index(items: list[dict]) -> None:
    _ensure_dirs()
    INDEX_PATH.write_text(json.dumps(items, indent=2))


def get_book(book_id: str) -> dict | None:
    path = book_dir(book_id) / "book.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_book(book_id: str, filename: str, data: dict) -> dict:
    _ensure_dirs()
    bdir = book_dir(book_id)
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "book.json").write_text(json.dumps(data))

    index = load_index()
    entry = {
        "id": book_id,
        "filename": filename,
        "added_at": int(time.time()),
        "card_count": len(data.get("cards", [])),
        "chapter_count": len(data.get("chapters", [])),
        "page_count": data.get("page_count", 0),
    }
    # Upsert by id.
    index = [e for e in index if e.get("id") != book_id]
    index.insert(0, entry)
    save_index(index)
    return entry


def delete_book(book_id: str) -> bool:
    bdir = book_dir(book_id)
    if bdir.exists():
        shutil.rmtree(bdir)
    index = load_index()
    new_index = [e for e in index if e.get("id") != book_id]
    if len(new_index) != len(index):
        save_index(new_index)
        return True
    return False
