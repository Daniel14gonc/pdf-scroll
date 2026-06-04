"""FastAPI entrypoint: library + parse + per-book static images + frontend."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import storage
from chunker import build_cards, cards_to_dicts, map_chapters
from parser import extract_elements

app = FastAPI(title="ScrollRead")

# CORS only matters when frontend is served from a different origin (local dev).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

storage.DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/books", StaticFiles(directory=storage.DATA_DIR), name="books")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/books")
def list_books():
    return {"books": storage.load_index()}


@app.get("/books/{book_id}")
def get_book(book_id: str):
    book = storage.get_book(book_id)
    if book is None:
        raise HTTPException(404, "Book not found")
    return book


@app.delete("/books/{book_id}")
def delete_book(book_id: str):
    ok = storage.delete_book(book_id)
    if not ok:
        raise HTTPException(404, "Book not found")
    return {"deleted": book_id}


@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files supported for now")

    pdf_bytes = await file.read()
    book_id = storage.book_id_for(pdf_bytes)

    # If we already have it, return the existing record without re-parsing.
    existing = storage.get_book(book_id)
    if existing is not None:
        return {"id": book_id, "reused": True, **existing}

    elements, toc, page_count = extract_elements(pdf_bytes, storage.image_dir(book_id))
    cards = build_cards(elements)
    chapters = map_chapters(cards, toc, page_count)

    book_data = {
        "id": book_id,
        "filename": file.filename,
        "page_count": page_count,
        "cards": cards_to_dicts(cards),
        "chapters": chapters,
    }
    storage.save_book(book_id, file.filename, book_data)
    return {"id": book_id, "reused": False, **book_data}


# Serve the frontend at "/" — must be mounted AFTER all API routes so it
# only catches whatever didn't match an endpoint.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
