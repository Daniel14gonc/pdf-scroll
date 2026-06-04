# ScrollRead

Lee libros (PDF) como si fueran un feed de Instagram. Cada "tarjeta" es un
trozo de un único párrafo — nunca mezcla párrafos distintos, así no pierdes
el hilo.

## Estructura

- `backend/` — FastAPI + PyMuPDF + spaCy. Recibe el PDF, devuelve JSON de tarjetas.
- `frontend/index.html` — Un solo HTML con feed vertical y `scroll-snap`. Sin build step.

## Cómo correrlo

### 1. Backend

```bash
cd backend
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
python3 -m http.server 5173
```

Abre <http://localhost:5173> y sube un PDF.

## Parámetros del chunker

En `backend/chunker.py`:

- `TARGET_WORDS = 60` — tamaño deseado de tarjeta.
- `MAX_WORDS = 85` — tope duro antes de forzar corte.
- `MIN_TAIL_WORDS = 15` — si lo que queda del párrafo es muy poco, lo mete
  en la tarjeta actual en vez de dejar una tarjeta huérfana minúscula.

Si quieres tarjetas más cortas (más "swipes" estilo TikTok), baja `TARGET_WORDS`
a 30-40. Para lectura más pausada, sube a 80-100.
