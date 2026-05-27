import asyncio
import base64
import io
import json

import fitz
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response

from ocr_helpers import OCR_BATCH, _ocr_batch, embed_text_layer

app = FastAPI(title="ocr-embedder")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(pdf: UploadFile = File(...)):
    pdf_bytes = await pdf.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="pdf field is empty")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = doc.page_count
    finally:
        doc.close()

    batches = [
        list(range(i, min(i + OCR_BATCH, total_pages)))
        for i in range(0, total_pages, OCR_BATCH)
    ]

    results = await asyncio.gather(
        *[_ocr_batch(pdf_bytes, batch) for batch in batches],
        return_exceptions=True,
    )

    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise HTTPException(status_code=502, detail=f"OCR batch error: {errors[0]}")

    page_texts: dict[int, str] = {}
    for r in results:
        page_texts.update(r)

    missing = [n for n in range(1, total_pages + 1) if n not in page_texts]
    if missing:
        raise HTTPException(status_code=502, detail=f"Missing OCR output for pages: {missing}")

    ordered_texts = [page_texts[n] for n in range(1, total_pages + 1)]
    output_bytes = embed_text_layer(pdf_bytes, ordered_texts)
    return Response(content=output_bytes, media_type="application/pdf")


@app.post("/rasterize")
async def rasterize(
    pdf: UploadFile = File(...),
    dpi: int = Query(200, ge=72, le=600),
):
    pdf_bytes = await pdf.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pix.tobytes(output="png")
        pages.append({
            "page": i + 1,
            "width": pix.width,
            "height": pix.height,
            "image_b64": base64.b64encode(png_bytes).decode("ascii"),
        })

    total = len(doc)
    doc.close()
    return {"total_pages": total, "pages": pages}


@app.post("/embed")
async def embed(
    pdf: UploadFile = File(...),
    page_texts: str = Form(...),
):
    try:
        texts = json.loads(page_texts)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"page_texts must be JSON array: {e}")
    if not isinstance(texts, list):
        raise HTTPException(400, "page_texts must be a JSON array")

    pdf_bytes = await pdf.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if len(texts) < len(doc):
        texts = texts + [""] * (len(doc) - len(texts))
    elif len(texts) > len(doc):
        texts = texts[: len(doc)]

    for page, text in zip(doc, texts):
        if not text or not text.strip():
            continue
        page.insert_textbox(
            page.rect,
            text,
            fontsize=8,
            fontname="helv",
            render_mode=3,
            align=0,
            overlay=True,
        )

    out = io.BytesIO()
    doc.save(out, garbage=4, deflate=True, clean=True)
    doc.close()
    return Response(content=out.getvalue(), media_type="application/pdf")
