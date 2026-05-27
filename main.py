import base64
import io
import json
import logging

import fitz
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse

from ocr_helpers import embed_text_layer, has_text_layer, ocr_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ocr-embedder")

app = FastAPI(title="ocr-embedder")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(pdf: UploadFile = File(...)):
    pdf_bytes = await pdf.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="pdf field is empty")

    filename = pdf.filename or "document.pdf"

    if has_text_layer(pdf_bytes):
        log.info("skip: %s already has a text layer", filename)
        return _pdf_response(pdf_bytes, filename, skipped=True)

    try:
        texts = await ocr_pdf(pdf_bytes)
    except Exception as exc:
        log.exception("ocr failed for %s", filename)
        raise HTTPException(status_code=502, detail=f"OCR failed: {exc}")

    output_bytes = embed_text_layer(pdf_bytes, texts)
    log.info("ocr done: %s (%d pages, %d bytes out)", filename, len(texts), len(output_bytes))
    return _pdf_response(output_bytes, filename, skipped=False)


def _pdf_response(pdf_bytes: bytes, filename: str, *, skipped: bool) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
            "X-OCR-Skipped": "true" if skipped else "false",
        },
    )


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
