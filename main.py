import base64
import io
import json
import asyncio
import httpx
import pypdfium2 as pdfium
import ocr_helpers
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
import pymupdf
import logging
logger = logging.getLogger(__name__)

app = FastAPI(title="ocr-embedder")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rasterize")
async def rasterize(
    pdf: UploadFile = File(...),
    dpi: int = Query(200, ge=72, le=600),
):
    """Render each PDF page as a PNG. Returns JSON with base64 images in order."""
    pdf_bytes = await pdf.read()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    # PyMuPDF uses zoom matrix, not DPI directly. 72 DPI = zoom 1.0.
    zoom = dpi / 72
    matrix = pymupdf.Matrix(zoom, zoom)

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
async def embed(pdf: UploadFile = File(...), page_texts: str = Form(...)):
    try:
        texts = json.loads(page_texts)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"page_texts must be JSON array: {e}")
    if not isinstance(texts, list):
        raise HTTPException(400, "page_texts must be a JSON array")

    pdf_bytes = await pdf.read()
    out_pdf = ocr_helpers.embed_text_layer(pdf_bytes, texts)
    return Response(content=out_pdf, media_type="application/pdf")


@app.post("/ocr")
async def ocr(pdf: UploadFile):
    pdf_bytes = await pdf.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty",
        )

    try:
        doc = pdfium.PdfDocument(pdf_bytes)
    except Exception as e:
        logger.exception("Failed to open uploaded PDF")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to open uploaded PDF: {type(e).__name__}",
        )

    total = len(doc)

    if total == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF has no pages",
        )

    # Build batches of zero-based page indices.
    batches = [
        list(range(i, min(i + ocr_helpers.BATCH, total)))
        for i in range(0, total, ocr_helpers.BATCH)
    ]

    sem = asyncio.Semaphore(ocr_helpers.CONCURRENCY)

    # page_texts is keyed by 1-based page number.
    # Every page should get an entry, even if OCR text is empty.
    page_texts: dict[int, str] = {}

    timeout = httpx.Timeout(
        connect=30.0,
        read=300.0,
        write=300.0,
        pool=300.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def run_batch(idxs: list[int]):
            async with sem:
                rendered: list[tuple[int, bytes]] = []

    