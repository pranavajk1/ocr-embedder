import base64
import io
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import Response
import pymupdf

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
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    if len(texts) < len(doc):
        # Pad with empty so we don't crash on partial coverage
        texts = texts + [""] * (len(doc) - len(texts))
    elif len(texts) > len(doc):
        texts = texts[: len(doc)]

    for page, text in zip(doc, texts):
        if not text or not text.strip():
            continue
        # Invisible text layer: render_mode=3 (neither fill nor stroke).
        # We dump the whole page's text into a single textbox covering the
        # page area. No positional accuracy, but full-text search and ctrl-F
        # both work in any PDF viewer, and paperless extracts it as content.
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