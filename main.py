import base64
import io
import json
import logging

import fitz
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse

from ocr_helpers import embed_text_layer, ocr_pdf

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

    try:
        texts = await ocr_pdf(pdf_bytes)
    except Exception as exc:
        log.exception("ocr failed for %s", filename)
        raise HTTPException(status_code=502, detail=f"OCR failed: {exc}")

    output_bytes = embed_text_layer(pdf_bytes, texts)

    # What the model actually extracted, reported alongside the PDF.
    #
    # The caller cannot get this from the response body: the text layer is
    # embedded invisibly and reading it back means re-parsing the PDF. Without
    # it, a run where the vision model returned nothing for every page -- an
    # unloaded model, an exhausted quota, a refusal -- is indistinguishable from
    # a good one, because the response is still a valid PDF of the right size.
    # The Windmill flow refuses to replace an original on these numbers; see
    # f/paperless_ocr/steps/verify_ocr.ts in the sigma-lab-windmill repo.
    chars = sum(len(t) for t in texts)
    empty_pages = sum(1 for t in texts if not t.strip())

    log.info(
        "ocr done: %s (%d pages, %d empty, %d chars, %d bytes out)",
        filename, len(texts), empty_pages, chars, len(output_bytes),
    )
    return StreamingResponse(
        io.BytesIO(output_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(output_bytes)),
            "X-OCR-Pages": str(len(texts)),
            "X-OCR-Chars": str(chars),
            "X-OCR-Empty-Pages": str(empty_pages),
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
