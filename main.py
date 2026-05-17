import io
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import pymupdf  # PyMuPDF >= 1.24

app = FastAPI(title="ocr-embedder")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
async def embed(
    pdf: UploadFile = File(...),
    page_texts: str = Form(...),  # JSON array of strings, one per page
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