import asyncio
import base64
import logging
import os
import re

import fitz
import httpx

log = logging.getLogger("ocr")

LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")
if not LITELLM_API_KEY:
    raise RuntimeError("LITELLM_API_KEY environment variable is required")

OCR_MODEL = os.environ.get("OCR_MODEL", "qwen3.6-35b")
OCR_DPI = int(os.environ.get("OCR_DPI", "150"))
OCR_CONCURRENCY = int(os.environ.get("OCR_CONCURRENCY", "3"))
OCR_TIMEOUT = float(os.environ.get("OCR_TIMEOUT", "600"))
HAS_TEXT_THRESHOLD = int(os.environ.get("OCR_TEXT_THRESHOLD", "50"))

LITELLM_URL = "http://litellm-svc.vllm.svc.cluster.local:4000/v1/chat/completions"

OCR_PROMPT = (
    "You are an OCR engine. Extract ALL visible text from this rasterized PDF page image "
    "in reading order. Preserve paragraph breaks, lists, columns, checkboxes, and tables using "
    "markdown. Do not summarize, do not add commentary, do not include any preface or explanation. "
    "Output only the extracted text. If the page has no readable text, output an empty response."
)


def has_text_layer(pdf_bytes: bytes) -> bool:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total = 0
        for page in doc:
            total += len(page.get_text("text").strip())
            if total >= HAS_TEXT_THRESHOLD:
                return True
        return False
    finally:
        doc.close()


def _rasterize_page(pdf_bytes: bytes, page_index: int) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        matrix = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
        pix = doc[page_index].get_pixmap(matrix=matrix, alpha=False)
        return base64.b64encode(pix.tobytes(output="jpeg", jpg_quality=85)).decode()
    finally:
        doc.close()


async def _ocr_page(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    pdf_bytes: bytes,
    page_index: int,
) -> str:
    async with sem:
        b64 = await asyncio.to_thread(_rasterize_page, pdf_bytes, page_index)

        payload = {
            "model": OCR_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            "temperature": 0,
            "max_tokens": 8192,
        }

        r = await client.post(
            LITELLM_URL,
            json=payload,
            headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
            timeout=OCR_TIMEOUT,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"] or ""
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        log.info("ocr page %d ok (%d chars)", page_index + 1, len(raw))
        return raw


async def ocr_pdf(pdf_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = doc.page_count
    finally:
        doc.close()

    if total_pages == 0:
        return []

    sem = asyncio.Semaphore(OCR_CONCURRENCY)
    async with httpx.AsyncClient() as client:
        tasks = [_ocr_page(client, sem, pdf_bytes, i) for i in range(total_pages)]
        return await asyncio.gather(*tasks)


def embed_text_layer(pdf_bytes: bytes, texts: list[str]) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for i, page in enumerate(doc):
            if i < len(texts) and texts[i].strip():
                page.insert_textbox(
                    page.rect,
                    texts[i],
                    fontname="helv",
                    fontsize=8,
                    render_mode=3,
                    align=0,
                    overlay=True,
                )
        return doc.tobytes(garbage=4, deflate=True, clean=True)
    finally:
        doc.close()
