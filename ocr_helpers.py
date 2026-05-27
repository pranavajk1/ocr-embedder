import asyncio
import base64
import os
import re

import fitz
import httpx

LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")
if not LITELLM_API_KEY:
    raise RuntimeError("LITELLM_API_KEY environment variable is required")

OCR_MODEL = os.environ.get("OCR_MODEL", "qwen3.6-35b")
OCR_DPI = int(os.environ.get("OCR_DPI", "150"))
OCR_BATCH = int(os.environ.get("OCR_BATCH", "2"))
OCR_CONCURRENCY = int(os.environ.get("OCR_CONCURRENCY", "3"))

LITELLM_URL = "http://litellm-svc.vllm.svc.cluster.local:4000/v1/chat/completions"

OCR_PROMPT = (
    "You are an OCR engine. Extract ALL visible text from these rasterized PDF page image(s), "
    "in reading order. Preserve paragraph breaks, lists, columns, checkboxes, and tables using "
    "markdown. Do not summarize. Do not add commentary. For every page, output exactly one header "
    'in the form "=== Page N ===" where N is the page number provided before the image.'
)

_semaphore = asyncio.Semaphore(OCR_CONCURRENCY)


async def _ocr_batch(pdf_bytes: bytes, page_indices: list[int]) -> dict[int, str]:
    async with _semaphore:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            jpeg_b64s: list[str] = []
            for idx in page_indices:
                matrix = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
                pix = doc[idx].get_pixmap(matrix=matrix, alpha=False)
                jpeg_b64s.append(
                    base64.b64encode(pix.tobytes(output="jpeg", jpg_quality=85)).decode()
                )
        finally:
            doc.close()

        content: list[dict] = [{"type": "text", "text": OCR_PROMPT}]
        for idx, b64 in zip(page_indices, jpeg_b64s):
            page_num = idx + 1
            content.append({"type": "text", "text": f"=== Page {page_num} image follows ==="})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        payload = {
            "model": OCR_MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 8192,
        }

        async with httpx.AsyncClient() as client:
            r = await client.post(
                LITELLM_URL,
                json=payload,
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                timeout=600.0,
            )
            r.raise_for_status()

    raw = r.json()["choices"][0]["message"]["content"]
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

    matches = list(re.finditer(r"=== Page (\d+) ===", raw))
    results: dict[int, str] = {}
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        results[page_num] = raw[start:end].strip()

    return results


def embed_text_layer(pdf_bytes: bytes, texts: list[str]) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for i, page in enumerate(doc):
            if i < len(texts) and texts[i]:
                page.insert_textbox(
                    page.rect,
                    texts[i],
                    fontname="helv",
                    fontsize=8,
                    render_mode=3,
                    align=0,
                    overlay=True,
                )
        return doc.tobytes()
    finally:
        doc.close()
