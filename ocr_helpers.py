import base64
import io
import os
import re
import httpx
import pypdfium2 as pdfium


LITELLM_URL = "http://litellm-svc.vllm.svc.cluster.local:4000/v1/chat/completions"
LITELLM_KEY = os.environ["LITELLM_API_KEY"]
MODEL = os.environ.get("OCR_MODEL", "qwen3.6-35b")
DPI = int(os.environ.get("OCR_DPI", "150"))
BATCH = int(os.environ.get("OCR_BATCH", "2"))
CONCURRENCY = int(os.environ.get("OCR_CONCURRENCY", "3"))

PROMPT = (
    "You are an OCR engine. Extract ALL visible text from these rasterized PDF "
    "page image(s), in reading order. Preserve paragraph breaks, lists, columns, "
    "checkboxes, and tables using markdown. Do not summarize. Do not add commentary. "
    'For every page, output exactly one header in the form "=== Page N ===" '
    "where N is the page number provided before the image."
)


def _render_page_jpeg(pdf: pdfium.PdfDocument, idx: int) -> bytes:
    bitmap = pdf[idx].render(scale=DPI / 72)
    pil = bitmap.to_pil()
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


async def _ocr_batch(client: httpx.AsyncClient, pages: list[tuple[int, bytes]]) -> dict[int, str]:
    content = [{"type": "text", "text": PROMPT}]
    for page_num, jpeg in pages:
        b64 = base64.b64encode(jpeg).decode()
        content.append({"type": "text", "text": f"=== Page {page_num} image follows ==="})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    r = await client.post(
        LITELLM_URL,
        headers={"Authorization": f"Bearer {LITELLM_KEY}"},
        json={"model": MODEL, "messages": [{"role": "user", "content": content}],
              "temperature": 0, "max_tokens": 8192},
        timeout=600.0,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    matches = list(re.finditer(r"===\s*Page\s+(\d+)\s*===", text, re.I))
    out = {}
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[page_num] = text[start:end].strip()
    return out
