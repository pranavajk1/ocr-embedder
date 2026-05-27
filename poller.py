import asyncio
import logging
import os

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("poller")

PAPERLESS_URL   = os.environ["PAPERLESS_URL"]
PAPERLESS_TOKEN = os.environ["PAPERLESS_API_TOKEN"]
N8N_WEBHOOK_URL = os.environ["N8N_WEBHOOK_URL"]
TAG_PROCESSING  = os.environ.get("TAG_PROCESSING", "ocr-processing")
TAG_DONE        = os.environ.get("TAG_DONE", "ocr-done")
PAGE_SIZE       = int(os.environ.get("POLL_PAGE_SIZE", "10"))


async def resolve_tag(client: httpx.AsyncClient, name: str) -> int:
    r = await client.get("/api/tags/", params={"name__iexact": name})
    r.raise_for_status()
    for t in r.json()["results"]:
        if t["name"].lower() == name.lower():
            return t["id"]
    r = await client.post("/api/tags/", json={"name": name})
    r.raise_for_status()
    return r.json()["id"]


async def main() -> None:
    headers = {"Authorization": f"Token {PAPERLESS_TOKEN}"}
    async with httpx.AsyncClient(base_url=PAPERLESS_URL, headers=headers, timeout=30.0) as client:
        processing_id = await resolve_tag(client, TAG_PROCESSING)
        done_id       = await resolve_tag(client, TAG_DONE)

        r = await client.get("/api/documents/", params={
            "tags__id__none": f"{processing_id},{done_id}",
            "page_size": PAGE_SIZE,
            "ordering": "created",
        })
        r.raise_for_status()
        docs = r.json()["results"]

        log.info("found %d pending doc(s)", len(docs))
        for doc in docs:
            doc_id   = doc["id"]
            cur_tags = list(doc["tags"])
            await client.patch(f"/api/documents/{doc_id}/", json={"tags": cur_tags + [processing_id]})

            try:
                async with httpx.AsyncClient(timeout=30.0) as n8n:
                    resp = await n8n.post(N8N_WEBHOOK_URL, json={
                        "document_id": doc_id,
                        "title": doc.get("title", ""),
                    })
                    resp.raise_for_status()
                log.info("doc %d fired", doc_id)
            except Exception as exc:
                log.error("doc %d failed: %s — releasing tag", doc_id, exc)
                await client.patch(f"/api/documents/{doc_id}/", json={"tags": cur_tags})


if __name__ == "__main__":
    asyncio.run(main())