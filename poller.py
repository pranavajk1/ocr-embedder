import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("poller")

PAPERLESS_URL   = os.environ["PAPERLESS_URL"]
PAPERLESS_TOKEN = os.environ["PAPERLESS_API_TOKEN"]
N8N_TOKEN       = os.environ["N8N_WEBHOOK_TOKEN"]
N8N_WEBHOOK_URL = os.environ["N8N_WEBHOOK_URL"]
TAG_TRIGGER     = os.environ.get("TAG_PROCESSING", "ocr-processing")
TAG_INFLIGHT    = os.environ.get("TAG_INFLIGHT", "ocr-inflight")
TAG_DONE        = os.environ.get("TAG_DONE", "ocr-done")
PAGE_SIZE       = int(os.environ.get("POLL_PAGE_SIZE", "10"))
STALE_INFLIGHT_MINUTES = int(os.environ.get("STALE_INFLIGHT_MINUTES", "30"))


async def resolve_tag(client: httpx.AsyncClient, name: str) -> int:
    r = await client.get("/api/tags/", params={"name__iexact": name})
    r.raise_for_status()
    for t in r.json()["results"]:
        if t["name"].lower() == name.lower():
            return t["id"]
    r = await client.post("/api/tags/", json={"name": name})
    r.raise_for_status()
    return r.json()["id"]


async def sweep_stale_inflight(client: httpx.AsyncClient, inflight_id: int) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_INFLIGHT_MINUTES)
    r = await client.get("/api/documents/", params={
        "tags__id__all": f"{inflight_id}",
        "page_size": 100,
    })
    r.raise_for_status()

    released = 0
    for doc in r.json()["results"]:
        modified_str = doc.get("modified")
        if not modified_str:
            continue
        modified = datetime.fromisoformat(modified_str.replace("Z", "+00:00"))
        if modified > cutoff:
            continue
        new_tags = [t for t in doc["tags"] if t != inflight_id]
        await client.patch(f"/api/documents/{doc['id']}/", json={"tags": new_tags})
        log.warning("released stale inflight from doc %d (modified %s)", doc["id"], modified_str)
        released += 1

    if released:
        log.info("sweep released %d stale inflight tag(s)", released)


async def main() -> None:
    headers = {"Authorization": f"Token {PAPERLESS_TOKEN}"}
    async with httpx.AsyncClient(base_url=PAPERLESS_URL, headers=headers, timeout=30.0) as client:
        trigger_id  = await resolve_tag(client, TAG_TRIGGER)
        inflight_id = await resolve_tag(client, TAG_INFLIGHT)
        done_id     = await resolve_tag(client, TAG_DONE)

        await sweep_stale_inflight(client, inflight_id)

        r = await client.get("/api/documents/", params={
            "tags__id__all": f"{trigger_id}",
            "tags__id__none": f"{done_id},{inflight_id}",
            "page_size": PAGE_SIZE,
            "ordering": "created",
        })
        r.raise_for_status()
        docs = r.json()["results"]

        log.info("found %d pending doc(s)", len(docs))
        for doc in docs:
            doc_id   = doc["id"]
            cur_tags = list(doc["tags"])
            if inflight_id in cur_tags:
                continue
            await client.patch(
                f"/api/documents/{doc_id}/",
                json={"tags": cur_tags + [inflight_id]},
            )

            try:
                async with httpx.AsyncClient(timeout=30.0) as n8n:
                    resp = await n8n.post(N8N_WEBHOOK_URL, json={
                        "document_id": doc_id,
                        "title": doc.get("title", ""),
                    }, headers={
                        "X-Webhook-Secret": N8N_TOKEN
                    })
                    resp.raise_for_status()
                log.info("doc %d fired", doc_id)
            except Exception as exc:
                log.error("doc %d failed: %s — releasing inflight tag", doc_id, exc)
                await client.patch(
                    f"/api/documents/{doc_id}/",
                    json={"tags": cur_tags},
                )


if __name__ == "__main__":
    asyncio.run(main())
