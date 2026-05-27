import asyncio
import httpx

PAPERLESS_URL = "http://localhost:37671"
PAPERLESS_TOKEN = "e2d832d425784d4e3a42aa023ff762132c032d39"
TAG_PROCESSING = "ocr-processing"
TAG_DONE = "ocr-done"
PAGE_SIZE = 10


async def resolve_tag(client: httpx.AsyncClient, name: str) -> int:
    """Resolve a tag name to its ID; create if not found."""
    r = await client.get("/api/tags/", params={"name__iexact": name})
    r.raise_for_status()
    for t in r.json()["results"]:
        if t["name"].lower() == name.lower():
            print(f"✓ Found tag '{name}' with ID {t['id']}")
            return t["id"]
    
    # Tag not found, create it
    r = await client.post("/api/tags/", json={"name": name})
    r.raise_for_status()
    tag_id = r.json()["id"]
    print(f"✓ Created tag '{name}' with ID {tag_id}")
    return tag_id


async def test_paperless_api():
    """Test Paperless API calls: resolve tags and query documents."""
    headers = {"Authorization": f"Token {PAPERLESS_TOKEN}"}
    
    async with httpx.AsyncClient(base_url=PAPERLESS_URL, headers=headers, timeout=30.0) as client:
        # Step 1: Resolve tags
        print("\n1. Resolving tags...")
        processing_id = await resolve_tag(client, TAG_PROCESSING)
        done_id = await resolve_tag(client, TAG_DONE)
        
        print(f"\n  TAG_PROCESSING ID: {processing_id}")
        print(f"  TAG_DONE ID: {done_id}")
        
        # Step 2: Query documents with processing tag but NOT done tag
        print(f"\n2. Querying documents with tag {TAG_PROCESSING} and without tag {TAG_DONE}...")
        r = await client.get("/api/documents/", params={
            "tags__id__all": f"{processing_id}",
            "tags__id__none": f"{done_id}",
            "page_size": PAGE_SIZE,
            "ordering": "created",
        })
        r.raise_for_status()
        
        data = r.json()
        docs = data["results"]
        total = data.get("count", len(docs))
        
        print(f"\n✓ Query successful!")
        print(f"  Total documents: {total}")
        print(f"  Returned: {len(docs)}")
        
        # Step 3: Display results
        if docs:
            print(f"\n3. Documents found ({len(docs)}):")
            for i, doc in enumerate(docs, 1):
                doc_id = doc["id"]
                title = doc.get("title", "(no title)")
                tags = doc.get("tags", [])
                print(f"   [{i}] ID: {doc_id}, Title: {title}, Tags: {tags}")
        else:
            print("\n⚠ No documents found!")
        
        # Assertion check
        assert len(docs) == 9, f"Expected 9 documents, got {len(docs)}"
        print(f"\n✅ PASS: Got exactly 9 documents as expected!")


if __name__ == "__main__":
    asyncio.run(test_paperless_api())
