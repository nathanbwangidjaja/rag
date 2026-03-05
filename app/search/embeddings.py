import httpx
import asyncio
from app.config import MISTRAL_API_KEY, MISTRAL_BASE_URL, EMBED_MODEL

BATCH_SIZE = 16  # mistral endpoint handles up to 16 inputs per call


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Get embeddings from Mistral for a list of texts.
    Splits into batches to respect API limits.
    """
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        emb = await _embed_batch(batch)
        all_embeddings.extend(emb)
        # small pause between batches to avoid rate limits
        if i + BATCH_SIZE < len(texts):
            await asyncio.sleep(0.2)

    return all_embeddings


async def get_single_embedding(text: str) -> list[float]:
    """Embed a single piece of text. Convenience wrapper."""
    result = await _embed_batch([text])
    return result[0]


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{MISTRAL_BASE_URL}/embeddings",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()

    data = resp.json()
    # sort by index to guarantee order matches input
    sorted_items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_items]
