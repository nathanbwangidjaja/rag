import httpx
from app.config import MISTRAL_API_KEY, MISTRAL_BASE_URL, CHAT_MODEL

REWRITE_PROMPT = """You are a search query optimizer. Rewrite the user's question into a better search query for finding relevant passages in a document collection.

Rules:
- Expand abbreviations and vague references
- Add relevant keywords that would appear in source documents
- Keep it concise, under 2 sentences
- Output ONLY the rewritten query, nothing else

User question: {query}

Rewritten query:"""


async def transform_query(query: str) -> str:
    """
    Rewrite the user's raw question into something that embeds better.
    Vague stuff like "what does it say about that?" becomes more specific.
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "user", "content": REWRITE_PROMPT.format(query=query)}
        ],
        "max_tokens": 128,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{MISTRAL_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()

    rewritten = resp.json()["choices"][0]["message"]["content"].strip()

    # sanity check: if the rewrite is empty or way too long, just use the original
    if not rewritten or len(rewritten) > 500:
        return query

    return rewritten
