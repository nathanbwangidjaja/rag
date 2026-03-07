import httpx
from app.config import MISTRAL_API_KEY, MISTRAL_BASE_URL, CHAT_MODEL


async def chat(messages: list[dict], temperature=0.3, max_tokens=1024) -> str:
    """Send messages to Mistral chat and return the response text."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{MISTRAL_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"].strip()
