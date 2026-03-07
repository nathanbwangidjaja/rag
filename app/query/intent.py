import httpx
from app.config import MISTRAL_API_KEY, MISTRAL_BASE_URL, CHAT_MODEL

INTENT_PROMPT = """Classify the user message into exactly one intent. Reply with ONLY the intent label, nothing else.

Intents:
- "knowledge" : user is asking a factual question that needs document lookup
- "list" : user wants a list, comparison, or table from the documents
- "summary" : user wants a summary or overview of document content
- "casual" : greetings, thanks, small talk, not a real question
- "refuse" : user is asking for legal advice, medical advice, or sharing PII like SSNs or credit card numbers

User message: {query}

Intent:"""

VALID_INTENTS = {"knowledge", "list", "summary", "casual", "refuse"}


async def detect_intent(query: str) -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "user", "content": INTENT_PROMPT.format(query=query)}
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{MISTRAL_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"].strip().lower()

    # strip quotes
    raw = raw.strip('"').strip("'")

    if raw in VALID_INTENTS:
        return raw

    # default to knowledge
    return "knowledge"
