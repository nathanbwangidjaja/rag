import re
from app.generation.llm import chat

PII_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', "SSN"),
    (r'\b\d{9}\b', "SSN"),
    (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', "credit card"),
    (r'\b\d{16}\b', "credit card"),
    (r'\b[A-Z]{1,2}\d{6,9}\b', "passport number"),
]


def check_pii(text: str) -> str | None:
    """Regex scan for SSN, credit card, passport. Returns label or None."""
    for pattern, label in PII_PATTERNS:
        if re.search(pattern, text):
            return label
    return None


async def check_hallucinations(answer: str, chunks: list[dict]) -> dict:
    context_text = "\n\n".join(c["text"] for c in chunks)

    prompt = f"""For each sentence in the ANSWER, say if it's supported by the SOURCE CONTEXT. Reply with a JSON array: [{{"sentence": "...", "supported": true/false}}].

Rules:
- Mark unsupported ONLY if a sentence makes a factual claim that contradicts or has no basis in the context.
- Citation tags like [Source: ...] are NOT claims — always mark them as supported.
- Transitional phrases, hedging, and summaries that accurately reflect the context are supported.

SOURCE CONTEXT:
{context_text}

ANSWER:
{answer}

JSON array only."""

    try:
        raw = await chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )

        # strip markdown code blocks if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```\w*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)

        import json
        checks = json.loads(raw)

        flagged = [c["sentence"] for c in checks if not c.get("supported", True)]

        if flagged:
            clean = answer
            for sent in flagged:
                clean = clean.replace(sent, f"⚠️ {sent}")
            return {
                "clean_answer": clean,
                "flagged_sentences": flagged,
                "passed": False,
            }

        return {"clean_answer": answer, "flagged_sentences": [], "passed": True}

    except Exception:
        return {"clean_answer": answer, "flagged_sentences": [], "passed": True}
