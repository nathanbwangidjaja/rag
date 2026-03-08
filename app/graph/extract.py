import re
import json
from app.generation.llm import chat


EXTRACTION_PROMPT = """Extract entities and relationships from this text. Return valid JSON only.

Rules:
- Entity types: PERSON, ORG, CONCEPT, DATE, LOCATION, METRIC, DOCUMENT
- Normalize names: full names preferred, consistent casing
- Only extract relationships that are explicitly stated, not inferred
- Keep relationship labels short and verb-based (e.g., "manages", "located_in", "part_of")

Text:
{text}

Source: {source}, Page {page}

Return this exact format:
{{
  "entities": [
    {{"name": "...", "type": "PERSON|ORG|CONCEPT|...", "description": "one sentence"}}
  ],
  "relationships": [
    {{"source": "entity_name_1", "target": "entity_name_2", "label": "verb_phrase"}}
  ]
}}

JSON only, no explanation."""


async def extract_entities_and_relationships(chunk: dict) -> dict:
    """Extract entities and relationships from a single chunk via LLM."""
    prompt = EXTRACTION_PROMPT.format(
        text=chunk["text"],
        source=chunk.get("source", "unknown"),
        page=chunk.get("page", 0),
    )

    raw = await chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )

    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```\w*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)

    parsed = json.loads(raw)

    entities = parsed.get("entities", [])
    relationships = parsed.get("relationships", [])

    return {"entities": entities, "relationships": relationships}
