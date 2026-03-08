# prompt templates for knowledge/list/summary intents


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source", "unknown")
        page = c.get("page", "?")
        parts.append(f"[{i}] (Source: {source}, p.{page})\n{c['text']}")
    return "\n\n".join(parts)


KNOWLEDGE_TEMPLATE = """Answer the user's question using ONLY the context below. If the context doesn't contain enough information, say so. Don't make things up.

Cite your sources inline like [Source: filename.pdf, p.X] when you reference specific information.

Context:
{context}

Question: {question}

Answer:"""


LIST_TEMPLATE = """Based on the context below, provide a structured answer to the user's question. Use bullet points or a table where appropriate.

Only use information from the provided context. Cite sources inline like [Source: filename.pdf, p.X].

Context:
{context}

Question: {question}

Answer:"""


SUMMARY_TEMPLATE = """Summarize the relevant information from the context below to address the user's question. Cover the key points without unnecessary detail.

Only use information from the provided context. Cite sources inline like [Source: filename.pdf, p.X].

Context:
{context}

Question: {question}

Summary:"""


CASUAL_RESPONSES = {
    "greeting": "Hey! I'm here to help you find information in your uploaded documents. Ask me anything about them.",
    "thanks": "No problem! Let me know if you have more questions.",
    "default": "I'm a document Q&A assistant. Upload some PDFs and ask me questions about them!",
}


REFUSE_TEMPLATE = """I can't help with that. {reason}

If you have questions about your uploaded documents, I'm happy to help with those instead."""

REFUSE_REASONS = {
    "pii": "I don't process or store personal information like SSNs, credit card numbers, or similar data.",
    "legal": "I'm not qualified to give legal advice. Please consult a licensed attorney.",
    "medical": "I'm not qualified to give medical advice. Please consult a healthcare professional.",
    "default": "That type of request is outside what I can help with.",
}


INSUFFICIENT_EVIDENCE = "I couldn't find enough relevant information in the uploaded documents to answer that. Try rephrasing your question, or make sure the right files have been ingested."


def _format_graph_context(relationships: list[dict]) -> str:
    lines = []
    for rel in relationships:
        lines.append(f"- {rel['source']} --[{rel['label']}]--> {rel['target']}")
    return "\n".join(lines)


def build_prompt(
    intent: str,
    question: str,
    chunks: list[dict],
    graph_context: list[dict] | None = None,
) -> list[dict]:

    if intent == "casual":
        return None

    if intent == "refuse":
        return None

    context = _format_context(chunks)

    # inject graph relationship context when available
    if graph_context:
        rel_text = _format_graph_context(graph_context)
        context += f"\n\nEntity Relationships (from knowledge graph):\n{rel_text}"

    if intent == "list":
        template = LIST_TEMPLATE
    elif intent == "summary":
        template = SUMMARY_TEMPLATE
    else:
        template = KNOWLEDGE_TEMPLATE

    prompt = template.format(context=context, question=question)

    return [
        {"role": "system", "content": "You are a helpful document Q&A assistant. You only answer based on provided context and always cite your sources."},
        {"role": "user", "content": prompt},
    ]
