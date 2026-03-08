from app.graph.knowledge_graph import KnowledgeGraph


MULTI_HOP_SIGNALS = [
    "related to", "connected to", "between", "relationship",
    "linked", "associated with", "works with", "partners of",
    "who is", "who are", "involved in", "part of",
]


def should_use_graph(query: str, kg: KnowledgeGraph) -> tuple[bool, list[str]]:
    """
    Check if the query references known entities in the graph.
    Returns (should_use, list_of_matched_entity_names).
    """
    if not kg.nodes:
        return False, []

    q_lower = query.lower()
    matched = []

    for key, node in kg.nodes.items():
        # skip very short entity names to avoid false positives (e.g. "p." or "1")
        if len(key) < 3:
            continue

        # skip the document name itself — not useful as a graph signal
        if node["type"] == "DOCUMENT":
            continue

        if key in q_lower:
            matched.append(key)

    if not matched:
        # check for multi-hop signal words even without entity matches
        has_signal = any(s in q_lower for s in MULTI_HOP_SIGNALS)
        if has_signal:
            # grab top entities by mention count as fallback
            top = sorted(
                kg.nodes.values(),
                key=lambda n: n["mentions"],
                reverse=True,
            )
            fallback = [
                kg._normalize(n["name"]) for n in top[:3]
                if n["type"] != "DOCUMENT"
            ]
            return bool(fallback), fallback

    return bool(matched), matched
