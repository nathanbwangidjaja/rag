from app.graph.knowledge_graph import KnowledgeGraph
from app.search.vector_store import VectorStore
from app.config import GRAPH_MAX_HOPS, GRAPH_MAX_RESULTS


def graph_search(
    entity_names: list[str],
    kg: KnowledgeGraph,
    chunk_store: VectorStore,
    max_hops: int = GRAPH_MAX_HOPS,
    max_results: int = GRAPH_MAX_RESULTS,
) -> list[dict]:
    """
    Traverse the knowledge graph from matched entities,
    collect related chunks, return them scored by hop distance.
    """
    chunk_scores = kg.get_related_chunk_ids(entity_names, max_hops)

    if not chunk_scores:
        return []

    # look up actual chunk dicts from the vector store
    results = []
    for chunk in chunk_store.chunks:
        cid = chunk["id"]
        if cid in chunk_scores:
            c = chunk.copy()
            c["graph_score"] = chunk_scores[cid]
            results.append(c)

    # sort by graph score descending
    results.sort(key=lambda c: c["graph_score"], reverse=True)

    return results[:max_results]
