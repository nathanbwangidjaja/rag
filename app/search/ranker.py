"""RRF to merge semantic + keyword results. No tuned weights, just rank position."""

from app.config import SIMILARITY_THRESHOLD

RRF_K = 60


def reciprocal_rank_fusion(*result_lists: list[dict]) -> list[dict]:
    """Fuse N ranked lists. Score = sum of 1/(k + rank) for each list."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, fused_score in ranked:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = fused_score
        results.append(chunk)

    return results


def deduplicate(results: list[dict], overlap_ratio=0.7) -> list[dict]:
    """Drop near-dupes from overlapping chunks. Keep higher-ranked."""
    kept = []
    seen_texts = []

    for chunk in results:
        text = chunk["text"]
        is_dup = False

        for prev in seen_texts:
            words_a = set(text.lower().split())
            words_b = set(prev.lower().split())
            if not words_a or not words_b:
                continue
            shared = len(words_a & words_b)
            # need both a high ratio AND enough absolute overlap
            # otherwise short chunks with a few common words get falsely flagged
            ratio = shared / min(len(words_a), len(words_b))
            if ratio > overlap_ratio and shared >= 10:
                is_dup = True
                break

        if not is_dup:
            kept.append(chunk)
            seen_texts.append(text)

    return kept


def filter_by_threshold(results: list[dict], threshold=None) -> list[dict]:
    """Drop chunks below similarity threshold. Uses 'score' field."""
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    filtered = []
    for chunk in results:
        # use semantic score if available
        sem_score = chunk.get("score", 1.0)
        if sem_score >= threshold:
            filtered.append(chunk)

    return filtered


def merge_and_rank(
    semantic_results: list[dict],
    keyword_results: list[dict],
    graph_results: list[dict] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Fuse, dedupe, filter, trim to top_k."""
    lists = [semantic_results, keyword_results]
    if graph_results:
        lists.append(graph_results)
    fused = reciprocal_rank_fusion(*lists)
    deduped = deduplicate(fused)
    filtered = filter_by_threshold(deduped)
    return filtered[:top_k]
