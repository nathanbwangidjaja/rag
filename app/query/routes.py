from fastapi import APIRouter
from pydantic import BaseModel
from app.query.intent import detect_intent
from app.query.transform import transform_query
from app.search.embeddings import get_single_embedding
from app.search.vector_store import store
from app.search.bm25 import bm25_index
from app.search.ranker import merge_and_rank
from app.generation.llm import chat
from app.generation.prompts import (
    build_prompt, CASUAL_RESPONSES, REFUSE_TEMPLATE,
    REFUSE_REASONS, INSUFFICIENT_EVIDENCE,
)
from app.config import TOP_K

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    intent: str
    sources: list[dict] = []
    transformed_query: str | None = None


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    question = req.question.strip()

    # step 1: figure out what they're asking
    intent = await detect_intent(question)

    # casual messages — no retrieval needed
    if intent == "casual":
        q_lower = question.lower()
        if any(w in q_lower for w in ["hi", "hello", "hey", "sup"]):
            msg = CASUAL_RESPONSES["greeting"]
        elif any(w in q_lower for w in ["thank", "thanks", "thx"]):
            msg = CASUAL_RESPONSES["thanks"]
        else:
            msg = CASUAL_RESPONSES["default"]
        return QueryResponse(answer=msg, intent=intent)

    # refused queries — PII, legal, medical
    if intent == "refuse":
        q_lower = question.lower()
        if any(w in q_lower for w in ["ssn", "social security", "credit card"]):
            reason = REFUSE_REASONS["pii"]
        elif any(w in q_lower for w in ["legal", "lawyer", "sue", "lawsuit"]):
            reason = REFUSE_REASONS["legal"]
        elif any(w in q_lower for w in ["diagnos", "symptom", "medical", "prescri"]):
            reason = REFUSE_REASONS["medical"]
        else:
            reason = REFUSE_REASONS["default"]
        return QueryResponse(
            answer=REFUSE_TEMPLATE.format(reason=reason),
            intent=intent,
        )

    # step 2: rewrite the query for better retrieval
    transformed = await transform_query(question)

    # step 3: run both searches
    query_embedding = await get_single_embedding(transformed)
    semantic_results = store.search(query_embedding, top_k=TOP_K * 2)
    keyword_results = bm25_index.search(transformed, top_k=TOP_K * 2)

    # step 4: merge and rank
    ranked = merge_and_rank(semantic_results, keyword_results, top_k=TOP_K)

    # nothing good enough? say so
    if not ranked:
        return QueryResponse(
            answer=INSUFFICIENT_EVIDENCE,
            intent=intent,
            transformed_query=transformed,
        )

    # step 5: build prompt and generate
    messages = build_prompt(intent, question, ranked)
    answer = await chat(messages)

    # collect sources for the response
    sources = []
    seen = set()
    for chunk in ranked:
        key = (chunk.get("source"), chunk.get("page"))
        if key not in seen:
            sources.append({"source": chunk.get("source"), "page": chunk.get("page")})
            seen.add(key)

    return QueryResponse(
        answer=answer,
        intent=intent,
        sources=sources,
        transformed_query=transformed,
    )
