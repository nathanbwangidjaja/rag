from fastapi import APIRouter
from pydantic import BaseModel
from app.graph.knowledge_graph import graph

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/stats")
def graph_stats():
    """Node/edge counts and top entities by mention count."""
    counts = graph.count()

    top_entities = sorted(
        [
            {"name": n["name"], "type": n["type"], "mentions": n["mentions"]}
            for n in graph.nodes.values()
            if n["type"] != "DOCUMENT"
        ],
        key=lambda x: x["mentions"],
        reverse=True,
    )[:20]

    return {
        "nodes": counts["nodes"],
        "edges": counts["edges"],
        "top_entities": top_entities,
    }


@router.get("/data")
def graph_data():
    """Full graph in vis-network format for the UI."""
    return graph.to_vis_format()


class GraphSearchRequest(BaseModel):
    entity: str
    max_hops: int = 2


@router.post("/search")
def graph_search_endpoint(req: GraphSearchRequest):
    """Subgraph around a specific entity."""
    neighbors = graph.get_neighbors(req.entity, max_hops=req.max_hops)

    if not neighbors:
        return {"nodes": [], "edges": [], "chunks": []}

    # collect node keys in the subgraph
    node_keys = {graph._normalize(n["name"]) for n in neighbors}

    # filter edges to only those within the subgraph
    sub_edges = []
    for i, edge in enumerate(graph.edges):
        if edge["source"] in node_keys and edge["target"] in node_keys:
            src_node = graph.nodes.get(edge["source"], {})
            tgt_node = graph.nodes.get(edge["target"], {})
            sub_edges.append({
                "id": i,
                "from": edge["source"],
                "to": edge["target"],
                "label": edge["label"],
                "weight": edge["weight"],
                "source_name": src_node.get("name", edge["source"]),
                "target_name": tgt_node.get("name", edge["target"]),
            })

    sub_nodes = [
        {
            "id": graph._normalize(n["name"]),
            "label": n["name"],
            "type": n["type"],
            "description": n.get("description", ""),
            "mentions": n["mentions"],
            "hop": n.get("hop", 0),
        }
        for n in neighbors
    ]

    # collect unique chunk ids referenced by these nodes
    chunk_ids = set()
    for n in neighbors:
        for cid in n.get("chunk_ids", []):
            chunk_ids.add(cid)

    return {
        "nodes": sub_nodes,
        "edges": sub_edges,
        "chunk_count": len(chunk_ids),
    }
