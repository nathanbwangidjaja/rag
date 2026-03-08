import os
import json
from collections import deque
from app.config import DATA_DIR


class KnowledgeGraph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.adjacency: dict[str, list[int]] = {}
        self._load()

    @staticmethod
    def _normalize(name: str) -> str:
        return " ".join(name.lower().strip().split())

    def add_extraction(self, chunk_id: str, entities: list[dict], relationships: list[dict]):
        for ent in entities:
            key = self._normalize(ent.get("name", ""))
            if not key:
                continue

            if key in self.nodes:
                node = self.nodes[key]
                if chunk_id not in node["chunk_ids"]:
                    node["chunk_ids"].append(chunk_id)
                node["mentions"] += 1
                # keep the longer description
                desc = ent.get("description", "")
                if len(desc) > len(node.get("description", "")):
                    node["description"] = desc
            else:
                self.nodes[key] = {
                    "name": ent.get("name", key),
                    "type": ent.get("type", "CONCEPT"),
                    "description": ent.get("description", ""),
                    "chunk_ids": [chunk_id],
                    "mentions": 1,
                }
                self.adjacency[key] = []

        for rel in relationships:
            src = self._normalize(rel.get("source", ""))
            tgt = self._normalize(rel.get("target", ""))
            label = rel.get("label", "related_to")

            if not src or not tgt:
                continue
            if src not in self.nodes or tgt not in self.nodes:
                continue

            # check if this edge already exists
            existing = None
            for idx in self.adjacency.get(src, []):
                e = self.edges[idx]
                if e["target"] == tgt and e["label"] == label:
                    existing = e
                    break

            if existing:
                existing["weight"] += 1
                if chunk_id not in existing["chunk_ids"]:
                    existing["chunk_ids"].append(chunk_id)
            else:
                edge_idx = len(self.edges)
                edge = {
                    "source": src,
                    "target": tgt,
                    "label": label,
                    "chunk_ids": [chunk_id],
                    "weight": 1,
                }
                self.edges.append(edge)
                self.adjacency.setdefault(src, []).append(edge_idx)
                self.adjacency.setdefault(tgt, []).append(edge_idx)

        self._save()

    def get_node(self, name: str) -> dict | None:
        return self.nodes.get(self._normalize(name))

    def get_neighbors(self, name: str, max_hops: int = 2) -> list[dict]:
        """BFS from a node, returns visited nodes within max_hops."""
        start = self._normalize(name)
        if start not in self.nodes:
            return []

        visited = {}
        queue = deque([(start, 0)])
        visited[start] = 0

        while queue:
            current, hop = queue.popleft()
            if hop >= max_hops:
                continue

            for edge_idx in self.adjacency.get(current, []):
                edge = self.edges[edge_idx]
                neighbor = edge["target"] if edge["source"] == current else edge["source"]

                if neighbor not in visited:
                    visited[neighbor] = hop + 1
                    queue.append((neighbor, hop + 1))

        results = []
        for node_key, hop in visited.items():
            node = self.nodes[node_key].copy()
            node["hop"] = hop
            results.append(node)

        return results

    def get_related_chunk_ids(self, entity_names: list[str], max_hops: int = 2) -> dict[str, float]:
        """Traverse from entities, return {chunk_id: score} where closer = higher score."""
        chunk_scores: dict[str, float] = {}

        for name in entity_names:
            neighbors = self.get_neighbors(name, max_hops)
            for node in neighbors:
                hop = node["hop"]
                score = 1.0 / (1 + hop)  # hop 0 = 1.0, hop 1 = 0.5, hop 2 = 0.33
                for cid in node["chunk_ids"]:
                    chunk_scores[cid] = max(chunk_scores.get(cid, 0), score)

        return chunk_scores

    def get_relationships_for_entities(self, entity_names: list[str]) -> list[dict]:
        """Get all edges connected to the given entities (1 hop)."""
        normalized = {self._normalize(n) for n in entity_names}
        results = []
        seen = set()

        for name in normalized:
            for edge_idx in self.adjacency.get(name, []):
                if edge_idx not in seen:
                    seen.add(edge_idx)
                    edge = self.edges[edge_idx]
                    # use display names
                    src_node = self.nodes.get(edge["source"], {})
                    tgt_node = self.nodes.get(edge["target"], {})
                    results.append({
                        "source": src_node.get("name", edge["source"]),
                        "target": tgt_node.get("name", edge["target"]),
                        "label": edge["label"],
                    })

        return results

    def to_vis_format(self) -> dict:
        """Format for vis-network rendering."""
        vis_nodes = []
        for key, node in self.nodes.items():
            vis_nodes.append({
                "id": key,
                "label": node["name"],
                "type": node["type"],
                "description": node.get("description", ""),
                "mentions": node["mentions"],
                "chunks": len(node["chunk_ids"]),
            })

        vis_edges = []
        for i, edge in enumerate(self.edges):
            vis_edges.append({
                "id": i,
                "from": edge["source"],
                "to": edge["target"],
                "label": edge["label"],
                "weight": edge["weight"],
            })

        return {"nodes": vis_nodes, "edges": vis_edges}

    def count(self) -> dict:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}

    def clear(self):
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
        self._save()

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, "graph_nodes.json"), "w") as f:
            json.dump(self.nodes, f)
        with open(os.path.join(DATA_DIR, "graph_edges.json"), "w") as f:
            json.dump(self.edges, f)

    def _load(self):
        nodes_path = os.path.join(DATA_DIR, "graph_nodes.json")
        edges_path = os.path.join(DATA_DIR, "graph_edges.json")

        if os.path.exists(nodes_path):
            with open(nodes_path) as f:
                self.nodes = json.load(f)
        if os.path.exists(edges_path):
            with open(edges_path) as f:
                self.edges = json.load(f)

        # rebuild adjacency from loaded edges
        self.adjacency = {}
        for key in self.nodes:
            self.adjacency[key] = []
        for idx, edge in enumerate(self.edges):
            self.adjacency.setdefault(edge["source"], []).append(idx)
            self.adjacency.setdefault(edge["target"], []).append(idx)


graph = KnowledgeGraph()
