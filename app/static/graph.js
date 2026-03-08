const TYPE_COLORS = {
    PERSON: { bg: "#3b82f6", text: "white" },
    ORG: { bg: "#22c55e", text: "white" },
    CONCEPT: { bg: "#a855f7", text: "white" },
    LOCATION: { bg: "#f97316", text: "white" },
    DATE: { bg: "#6b7280", text: "white" },
    METRIC: { bg: "#14b8a6", text: "white" },
    DOCUMENT: { bg: "#ef4444", text: "white" },
};

let network = null;
let allNodes = [];
let allEdges = [];

// --- init ---

async function init() {
    const statsResp = await fetch("/graph/stats");
    const stats = await statsResp.json();
    document.getElementById("stats-text").textContent =
        stats.nodes + " nodes · " + stats.edges + " edges";

    const dataResp = await fetch("/graph/data");
    const data = await dataResp.json();

    allNodes = data.nodes;
    allEdges = data.edges;

    renderLegend();
    renderGraph(data);
}

function renderLegend() {
    const legend = document.getElementById("legend");
    const seen = new Set();
    for (const n of allNodes) {
        if (!seen.has(n.type)) {
            seen.add(n.type);
            const color = TYPE_COLORS[n.type] || TYPE_COLORS.CONCEPT;
            const dot = document.createElement("div");
            dot.className = "flex items-center gap-1";
            dot.innerHTML = `
                <span class="h-2.5 w-2.5 rounded-full" style="background:${color.bg}"></span>
                <span class="text-[10px] font-medium text-slate-600">${n.type}</span>
            `;
            legend.appendChild(dot);
        }
    }
}

function renderGraph(data) {
    const container = document.getElementById("graph-canvas");

    const nodes = new vis.DataSet(
        data.nodes
            .filter((n) => n.type !== "DOCUMENT")
            .map((n) => {
                const color = TYPE_COLORS[n.type] || TYPE_COLORS.CONCEPT;
                const size = Math.max(10, Math.min(40, n.mentions * 5));
                return {
                    id: n.id,
                    label: n.label,
                    color: {
                        background: color.bg,
                        border: color.bg,
                        highlight: { background: color.bg, border: "#1e40af" },
                    },
                    font: { color: "#334155", size: 11, face: "Inter" },
                    size: size,
                    shape: "dot",
                    title: n.description || n.label,
                    _type: n.type,
                    _description: n.description,
                    _mentions: n.mentions,
                };
            })
    );

    const edges = new vis.DataSet(
        data.edges.map((e) => ({
            id: e.id,
            from: e.from,
            to: e.to,
            label: e.label,
            arrows: "to",
            color: { color: "#cbd5e1", highlight: "#3b82f6" },
            font: { size: 9, color: "#94a3b8", face: "Inter", strokeWidth: 0 },
            smooth: { type: "curvedCW", roundness: 0.15 },
        }))
    );

    const options = {
        physics: {
            solver: "forceAtlas2Based",
            forceAtlas2Based: {
                gravitationalConstant: -40,
                centralGravity: 0.005,
                springLength: 150,
                springConstant: 0.02,
                damping: 0.4,
            },
            stabilization: { iterations: 200 },
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            zoomView: true,
            dragView: true,
        },
        nodes: {
            borderWidth: 2,
            shadow: { enabled: true, size: 5, color: "rgba(0,0,0,0.1)" },
        },
        edges: {
            width: 1,
            selectionWidth: 2,
        },
    };

    network = new vis.Network(container, { nodes, edges }, options);

    network.on("click", (params) => {
        if (params.nodes.length > 0) {
            showDetail(params.nodes[0]);
        } else {
            closeDetail();
        }
    });
}

// --- detail panel ---

async function showDetail(nodeId) {
    const panel = document.getElementById("detail-panel");
    panel.classList.remove("hidden");

    // fetch subgraph for this entity
    const resp = await fetch("/graph/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entity: nodeId, max_hops: 1 }),
    });
    const data = await resp.json();

    const node = data.nodes.find((n) => n.id === nodeId);
    if (!node) return;

    const color = TYPE_COLORS[node.type] || TYPE_COLORS.CONCEPT;

    document.getElementById("detail-name").textContent = node.label;

    const typeEl = document.getElementById("detail-type");
    typeEl.textContent = node.type;
    typeEl.style.backgroundColor = color.bg;
    typeEl.style.color = color.text;

    document.getElementById("detail-desc").textContent = node.description || "No description";
    document.getElementById("detail-mentions").textContent =
        node.mentions + " mention" + (node.mentions !== 1 ? "s" : "") +
        " across " + data.chunk_count + " chunk" + (data.chunk_count !== 1 ? "s" : "");

    const connEl = document.getElementById("detail-connections");
    connEl.innerHTML = "";

    for (const edge of data.edges) {
        const isOutgoing = edge.from === nodeId;
        const otherName = isOutgoing ? edge.target_name : edge.source_name;
        const otherId = isOutgoing ? edge.to : edge.from;
        const direction = isOutgoing ? "->" : "<-";

        const div = document.createElement("div");
        div.className =
            "flex items-center gap-2 text-xs rounded-lg bg-slate-50 border border-slate-100 px-3 py-2 cursor-pointer hover:bg-primary/5 hover:border-primary/20 transition-colors";
        div.innerHTML = `
            <span class="text-slate-400">${direction}</span>
            <span class="font-medium text-slate-700">${escapeHtml(otherName)}</span>
            <span class="text-slate-400 ml-auto">${escapeHtml(edge.label)}</span>
        `;
        div.onclick = () => {
            network.selectNodes([otherId]);
            network.focus(otherId, { scale: 1.2, animation: true });
            showDetail(otherId);
        };
        connEl.appendChild(div);
    }
}

function closeDetail() {
    document.getElementById("detail-panel").classList.add("hidden");
}

// --- search ---

const searchInput = document.getElementById("search-input");
searchInput.addEventListener("input", () => {
    const query = searchInput.value.toLowerCase().trim();
    if (!query || !network) return;

    const match = allNodes.find(
        (n) => n.label.toLowerCase().includes(query) && n.type !== "DOCUMENT"
    );
    if (match) {
        network.selectNodes([match.id]);
        network.focus(match.id, { scale: 1.5, animation: true });
        showDetail(match.id);
    }
});

searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        const query = searchInput.value.toLowerCase().trim();
        if (!query || !network) return;

        const match = allNodes.find(
            (n) => n.label.toLowerCase().includes(query) && n.type !== "DOCUMENT"
        );
        if (match) {
            network.selectNodes([match.id]);
            network.focus(match.id, { scale: 1.5, animation: true });
            showDetail(match.id);
        }
    }
});

// --- helpers ---

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

init();
